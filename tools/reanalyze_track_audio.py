from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import librosa
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path("/Users/yanwei/dev/namibox/book-res-studio/data/library/book-res")
DEFAULT_ANALYSIS_DB = REPO_ROOT / "outputs" / "audio_analysis" / "book_res_full_20260418" / "audio_features.sqlite"


@dataclass
class ProgressState:
    phase: str
    processed: int
    total: int
    elapsed_seconds: float
    rate_per_second: float
    eta_seconds: float
    output_dir: str
    message: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reanalyze only book-res track_audio files with the current RVC auto-classifier."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--analysis-db", type=Path, default=DEFAULT_ANALYSIS_DB)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sample-seconds", type=float, default=15.0)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 4) // 2)))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=1.0,
        help="Random sample ratio in (0,1]. Example: 0.05 means analyze 1/20 of track_audio.",
    )
    parser.add_argument("--sample-seed", type=int, default=20260419)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--progress-every-seconds", type=float, default=15.0)
    return parser.parse_args(argv)


def timestamped_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "outputs" / "audio_analysis" / f"track_audio_reanalysis_{stamp}"


def create_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reanalysis (
            rel_path TEXT PRIMARY KEY,
            abs_path TEXT NOT NULL,
            book_id TEXT,
            duration_seconds REAL,
            analyzed_seconds REAL,
            classification TEXT,
            confidence REAL,
            dominant_route TEXT,
            music_risk REAL,
            voiced_ratio REAL,
            voiced_frames INTEGER,
            median_f0_hz REAL,
            override_reason TEXT,
            music_risk_overridden INTEGER,
            music_reasons TEXT,
            processing_mode TEXT,
            target_route TEXT,
            music_prediction_label TEXT,
            music_prediction_confidence REAL,
            gender_prediction_label TEXT,
            gender_prediction_confidence REAL,
            feature_summary TEXT,
            calibration TEXT,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS errors (
            rel_path TEXT PRIMARY KEY,
            abs_path TEXT NOT NULL,
            error_type TEXT NOT NULL,
            error_message TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()


def write_progress(output_dir: Path, state: ProgressState) -> None:
    payload = {
        "phase": state.phase,
        "processed": state.processed,
        "total": state.total,
        "elapsed_seconds": round(state.elapsed_seconds, 1),
        "rate_per_second": round(state.rate_per_second, 3),
        "eta_seconds": round(state.eta_seconds, 1),
        "output_dir": state.output_dir,
        "message": state.message,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (output_dir / "progress.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    ts = datetime.now().isoformat(timespec="seconds")
    line = (
        f"[{ts}] {state.phase}: {state.processed}/{state.total} | "
        f"elapsed={state.elapsed_seconds:.1f}s | rate={state.rate_per_second:.2f}/s | "
        f"eta={state.eta_seconds:.1f}s | {state.message}"
    )
    with open(output_dir / "progress.log", "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def iter_track_audio_from_db(analysis_db: Path) -> list[dict]:
    conn = sqlite3.connect(analysis_db)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT rel_path, abs_path, book_id, duration_seconds
            FROM files
            WHERE audio_kind = 'track_audio'
            ORDER BY rel_path
            """
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def iter_track_audio_from_fs(root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if "/track_audio/" not in path.as_posix():
            continue
        if path.suffix.lower() not in {".mp3", ".wav", ".m4a", ".flac", ".aac"}:
            continue
        rel_path = path.relative_to(root).as_posix()
        parts = Path(rel_path).parts
        rows.append(
            {
                "rel_path": rel_path,
                "abs_path": str(path.resolve()),
                "book_id": parts[0] if parts else "",
                "duration_seconds": 0.0,
            }
        )
    rows.sort(key=lambda item: item["rel_path"])
    return rows


def normalize_result(item: dict, analyzed_seconds: float, analysis: dict, plan: dict) -> dict:
    calibration = analysis.get("calibration") or {}
    music_prediction = calibration.get("music_prediction") or {}
    gender_prediction = calibration.get("gender_prediction") or {}
    return {
        "rel_path": item["rel_path"],
        "abs_path": item["abs_path"],
        "book_id": item.get("book_id") or "",
        "duration_seconds": float(analysis.get("duration_seconds") or item.get("duration_seconds") or 0.0),
        "analyzed_seconds": float(analyzed_seconds),
        "classification": str(analysis.get("classification") or ""),
        "confidence": float(analysis.get("confidence") or 0.0),
        "dominant_route": str(analysis.get("dominant_route") or ""),
        "music_risk": float(analysis.get("music_risk") or 0.0),
        "voiced_ratio": float(analysis.get("voiced_ratio") or 0.0),
        "voiced_frames": int(analysis.get("voiced_frames") or 0),
        "median_f0_hz": float(analysis.get("median_f0_hz") or 0.0),
        "override_reason": str(analysis.get("override_reason") or ""),
        "music_risk_overridden": 1 if analysis.get("music_risk_overridden") else 0,
        "music_reasons": json.dumps(analysis.get("music_reasons") or [], ensure_ascii=False),
        "processing_mode": str(plan.get("processing_mode") or ""),
        "target_route": str(plan.get("target_route") or ""),
        "music_prediction_label": str(music_prediction.get("label") or ""),
        "music_prediction_confidence": float(music_prediction.get("confidence") or 0.0),
        "gender_prediction_label": str(gender_prediction.get("label") or ""),
        "gender_prediction_confidence": float(gender_prediction.get("confidence") or 0.0),
        "feature_summary": json.dumps(analysis.get("feature_summary") or {}, ensure_ascii=False),
        "calibration": json.dumps(calibration, ensure_ascii=False),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def insert_result(conn: sqlite3.Connection, row: dict) -> None:
    columns = list(row.keys())
    conn.execute(
        f"""
        INSERT INTO reanalysis({",".join(columns)})
        VALUES({",".join("?" for _ in columns)})
        ON CONFLICT(rel_path) DO UPDATE SET
            {",".join(f"{col}=excluded.{col}" for col in columns if col != "rel_path")}
        """,
        [row[col] for col in columns],
    )


def insert_error(conn: sqlite3.Connection, rel_path: str, abs_path: str, exc: Exception) -> None:
    conn.execute(
        """
        INSERT INTO errors(rel_path, abs_path, error_type, error_message, updated_at)
        VALUES(?, ?, ?, ?, ?)
        ON CONFLICT(rel_path) DO UPDATE SET
            abs_path=excluded.abs_path,
            error_type=excluded.error_type,
            error_message=excluded.error_message,
            updated_at=excluded.updated_at
        """,
        (rel_path, abs_path, type(exc).__name__, str(exc), datetime.now().isoformat(timespec="seconds")),
    )


def analyze_job(item: dict, sample_seconds: float) -> tuple[str, dict]:
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from tools import rvc_auto_convert as auto

    abs_path = Path(item["abs_path"])
    duration = float(item.get("duration_seconds") or 0.0)
    if duration <= 0.0:
        duration = float(sf.info(abs_path).duration)
    y, sr = librosa.load(abs_path, sr=None, mono=True, duration=sample_seconds if sample_seconds > 0 else None)
    analyzed_seconds = min(duration, sample_seconds) if sample_seconds > 0 else duration
    feature_bundle = auto.extract_audio_feature_bundle(y, sr, duration, max_seconds=sample_seconds)

    music_risk, music_reasons = auto.estimate_music_risk_from_features(feature_bundle)
    dominant_route, voice_confidence, median_f0, voiced_frames, voiced_ratio = auto.classify_voice_from_features(
        feature_bundle
    )

    calibration_model = auto.get_sample_calibration_model()
    gender_prediction = None
    music_prediction = None
    if calibration_model:
        gender_task = calibration_model.get("gender")
        music_task = calibration_model.get("music")
        if (
            calibration_model.get("source") == "manual_sample_library"
            and int((calibration_model.get("stats") or {}).get("audio_feature_samples") or 0) == 0
        ):
            gender_task = None
        gender_label, gender_confidence, gender_scores = auto.predict_knn_label(
            auto._vector_for_calibration_task(feature_bundle, gender_task, "gender"),
            gender_task,
        )
        music_label, music_confidence, music_scores = auto.predict_knn_label(
            auto._vector_for_calibration_task(feature_bundle, music_task, "music"),
            music_task,
        )
        gender_prediction = {
            "label": gender_label,
            "confidence": round(gender_confidence, 4),
            "scores": {key: round(value, 4) for key, value in gender_scores.items()},
        }
        music_prediction = {
            "label": music_label,
            "confidence": round(music_confidence, 4),
            "scores": {key: round(value, 4) for key, value in music_scores.items()},
        }
        dominant_route, voice_confidence = auto.apply_calibrated_voice_route(
            dominant_route,
            voice_confidence,
            feature_bundle,
            gender_prediction,
        )
        music_risk, music_reasons, music_calibration = auto.apply_calibrated_music_adjustment(
            music_risk,
            music_reasons,
            feature_bundle,
            music_prediction,
        )
    else:
        music_calibration = {"used": False, "suppressed": False, "boosted": False}

    analysis = auto.build_analysis(
        duration=duration,
        music_risk=music_risk,
        music_reasons=music_reasons,
        dominant_route=dominant_route,
        voice_confidence=voice_confidence,
        median_f0=median_f0,
        voiced_frames=voiced_frames,
        voiced_ratio=voiced_ratio,
        thresholds=auto.ACTIVE_THRESHOLDS,
    )
    analysis["feature_summary"] = {
        "non_voiced_ratio": round(float(feature_bundle.get("non_voiced_ratio", 0.0)), 4),
        "rmvpe_p50": round(float(feature_bundle.get("rmvpe_p50", 0.0)), 4),
        "rmvpe_mean": round(float(feature_bundle.get("rmvpe_mean", 0.0)), 4),
        "yin_p50": round(float(feature_bundle.get("yin_p50", 0.0)), 4),
        "harmonic_ratio": round(float(feature_bundle.get("harmonic_ratio", 0.0)), 4),
        "percussive_ratio": round(float(feature_bundle.get("percussive_ratio", 0.0)), 4),
    }
    analysis["calibration"] = {
        "sample_library_used": bool(calibration_model),
        "gender_prediction": gender_prediction,
        "music_prediction": music_prediction,
        "music_adjustment": music_calibration,
    }
    plan = auto.select_processing_plan(analysis, profile=auto.DEFAULT_PROFILE)
    return "ok", normalize_result(item, analyzed_seconds, analysis, plan)


def summarize(conn: sqlite3.Connection) -> dict:
    total = conn.execute("SELECT COUNT(*) FROM reanalysis").fetchone()[0]
    errors = conn.execute("SELECT COUNT(*) FROM errors").fetchone()[0]
    by_classification = dict(
        conn.execute(
            "SELECT classification, COUNT(*) FROM reanalysis GROUP BY classification ORDER BY COUNT(*) DESC"
        ).fetchall()
    )
    by_mode = dict(
        conn.execute(
            "SELECT processing_mode, COUNT(*) FROM reanalysis GROUP BY processing_mode ORDER BY COUNT(*) DESC"
        ).fetchall()
    )
    by_route = dict(
        conn.execute(
            "SELECT dominant_route, COUNT(*) FROM reanalysis GROUP BY dominant_route ORDER BY COUNT(*) DESC"
        ).fetchall()
    )
    overridden = conn.execute(
        "SELECT COUNT(*) FROM reanalysis WHERE music_risk_overridden = 1"
    ).fetchone()[0]
    return {
        "total_files": total,
        "total_errors": errors,
        "by_classification": by_classification,
        "by_processing_mode": by_mode,
        "by_dominant_route": by_route,
        "music_risk_overridden": overridden,
    }


def write_summary(output_dir: Path, conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    summary = summarize(conn)
    summary["root"] = str(args.root)
    summary["analysis_db"] = str(args.analysis_db)
    summary["sample_seconds"] = args.sample_seconds
    summary["workers"] = args.workers
    summary["sample_ratio"] = args.sample_ratio
    summary["sample_seed"] = args.sample_seed
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Track Audio Reanalysis Summary",
        "",
        f"- Root: `{args.root}`",
        f"- Analysis DB: `{args.analysis_db}`",
        f"- Sample seconds: `{args.sample_seconds}`",
        f"- Workers: `{args.workers}`",
        f"- Sample ratio: `{args.sample_ratio}`",
        f"- Sample seed: `{args.sample_seed}`",
        f"- Reanalyzed track_audio files: `{summary['total_files']}`",
        f"- Errors: `{summary['total_errors']}`",
        f"- Music risk overridden: `{summary['music_risk_overridden']}`",
        "",
        "## Classification distribution",
    ]
    for key, value in summary["by_classification"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Processing mode distribution"])
    for key, value in summary["by_processing_mode"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Dominant route distribution"])
    for key, value in summary["by_dominant_route"].items():
        lines.append(f"- `{key}`: `{value}`")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir or timestamped_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "track_audio_reanalysis.sqlite"
    conn = sqlite3.connect(db_path)
    create_schema(conn)

    if args.analysis_db.exists():
        items = iter_track_audio_from_db(args.analysis_db)
    else:
        items = iter_track_audio_from_fs(args.root)
    if not (0.0 < args.sample_ratio <= 1.0):
        raise ValueError("--sample-ratio must be in (0, 1].")
    if args.sample_ratio < 1.0 and items:
        rng = random.Random(args.sample_seed)
        sample_size = max(1, round(len(items) * args.sample_ratio))
        items = rng.sample(items, sample_size)
        items.sort(key=lambda item: item["rel_path"])
    if args.limit > 0:
        items = items[: args.limit]

    if args.resume:
        done = {
            row[0]
            for row in conn.execute("SELECT rel_path FROM reanalysis").fetchall()
        }
        items = [item for item in items if item["rel_path"] not in done]

    total = len(items)
    start_time = time.time()
    last_progress_time = 0.0
    processed = 0
    errors = 0
    write_progress(
        output_dir,
        ProgressState(
            phase="track_reanalysis",
            processed=0,
            total=total,
            elapsed_seconds=0.0,
            rate_per_second=0.0,
            eta_seconds=0.0,
            output_dir=str(output_dir),
            message="starting track_audio reanalysis",
        ),
    )

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(analyze_job, item, args.sample_seconds): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                status, payload = future.result()
                if status == "ok":
                    insert_result(conn, payload)
                else:
                    raise RuntimeError(f"unexpected status: {status}")
            except Exception as exc:
                errors += 1
                insert_error(conn, item["rel_path"], item["abs_path"], exc)
            processed += 1
            if processed % 50 == 0:
                conn.commit()
            elapsed = max(time.time() - start_time, 1e-6)
            now = time.time()
            should_report = (
                processed == total
                or processed % max(1, args.progress_every) == 0
                or (now - last_progress_time) >= max(1.0, args.progress_every_seconds)
            )
            if should_report:
                rate = processed / elapsed
                remaining = max(total - processed, 0)
                eta = remaining / max(rate, 1e-6)
                write_progress(
                    output_dir,
                    ProgressState(
                        phase="track_reanalysis",
                        processed=processed,
                        total=total,
                        elapsed_seconds=elapsed,
                        rate_per_second=rate,
                        eta_seconds=eta,
                        output_dir=str(output_dir),
                        message=f"errors={errors}",
                    ),
                )
                last_progress_time = now

    conn.commit()
    write_summary(output_dir, conn, args)
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
