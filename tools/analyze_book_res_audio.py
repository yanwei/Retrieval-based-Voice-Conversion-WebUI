from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sqlite3
import statistics
import sys
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path("/Users/yanwei/dev/namibox/book-res-studio/data/library/book-res")
AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".flac", ".aac"}
RMVPE_ROOT = REPO_ROOT / "assets" / "rmvpe" / "rmvpe.pt"
DURATION_BUCKETS = (
    ("<1s", 0.0, 1.0),
    ("1-3s", 1.0, 3.0),
    ("3-8s", 3.0, 8.0),
    ("8-25s", 8.0, 25.0),
    ("25-60s", 25.0, 60.0),
    ("60-180s", 60.0, 180.0),
    (">180s", 180.0, math.inf),
)


@dataclass
class AudioFeature:
    rel_path: str
    abs_path: str
    book_id: str
    audio_kind: str
    suffix: str
    file_size: int
    sample_rate: int
    channels: int
    frames: int
    duration_seconds: float
    bitrate_kbps: float
    analyzed_seconds: float
    peak: float
    rms: float
    loudness_db: float
    voiced_coverage: float
    segment_count: int
    short_segment_count: int
    short_segment_ratio: float
    silence_ratio: float
    spectral_flatness: float
    onset_strength: float
    harmonic_ratio: float
    percussive_ratio: float
    duration_bucket: str
    audio_type: str
    flags: str


@dataclass
class DeepFeature:
    rel_path: str
    vad_segment_count: int
    vad_total_seconds: float
    vad_median_segment_seconds: float
    vad_median_gap_seconds: float
    f0_median_hz: float
    f0_p10_hz: float
    f0_p90_hz: float
    f0_voiced_frames: int
    f0_voiced_ratio: float
    gender_guess: str
    gender_confidence: float
    mfcc_cluster_hint: str
    transient_event_count: int
    strategy_suggestion: str
    notes: str


@dataclass
class AnalysisError:
    rel_path: str
    abs_path: str
    error_type: str
    error_message: str


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


RMVPE_ANALYZER = None
RMVPE_ANALYZER_DEVICE = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze book-res audio files into SQLite plus summary reports."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--analysis-mode",
        choices=("all", "light", "deep"),
        default="all",
        help="Run both phases, only light analysis, or only deep analysis from existing SQLite.",
    )
    parser.add_argument("--deep-sample-size", type=int, default=8)
    parser.add_argument("--sample-seconds", type=float, default=15.0)
    parser.add_argument("--deep-sample-seconds", type=float, default=90.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-deep", action="store_true")
    parser.add_argument(
        "--deep-f0-backend",
        choices=("librosa", "rmvpe"),
        default="librosa",
        help="Pitch backend for deep analysis. rmvpe can use GPU/MPS if available.",
    )
    parser.add_argument(
        "--deep-device",
        choices=("auto", "cpu"),
        default="auto",
        help="Device for deep RMVPE analysis. auto prefers cuda/mps, cpu disables acceleration.",
    )
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--progress-every-seconds", type=float, default=30.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 4) // 2)),
    )
    return parser.parse_args(argv)


def timestamped_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "outputs" / "audio_analysis" / f"book_res_{stamp}"


def iter_audio_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
            yield path


def classify_duration_bucket(duration_seconds: float) -> str:
    for label, lower, upper in DURATION_BUCKETS:
        if lower <= duration_seconds < upper:
            return label
    return ">180s"


def path_parts(root: Path, path: Path) -> tuple[str, str, str]:
    rel_path = path.relative_to(root).as_posix()
    parts = Path(rel_path).parts
    book_id = parts[0] if parts else ""
    audio_kind = parts[1] if len(parts) > 2 else ""
    return rel_path, book_id, audio_kind


def safe_float(value: float | np.floating, digits: int = 6) -> float:
    if value is None:
        return 0.0
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return round(value, digits)


def choose_audio_type(
    *,
    duration_seconds: float,
    voiced_coverage: float,
    rms: float,
    spectral_flatness: float,
    onset_strength: float,
    harmonic_ratio: float,
    percussive_ratio: float,
    segment_count: int,
    short_segment_ratio: float,
) -> str:
    if duration_seconds <= 0 or rms < 1e-5:
        return "invalid_or_silent"
    if duration_seconds < 0.75:
        return "short_speech_or_sfx"
    music_like = (
        (harmonic_ratio >= 0.68 and percussive_ratio >= 0.10)
        or (spectral_flatness <= 0.03 and onset_strength >= 0.45)
        or percussive_ratio >= 0.22
    )
    if music_like and duration_seconds >= 3.0:
        return "music_or_bgm_candidate"
    if voiced_coverage >= 0.70 and segment_count <= 3 and duration_seconds < 25.0:
        return "clean_speech_candidate"
    if voiced_coverage >= 0.45 and duration_seconds < 25.0:
        return "clean_or_mixed_speech_candidate"
    if duration_seconds >= 60.0:
        return "long_mixed_candidate"
    if short_segment_ratio >= 0.50 and segment_count >= 3:
        return "fragmented_or_sfx_candidate"
    return "low_confidence"


def choose_deep_device(device_mode: str) -> str:
    if device_mode == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                torch.zeros(1).to(torch.device("mps"))
                return "mps"
            except Exception:
                pass
    except Exception:
        pass
    return "cpu"


def get_rmvpe_analyzer(device_mode: str):
    global RMVPE_ANALYZER, RMVPE_ANALYZER_DEVICE
    device = choose_deep_device(device_mode)
    if RMVPE_ANALYZER is not None and RMVPE_ANALYZER_DEVICE == device:
        return RMVPE_ANALYZER, device
    from infer.lib.rmvpe import RMVPE

    RMVPE_ANALYZER = RMVPE(str(RMVPE_ROOT), is_half=False, device=device)
    RMVPE_ANALYZER_DEVICE = device
    return RMVPE_ANALYZER, device


def analyze_pitch_distribution(
    y: np.ndarray,
    sr: int,
    backend: str,
    device_mode: str,
) -> tuple[np.ndarray, str]:
    if y.size == 0:
        return np.asarray([], dtype=np.float32), backend
    if backend == "rmvpe":
        rmvpe, device = get_rmvpe_analyzer(device_mode)
        segment_16k = y if sr == 16000 else librosa.resample(y, orig_sr=sr, target_sr=16000)
        f0 = rmvpe.infer_from_audio(segment_16k.astype(np.float32), thred=0.03)
        voiced = f0[f0 > 0]
        return voiced.astype(np.float32), f"rmvpe:{device}"
    f0 = librosa.yin(y, fmin=60, fmax=450, sr=sr)
    voiced = f0[np.isfinite(f0)]
    voiced = voiced[(voiced >= 60) & (voiced <= 450)]
    return voiced.astype(np.float32), "librosa:cpu"


def build_flags(feature: AudioFeature) -> str:
    flags = {feature.audio_type, feature.duration_bucket}
    if feature.duration_seconds < 1.0:
        flags.add("very_short")
    if feature.duration_seconds >= 180.0:
        flags.add("very_long")
    if feature.segment_count >= 8:
        flags.add("many_segments")
    if feature.short_segment_ratio >= 0.5:
        flags.add("many_short_segments")
    if feature.percussive_ratio >= 0.18:
        flags.add("percussive")
    if feature.harmonic_ratio >= 0.68:
        flags.add("harmonic")
    if feature.voiced_coverage >= 0.70:
        flags.add("high_voiced_coverage")
    if feature.voiced_coverage < 0.10:
        flags.add("low_voiced_coverage")
    return ",".join(sorted(flags))


def analyze_audio_file(root: Path, path: Path, sample_seconds: float = 60.0) -> AudioFeature:
    rel_path, book_id, audio_kind = path_parts(root, path)
    info = sf.info(str(path))
    file_size = path.stat().st_size
    duration_seconds = float(info.duration)
    bitrate_kbps = (file_size * 8.0 / 1000.0 / duration_seconds) if duration_seconds else 0.0
    load_duration = min(sample_seconds, duration_seconds) if sample_seconds > 0 else None
    with warnings.catch_warnings(), contextlib.redirect_stderr(sys.stderr):
        warnings.simplefilter("ignore")
        y, sr = librosa.load(str(path), sr=16000, mono=True, duration=load_duration)

    y = y.astype(np.float32, copy=False)
    analyzed_seconds = len(y) / sr if sr and y.size else 0.0
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    rms = float(np.sqrt(np.mean(y * y))) if y.size else 0.0
    loudness_db = 20.0 * math.log10(max(rms, 1e-8))
    intervals = librosa.effects.split(y, top_db=35) if y.size else np.empty((0, 2), dtype=int)
    voiced_samples = int(sum(end - start for start, end in intervals))
    voiced_coverage = voiced_samples / max(len(y), 1)
    segment_lengths = [(end - start) / sr for start, end in intervals] if sr else []
    short_segment_count = sum(1 for duration in segment_lengths if duration < 0.6)
    segment_count = len(segment_lengths)
    short_segment_ratio = short_segment_count / max(segment_count, 1)
    silence_ratio = 1.0 - voiced_coverage

    if y.size:
        normalized = y / max(peak, 1e-6) if peak else y
        flatness = librosa.feature.spectral_flatness(y=normalized)[0]
        onset_env = librosa.onset.onset_strength(y=normalized, sr=sr)
        harmonic, percussive = librosa.effects.hpss(normalized)
        total_rms = float(np.sqrt(np.mean(normalized * normalized))) if normalized.size else 0.0
        harmonic_rms = float(np.sqrt(np.mean(harmonic * harmonic))) if harmonic.size else 0.0
        percussive_rms = float(np.sqrt(np.mean(percussive * percussive))) if percussive.size else 0.0
        spectral_flatness = float(np.mean(flatness)) if flatness.size else 0.0
        onset_strength = float(np.mean(onset_env)) if onset_env.size else 0.0
        harmonic_ratio = harmonic_rms / max(total_rms, 1e-8)
        percussive_ratio = percussive_rms / max(total_rms, 1e-8)
    else:
        spectral_flatness = 0.0
        onset_strength = 0.0
        harmonic_ratio = 0.0
        percussive_ratio = 0.0

    duration_bucket = classify_duration_bucket(duration_seconds)
    audio_type = choose_audio_type(
        duration_seconds=duration_seconds,
        voiced_coverage=voiced_coverage,
        rms=rms,
        spectral_flatness=spectral_flatness,
        onset_strength=onset_strength,
        harmonic_ratio=harmonic_ratio,
        percussive_ratio=percussive_ratio,
        segment_count=segment_count,
        short_segment_ratio=short_segment_ratio,
    )
    feature = AudioFeature(
        rel_path=rel_path,
        abs_path=str(path),
        book_id=book_id,
        audio_kind=audio_kind,
        suffix=path.suffix.lower(),
        file_size=file_size,
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
        frames=int(info.frames),
        duration_seconds=safe_float(duration_seconds),
        bitrate_kbps=safe_float(bitrate_kbps, 3),
        analyzed_seconds=safe_float(analyzed_seconds),
        peak=safe_float(peak),
        rms=safe_float(rms),
        loudness_db=safe_float(loudness_db, 3),
        voiced_coverage=safe_float(voiced_coverage),
        segment_count=segment_count,
        short_segment_count=short_segment_count,
        short_segment_ratio=safe_float(short_segment_ratio),
        silence_ratio=safe_float(silence_ratio),
        spectral_flatness=safe_float(spectral_flatness),
        onset_strength=safe_float(onset_strength),
        harmonic_ratio=safe_float(harmonic_ratio),
        percussive_ratio=safe_float(percussive_ratio),
        duration_bucket=duration_bucket,
        audio_type=audio_type,
        flags="",
    )
    feature.flags = build_flags(feature)
    return feature


def create_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        create table if not exists files (
            rel_path text primary key,
            abs_path text not null,
            book_id text,
            audio_kind text,
            suffix text,
            file_size integer,
            sample_rate integer,
            channels integer,
            frames integer,
            duration_seconds real,
            bitrate_kbps real,
            analyzed_seconds real,
            peak real,
            rms real,
            loudness_db real,
            voiced_coverage real,
            segment_count integer,
            short_segment_count integer,
            short_segment_ratio real,
            silence_ratio real,
            spectral_flatness real,
            onset_strength real,
            harmonic_ratio real,
            percussive_ratio real,
            duration_bucket text,
            audio_type text,
            flags text,
            updated_at text not null
        )
        """
    )
    conn.execute(
        """
        create table if not exists deep_features (
            rel_path text primary key,
            vad_segment_count integer,
            vad_total_seconds real,
            vad_median_segment_seconds real,
            vad_median_gap_seconds real,
            f0_median_hz real,
            f0_p10_hz real,
            f0_p90_hz real,
            f0_voiced_frames integer,
            f0_voiced_ratio real,
            gender_guess text,
            gender_confidence real,
            mfcc_cluster_hint text,
            transient_event_count integer,
            strategy_suggestion text,
            notes text,
            updated_at text not null
        )
        """
    )
    conn.execute(
        """
        create table if not exists errors (
            rel_path text primary key,
            abs_path text not null,
            error_type text not null,
            error_message text not null,
            updated_at text not null default current_timestamp
        )
        """
    )
    conn.execute("create index if not exists idx_files_audio_type on files(audio_type)")
    conn.execute("create index if not exists idx_files_duration_bucket on files(duration_bucket)")
    conn.execute("create index if not exists idx_files_audio_kind on files(audio_kind)")
    conn.commit()


def insert_file_features(conn: sqlite3.Connection, feature: AudioFeature) -> None:
    values = asdict(feature)
    values["updated_at"] = datetime.now().isoformat(timespec="seconds")
    columns = list(values.keys())
    placeholders = ",".join("?" for _ in columns)
    update_clause = ",".join(f"{column}=excluded.{column}" for column in columns if column != "rel_path")
    conn.execute(
        f"""
        insert into files({",".join(columns)})
        values({placeholders})
        on conflict(rel_path) do update set {update_clause}
        """,
        [values[column] for column in columns],
    )


def insert_error(conn: sqlite3.Connection, root: Path, path: Path, exc: Exception) -> None:
    rel_path, _, _ = path_parts(root, path)
    conn.execute(
        """
        insert into errors(rel_path, abs_path, error_type, error_message, updated_at)
        values(?, ?, ?, ?, ?)
        on conflict(rel_path) do update set
            abs_path=excluded.abs_path,
            error_type=excluded.error_type,
            error_message=excluded.error_message,
            updated_at=excluded.updated_at
        """,
        (
            rel_path,
            str(path),
            type(exc).__name__,
            str(exc)[:1000],
            datetime.now().isoformat(timespec="seconds"),
        ),
    )


def insert_analysis_error(conn: sqlite3.Connection, error: AnalysisError) -> None:
    conn.execute(
        """
        insert into errors(rel_path, abs_path, error_type, error_message, updated_at)
        values(?, ?, ?, ?, ?)
        on conflict(rel_path) do update set
            abs_path=excluded.abs_path,
            error_type=excluded.error_type,
            error_message=excluded.error_message,
            updated_at=excluded.updated_at
        """,
        (
            error.rel_path,
            error.abs_path,
            error.error_type,
            error.error_message[:1000],
            datetime.now().isoformat(timespec="seconds"),
        ),
    )


def already_done(conn: sqlite3.Connection, rel_path: str) -> bool:
    file_row = conn.execute("select 1 from files where rel_path=?", (rel_path,)).fetchone()
    error_row = conn.execute("select 1 from errors where rel_path=?", (rel_path,)).fetchone()
    return file_row is not None or error_row is not None


def select_deep_samples(conn: sqlite3.Connection, sample_size: int) -> list[str]:
    if sample_size <= 0:
        return []
    rows = conn.execute(
        """
        select rel_path, duration_bucket, audio_kind, audio_type, duration_seconds,
               short_segment_ratio, voiced_coverage
        from files
        order by duration_bucket, audio_kind, audio_type, duration_seconds
        """
    ).fetchall()
    groups: dict[tuple[str, str, str], list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        groups[(row[1], row[2], row[3])].append(row)

    selected: list[str] = []
    seen: set[str] = set()
    per_group = max(1, sample_size)
    for group_rows in groups.values():
        candidates = [group_rows[len(group_rows) // 2], group_rows[0], group_rows[-1]]
        risky = sorted(
            group_rows,
            key=lambda item: (float(item[5]), -float(item[6])),
            reverse=True,
        )
        group_selected = 0
        for row in candidates + risky:
            if group_selected >= per_group:
                break
            rel_path = row[0]
            if rel_path not in seen:
                selected.append(rel_path)
                seen.add(rel_path)
                group_selected += 1
    return selected


def analyze_deep_feature(
    root: Path,
    rel_path: str,
    sample_seconds: float = 90.0,
    f0_backend: str = "librosa",
    device_mode: str = "auto",
) -> DeepFeature:
    path = root / rel_path
    y, sr = librosa.load(str(path), sr=16000, mono=True, duration=sample_seconds)
    y = y.astype(np.float32, copy=False)
    intervals = librosa.effects.split(y, top_db=35) if y.size else np.empty((0, 2), dtype=int)
    durations = [(end - start) / sr for start, end in intervals]
    gaps = [
        (intervals[idx][0] - intervals[idx - 1][1]) / sr
        for idx in range(1, len(intervals))
    ]
    vad_total_seconds = sum(durations)
    if y.size:
        try:
            voiced, pitch_backend_used = analyze_pitch_distribution(
                y, sr, f0_backend, device_mode
            )
        except Exception:
            voiced = np.asarray([], dtype=np.float32)
            pitch_backend_used = f"{f0_backend}:failed"
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        transient_event_count = int(
            np.sum(
                onset_env
                > max(float(np.mean(onset_env)) + 2.5 * float(np.std(onset_env)), 0.5)
            )
        )
    else:
        voiced = np.asarray([], dtype=np.float32)
        transient_event_count = 0
        pitch_backend_used = f"{f0_backend}:empty"
    voiced_frames = int(voiced.shape[0])
    voiced_ratio = voiced_frames / max(len(y) // 512, 1)
    f0_median = float(np.median(voiced)) if voiced_frames else 0.0
    f0_p10 = float(np.percentile(voiced, 10)) if voiced_frames else 0.0
    f0_p90 = float(np.percentile(voiced, 90)) if voiced_frames else 0.0
    if voiced_frames < 10:
        gender_guess = "unknown"
        gender_confidence = 0.0
    else:
        gender_guess = "female" if f0_median >= 190.0 else "male"
        gender_confidence = min(0.98, 0.55 + abs(f0_median - 190.0) / 160.0)
    mfcc_cluster_hint = "insufficient_audio"
    if y.size and len(durations) >= 2:
        mfcc_cluster_hint = "multi_segment_same_file"
    strategy_suggestion = suggest_strategy(
        vad_segment_count=len(durations),
        vad_total_seconds=vad_total_seconds,
        duration_seconds=len(y) / sr if sr else 0.0,
        gender_guess=gender_guess,
        transient_event_count=transient_event_count,
    )
    return DeepFeature(
        rel_path=rel_path,
        vad_segment_count=len(durations),
        vad_total_seconds=safe_float(vad_total_seconds),
        vad_median_segment_seconds=safe_float(statistics.median(durations) if durations else 0.0),
        vad_median_gap_seconds=safe_float(statistics.median(gaps) if gaps else 0.0),
        f0_median_hz=safe_float(f0_median, 3),
        f0_p10_hz=safe_float(f0_p10, 3),
        f0_p90_hz=safe_float(f0_p90, 3),
        f0_voiced_frames=voiced_frames,
        f0_voiced_ratio=safe_float(voiced_ratio),
        gender_guess=gender_guess,
        gender_confidence=safe_float(gender_confidence),
        mfcc_cluster_hint=mfcc_cluster_hint,
        transient_event_count=transient_event_count,
        strategy_suggestion=strategy_suggestion,
        notes=f"pitch_backend={pitch_backend_used}",
    )


def suggest_strategy(
    *,
    vad_segment_count: int,
    vad_total_seconds: float,
    duration_seconds: float,
    gender_guess: str,
    transient_event_count: int,
) -> str:
    if duration_seconds <= 0 or vad_total_seconds <= 0:
        return "passthrough_original"
    if transient_event_count >= 3 and vad_total_seconds / max(duration_seconds, 1e-6) < 0.75:
        return "speech_sfx_segments"
    if duration_seconds >= 60.0:
        return "long_mixed_lesson"
    if vad_segment_count <= 2 and gender_guess in {"male", "female"}:
        return "single_voice"
    if vad_segment_count > 2 and gender_guess in {"male", "female"}:
        return "clean_voice_segments"
    return "passthrough_original"


def analyze_audio_file_job(root_str: str, path_str: str, sample_seconds: float) -> tuple[str, dict]:
    root = Path(root_str)
    path = Path(path_str)
    try:
        feature = analyze_audio_file(root, path, sample_seconds=sample_seconds)
        return "feature", asdict(feature)
    except Exception as exc:
        rel_path, _, _ = path_parts(root, path)
        error = AnalysisError(
            rel_path=rel_path,
            abs_path=str(path),
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        return "error", asdict(error)


def insert_deep_feature(conn: sqlite3.Connection, feature: DeepFeature) -> None:
    values = asdict(feature)
    values["updated_at"] = datetime.now().isoformat(timespec="seconds")
    columns = list(values.keys())
    placeholders = ",".join("?" for _ in columns)
    update_clause = ",".join(f"{column}=excluded.{column}" for column in columns if column != "rel_path")
    conn.execute(
        f"""
        insert into deep_features({",".join(columns)})
        values({placeholders})
        on conflict(rel_path) do update set {update_clause}
        """,
        [values[column] for column in columns],
    )


def summarize_database(conn: sqlite3.Connection) -> dict:
    conn.row_factory = sqlite3.Row
    total_files = conn.execute("select count(*) from files").fetchone()[0]
    total_errors = conn.execute("select count(*) from errors").fetchone()[0]

    def count_by(column: str) -> dict[str, int]:
        return {
            str(row[0]): int(row[1])
            for row in conn.execute(f"select {column}, count(*) from files group by {column} order by count(*) desc")
        }

    duration_values = [float(row[0]) for row in conn.execute("select duration_seconds from files order by duration_seconds")]
    short_segment_values = [float(row[0]) for row in conn.execute("select short_segment_ratio from files")]
    voiced_values = [float(row[0]) for row in conn.execute("select voiced_coverage from files")]
    percentiles = {}
    if duration_values:
        for label, q in [("min", 0), ("p01", 0.01), ("p05", 0.05), ("p25", 0.25), ("p50", 0.5), ("p75", 0.75), ("p90", 0.9), ("p95", 0.95), ("p99", 0.99), ("max", 1.0)]:
            idx = min(len(duration_values) - 1, int((len(duration_values) - 1) * q))
            percentiles[label] = round(duration_values[idx], 3)
        percentiles["mean"] = round(statistics.mean(duration_values), 3)
    strategy_counts = {
        str(row[0]): int(row[1])
        for row in conn.execute(
            "select strategy_suggestion, count(*) from deep_features group by strategy_suggestion order by count(*) desc"
        )
    }
    error_examples = [
        {"rel_path": row[0], "error_type": row[1], "error_message": row[2]}
        for row in conn.execute(
            "select rel_path, error_type, error_message from errors order by rel_path limit 20"
        )
    ]
    longest_examples = [
        {"rel_path": row[0], "duration_seconds": float(row[1])}
        for row in conn.execute(
            "select rel_path, duration_seconds from files order by duration_seconds desc limit 20"
        )
    ]
    suspicious_examples = [
        {"rel_path": row[0], "audio_type": row[1], "flags": row[2]}
        for row in conn.execute(
            """
            select rel_path, audio_type, flags
            from files
            where flags like '%many_short_segments%'
               or flags like '%very_long%'
               or flags like '%low_voiced_coverage%'
            order by duration_seconds desc
            limit 40
            """
        )
    ]
    samples = {}
    for audio_type in count_by("audio_type").keys():
        rows = conn.execute(
            "select rel_path from files where audio_type=? order by duration_seconds limit 20",
            (audio_type,),
        ).fetchall()
        samples[audio_type] = [row[0] for row in rows]
    return {
        "total_files": total_files,
        "total_errors": total_errors,
        "by_audio_kind": count_by("audio_kind"),
        "by_duration_bucket": count_by("duration_bucket"),
        "by_audio_type": count_by("audio_type"),
        "duration_seconds": percentiles,
        "voiced_coverage_mean": round(statistics.mean(voiced_values), 4) if voiced_values else 0.0,
        "short_segment_ratio_mean": round(statistics.mean(short_segment_values), 4) if short_segment_values else 0.0,
        "deep_features": conn.execute("select count(*) from deep_features").fetchone()[0],
        "strategy_suggestions": strategy_counts,
        "error_examples": error_examples,
        "longest_examples": longest_examples,
        "suspicious_examples": suspicious_examples,
        "threshold_draft": {
            "light_sample_seconds": 15.0,
            "vocal_split_top_db": 35,
            "very_short_seconds": 0.75,
            "music_candidate_harmonic_ratio": 0.68,
            "music_candidate_percussive_ratio": 0.22,
            "clean_speech_voiced_coverage": 0.70,
            "clean_mixed_voiced_coverage": 0.45,
        },
        "samples": samples,
    }


def write_summary_files(output_dir: Path, summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with open(output_dir / "summary.md", "w", encoding="utf-8") as handle:
        handle.write("# Book Res Audio Analysis Summary\n\n")
        handle.write(f"- Total analyzed files: {summary['total_files']}\n")
        handle.write(f"- Total errors: {summary['total_errors']}\n")
        handle.write(f"- Deep analyzed files: {summary['deep_features']}\n\n")
        handle.write("## Quick Signals\n\n")
        handle.write(f"- Mean voiced coverage: {summary['voiced_coverage_mean']}\n")
        handle.write(f"- Mean short-segment ratio: {summary['short_segment_ratio_mean']}\n\n")
        handle.write("## Duration Seconds\n\n")
        for key, value in summary["duration_seconds"].items():
            handle.write(f"- {key}: {value}\n")
        handle.write("\n## By Audio Kind\n\n")
        for key, value in summary["by_audio_kind"].items():
            handle.write(f"- {key}: {value}\n")
        handle.write("\n## By Duration Bucket\n\n")
        for key, value in summary["by_duration_bucket"].items():
            handle.write(f"- {key}: {value}\n")
        handle.write("\n## By Audio Type\n\n")
        for key, value in summary["by_audio_type"].items():
            handle.write(f"- {key}: {value}\n")
        handle.write("\n## Deep Strategy Suggestions\n\n")
        for key, value in summary["strategy_suggestions"].items():
            handle.write(f"- {key}: {value}\n")
        handle.write("\n## Threshold Draft\n\n")
        for key, value in summary["threshold_draft"].items():
            handle.write(f"- {key}: {value}\n")
        handle.write("\n## Longest Examples\n\n")
        for item in summary["longest_examples"]:
            handle.write(f"- `{item['rel_path']}`: {item['duration_seconds']}s\n")
        handle.write("\n## Suspicious Examples\n\n")
        for item in summary["suspicious_examples"][:40]:
            handle.write(f"- `{item['rel_path']}` | {item['audio_type']} | {item['flags']}\n")
        if summary["error_examples"]:
            handle.write("\n## Error Examples\n\n")
            for item in summary["error_examples"]:
                handle.write(
                    f"- `{item['rel_path']}` | {item['error_type']} | {item['error_message']}\n"
                )
        handle.write("\n## Review Samples\n\n")
        for audio_type, rel_paths in summary["samples"].items():
            handle.write(f"### {audio_type}\n\n")
            for rel_path in rel_paths[:20]:
                handle.write(f"- `{rel_path}`\n")
            handle.write("\n")


def write_sample_lists(output_dir: Path, summary: dict) -> None:
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    for audio_type, rel_paths in summary["samples"].items():
        safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in audio_type)
        with open(samples_dir / f"{safe_name}.txt", "w", encoding="utf-8") as handle:
            for rel_path in rel_paths:
                handle.write(f"{rel_path}\n")


def build_progress_state(
    *,
    phase: str,
    processed: int,
    total: int,
    started: float,
    output_dir: Path,
    message: str,
) -> ProgressState:
    elapsed = max(time.time() - started, 1e-6)
    rate = processed / elapsed if processed > 0 else 0.0
    remaining = max(total - processed, 0)
    eta = (remaining / rate) if rate > 0 else 0.0
    return ProgressState(
        phase=phase,
        processed=processed,
        total=total,
        elapsed_seconds=round(elapsed, 2),
        rate_per_second=round(rate, 3),
        eta_seconds=round(eta, 2),
        output_dir=str(output_dir),
        message=message,
    )


def write_progress(output_dir: Path, state: ProgressState) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    line = (
        f"[{timestamp}] {state.phase}: {state.processed}/{state.total} | "
        f"elapsed={state.elapsed_seconds}s | rate={state.rate_per_second}/s | "
        f"eta={state.eta_seconds}s | {state.message}"
    )
    with open(output_dir / "progress.log", "a", encoding="utf-8") as handle:
        handle.write(f"{line}\n")
    with open(output_dir / "progress.json", "w", encoding="utf-8") as handle:
        json.dump(asdict(state), handle, ensure_ascii=False, indent=2)
    print(line, flush=True)


def chunk_paths(paths: list[Path], batch_size: int) -> Iterable[list[Path]]:
    for idx in range(0, len(paths), batch_size):
        yield paths[idx : idx + batch_size]


def run_light_analysis(
    conn: sqlite3.Connection,
    root: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    paths = list(iter_audio_files(root))
    if args.limit > 0:
        paths = paths[: args.limit]
    if args.resume:
        paths = [path for path in paths if not already_done(conn, path_parts(root, path)[0])]
    started = time.time()
    last_progress_time = started
    processed = 0
    write_progress(
        output_dir,
        build_progress_state(
            phase="light_analysis",
            processed=0,
            total=len(paths),
            started=started,
            output_dir=output_dir,
            message="starting light analysis",
        ),
    )
    if paths:
        executor_cls = ProcessPoolExecutor
        pool = None
        try:
            pool = executor_cls(max_workers=max(1, args.workers))
        except PermissionError:
            pool = None
        if pool is None:
            for path in paths:
                status, payload = analyze_audio_file_job(str(root), str(path), args.sample_seconds)
                if status == "feature":
                    insert_file_features(conn, AudioFeature(**payload))
                else:
                    insert_analysis_error(conn, AnalysisError(**payload))
                processed += 1
                should_report = processed % max(args.progress_every, 1) == 0
                if time.time() - last_progress_time >= max(args.progress_every_seconds, 1.0):
                    should_report = True
                if should_report:
                    conn.commit()
                    write_progress(
                        output_dir,
                        build_progress_state(
                            phase="light_analysis",
                            processed=processed,
                            total=len(paths),
                            started=started,
                            output_dir=output_dir,
                            message="running light analysis",
                        ),
                    )
                    last_progress_time = time.time()
        else:
            with pool:
                batch_size = max(64, args.workers * 8)
                for batch in chunk_paths(paths, batch_size):
                    futures = [
                        pool.submit(
                            analyze_audio_file_job, str(root), str(path), args.sample_seconds
                        )
                        for path in batch
                    ]
                    for future in as_completed(futures):
                        status, payload = future.result()
                        if status == "feature":
                            insert_file_features(conn, AudioFeature(**payload))
                        else:
                            insert_analysis_error(conn, AnalysisError(**payload))
                        processed += 1
                        should_report = processed % max(args.progress_every, 1) == 0
                        if (
                            time.time() - last_progress_time
                            >= max(args.progress_every_seconds, 1.0)
                        ):
                            should_report = True
                        if should_report:
                            conn.commit()
                            write_progress(
                                output_dir,
                                build_progress_state(
                                    phase="light_analysis",
                                    processed=processed,
                                    total=len(paths),
                                    started=started,
                                    output_dir=output_dir,
                                    message="running light analysis",
                                ),
                            )
                            last_progress_time = time.time()
    conn.commit()
    write_progress(
        output_dir,
        build_progress_state(
            phase="light_analysis",
            processed=processed,
            total=len(paths),
            started=started,
            output_dir=output_dir,
            message="light analysis completed",
        ),
    )

 
def run_deep_analysis(
    conn: sqlite3.Connection,
    root: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    if args.skip_deep or args.deep_sample_size <= 0:
        return
    deep_samples = select_deep_samples(conn, args.deep_sample_size)
    deep_started = time.time()
    last_deep_progress_time = deep_started
    write_progress(
        output_dir,
        build_progress_state(
            phase="deep_analysis",
            processed=0,
            total=len(deep_samples),
            started=deep_started,
            output_dir=output_dir,
            message="starting deep analysis",
        ),
    )
    for idx, rel_path in enumerate(deep_samples, start=1):
        if args.resume and conn.execute("select 1 from deep_features where rel_path=?", (rel_path,)).fetchone():
            continue
        try:
            insert_deep_feature(
                conn,
                analyze_deep_feature(
                    root,
                    rel_path,
                    sample_seconds=args.deep_sample_seconds,
                    f0_backend=args.deep_f0_backend,
                    device_mode=args.deep_device,
                ),
            )
        except Exception as exc:
            path = root / rel_path
            insert_error(conn, root, path, exc)
        should_report = idx % max(min(args.progress_every, 25), 1) == 0
        if time.time() - last_deep_progress_time >= max(args.progress_every_seconds, 1.0):
            should_report = True
        if should_report:
            conn.commit()
            write_progress(
                output_dir,
                build_progress_state(
                    phase="deep_analysis",
                    processed=idx,
                    total=len(deep_samples),
                    started=deep_started,
                    output_dir=output_dir,
                    message=f"running deep analysis ({args.deep_f0_backend})",
                ),
            )
            last_deep_progress_time = time.time()
    conn.commit()
    write_progress(
        output_dir,
        build_progress_state(
            phase="deep_analysis",
            processed=len(deep_samples),
            total=len(deep_samples),
            started=deep_started,
            output_dir=output_dir,
            message="deep analysis completed",
        ),
    )


def run_analysis(args: argparse.Namespace) -> Path:
    root = args.root.expanduser().resolve()
    output_dir = (args.output_dir or timestamped_output_dir()).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(output_dir / "audio_features.sqlite")
    conn.row_factory = sqlite3.Row
    create_schema(conn)

    if args.analysis_mode in {"all", "light"}:
        run_light_analysis(conn, root, output_dir, args)

    if args.analysis_mode in {"all", "deep"}:
        if (
            conn.execute("select count(*) from files").fetchone()[0] == 0
            and args.analysis_mode == "deep"
        ):
            raise RuntimeError(
                "deep mode requires an existing audio_features.sqlite with files already analyzed"
            )
        run_deep_analysis(conn, root, output_dir, args)

    summary = summarize_database(conn)
    write_summary_files(output_dir, summary)
    write_sample_lists(output_dir, summary)
    write_progress(
        output_dir,
        build_progress_state(
            phase="finalizing",
            processed=1,
            total=1,
            started=time.time() - 0.001,
            output_dir=output_dir,
            message="summary files written",
        ),
    )
    conn.close()
    return output_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = run_analysis(args)
    print(f"analysis output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
