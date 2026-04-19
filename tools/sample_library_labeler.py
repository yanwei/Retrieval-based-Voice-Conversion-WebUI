from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flask import Flask, abort, jsonify, render_template, request, send_file

from tools.sample_library import LABELS_PATH, MANIFEST_PATH, load_manifest, write_label_record

REVIEW_CANDIDATES_PATH = REPO_ROOT / "assets" / "sample_library" / "review_candidates.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the sample library labeler.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--review-candidates", type=Path, default=REVIEW_CANDIDATES_PATH)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7868)
    return parser.parse_args()


def load_labeler_items(manifest_path: Path, review_candidates_path: Path | None = None) -> list[dict[str, object]]:
    if review_candidates_path and review_candidates_path.exists():
        return json.loads(review_candidates_path.read_text(encoding="utf-8"))
    return load_manifest(manifest_path)


def load_labeled_sample_ids(labels_path: Path) -> set[str]:
    if not labels_path.exists():
        return set()
    sample_ids: set[str] = set()
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        sample_id = str(payload.get("sample_id") or "").strip()
        if sample_id:
            sample_ids.add(sample_id)
    return sample_ids


def create_app(
    *,
    manifest_path: Path = MANIFEST_PATH,
    labels_path: Path = LABELS_PATH,
    review_candidates_path: Path = REVIEW_CANDIDATES_PATH,
    host: str = "127.0.0.1",
    port: int = 7868,
) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).resolve().parent / "templates"),
        static_folder=str(Path(__file__).resolve().parent / "static"),
    )
    app.config.update(
        MANIFEST_PATH=str(manifest_path),
        LABELS_PATH=str(labels_path),
        REVIEW_CANDIDATES_PATH=str(review_candidates_path),
        HOST=host,
        PORT=port,
    )

    @app.get("/")
    def index():
        return render_template("sample_library_labeler.html", host=host, port=port)

    @app.get("/api/items")
    def api_items():
        items = load_labeler_items(
            Path(app.config["MANIFEST_PATH"]),
            Path(app.config["REVIEW_CANDIDATES_PATH"]),
        )
        labeled_ids = load_labeled_sample_ids(Path(app.config["LABELS_PATH"]))
        pending_items = [item for item in items if str(item.get("sample_id")) not in labeled_ids]
        return jsonify({"items": pending_items, "skipped_labeled": len(items) - len(pending_items)})

    @app.post("/api/labels")
    def api_labels():
        payload = request.get_json(silent=True) or {}
        sample_id = str(payload.get("sample_id") or "").strip()
        labels = payload.get("labels") or {}
        item = payload.get("item") or {}
        if not sample_id:
            return jsonify({"error": "sample_id is required"}), 400
        write_label_record(
            Path(app.config["LABELS_PATH"]),
            {
                "sample_id": sample_id,
                "labels": labels,
                "item": item,
            },
        )
        return jsonify({"status": "ok", "sample_id": sample_id})

    @app.get("/static-proxy")
    def static_proxy():
        raw_path = str(request.args.get("path") or "").strip()
        if not raw_path:
            abort(400)
        path = Path(raw_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            abort(404)
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return send_file(path, mimetype=mime_type, conditional=True)

    return app


if __name__ == "__main__":
    args = parse_args()
    app = create_app(
        manifest_path=args.manifest,
        labels_path=args.labels,
        review_candidates_path=args.review_candidates,
        host=args.host,
        port=args.port,
    )
    app.run(host=app.config["HOST"], port=app.config["PORT"], debug=False)
