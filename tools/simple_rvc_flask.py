import os
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from dotenv import load_dotenv

from configs.config import Config
from infer.modules.vc.modules import VC
from infer.modules.vc.utils import get_index_path_from_model
from tools import process_mixed_long_audio as mixed_audio


APP_ROOT = REPO_ROOT / "outputs" / "simple_rvc_flask"
UPLOAD_ROOT = APP_ROOT / "uploads"
JOB_ROOT = APP_ROOT / "jobs"
HOST = "127.0.0.1"
PORT = 7867
UVR_MODELS = [
    "auto",
    "HP5_only_main_vocal",
    "HP3_all_vocals",
    "HP2_all_vocals",
]
DEFAULT_MALE_MODEL = "Myaz.pth"
DEFAULT_FEMALE_MODEL = "haoshengyin.pth"

VC_CACHE = {}
JOBS = {}
JOBS_LOCK = threading.Lock()
MODEL_PRESETS = {
    "mygf-jack.pth": {
        "f0_up_key": 0,
        "index_rate": 0.0,
        "rms_mix_rate": 0.86,
    },
    "haoshengyin.pth": {
        "f0_up_key": 0,
        "index_rate": 0.75,
        "protect": 0.25,
        "rms_mix_rate": 1.25,
    },
    "Myaz.pth": {
        "f0_up_key": 0,
        "index_rate": 0.75,
        "protect": 0.25,
        "rms_mix_rate": 1.25,
    },
}


def ensure_environment() -> None:
    load_dotenv(REPO_ROOT / ".env")
    os.environ.setdefault("weight_root", "assets/weights")
    os.environ.setdefault("weight_uvr5_root", "assets/uvr5_weights")
    os.environ.setdefault("index_root", "logs")
    os.environ.setdefault("outside_index_root", "assets/indices")
    os.environ.setdefault("rmvpe_root", "assets/rmvpe")
    os.environ.setdefault("TEMP", "/tmp")
    APP_ROOT.mkdir(parents=True, exist_ok=True)
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    JOB_ROOT.mkdir(parents=True, exist_ok=True)


def list_models() -> list[str]:
    ensure_environment()
    weight_root = (REPO_ROOT / os.environ["weight_root"]).resolve()
    if not weight_root.exists():
        return []
    return sorted(
        file.name for file in weight_root.iterdir() if file.is_file() and file.suffix == ".pth"
    )


def list_indices() -> list[str]:
    ensure_environment()
    roots = [
        (REPO_ROOT / os.environ["index_root"]).resolve(),
        (REPO_ROOT / os.environ["outside_index_root"]).resolve(),
    ]
    paths = []
    for root in roots:
        if not root.exists():
            continue
        for file in root.rglob("*.index"):
            if "trained" not in file.name:
                paths.append(str(file.resolve()))
    return sorted(set(paths))


def default_index_for_model(model_name: str) -> str:
    ensure_environment()
    model_name = (model_name or "").strip()
    if not model_name:
        return ""
    auto = get_index_path_from_model(model_name)
    if auto:
        auto_path = Path(auto)
        if auto_path.is_absolute():
            return str(auto_path)
        return str((REPO_ROOT / auto_path).resolve())

    stem = Path(model_name).stem.lower()
    for index_path in list_indices():
        index_stem = Path(index_path).stem.lower()
        if stem in index_stem or any(
            token and token in index_stem for token in stem.replace("-", "_").split("_")
        ):
            return index_path
    return ""


def default_model_choice(models: list[str], preferred: str) -> str:
    if preferred in models:
        return preferred
    return models[0] if models else ""


def get_vc(model_name: str) -> VC:
    model_name = (model_name or "").strip()
    if not model_name:
        raise ValueError("请选择模型。")
    if model_name not in VC_CACHE:
        vc = VC(Config())
        vc.get_vc(model_name)
        VC_CACHE[model_name] = vc
    return VC_CACHE[model_name]


def create_job(kind: str, input_name: str) -> tuple[str, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = uuid.uuid4().hex[:10]
    stem = Path(input_name).stem
    job_dir = JOB_ROOT / f"{kind}_{stem}_{stamp}_{job_id}"
    job_dir.mkdir(parents=True, exist_ok=True)
    job = {
        "id": job_id,
        "kind": kind,
        "status": "queued",
        "progress": 0.0,
        "message": "等待开始",
        "log": [],
        "job_dir": str(job_dir),
        "input_path": "",
        "original_audio": "",
        "result_audio": "",
        "converted_vocals": "",
        "downloads": [],
        "error": "",
        "started_at": datetime.now().timestamp(),
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
    return job_id, job_dir


def update_job(job_id: str, **fields) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        for key, value in fields.items():
            if key == "log_append":
                job["log"].append(value)
            else:
                job[key] = value


def export_mp3(input_wav: Path, output_mp3: Path) -> None:
    mixed_audio.export_mp3(input_wav, output_mp3)


def resolve_index(model_name: str, selected_index: str, use_index: bool = True) -> str:
    if not use_index:
        return ""
    selected_index = (selected_index or "").strip()
    if selected_index:
        return selected_index
    return default_index_for_model(model_name)


def run_single_job(job_id: str, payload: dict) -> None:
    try:
        input_path = payload["input_path"]
        model_name = payload["model_name"]
        index_path = resolve_index(model_name, payload.get("index_path", ""), payload["use_index"])
        job_dir = Path(payload["job_dir"])
        update_job(
            job_id,
            status="running",
            progress=0.05,
            message="加载模型",
            input_path=input_path,
            original_audio=input_path,
        )
        vc = get_vc(model_name)
        update_job(job_id, progress=0.15, message="开始转换", log_append="开始调用 vc_single")
        result = {"info": None, "audio_tuple": None, "error": None}

        def worker():
            try:
                info, audio_tuple = vc.vc_single(
                    0,
                    input_path,
                    int(payload["f0_up_key"]),
                    None,
                    payload["f0_method"],
                    index_path,
                    index_path,
                    float(payload["index_rate"]),
                    int(payload["filter_radius"]),
                    int(payload["resample_sr"]),
                    float(payload["rms_mix_rate"]),
                    float(payload["protect"]),
                )
                result["info"] = info
                result["audio_tuple"] = audio_tuple
            except Exception as exc:
                result["error"] = exc

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        stage_messages = [
            (0.24, "读取音频与准备特征"),
            (0.40, "提取 HuBERT 特征"),
            (0.58, "提取 F0"),
            (0.78, "模型推理中"),
            (0.86, "后处理音频"),
        ]
        stage_index = 0
        last_message = ""
        start_time = time.time()
        while thread.is_alive():
            elapsed = time.time() - start_time
            if elapsed < 3:
                stage_index = 0
            elif elapsed < 10:
                stage_index = 1
            elif elapsed < 20:
                stage_index = 2
            elif elapsed < 35:
                stage_index = 3
            else:
                stage_index = 4
            progress, message = stage_messages[stage_index]
            if message != last_message:
                update_job(job_id, progress=progress, message=message, log_append=message)
                last_message = message
            time.sleep(0.5)

        thread.join()
        if result["error"] is not None:
            raise result["error"]
        info = result["info"]
        audio_tuple = result["audio_tuple"]
        if audio_tuple is None:
            raise RuntimeError(info)
        update_job(job_id, progress=0.92, message="导出目标音频")
        output_wav = job_dir / "converted.wav"
        output_mp3 = job_dir / "converted.mp3"
        sr, audio = audio_tuple
        import soundfile as sf

        sf.write(output_wav, audio, sr)
        export_mp3(output_wav, output_mp3)
        downloads = [str(output_mp3)]
        update_job(
            job_id,
            status="done",
            progress=1.0,
            message="处理完成",
            result_audio=str(output_mp3),
            downloads=downloads,
            log_append=info.strip(),
        )
    except Exception as exc:
        update_job(
            job_id,
            status="error",
            message="处理失败",
            error=f"{exc}\n{traceback.format_exc()}",
            log_append=f"ERROR: {exc}",
        )


def run_long_job(job_id: str, payload: dict) -> None:
    try:
        input_path = payload["input_path"]
        job_dir = Path(payload["job_dir"])
        male_model = payload["male_model"]
        female_model = payload["female_model"]
        male_index = resolve_index(male_model, payload.get("male_index", ""), True)
        female_index = resolve_index(female_model, payload.get("female_index", ""), True)
        update_job(
            job_id,
            status="running",
            progress=0.02,
            message="准备开始",
            input_path=input_path,
            original_audio=input_path,
        )
        mixed_audio.ensure_environment()
        mixed_audio.MALE_MODEL = male_model
        mixed_audio.MALE_INDEX = male_index
        mixed_audio.FEMALE_MODEL = female_model
        mixed_audio.FEMALE_INDEX = female_index
        mixed_audio.UVR_MODEL = payload["uvr_model"]
        mixed_audio.READING_MODE = bool(payload["reading_mode"])
        if not payload["speaker_embedding"]:
            mixed_audio.SPEAKER_ENCODER_FAILED = True
            mixed_audio.SPEAKER_ENCODER = None
        mixed_audio.MALE_PARAMS = {
            "f0_up_key": int(payload["male_f0_up_key"]),
            "f0_method": payload["f0_method"],
            "index_rate": float(payload["male_index_rate"]),
            "protect": float(payload["male_protect"]),
            "rms_mix_rate": float(payload["male_rms_mix_rate"]),
            "filter_radius": int(payload["filter_radius"]),
            "resample_sr": int(payload["resample_sr"]),
        }
        mixed_audio.FEMALE_PARAMS = {
            "f0_up_key": int(payload["female_f0_up_key"]),
            "f0_method": payload["f0_method"],
            "index_rate": float(payload["female_index_rate"]),
            "protect": float(payload["female_protect"]),
            "rms_mix_rate": float(payload["female_rms_mix_rate"]),
            "filter_radius": int(payload["filter_radius"]),
            "resample_sr": int(payload["resample_sr"]),
        }

        def on_progress(value: float, message: str) -> None:
            update_job(job_id, progress=value, message=message, log_append=message)

        mixed_audio.process_audio(
            Path(input_path),
            job_dir,
            remix=bool(payload["remix"]),
            progress_callback=on_progress,
        )
        converted_vocals = job_dir / "converted_vocals.mp3"
        final_mix = job_dir / "final_mix.mp3"
        segments_json = job_dir / "segments.json"
        segments_csv = job_dir / "segments.csv"
        downloads = [str(path) for path in [converted_vocals, final_mix, segments_json, segments_csv] if path.exists()]
        result_audio = str(final_mix if payload["remix"] and final_mix.exists() else converted_vocals)
        update_job(
            job_id,
            status="done",
            progress=1.0,
            message="处理完成",
            converted_vocals=str(converted_vocals) if converted_vocals.exists() else "",
            result_audio=result_audio,
            downloads=downloads,
        )
    except Exception as exc:
        update_job(
            job_id,
            status="error",
            message="处理失败",
            error=f"{exc}\n{traceback.format_exc()}",
            log_append=f"ERROR: {exc}",
        )


ensure_environment()
app = Flask(
    __name__,
    template_folder=str(REPO_ROOT / "tools" / "templates"),
    static_folder=str(REPO_ROOT / "tools" / "static"),
)


@app.get("/")
def index():
    models = list_models()
    return render_template(
        "simple_rvc_flask.html",
        models=models,
        indices=list_indices(),
        default_model=models[0] if models else "",
        default_male=default_model_choice(models, DEFAULT_MALE_MODEL),
        default_female=default_model_choice(models, DEFAULT_FEMALE_MODEL),
    )


@app.get("/api/options")
def api_options():
    models = list_models()
    return jsonify(
        {
            "models": models,
            "indices": list_indices(),
            "model_presets": MODEL_PRESETS,
            "uvr_models": UVR_MODELS,
            "default_uvr_model": UVR_MODELS[0],
            "default_male": default_model_choice(models, DEFAULT_MALE_MODEL),
            "default_female": default_model_choice(models, DEFAULT_FEMALE_MODEL),
        }
    )


@app.get("/api/default-index")
def api_default_index():
    return jsonify({"index": default_index_for_model(request.args.get("model", ""))})


def save_uploaded_file(file_storage, job_dir: Path) -> str:
    original_name = file_storage.filename or "audio.wav"
    original_path = Path(original_name)
    safe_stem = secure_filename(original_path.stem) or "audio"
    suffix = original_path.suffix or ".wav"
    target = job_dir / f"{safe_stem}_{uuid.uuid4().hex[:6]}{suffix}"
    file_storage.save(target)
    return str(target.resolve())


@app.post("/api/jobs/single")
def api_jobs_single():
    file_storage = request.files.get("audio")
    if not file_storage:
        return jsonify({"error": "缺少音频文件"}), 400
    job_id, job_dir = create_job("single", file_storage.filename or "audio")
    input_path = save_uploaded_file(file_storage, job_dir)
    payload = {
        "job_dir": str(job_dir),
        "input_path": input_path,
        "model_name": request.form.get("model_name", ""),
        "index_path": request.form.get("index_path", ""),
        "use_index": request.form.get("use_index", "true") == "true",
        "f0_up_key": int(request.form.get("f0_up_key", 0)),
        "f0_method": request.form.get("f0_method", "rmvpe"),
        "index_rate": float(request.form.get("index_rate", 0.75)),
        "protect": float(request.form.get("protect", 0.25)),
        "rms_mix_rate": float(request.form.get("rms_mix_rate", 0.18)),
        "filter_radius": int(request.form.get("filter_radius", 3)),
        "resample_sr": int(request.form.get("resample_sr", 0)),
    }
    thread = threading.Thread(target=run_single_job, args=(job_id, payload), daemon=True)
    thread.start()
    return jsonify({"job_id": job_id})


@app.post("/api/jobs/long")
def api_jobs_long():
    file_storage = request.files.get("audio")
    if not file_storage:
        return jsonify({"error": "缺少音频文件"}), 400
    job_id, job_dir = create_job("long", file_storage.filename or "audio")
    input_path = save_uploaded_file(file_storage, job_dir)
    uvr_model = request.form.get("uvr_model", UVR_MODELS[0])
    if uvr_model not in UVR_MODELS:
        uvr_model = UVR_MODELS[0]
    payload = {
        "job_dir": str(job_dir),
        "input_path": input_path,
        "male_model": request.form.get("male_model", ""),
        "male_index": request.form.get("male_index", ""),
        "male_f0_up_key": int(request.form.get("male_f0_up_key", 0)),
        "male_index_rate": float(request.form.get("male_index_rate", 0.75)),
        "male_protect": float(request.form.get("male_protect", 0.30)),
        "male_rms_mix_rate": float(request.form.get("male_rms_mix_rate", 0.20)),
        "female_model": request.form.get("female_model", ""),
        "female_index": request.form.get("female_index", ""),
        "female_f0_up_key": int(request.form.get("female_f0_up_key", 0)),
        "female_index_rate": float(request.form.get("female_index_rate", 0.75)),
        "female_protect": float(request.form.get("female_protect", 0.25)),
        "female_rms_mix_rate": float(request.form.get("female_rms_mix_rate", 0.18)),
        "uvr_model": uvr_model,
        "f0_method": request.form.get("f0_method", "rmvpe"),
        "filter_radius": int(request.form.get("filter_radius", 3)),
        "resample_sr": int(request.form.get("resample_sr", 0)),
        "remix": request.form.get("remix", "true") == "true",
        "reading_mode": request.form.get("reading_mode", "true") == "true",
        "speaker_embedding": request.form.get("speaker_embedding", "false") == "true",
    }
    thread = threading.Thread(target=run_long_job, args=(job_id, payload), daemon=True)
    thread.start()
    return jsonify({"job_id": job_id})


@app.get("/api/jobs/<job_id>")
def api_job_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "job not found"}), 404
        return jsonify(job)


@app.get("/download")
def download():
    path = request.args.get("path", "")
    if not path:
        return jsonify({"error": "missing path"}), 400
    file_path = Path(path).resolve()
    if not file_path.exists():
        return jsonify({"error": "file not found"}), 404
    return send_file(file_path, as_attachment=True)


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
