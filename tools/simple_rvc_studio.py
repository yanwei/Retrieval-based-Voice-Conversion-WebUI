import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import gradio as gr
import soundfile as sf
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from configs.config import Config
from infer.modules.vc.modules import VC
from infer.modules.vc.utils import get_index_path_from_model
from tools import process_mixed_long_audio as mixed_audio


APP_OUTPUT_ROOT = REPO_ROOT / "outputs" / "simple_rvc_studio"
DEFAULT_PORT = 7866
VC_CACHE: Dict[str, VC] = {}


def ensure_environment() -> None:
    load_dotenv(REPO_ROOT / ".env")
    os.environ.setdefault("weight_root", "assets/weights")
    os.environ.setdefault("weight_uvr5_root", "assets/uvr5_weights")
    os.environ.setdefault("index_root", "logs")
    os.environ.setdefault("outside_index_root", "assets/indices")
    os.environ.setdefault("rmvpe_root", "assets/rmvpe")
    os.environ.setdefault("TEMP", "/tmp")
    APP_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


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


def make_job_dir(prefix: str, input_path: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(input_path).stem
    job_dir = APP_OUTPUT_ROOT / f"{prefix}_{stem}_{stamp}"
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def get_vc(model_name: str) -> VC:
    model_name = (model_name or "").strip()
    if not model_name:
        raise ValueError("请选择模型。")
    if model_name not in VC_CACHE:
        vc = VC(Config())
        vc.get_vc(model_name)
        VC_CACHE[model_name] = vc
    return VC_CACHE[model_name]


def export_mp3(input_wav: Path, output_mp3: Path) -> None:
    mixed_audio.export_mp3(input_wav, output_mp3)


def save_audio_tuple(audio_tuple: Tuple[int, object], output_wav: Path) -> Path:
    sr, audio = audio_tuple
    sf.write(output_wav, audio, sr)
    return output_wav


def normalize_input_path(input_audio) -> str:
    if input_audio is None:
        return ""
    if isinstance(input_audio, str):
        return str(Path(input_audio).expanduser().resolve())
    if hasattr(input_audio, "name"):
        return str(Path(input_audio.name).expanduser().resolve())
    raise ValueError(f"不支持的输入文件类型: {type(input_audio)!r}")


def resolve_index(model_name: str, selected_index: str, use_index: bool) -> str:
    if not use_index:
        return ""
    selected_index = (selected_index or "").strip()
    if selected_index:
        return selected_index
    return default_index_for_model(model_name)


def convert_single(
    input_audio: str,
    model_name: str,
    index_path: str,
    use_index: bool,
    f0_up_key: int,
    f0_method: str,
    index_rate: float,
    protect: float,
    rms_mix_rate: float,
    filter_radius: int,
    resample_sr: int,
    progress=gr.Progress(),
):
    ensure_environment()
    if not input_audio:
        raise gr.Error("请先选择原始音频文件。")

    input_path = normalize_input_path(input_audio)
    progress(0.05, desc="加载模型")
    vc = get_vc(model_name)
    resolved_index = resolve_index(model_name, index_path, use_index)
    progress(0.15, desc="开始转换")
    info, audio_tuple = vc.vc_single(
        0,
        input_path,
        int(f0_up_key),
        None,
        f0_method,
        resolved_index,
        resolved_index,
        float(index_rate),
        int(filter_radius),
        int(resample_sr),
        float(rms_mix_rate),
        float(protect),
    )
    if audio_tuple is None:
        raise gr.Error(info)

    job_dir = make_job_dir("single", input_path)
    output_wav = job_dir / "converted.wav"
    output_mp3 = job_dir / "converted.mp3"
    progress(0.92, desc="导出音频")
    save_audio_tuple(audio_tuple, output_wav)
    export_mp3(output_wav, output_mp3)
    progress(1.0, desc="处理完成")

    summary = "\n".join(
        [
            info.strip(),
            f"模型: {model_name}",
            f"索引: {resolved_index or '未使用'}",
            f"输出目录: {job_dir}",
        ]
    )
    return summary, input_path, str(output_mp3), str(output_mp3)


def run_long_audio(
    input_audio: str,
    male_model: str,
    male_index: str,
    male_f0_up_key: int,
    male_index_rate: float,
    male_protect: float,
    male_rms_mix_rate: float,
    female_model: str,
    female_index: str,
    female_f0_up_key: int,
    female_index_rate: float,
    female_protect: float,
    female_rms_mix_rate: float,
    f0_method: str,
    filter_radius: int,
    resample_sr: int,
    remix: bool,
    progress=gr.Progress(),
):
    ensure_environment()
    if not input_audio:
        raise gr.Error("请先选择原始音频文件。")

    input_path = normalize_input_path(input_audio)
    output_dir = make_job_dir("long", input_path)

    male_index = resolve_index(male_model, male_index, True)
    female_index = resolve_index(female_model, female_index, True)

    mixed_audio.ensure_environment()
    mixed_audio.MALE_MODEL = male_model
    mixed_audio.MALE_INDEX = Path(male_index).name if male_index else ""
    mixed_audio.FEMALE_MODEL = female_model
    mixed_audio.FEMALE_INDEX = Path(female_index).name if female_index else ""
    mixed_audio.MALE_PARAMS = {
        "f0_up_key": int(male_f0_up_key),
        "f0_method": f0_method,
        "index_rate": float(male_index_rate),
        "protect": float(male_protect),
        "rms_mix_rate": float(male_rms_mix_rate),
        "filter_radius": int(filter_radius),
        "resample_sr": int(resample_sr),
    }
    mixed_audio.FEMALE_PARAMS = {
        "f0_up_key": int(female_f0_up_key),
        "f0_method": f0_method,
        "index_rate": float(female_index_rate),
        "protect": float(female_protect),
        "rms_mix_rate": float(female_rms_mix_rate),
        "filter_radius": int(filter_radius),
        "resample_sr": int(resample_sr),
    }
    log_lines: list[str] = []

    def on_progress(value: float, message: str) -> None:
        progress(value, desc=message)
        log_lines.append(message)

    mixed_audio.process_audio(
        Path(input_path),
        output_dir,
        remix=bool(remix),
        progress_callback=on_progress,
        log_callback=log_lines.append,
    )

    converted_vocals = output_dir / "converted_vocals.mp3"
    final_mix = output_dir / "final_mix.mp3"
    segments_json = output_dir / "segments.json"
    segments_csv = output_dir / "segments.csv"

    lines = [
        "长音频处理完成。",
        f"男声模型: {male_model}",
        f"男声索引: {male_index or '未使用'}",
        f"女声模型: {female_model}",
        f"女声索引: {female_index or '未使用'}",
        f"输出目录: {output_dir}",
    ]
    if log_lines:
        lines.append("处理过程:")
        lines.extend(f"- {line}" for line in log_lines[:30])
    if segments_json.exists():
        lines.append(f"分段日志: {segments_json}")

    final_audio = str(final_mix if remix and final_mix.exists() else converted_vocals)
    downloads = [str(converted_vocals)] if converted_vocals.exists() else []
    if final_mix.exists():
        downloads.append(str(final_mix))
    if segments_json.exists():
        downloads.append(str(segments_json))
    if segments_csv.exists():
        downloads.append(str(segments_csv))

    return "\n".join(lines), input_path, str(converted_vocals), final_audio, downloads


def refresh_choices():
    models = list_models()
    indices = list_indices()
    default_model = models[0] if models else None
    default_index = default_index_for_model(default_model) if default_model else ""
    model_update = gr.Dropdown.update(choices=models, value=default_model)
    index_update = gr.Dropdown.update(choices=indices, value=default_index)
    return (
        model_update,
        index_update,
        model_update,
        index_update,
        model_update,
        index_update,
    )


def update_index_for_model(model_name: str):
    return gr.Dropdown.update(value=default_index_for_model(model_name))


def build_app() -> gr.Blocks:
    ensure_environment()
    models = list_models()
    indices = list_indices()
    default_model = models[0] if models else None
    default_index = default_index_for_model(default_model) if default_model else ""

    with gr.Blocks(title="Simple RVC Studio") as app:
        gr.Markdown("## Simple RVC Studio")
        gr.Markdown("用于快速试模型、调参数、试听和下载结果。单文件和长音频/带音乐流程分开。")

        refresh = gr.Button("刷新模型和索引列表", variant="secondary")

        with gr.Tabs():
            with gr.Tab("单文件转换"):
                with gr.Row():
                    with gr.Column():
                        single_input = gr.Audio(
                            label="原始音频试听",
                            interactive=False,
                        )
                        single_input_file = gr.File(
                            label="选择原始音频文件",
                            file_types=["audio"],
                            type="file",
                        )
                        single_model = gr.Dropdown(
                            label="模型",
                            choices=models,
                            value=default_model,
                            interactive=True,
                        )
                        single_index = gr.Dropdown(
                            label="索引",
                            choices=indices,
                            value=default_index,
                            allow_custom_value=True,
                            interactive=True,
                        )
                        single_use_index = gr.Checkbox(label="使用索引", value=True)
                        single_f0_method = gr.Radio(
                            label="音高算法",
                            choices=["pm", "harvest", "crepe", "rmvpe"],
                            value="rmvpe",
                        )
                        single_f0_up_key = gr.Slider(
                            label="变调",
                            minimum=-24,
                            maximum=24,
                            step=1,
                            value=0,
                        )
                        single_index_rate = gr.Slider(
                            label="检索特征占比",
                            minimum=0,
                            maximum=1,
                            step=0.01,
                            value=0.75,
                        )
                        single_protect = gr.Slider(
                            label="protect",
                            minimum=0,
                            maximum=0.5,
                            step=0.01,
                            value=0.25,
                        )
                        single_rms = gr.Slider(
                            label="rms_mix_rate",
                            minimum=0,
                            maximum=1,
                            step=0.01,
                            value=0.18,
                        )
                        single_filter_radius = gr.Slider(
                            label="filter_radius",
                            minimum=0,
                            maximum=7,
                            step=1,
                            value=3,
                        )
                        single_resample_sr = gr.Slider(
                            label="resample_sr",
                            minimum=0,
                            maximum=48000,
                            step=1,
                            value=0,
                        )
                        single_run = gr.Button("开始转换", variant="primary")

                    with gr.Column():
                        single_info = gr.Textbox(label="处理信息", lines=8)
                        single_original_preview = gr.Audio(label="原始音频试听")
                        single_output_preview = gr.Audio(label="目标音频试听")
                        single_download = gr.File(label="下载目标音频")

            with gr.Tab("长音频 / 音乐"):
                with gr.Row():
                    with gr.Column():
                        long_input = gr.Audio(
                            label="原始音频试听",
                            interactive=False,
                        )
                        long_input_file = gr.File(
                            label="选择原始音频文件",
                            file_types=["audio"],
                            type="file",
                        )
                        long_f0_method = gr.Radio(
                            label="音高算法",
                            choices=["pm", "harvest", "crepe", "rmvpe"],
                            value="rmvpe",
                        )
                        long_filter_radius = gr.Slider(
                            label="filter_radius",
                            minimum=0,
                            maximum=7,
                            step=1,
                            value=3,
                        )
                        long_resample_sr = gr.Slider(
                            label="resample_sr",
                            minimum=0,
                            maximum=48000,
                            step=1,
                            value=0,
                        )
                        long_remix = gr.Checkbox(label="分离人声后转换并混回伴奏", value=True)

                        with gr.Accordion("男声输入分路参数", open=True):
                            male_model = gr.Dropdown(
                                label="男声输入用模型",
                                choices=models,
                                value="man-B.pth" if "man-B.pth" in models else default_model,
                                interactive=True,
                            )
                            male_index = gr.Dropdown(
                                label="男声输入用索引",
                                choices=indices,
                                value=default_index_for_model("man-B.pth")
                                if "man-B.pth" in models
                                else default_index,
                                allow_custom_value=True,
                                interactive=True,
                            )
                            male_f0_up_key = gr.Slider(
                                label="男声分路变调",
                                minimum=-24,
                                maximum=24,
                                step=1,
                                value=0,
                            )
                            male_index_rate = gr.Slider(
                                label="男声 index_rate",
                                minimum=0,
                                maximum=1,
                                step=0.01,
                                value=0.75,
                            )
                            male_protect = gr.Slider(
                                label="男声 protect",
                                minimum=0,
                                maximum=0.5,
                                step=0.01,
                                value=0.30,
                            )
                            male_rms = gr.Slider(
                                label="男声 rms_mix_rate",
                                minimum=0,
                                maximum=1,
                                step=0.01,
                                value=0.20,
                            )

                        with gr.Accordion("女声输入分路参数", open=False):
                            female_model = gr.Dropdown(
                                label="女声输入用模型",
                                choices=models,
                                value=(
                                    "splicegirl_v3_e130_s26520.pth"
                                    if "splicegirl_v3_e130_s26520.pth" in models
                                    else default_model
                                ),
                                interactive=True,
                            )
                            female_index = gr.Dropdown(
                                label="女声输入用索引",
                                choices=indices,
                                value=default_index_for_model("splicegirl_v3_e130_s26520.pth")
                                if "splicegirl_v3_e130_s26520.pth" in models
                                else default_index,
                                allow_custom_value=True,
                                interactive=True,
                            )
                            female_f0_up_key = gr.Slider(
                                label="女声分路变调",
                                minimum=-24,
                                maximum=24,
                                step=1,
                                value=0,
                            )
                            female_index_rate = gr.Slider(
                                label="女声 index_rate",
                                minimum=0,
                                maximum=1,
                                step=0.01,
                                value=0.75,
                            )
                            female_protect = gr.Slider(
                                label="女声 protect",
                                minimum=0,
                                maximum=0.5,
                                step=0.01,
                                value=0.25,
                            )
                            female_rms = gr.Slider(
                                label="女声 rms_mix_rate",
                                minimum=0,
                                maximum=1,
                                step=0.01,
                                value=0.18,
                            )
                        long_run = gr.Button("开始长音频处理", variant="primary")

                    with gr.Column():
                        long_info = gr.Textbox(label="处理信息", lines=10)
                        long_original_preview = gr.Audio(label="原始音频试听")
                        long_vocals_preview = gr.Audio(label="转换后人声试听")
                        long_final_preview = gr.Audio(label="最终成品试听")
                        long_downloads = gr.File(label="下载输出文件", file_count="multiple")

        refresh.click(
            refresh_choices,
            outputs=[single_model, single_index, male_model, male_index, female_model, female_index],
        )
        single_model.change(update_index_for_model, inputs=[single_model], outputs=[single_index])
        male_model.change(update_index_for_model, inputs=[male_model], outputs=[male_index])
        female_model.change(update_index_for_model, inputs=[female_model], outputs=[female_index])

        single_run.click(
            convert_single,
            inputs=[
                single_input_file,
                single_model,
                single_index,
                single_use_index,
                single_f0_up_key,
                single_f0_method,
                single_index_rate,
                single_protect,
                single_rms,
                single_filter_radius,
                single_resample_sr,
            ],
            outputs=[single_info, single_original_preview, single_output_preview, single_download],
        )
        long_run.click(
            run_long_audio,
            inputs=[
                long_input_file,
                male_model,
                male_index,
                male_f0_up_key,
                male_index_rate,
                male_protect,
                male_rms,
                female_model,
                female_index,
                female_f0_up_key,
                female_index_rate,
                female_protect,
                female_rms,
                long_f0_method,
                long_filter_radius,
                long_resample_sr,
                long_remix,
            ],
            outputs=[
                long_info,
                long_original_preview,
                long_vocals_preview,
                long_final_preview,
                long_downloads,
            ],
        )
        single_input_file.change(
            lambda file_obj: normalize_input_path(file_obj) if file_obj else None,
            inputs=[single_input_file],
            outputs=[single_input],
        )
        long_input_file.change(
            lambda file_obj: normalize_input_path(file_obj) if file_obj else None,
            inputs=[long_input_file],
            outputs=[long_input],
        )
    return app


if __name__ == "__main__":
    build_app().queue().launch(server_name="127.0.0.1", server_port=DEFAULT_PORT, share=False)
