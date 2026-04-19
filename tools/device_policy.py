from __future__ import annotations


VALID_DEVICE_MODES = {"auto", "cpu", "gpu"}


def normalize_device_mode(mode: str | None) -> str:
    value = (mode or "auto").strip().lower()
    if value not in VALID_DEVICE_MODES:
        return "auto"
    return value


def choose_rvc_force_cpu(mode: str | None) -> bool:
    return normalize_device_mode(mode) == "cpu"


def choose_uvr_device(
    mode: str | None, config_device: str, config_is_half: bool = False
) -> tuple[str, bool]:
    normalized = normalize_device_mode(mode)
    device = str(config_device or "cpu")
    if normalized == "cpu":
        return "cpu", False
    if normalized in {"auto", "gpu"} and device != "cpu":
        if device == "mps":
            return "mps", False
        return device, bool(config_is_half)
    return "cpu", False
