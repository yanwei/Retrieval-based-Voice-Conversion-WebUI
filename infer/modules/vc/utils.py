import os

from fairseq import checkpoint_utils


def get_index_path_from_model(sid):
    stem = sid.split(".")[0]
    roots = [os.getenv("outside_index_root"), os.getenv("index_root")]
    candidates = []
    for base in roots:
        if not base or not os.path.exists(base):
            continue
        for root, _, files in os.walk(base, topdown=False):
            for name in files:
                if name.endswith(".index") and "trained" not in name:
                    candidates.append(os.path.join(root, name))

    exact = next(
        (
            path
            for path in candidates
            if os.path.splitext(os.path.basename(path))[0].lower() == stem.lower()
        ),
        "",
    )
    if exact:
        return exact

    return next((path for path in candidates if stem.lower() in path.lower()), "")


def load_hubert(config):
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["assets/hubert/hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()
