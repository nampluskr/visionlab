# src/visionlab/utils/io.py: local teacher weights, pure exports and reproducible run metadata

import hashlib
import json
import os
import re
import socket
import tomllib

PATHS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "paths.toml"))


def read_backbone_path(name):
    """Resolve a required local backbone file from the project mapping."""
    with open(PATHS_FILE, "rb") as stream:
        backbones = tomllib.load(stream).get("backbones", {})
    if name not in backbones:
        raise FileNotFoundError(f"Missing backbone '{name}' in {PATHS_FILE}")
    path = os.path.abspath(backbones[name])
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def load_teacher_weights(feature_extractor, backbone):
    """Load local full-backbone weights while allowing only omitted top-level modules."""
    import torch

    path = read_backbone_path(backbone)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise RuntimeError("Backbone file must contain a state dict")
    target = feature_extractor.feature_extractor
    expected = set(target.state_dict())
    # Older torchvision weights omit BatchNorm counters, which retain their initial zero value.
    missing = [key for key in expected - set(state) if not key.endswith(".num_batches_tracked")]
    unexpected = set(state) - expected
    invalid = [key for key in unexpected if "." not in key or key.split(".", 1)[0] in target._modules]
    if missing or invalid:
        raise RuntimeError(f"Invalid teacher weights: missing={sorted(missing)}, unexpected={sorted(invalid)}")
    result = target.load_state_dict(state, strict=False)
    missing = [key for key in result.missing_keys if not key.endswith(".num_batches_tracked")]
    if missing:
        raise RuntimeError(f"Missing teacher weights: {missing}")
    return path


def file_sha256(path):
    """Hash a file in bounded chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pretrained_teacher(module, key):
    """Inject a required local teacher state with strict key validation."""
    import torch

    with open(PATHS_FILE, "rb") as stream:
        teachers = tomllib.load(stream).get("teachers", {})
    if key not in teachers:
        raise FileNotFoundError(f"Missing teacher '{key}' in {PATHS_FILE}")
    path = os.path.abspath(teachers[key])
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    module.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return path


def _deny_connection(*args, **kwargs):
    """Reject every socket connection signature."""
    raise RuntimeError("network access is blocked")


def block_network():
    """Disable network connections before importing training or model libraries."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TEMP", "."), "visionlab-mpl"))
    socket.socket.connect = _deny_connection
    socket.socket.connect_ex = _deny_connection
    socket.create_connection = _deny_connection


def create_versioned_dir(root):
    """Create a new numbered run directory without symlinks."""
    root = os.path.abspath(root)
    os.makedirs(root, exist_ok=True)
    versions = [int(match.group(1)) for name in os.listdir(root)
                if os.path.isdir(os.path.join(root, name)) and (match := re.fullmatch(r"v(\d+)", name))]
    version = max(versions, default=-1) + 1
    while True:
        path = os.path.join(root, f"v{version}")
        try:
            os.mkdir(path)
            return path
        except FileExistsError:
            version += 1


def save_export(path, model, model_name, model_args, pre_processor, post_processor):
    """Save teacher-inclusive pure model weights and processing state."""
    import torch

    mean = None if pre_processor is None or pre_processor.mean is None else pre_processor.mean.flatten().tolist()
    std = None if pre_processor is None or pre_processor.std is None else pre_processor.std.flatten().tolist()
    payload = {
        "format_version": 1,
        "model_name": model_name,
        "model_args": model_args,
        "model_state": {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()},
        "pre_processor": {"mean": mean, "std": std,
                          "center_crop": None if pre_processor is None or pre_processor.center_crop is None
                          else list(pre_processor.center_crop)},
        "post_processor": None if post_processor is None else post_processor.state_dict(),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(payload, path)


def load_export(path):
    """Load a tensor-only export without importing any training framework."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"format_version", "model_name", "model_args", "model_state", "pre_processor", "post_processor"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("Invalid model export fields")
    if payload["format_version"] != 1 or payload["model_name"] not in (
            "stfpm", "efficient_ad", "padim", "dfm", "dfkde", "patchcore", "reverse_distillation", "cfa", "fre",
            "supersimplenet", "draem", "fastflow", "cflow", "uflow", "csflow", "ganomaly", "dsr", "winclip"):
        raise ValueError("Unsupported model export format")
    payload["pre_processor"].setdefault("center_crop", None)
    return payload


def write_run_info(run_dir, config, datamodule, teacher_path, seed, checkpoint_path=None):
    """Record final configuration, exact split paths and local asset identities."""
    splits = {}
    for stage in ("train", "val", "test"):
        dataset = getattr(datamodule, f"{stage}_data", None)
        splits[stage] = None if dataset is None else [sample["image_path"] for sample in dataset.samples]
    checkpoint = None if not checkpoint_path else {
        "path": os.path.abspath(checkpoint_path), "sha256": file_sha256(checkpoint_path),
    }
    record = {"config": config, "splits": splits, "seed": seed,
              "teacher_weights": None if teacher_path is None else {
                  "path": os.path.abspath(teacher_path), "sha256": file_sha256(teacher_path)},
              "checkpoint": checkpoint}
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")
