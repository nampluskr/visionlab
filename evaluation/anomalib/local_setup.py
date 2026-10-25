# evaluation/anomalib/local_setup.py: offline setup, Windows run folder patch and local timm backbones for the anomalib scripts

import os

# Set before anomalib/timm import: no Hugging Face downloads, writable mpl cache.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TEMP", "."), "anomalib-mpl"))
# Required by torch.use_deterministic_algorithms for cuBLAS on CUDA >= 10.2.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import re
import socket
import tomllib
from pathlib import Path

import timm.models
from timm.models import register_model

import anomalib.engine.engine as anomalib_engine


PATHS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "paths.toml")


def block_network(*args, **kwargs):
    """Raise on any outgoing connection attempt."""
    raise RuntimeError("network access is blocked")


def create_versioned_dir(root_dir):
    """anomalib's create_versioned_dir with a junction fallback for Windows."""
    # A symlink needs admin rights or developer mode on Windows (WinError 1314);
    # a directory junction does not.
    root_dir = os.path.abspath(root_dir)
    os.makedirs(root_dir, exist_ok=True)
    versions = [int(m.group(1)) for name in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, name)) and (m := re.match(r"^v(\d+)$", name))]
    new_version_dir = os.path.join(root_dir, f"v{max(versions, default=-1) + 1}")
    os.mkdir(new_version_dir)

    latest_link_path = os.path.join(root_dir, "latest")
    if os.path.lexists(latest_link_path):
        try:
            os.unlink(latest_link_path)
        except (IsADirectoryError, PermissionError):
            os.rmdir(latest_link_path)  # junction
    try:
        os.symlink(new_version_dir, latest_link_path, target_is_directory=True)
    except OSError:
        import _winapi
        _winapi.CreateJunction(new_version_dir, latest_link_path)
    # Engine joins the result with "/", so it must be a Path.
    return Path(latest_link_path)


def register_local_backbone(name, weights_file):
    """Replace timm's '<name>' entrypoint with one that loads the given local weights file."""
    factory = timm.models.model_entrypoint(name)

    def local_model(pretrained=False, **kwargs):
        if pretrained and not os.path.isfile(weights_file):
            raise FileNotFoundError(weights_file)
        kwargs["pretrained_cfg_overlay"] = {"file": weights_file, "hf_hub_id": None, "url": ""}
        return factory(pretrained=pretrained, **kwargs)

    # register_model names the model after the function. The original name is kept (timm warns
    # about the overwrite) because anomalib checks backbone names, e.g. fastflow and reverse_distillation.
    local_model.__name__ = name
    register_model(local_model)


def setup_local():
    """Block downloads, patch the Windows run folder and register the local timm backbones."""
    with open(PATHS_FILE, "rb") as f:
        backbones = tomllib.load(f)["backbones"]
    socket.socket.connect = block_network
    socket.socket.connect_ex = block_network
    socket.create_connection = block_network
    anomalib_engine.create_versioned_dir = create_versioned_dir
    for name, weights_file in backbones.items():
        register_local_backbone(name, weights_file)
