# src/visionlab/bootstrap.py: load local asset paths, block downloads and record a run

import hashlib
import json
import os
import socket
import tomllib


def block_network(*args, **kwargs):
    """Raise on any outgoing connection attempt."""
    raise RuntimeError("network access is blocked")


def bootstrap(paths_file, required, run_dir, offline=True):
    """Load paths.toml, check required assets, block downloads and create the run folder."""
    with open(paths_file, "rb") as f:
        paths = tomllib.load(f)
    for key in required:
        section, name = key.split(".")
        if not os.path.exists(paths[section][name]):
            raise FileNotFoundError(f"{key}: {paths[section][name]}")
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        socket.socket.connect = block_network
        socket.socket.connect_ex = block_network
        socket.create_connection = block_network
    os.makedirs(run_dir, exist_ok=True)
    return {"paths": paths, "required": list(required), "run_dir": run_dir}


def file_sha256(path):
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_run(ctx, seed, metrics, files=None):
    """Write resolved paths, file hashes, seed and metrics to run.json."""
    hashes = {}
    for key in ctx["required"]:
        section, name = key.split(".")
        path = ctx["paths"][section][name]
        if os.path.isfile(path):
            hashes[key] = file_sha256(path)
    # Extra files outside paths.toml, e.g. the checkpoint under evaluation.
    for name, path in (files or {}).items():
        hashes[name] = file_sha256(path)
    record = {"paths": ctx["paths"], "hashes": hashes, "seed": seed, "metrics": metrics}
    with open(os.path.join(ctx["run_dir"], "run.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, indent=2)
