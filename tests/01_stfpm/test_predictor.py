# tests/01_stfpm/test_predictor.py: pytest export-only inference across isolated native and pure processes

import importlib.abc
import os
import runpy
import subprocess
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BlockImports(importlib.abc.MetaPathFinder):
    """Reject framework imports before they can enter either subprocess."""

    def __init__(self, forbidden):
        self.forbidden = forbidden

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.forbidden:
            raise ModuleNotFoundError(f"Forbidden import: {fullname}")
        return None


def produce_export(root):
    """Calibrate a native wrapper on temporary images and save its expected predictions."""
    sys.meta_path.insert(0, BlockImports({"anomalib"}))
    from visionlab.utils import io as io_utils

    io_utils.block_network()
    import torch
    from torch.utils.data import DataLoader

    from visionlab.anomaly_detection.data.mvtec.dataset import MVTecDataset
    from visionlab.anomaly_detection.models.stfpm.lightning_model import StfpmLitModel

    torch.manual_seed(42)
    helpers = runpy.run_path(os.path.join(PROJECT_ROOT, "tests", "01_stfpm", "test_lightning.py"))
    paths_file = helpers["make_assets"](root)
    data_root = os.path.join(root, "data")
    helpers["make_fixture"](data_root)
    dataset = MVTecDataset(root=data_root, category="bottle", split="test")
    batch = next(iter(DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(io_utils, "PATHS_FILE", paths_file)
        wrapper = StfpmLitModel(evaluator=False, visualizer=False).eval()
    with torch.no_grad():
        raw = wrapper.predict_step(batch, 0)
        wrapper.post_processor.update(raw)
        wrapper.post_processor.finalize()
        outputs = wrapper.predict_step(batch, 0)
    names = ("pred_score", "anomaly_map", "normalized_pred_score", "normalized_anomaly_map", "pred_label", "pred_mask")
    torch.save({"images": batch[0], "outputs": {name: outputs[name] for name in names}},
               os.path.join(root, "expected.pt"))
    export_path = os.path.join(root, "export", "model.pt")
    io_utils.save_export(path=export_path, model=wrapper.model, model_name="stfpm",
                         model_args={"backbone": wrapper.backbone, "layers": list(wrapper.layers)},
                         pre_processor=wrapper.pre_processor, post_processor=wrapper.post_processor)
    exported = io_utils.load_export(export_path)
    del exported["model_state"][next(iter(exported["model_state"]))]
    torch.save(exported, os.path.join(root, "broken.pt"))
    os.remove(paths_file)
    os.remove(os.path.join(root, "resnet18.pth"))
    print("PASS native wrapper export and expected predictions")


def consume_export(root):
    """Restore exports with dependency, file-open and socket guards in a fresh process."""
    forbidden = {"anomalib", "lightning", "pytorch_lightning"}
    sys.meta_path.insert(0, BlockImports(forbidden))
    from visionlab.utils import io as io_utils

    io_utils.block_network()
    import socket
    import torch

    from visionlab.inference.predictor import Predictor

    expected = torch.load(os.path.join(root, "expected.pt"), map_location="cpu", weights_only=True)
    export_path = os.path.join(root, "export", "model.pt")
    broken_path = os.path.join(root, "broken.pt")
    exported = io_utils.load_export(export_path)
    assert any(name.startswith("teacher_model.") for name in exported["model_state"])
    assert not os.path.exists(os.path.join(root, "paths.toml"))
    assert not os.path.exists(os.path.join(root, "resnet18.pth"))
    allowed = {os.path.normcase(os.path.abspath(path)) for path in (export_path, broken_path)}
    active = [False]
    opened = []

    def audit_open(event, args):
        if active[0] and event == "open":
            path, mode, flags = args
            normalized = os.path.normcase(os.path.abspath(path)) if isinstance(path, (str, bytes)) else None
            if normalized not in allowed or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                raise AssertionError(f"Unexpected file access during inference: {path}")
            opened.append(normalized)

    sys.addaudithook(audit_open)
    images = expected["images"]
    original = images.clone()
    network_attempts = []

    def reject_network(*args, **kwargs):
        network_attempts.append(True)
        raise AssertionError("Network access")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(socket.socket, "connect", reject_network)
        patch.setattr(socket.socket, "connect_ex", reject_network)
        patch.setattr(socket, "create_connection", reject_network)
        active[0] = True
        try:
            predictor = Predictor.from_export(export_path)
            with pytest.raises(RuntimeError, match=r"Missing key\(s\)"):
                Predictor.from_export(broken_path)
        finally:
            active[0] = False
        actual = predictor.predict(images)
        assert not network_attempts
    assert opened and set(opened) == allowed
    assert not predictor.model.training
    assert torch.equal(images, original)
    assert set(actual) == set(expected["outputs"])
    for name, value in actual.items():
        assert not value.requires_grad, name
        torch.testing.assert_close(value, expected["outputs"][name], rtol=1e-4, atol=1e-5, equal_nan=False)
    assert not any(name.split(".")[0] in forbidden for name in sys.modules)
    print("PASS FR-14 FR-19 FR-22 strict export-only inference without training dependencies or assets")


def test_export_round_trip_in_isolated_processes(tmp_path):
    """Compare native exports and pure predictions in separate cache-free interpreters."""
    script = '''
import runpy
import sys
sys.path.insert(0, sys.argv[2])
runpy.run_path(sys.argv[1])[sys.argv[3]](sys.argv[4])
'''
    for phase in ("produce_export", "consume_export"):
        result = subprocess.run([sys.executable, "-c", script, os.path.abspath(__file__),
                                 os.path.join(PROJECT_ROOT, "src"), phase, str(tmp_path)],
                                capture_output=True, text=True, check=False)
        assert result.returncode == 0, f"{phase}\n{result.stdout}\n{result.stderr}"
