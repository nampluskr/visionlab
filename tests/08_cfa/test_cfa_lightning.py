# tests/08_cfa/test_cfa_lightning.py: pytest native CFA centroid lifecycle, learning, checkpoints and pure exports

import inspect
import json
import os
import subprocess
import sys
import runpy
from types import SimpleNamespace

import pytest
import torch
import torchvision

from visionlab.anomaly_detection.common.pre_processor import IMAGENET_MEAN, IMAGENET_STD, PreProcessor
from visionlab.utils import io as io_utils

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_cfa_data():
    """Provide 256-pixel raw RGB images with a batch-one evaluation tail."""
    from lightning.pytorch import LightningDataModule

    class TinyData(LightningDataModule):
        """Supply small fixed train, validation, test and prediction datasets."""

        def __init__(self):
            super().__init__()
            self.train_data = [(torch.rand(3, 256, 256), {"label": 0}) for _ in range(2)]
            self.val_data = []
            for index in range(3):
                mask = torch.zeros(256, 256, dtype=torch.int64)
                if index % 2:
                    mask[64:128, 64:128] = 1
                self.val_data.append((torch.rand(3, 256, 256),
                                      {"label": index % 2, "mask": mask, "image_path": f"{index}.png",
                                       "sample_id": f"fixture/{index}.png"}))

        def train_dataloader(self):
            return torch.utils.data.DataLoader(self.train_data, batch_size=1, shuffle=False, num_workers=0)

        def val_dataloader(self):
            return torch.utils.data.DataLoader(self.val_data, batch_size=2, shuffle=False, num_workers=0)

        def test_dataloader(self):
            return self.val_dataloader()

        def predict_dataloader(self):
            return self.val_dataloader()

    return TinyData()


@pytest.fixture(scope="module")
def cfa_assets(tmp_path_factory):
    """Map an actual torchvision-format local ResNet18 state for mandatory teacher injection."""
    root = str(tmp_path_factory.mktemp("cfa_assets"))
    weights = os.path.join(root, "resnet18.pth")
    torch.save(torchvision.models.resnet18(weights=None).state_dict(), weights)
    mapping = os.path.join(root, "paths.toml")
    with open(mapping, "w", encoding="utf-8", newline="\n") as stream:
        stream.write('[backbones]\nresnet18 = "' + weights.replace("\\", "/") + '"\n')
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(io_utils, "PATHS_FILE", mapping)
        yield root


def test_cfa_native_signature_local_weights_and_missing_assets(cfa_assets, tmp_path, monkeypatch):
    """Keep ten constructor defaults and require exact local torchvision weights."""
    from visionlab.anomaly_detection.models.cfa.lightning_model import CfaLitModel

    expected = {"backbone": "wide_resnet50_2", "gamma_c": 1, "gamma_d": 1, "num_nearest_neighbors": 3,
                "num_hard_negative_features": 3, "radius": 1e-5, "pre_processor": True,
                "post_processor": True, "evaluator": True, "visualizer": True}
    signature = inspect.signature(CfaLitModel.__init__)
    assert list(signature.parameters) == ["self"] + list(expected)
    assert {name: signature.parameters[name].default for name in expected} == expected
    model = CfaLitModel(backbone="resnet18", gamma_d=56, visualizer=False)
    source = torch.load(os.path.join(cfa_assets, "resnet18.pth"), map_location="cpu", weights_only=True)
    for name, value in model.model.feature_extractor.state_dict().items():
        assert torch.equal(value, source[name]), name
    assert model.teacher_weights_path == os.path.join(cfa_assets, "resnet18.pth")
    assert model.pre_processor.center_crop is None and model.learning_type.value == "one_class"
    assert model.trainer_arguments == {"gradient_clip_val": 0, "num_sanity_val_steps": 0}
    with pytest.raises(ValueError, match="Memory bank is not initialized"):
        model(torch.rand(1, 3, 256, 256))
    monkeypatch.setattr(io_utils, "PATHS_FILE", os.path.join(tmp_path, "missing.toml"))
    with pytest.raises(FileNotFoundError):
        CfaLitModel(backbone="resnet18", gamma_d=56, visualizer=False)
    mapping = os.path.join(tmp_path, "paths.toml")
    with open(mapping, "w", encoding="utf-8", newline="\n") as stream:
        stream.write('[backbones]\nresnet18 = "missing.pth"\n')
    monkeypatch.setattr(io_utils, "PATHS_FILE", mapping)
    with pytest.raises(FileNotFoundError):
        CfaLitModel(backbone="resnet18", gamma_d=56, visualizer=False)


def test_cfa_native_repeated_raw_centroid_start(cfa_assets, monkeypatch):
    """Rebuild a fitted bank from the same raw loader without invoking normalization."""
    from visionlab.anomaly_detection.models.cfa.lightning_model import CfaLitModel

    class ForbiddenPreProcessor(PreProcessor):
        """Reject normalization during the separate centroid traversal."""

        def __call__(self, image):
            raise AssertionError("Centroid initialization must not call PreProcessor")

    model = CfaLitModel(backbone="resnet18", gamma_d=56, pre_processor=ForbiddenPreProcessor(), visualizer=False)
    data = make_cfa_data()
    model._trainer = SimpleNamespace(datamodule=data)
    seen, resets = [], []
    initialize = model.model.initialize_centroid

    def capture(data_loader):
        resets.append((model.model.memory_bank.ndim, model.model.memory_bank.item()))
        return initialize(data_loader=data_loader)

    monkeypatch.setattr(model.model, "initialize_centroid", capture)
    handle = model.model.feature_extractor.register_forward_pre_hook(
        lambda module, inputs: seen.append(inputs[0].detach().clone()))
    try:
        model.on_train_start()
        first = model.model.memory_bank.clone()
        model.on_train_start()
    finally:
        handle.remove()
    assert resets == [(0, 0.0), (0, 0.0)] and first.shape == (8, 4096)
    torch.testing.assert_close(model.model.memory_bank, first)
    for actual, (image, _) in zip(seen[:2], data.train_dataloader(), strict=True):
        assert torch.equal(actual, image)


def test_cfa_native_training_contract_and_retained_backward(cfa_assets, monkeypatch):
    """Normalize once, return only loss and retain the graph without emitting train_loss."""
    from visionlab.anomaly_detection.models.cfa.lightning_model import CfaLitModel

    model = CfaLitModel(backbone="resnet18", gamma_d=56, evaluator=False, visualizer=False)
    seen = []

    def distance(image):
        seen.append(image.detach().clone())
        return (torch.rand(image.shape[0], 4, 8) + 0.1).requires_grad_()

    def forbidden(*args, **kwargs):
        raise AssertionError("CFA training_step must not log metrics")

    monkeypatch.setattr(model.model, "forward", distance)
    monkeypatch.setattr(model, "log", forbidden)
    image = torch.rand(1, 3, 256, 256)
    output = model.training_step((image, {}), 0)
    assert set(output) == {"loss"} and output["loss"].requires_grad
    expected = (image - torch.tensor(IMAGENET_MEAN).reshape(1, 3, 1, 1)) / torch.tensor(IMAGENET_STD).reshape(1, 3, 1, 1)
    torch.testing.assert_close(seen[0], expected)
    calls = []
    loss = SimpleNamespace(backward=lambda **options: calls.append(options))
    model.backward(loss, "unused", unused=True)
    assert calls == [{"retain_graph": True}]


@pytest.fixture(scope="module")
def fitted_cfa(tmp_path_factory, cfa_assets):
    """Fit two real epochs while retaining a batch-one validation tail."""
    from lightning.pytorch import Trainer
    from visionlab.anomaly_detection.models.cfa.lightning_model import CfaLitModel

    root = str(tmp_path_factory.mktemp("fitted_cfa"))
    data = make_cfa_data()
    model = CfaLitModel(backbone="resnet18", gamma_d=56, visualizer=False)
    before = {name: parameter.detach().clone() for name, parameter in model.model.feature_extractor.named_parameters()}
    descriptor_before = {name: parameter.detach().clone() for name, parameter in model.model.descriptor.named_parameters()}
    counts = {"update": 0, "finalize": 0}
    update, finalize = model.post_processor.update, model.post_processor.finalize

    def capture_update(outputs):
        counts["update"] += 1
        return update(outputs)

    def capture_finalize():
        counts["finalize"] += 1
        return finalize()

    trainer = Trainer(default_root_dir=root, accelerator="cpu", devices=1, max_epochs=2, logger=False,
                      num_sanity_val_steps=0, enable_checkpointing=False,
                      enable_progress_bar=False, enable_model_summary=False)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(model.post_processor, "update", capture_update)
        patch.setattr(model.post_processor, "finalize", capture_finalize)
        trainer.fit(model, datamodule=data)
    path = os.path.join(root, "model.ckpt")
    trainer.save_checkpoint(path)
    return root, model, data, path, trainer, before, descriptor_before, counts


def test_cfa_native_learning_adamw_and_fit_only_calibration(fitted_cfa):
    """Update descriptors while freezing backbone parameters and omit train_loss logging."""
    _, model, data, _, trainer, before, descriptor_before, counts = fitted_cfa
    assert model.model.memory_bank.shape == (8, 4096) and model.post_processor.is_calibrated
    assert counts == {"update": 2 * len(data.val_dataloader()), "finalize": 2}
    for name, parameter in model.model.feature_extractor.named_parameters():
        assert torch.equal(parameter.detach(), before[name]), name
    assert any(not torch.equal(parameter.detach(), descriptor_before[name])
               for name, parameter in model.model.descriptor.named_parameters())
    optimizer = trainer.optimizers[0]
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["lr"] == 1e-3 and optimizer.defaults["weight_decay"] == 5e-4 and optimizer.defaults["amsgrad"]
    actual = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert actual == {id(parameter) for parameter in model.model.parameters()}
    assert "max_epochs" not in model.trainer_arguments
    assert not any(name.startswith("train_loss") for name in trainer.logged_metrics)
    assert not any(name.startswith("train_loss") for name in trainer.callback_metrics)


def test_cfa_native_checkpoint_frozen_evaluation_and_batch_one_visualization(fitted_cfa, tmp_path, monkeypatch):
    """Restore memory and calibration and evaluate squeezed batch-one outputs without refitting."""
    from lightning.pytorch import Trainer
    from visionlab.anomaly_detection.models.cfa.lightning_model import CfaLitModel

    _, model, data, path, _, _, _, _ = fitted_cfa
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert "model.memory_bank" in saved["state_dict"] and saved["post_processor"] is not None
    restored = CfaLitModel(backbone="resnet18", gamma_d=56, visualizer=False)
    assert restored.model.memory_bank.ndim == 0
    restored.load_state_dict(saved["state_dict"], strict=True)
    restored.on_load_checkpoint(saved)
    assert restored.model.memory_bank.shape == (8, 4096)
    model.eval()
    restored.eval()
    batch = list(data.val_dataloader())[-1]
    with torch.no_grad():
        expected, actual = model.predict_step(batch, 0), restored.predict_step(batch, 0)
    assert actual["pred_score"].ndim == 0 and actual["anomaly_map"].shape == (256, 256)
    for name in ("pred_score", "anomaly_map", "normalized_pred_score", "normalized_anomaly_map", "pred_label", "pred_mask"):
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-5, atol=1e-6)
    before = restored.post_processor.state_dict()

    def forbidden(*args, **kwargs):
        raise AssertionError("Standalone evaluation must not initialize centroids or update calibration")

    monkeypatch.setattr(restored.model, "initialize_centroid", forbidden)
    monkeypatch.setattr(restored.post_processor, "update", forbidden)
    monkeypatch.setattr(restored.post_processor, "finalize", forbidden)
    trainer = Trainer(default_root_dir=str(tmp_path), accelerator="cpu", devices=1, logger=False,
                      enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
    trainer.validate(restored, datamodule=data, ckpt_path=path)
    trainer.test(restored, datamodule=data, ckpt_path=path)
    predictions = trainer.predict(restored, datamodule=data)
    torch.testing.assert_close(predictions[-1]["anomaly_map"], expected["anomaly_map"], rtol=1e-5, atol=1e-6)
    for name, value in restored.post_processor.state_dict().items():
        torch.testing.assert_close(value, before[name], equal_nan=True)
    from visionlab.anomaly_detection.common.visualizer import ImageVisualizer

    restored.visualizer = ImageVisualizer()
    callback = restored.configure_callbacks()[0]
    callback.on_predict_batch_end(SimpleNamespace(is_global_zero=True, default_root_dir=str(tmp_path)),
                                  restored, actual, batch, 0)
    assert os.path.isfile(os.path.join(tmp_path, "images", "fixture", "2.png"))
    assert actual["pred_score"].ndim == 0 and actual["anomaly_map"].ndim == 2


def consume_cfa_export(root):
    """Restore every pure weight and buffer with asset reads and centroid recomputation forbidden."""
    from visionlab.inference.predictor import Predictor
    from visionlab.anomaly_detection.models.cfa.model import CfaModel

    io_utils.block_network()
    path = os.path.join(root, "export", "model.pt")
    expected = torch.load(os.path.join(root, "expected.pt"), map_location="cpu", weights_only=True)
    exported = io_utils.load_export(path)
    assert exported["model_name"] == "cfa"
    assert all(any(name.startswith(prefix) for name in exported["model_state"])
               for prefix in ("feature_extractor.", "descriptor.", "memory_bank"))

    def forbidden(*args, **kwargs):
        raise AssertionError("Inference must restore its saved memory bank")

    CfaModel.initialize_centroid = forbidden

    def audit_open(event, args):
        if event == "open":
            opened, mode, flags = args
            assert isinstance(opened, (str, bytes)) and os.path.abspath(opened) == path, opened
            assert not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)

    # Warm FX code generation before testing constructor I/O, as in the pure-model test.
    from visionlab.anomaly_detection.models.cfa.model import get_feature_extractor, get_return_nodes
    get_feature_extractor("resnet18", get_return_nodes("resnet18"))
    sys.addaudithook(audit_open)
    predictor = Predictor.from_export(path)
    for name, value in predictor.model.state_dict().items():
        assert torch.equal(value, exported["model_state"][name]), name
    for batch in expected["batches"]:
        outputs = predictor.predict(batch["images"])
        for name, value in outputs.items():
            torch.testing.assert_close(value, batch["outputs"][name], rtol=1e-5, atol=1e-6)


def test_cfa_native_export_in_isolated_process(fitted_cfa):
    """Match native batch-two and batch-one inference without Lightning, reference imports or asset files."""
    root, model, data, _, _, _, _, _ = fitted_cfa
    model.eval()
    expected = []
    names = ("pred_score", "anomaly_map", "normalized_pred_score", "normalized_anomaly_map", "pred_label", "pred_mask")
    for batch in data.val_dataloader():
        with torch.no_grad():
            outputs = model.predict_step(batch, 0)
        expected.append({"images": batch[0], "outputs": {name: outputs[name] for name in names}})
    torch.save({"batches": expected}, os.path.join(root, "expected.pt"))
    script = '''
import importlib.abc
import runpy
import sys
import torch
import torchvision
import timm
class Block(importlib.abc.MetaPathFinder):
    """Reject reference and training imports before the pure CFA consumer is loaded."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("anomalib", "lightning", "pytorch_lightning"):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, Block())
sys.path.insert(0, sys.argv[2])
runpy.run_path(sys.argv[1])["consume_cfa_export"](sys.argv[3])
assert not {"anomalib", "lightning", "pytorch_lightning"} & set(sys.modules)
'''
    result = subprocess.run([sys.executable, "-c", script, os.path.abspath(__file__), os.path.join(PROJECT_ROOT, "src"), root],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_cfa_stats_reports_memory_and_each_descriptor_tensor(tmp_path, capsys):
    """Record values and incompatible shapes without making an acceptance decision."""
    namespace = runpy.run_path(os.path.join(PROJECT_ROOT, "evaluation", "anomalib", "compare_cfa_stats.py"))
    reference_path, native_path = os.path.join(tmp_path, "reference.ckpt"), os.path.join(tmp_path, "native.ckpt")
    reference = {"model.memory_bank": torch.ones(2, 4), "model.descriptor.layer.weight": torch.ones(2, 3, 1, 1),
                 "model.descriptor.layer.conv2d.bias": torch.ones(2)}
    native = {name: value + 0.25 for name, value in reference.items()}
    torch.save({"state_dict": reference}, reference_path)
    torch.save({"state_dict": native}, native_path)
    output = os.path.join(tmp_path, "stats.json")
    report = namespace["main"](["--reference", reference_path, "--native", native_path, "--out", output])
    assert report["informational_only"] and "passed" not in report
    assert report["buffers"]["memory_bank"]["max_absolute_difference"] == 0.25
    assert report["buffers"]["memory_bank"]["max_relative_difference"] == 0.25
    assert all(value["max_absolute_difference"] == 0.25 for value in report["buffers"].values())
    assert "Informational comparison only" in capsys.readouterr().out
    with open(output, encoding="utf-8") as stream:
        assert json.load(stream) == report
    native["model.memory_bank"] = torch.ones(2, 3)
    torch.save({"state_dict": native}, native_path)
    report = namespace["compare_stats"](reference_path, native_path)
    assert report["buffers"]["memory_bank"]["not_comparable_reason"] == "shape_mismatch"
