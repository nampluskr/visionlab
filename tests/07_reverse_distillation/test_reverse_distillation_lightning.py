# tests/07_reverse_distillation/test_reverse_distillation_lightning.py: pytest native training, frozen encoder and independent exports

import csv
import inspect
import math
import os
import runpy
import subprocess
import sys

import pytest
import torch

from visionlab.anomaly_detection.common.pre_processor import IMAGENET_MEAN, IMAGENET_STD
from visionlab.anomaly_detection.models.reverse_distillation.anomaly_map import AnomalyMapGenerationMode
from visionlab.utils import io as io_utils

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def padim_helpers():
    """Reuse local weight and small DataModule builders without running earlier model tests."""
    return runpy.run_path(os.path.join(PROJECT_ROOT, "tests", "03_padim", "test_padim_lightning.py"))


@pytest.fixture(scope="module")
def reverse_assets(tmp_path_factory):
    """Map one random local ResNet18 state for all native constructor checks."""
    root = str(tmp_path_factory.mktemp("reverse_assets"))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(io_utils, "PATHS_FILE", padim_helpers()["make_padim_assets"](root))
        yield root


def test_reverse_native_signature_and_pretrained_injection(reverse_assets, tmp_path, monkeypatch):
    """Preserve eight defaults and inject encoder weights only when requested."""
    from visionlab.anomaly_detection.models.reverse_distillation.lightning_model import ReverseDistillationLitModel

    expected = {"backbone": "wide_resnet50_2", "layers": ("layer1", "layer2", "layer3"),
                "anomaly_map_mode": AnomalyMapGenerationMode.ADD, "pre_trained": True,
                "pre_processor": True, "post_processor": True, "evaluator": True, "visualizer": True}
    signature = inspect.signature(ReverseDistillationLitModel.__init__)
    assert list(signature.parameters) == ["self"] + list(expected)
    assert {name: signature.parameters[name].default for name in expected} == expected
    model = ReverseDistillationLitModel(backbone="resnet18", visualizer=False)
    source = torch.load(os.path.join(reverse_assets, "resnet18.pth"), map_location="cpu", weights_only=True)
    for name, value in model.model.encoder.feature_extractor.state_dict().items():
        assert torch.equal(value, source[name]), name
    assert model.teacher_weights_path == os.path.join(reverse_assets, "resnet18.pth")
    assert model.model.anomaly_map_generator.image_size == (256, 256) and model.pre_processor.center_crop is None
    assert model.learning_type.value == "one_class"
    assert model.trainer_arguments == {"gradient_clip_val": 0, "num_sanity_val_steps": 0}
    monkeypatch.setattr(io_utils, "PATHS_FILE", os.path.join(tmp_path, "missing.toml"))
    with pytest.raises(FileNotFoundError):
        ReverseDistillationLitModel(backbone="resnet18", visualizer=False)
    mapping = os.path.join(tmp_path, "paths.toml")
    with open(mapping, "w", encoding="utf-8", newline="\n") as stream:
        stream.write('[backbones]\nresnet18 = "missing.pth"\n')
    monkeypatch.setattr(io_utils, "PATHS_FILE", mapping)
    with pytest.raises(FileNotFoundError):
        ReverseDistillationLitModel(backbone="resnet18", visualizer=False)
    untrained = ReverseDistillationLitModel(backbone="resnet18", pre_trained=False, visualizer=False)
    assert untrained.teacher_weights_path is None


def test_reverse_native_training_loss_log_and_single_normalization(monkeypatch):
    """Call the original positional loss contract and log a scalar with the reference flags."""
    from visionlab.anomaly_detection.models.reverse_distillation.lightning_model import ReverseDistillationLitModel

    model = ReverseDistillationLitModel(backbone="resnet18", pre_trained=False, evaluator=False, visualizer=False)
    seen, logged = [], []

    def features(image):
        seen.append(image.clone())
        return [torch.ones(2, 3, 2, 2)], [torch.zeros(2, 3, 2, 2, requires_grad=True)]

    monkeypatch.setattr(model.model, "forward", features)
    monkeypatch.setattr(model, "log", lambda name, value, **options: logged.append((name, value, options)))
    image = torch.rand(2, 3, 256, 256)
    outputs = model.training_step((image, {}), 0)
    assert set(outputs) == {"loss"} and outputs["loss"].requires_grad and outputs["loss"].ndim == 0
    expected = (image - torch.tensor(IMAGENET_MEAN).reshape(1, 3, 1, 1)) / torch.tensor(IMAGENET_STD).reshape(1, 3, 1, 1)
    torch.testing.assert_close(seen[0], expected)
    assert logged == [("train_loss", outputs["loss"].item(),
                       {"on_epoch": True, "prog_bar": True, "logger": True, "batch_size": 1})]


@pytest.fixture(scope="module")
def fitted_reverse(tmp_path_factory, reverse_assets):
    """Train two real epochs on small 256 batches and persist optimizer-free evaluation weights."""
    from lightning.pytorch import Trainer
    from lightning.pytorch.loggers import CSVLogger
    from visionlab.anomaly_detection.models.reverse_distillation.lightning_model import ReverseDistillationLitModel

    root = str(tmp_path_factory.mktemp("fitted_reverse"))
    data = padim_helpers()["make_padim_data"]()
    for dataset in (data.train_data, data.val_data):
        for index, (image, target) in enumerate(dataset):
            image = torch.nn.functional.interpolate(image[None], size=(256, 256), mode="bilinear",
                                                   align_corners=False)[0]
            target["mask"] = torch.nn.functional.interpolate(target["mask"][None, None].float(),
                                                            size=(256, 256), mode="nearest")[0, 0].long()
            dataset[index] = image, target
    model = ReverseDistillationLitModel(backbone="resnet18", visualizer=False)
    encoder_before = {name: value.clone() for name, value in model.model.encoder.state_dict().items()}
    reconstruction_before = {branch: next(getattr(model.model, branch).parameters()).detach().clone()
                             for branch in ("decoder", "bottleneck")}
    counts = {"update": 0, "finalize": 0}
    update, finalize = model.post_processor.update, model.post_processor.finalize

    def capture_update(outputs):
        counts["update"] += 1
        return update(outputs)

    def capture_finalize():
        counts["finalize"] += 1
        return finalize()

    logger = CSVLogger(save_dir=root, name="metrics")
    trainer = Trainer(default_root_dir=root, accelerator="cpu", devices=1, max_epochs=2, logger=logger,
                      log_every_n_steps=1, num_sanity_val_steps=0, enable_checkpointing=False,
                      enable_progress_bar=False, enable_model_summary=False)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(model.post_processor, "update", capture_update)
        patch.setattr(model.post_processor, "finalize", capture_finalize)
        trainer.fit(model, datamodule=data)
    path = os.path.join(root, "evaluation.ckpt")
    trainer.save_checkpoint(path, weights_only=True)
    return root, model, data, path, trainer, encoder_before, reconstruction_before, counts


def test_reverse_native_two_epoch_learning_and_adam_parameters(fitted_reverse):
    """Train only reconstruction parameters and preserve every encoder weight and BatchNorm buffer."""
    root, model, data, _, trainer, before, reconstruction, counts = fitted_reverse
    assert counts == {"update": 2 * len(data.val_dataloader()), "finalize": 2}
    assert model.post_processor.is_calibrated and trainer.current_epoch == 2
    for name, value in model.model.encoder.state_dict().items():
        assert torch.equal(value, before[name]), name
    assert any(not torch.equal(next(getattr(model.model, branch).parameters()).detach(), value)
               for branch, value in reconstruction.items())
    optimizer = trainer.optimizers[0]
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.defaults["lr"] == 0.005 and optimizer.defaults["betas"] == (0.5, 0.99)
    actual = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    expected = {id(parameter) for parameter in list(model.model.decoder.parameters()) + list(model.model.bottleneck.parameters())}
    assert actual == expected and actual.isdisjoint(id(parameter) for parameter in model.model.encoder.parameters())
    assert "max_epochs" not in model.trainer_arguments
    with open(os.path.join(root, "metrics", "version_0", "metrics.csv"), newline="", encoding="utf-8") as stream:
        losses = [row for row in csv.DictReader(stream) if row.get("train_loss_epoch")]
    assert len(losses) == 2 and {int(float(row["epoch"])) for row in losses} == {0, 1}
    assert all(math.isfinite(float(row["train_loss_epoch"])) for row in losses)


def test_reverse_native_checkpoint_without_optimizer_and_frozen_evaluation(fitted_reverse, tmp_path, monkeypatch):
    """Restore model and calibration alone and keep standalone validation and testing read-only."""
    from lightning.pytorch import Trainer
    from visionlab.anomaly_detection.models.reverse_distillation.lightning_model import ReverseDistillationLitModel

    _, model, data, path, _, _, _, _ = fitted_reverse
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert "optimizer_states" not in saved and saved["post_processor"] is not None
    for prefix in ("model.encoder.", "model.bottleneck.", "model.decoder."):
        assert any(name.startswith(prefix) for name in saved["state_dict"])
    restored = ReverseDistillationLitModel(backbone="resnet18", pre_trained=False, visualizer=False)
    restored.load_state_dict(saved["state_dict"], strict=True)
    restored.on_load_checkpoint(saved)
    model.eval()
    restored.eval()
    batch = next(iter(data.val_dataloader()))
    with torch.no_grad():
        expected, actual = model.predict_step(batch, 0), restored.predict_step(batch, 0)
    for name in ("pred_score", "anomaly_map", "normalized_pred_score", "normalized_anomaly_map", "pred_label", "pred_mask"):
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-5, atol=1e-6)
    before = restored.post_processor.state_dict()

    def forbidden(*args, **kwargs):
        raise AssertionError("Standalone evaluation must not update calibration")

    monkeypatch.setattr(restored.post_processor, "update", forbidden)
    monkeypatch.setattr(restored.post_processor, "finalize", forbidden)
    trainer = Trainer(default_root_dir=str(tmp_path), accelerator="cpu", devices=1, logger=False,
                      enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
    trainer.validate(restored, datamodule=data, ckpt_path=path)
    trainer.test(restored, datamodule=data, ckpt_path=path)
    predictions = trainer.predict(restored, datamodule=data, ckpt_path=path)
    torch.testing.assert_close(predictions[0]["pred_score"], expected["pred_score"], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(predictions[0]["anomaly_map"], expected["anomaly_map"], rtol=1e-5, atol=1e-6)
    for name, value in restored.post_processor.state_dict().items():
        torch.testing.assert_close(value, before[name], equal_nan=True)


def consume_reverse_export(root):
    """Restore all reconstruction weights and outputs while allowing only export file reads."""
    from visionlab.inference.predictor import Predictor

    io_utils.block_network()
    path = os.path.join(root, "export", "model.pt")
    expected = torch.load(os.path.join(root, "expected.pt"), map_location="cpu", weights_only=True)
    exported = io_utils.load_export(path)
    assert exported["model_name"] == "reverse_distillation" and exported["model_args"]["pre_trained"] is False
    assert type(exported["model_args"]["anomaly_map_mode"]) is str and exported["model_args"]["anomaly_map_mode"] == "add"
    assert type(exported["model_args"]["input_size"]) is list and exported["model_args"]["input_size"] == [256, 256]
    assert all(any(name.startswith(prefix) for name in exported["model_state"])
               for prefix in ("encoder.", "bottleneck.", "decoder."))

    def audit_open(event, args):
        if event == "open":
            opened, mode, flags = args
            assert isinstance(opened, (str, bytes)) and os.path.abspath(opened) == path, opened
            assert not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)

    sys.addaudithook(audit_open)
    predictor = Predictor.from_export(path)
    assert predictor.model.anomaly_map_generator.mode is AnomalyMapGenerationMode.ADD
    assert predictor.model.anomaly_map_generator.image_size == (256, 256)
    for name, value in predictor.model.state_dict().items():
        assert torch.equal(value, exported["model_state"][name]), name
    for name, value in predictor.predict(expected["images"]).items():
        torch.testing.assert_close(value, expected["outputs"][name], rtol=1e-5, atol=1e-6)


def test_reverse_native_export_in_isolated_process(fitted_reverse):
    """Match native evaluation without reference frameworks, Lightning or access to teacher assets."""
    root, model, data, _, _, _, _, _ = fitted_reverse
    model.eval()
    batch = next(iter(data.val_dataloader()))
    with torch.no_grad():
        outputs = model.predict_step(batch, 0)
    names = ("pred_score", "anomaly_map", "normalized_pred_score", "normalized_anomaly_map", "pred_label", "pred_mask")
    torch.save({"images": batch[0], "outputs": {name: outputs[name] for name in names}}, os.path.join(root, "expected.pt"))
    script = '''
import importlib.abc
import runpy
import sys
import torch
import timm
class Block(importlib.abc.MetaPathFinder):
    """Reject reference and training frameworks before importing the export consumer."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("anomalib", "lightning", "pytorch_lightning"):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, Block())
sys.path.insert(0, sys.argv[2])
runpy.run_path(sys.argv[1])["consume_reverse_export"](sys.argv[3])
assert not {"anomalib", "lightning", "pytorch_lightning"} & set(sys.modules)
'''
    result = subprocess.run([sys.executable, "-c", script, os.path.abspath(__file__), os.path.join(PROJECT_ROOT, "src"), root],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
