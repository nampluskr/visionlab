# tests/09_fre/test_fre_lightning.py: pytest native FRE learning, equal-step loss logging and independent exports

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
from visionlab.utils import io as io_utils

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_fre_data():
    """Provide uneven 256-pixel batches with a high-intensity final training sample."""
    from lightning.pytorch import LightningDataModule

    class TinyData(LightningDataModule):
        """Supply five training samples and three labeled evaluation samples."""

        def __init__(self):
            super().__init__()
            self.train_data = [(torch.rand(3, 256, 256) * 0.02 + level, {"label": 0})
                               for level in (0.40, 0.42, 0.44, 0.46, 0.98)]
            self.val_data = []
            for index in range(3):
                mask = torch.zeros(256, 256, dtype=torch.int64)
                if index % 2:
                    mask[64:128, 64:128] = 1
                self.val_data.append((torch.rand(3, 256, 256),
                                      {"label": index % 2, "mask": mask, "image_path": f"{index}.png",
                                       "sample_id": f"fixture/{index}.png"}))

        def train_dataloader(self):
            return torch.utils.data.DataLoader(self.train_data, batch_size=2, shuffle=False, num_workers=0)

        def val_dataloader(self):
            return torch.utils.data.DataLoader(self.val_data, batch_size=2, shuffle=False, num_workers=0)

        def test_dataloader(self):
            return self.val_dataloader()

        def predict_dataloader(self):
            return self.val_dataloader()

    return TinyData()


@pytest.fixture(scope="module")
def fre_assets(tmp_path_factory):
    """Reuse a local full ResNet18 fixture without executing earlier model tests."""
    root = str(tmp_path_factory.mktemp("fre_assets"))
    helpers = runpy.run_path(os.path.join(PROJECT_ROOT, "tests", "03_padim", "test_padim_lightning.py"))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(io_utils, "PATHS_FILE", helpers["make_padim_assets"](root))
        yield root


def test_fre_native_signature_and_local_weight_injection(fre_assets, tmp_path, monkeypatch):
    """Match ten defaults and inject exact full-backbone weights through the timm wrapper."""
    from visionlab.anomaly_detection.models.fre.lightning_model import FreLitModel

    expected = {"backbone": "resnet50", "layer": "layer3", "pre_trained": True, "pooling_kernel_size": 2,
                "input_dim": 65536, "latent_dim": 220, "pre_processor": True, "post_processor": True,
                "evaluator": True, "visualizer": True}
    signature = inspect.signature(FreLitModel.__init__)
    assert list(signature.parameters) == ["self"] + list(expected)
    assert {name: signature.parameters[name].default for name in expected} == expected
    model = FreLitModel(backbone="resnet18", input_dim=16384, latent_dim=7, visualizer=False)
    source = torch.load(os.path.join(fre_assets, "resnet18.pth"), map_location="cpu", weights_only=True)
    for name, value in model.model.feature_extractor.feature_extractor.state_dict().items():
        assert torch.equal(value, source[name]), name
    assert model.teacher_weights_path == os.path.join(fre_assets, "resnet18.pth")
    assert model.pre_processor.center_crop is None and isinstance(model.loss_fn, torch.nn.MSELoss)
    assert model.trainer_arguments == {"gradient_clip_val": 0, "max_epochs": 10, "num_sanity_val_steps": 0}
    assert model.learning_type.value == "one_class"
    monkeypatch.setattr(io_utils, "PATHS_FILE", os.path.join(tmp_path, "missing.toml"))
    with pytest.raises(FileNotFoundError):
        FreLitModel(backbone="resnet18", input_dim=16384, latent_dim=7, visualizer=False)
    mapping = os.path.join(tmp_path, "paths.toml")
    with open(mapping, "w", encoding="utf-8", newline="\n") as stream:
        stream.write('[backbones]\nresnet18 = "missing.pth"\n')
    monkeypatch.setattr(io_utils, "PATHS_FILE", mapping)
    with pytest.raises(FileNotFoundError):
        FreLitModel(backbone="resnet18", input_dim=16384, latent_dim=7, visualizer=False)
    untrained = FreLitModel(backbone="resnet18", input_dim=16384, latent_dim=7, pre_trained=False, visualizer=False)
    assert untrained.teacher_weights_path is None


def test_fre_native_training_features_mse_and_log_contract(monkeypatch):
    """Normalize once before get_features and log an equal-step epoch loss."""
    from visionlab.anomaly_detection.models.fre.lightning_model import FreLitModel

    model = FreLitModel(backbone="resnet18", input_dim=16384, latent_dim=7, pre_trained=False, visualizer=False)
    seen, logged = [], []
    original, reconstructed = torch.rand(2, 6), torch.rand(2, 6, requires_grad=True)

    def features(image):
        seen.append(image.clone())
        return original, reconstructed, (2, 6, 1, 1)

    monkeypatch.setattr(model.model, "get_features", features)
    monkeypatch.setattr(model, "log", lambda name, value, **options: logged.append((name, value, options)))
    image = torch.rand(2, 3, 256, 256)
    output = model.training_step((image, {}), 0)
    assert set(output) == {"loss"} and output["loss"].requires_grad
    torch.testing.assert_close(output["loss"], torch.nn.functional.mse_loss(original, reconstructed))
    expected = (image - torch.tensor(IMAGENET_MEAN).reshape(1, 3, 1, 1)) / torch.tensor(IMAGENET_STD).reshape(1, 3, 1, 1)
    torch.testing.assert_close(seen[0], expected)
    assert logged == [("train_loss", output["loss"].item(),
                       {"on_epoch": True, "prog_bar": True, "logger": True, "batch_size": 1})]


@pytest.fixture(scope="module")
def fitted_fre(tmp_path_factory, fre_assets):
    """Fit two real epochs and record every pre-update loss for independent epoch aggregation checks."""
    from lightning.pytorch import Trainer
    from lightning.pytorch.loggers import CSVLogger
    from visionlab.anomaly_detection.models.fre.lightning_model import FreLitModel

    root = str(tmp_path_factory.mktemp("fitted_fre"))
    data = make_fre_data()
    model = FreLitModel(backbone="resnet18", input_dim=16384, latent_dim=7, visualizer=False)
    backbone_before = {name: value.clone() for name, value in model.model.feature_extractor.state_dict().items()}
    weight_before = model.model.fre_model.weight.detach().clone()
    counts, steps = {"update": 0, "finalize": 0}, []
    update, finalize, training_step = model.post_processor.update, model.post_processor.finalize, model.training_step

    def capture_update(outputs):
        counts["update"] += 1
        return update(outputs)

    def capture_finalize():
        counts["finalize"] += 1
        return finalize()

    def capture_step(batch, batch_idx):
        output = training_step(batch, batch_idx)
        steps.append((model.current_epoch, batch[0].shape[0], output["loss"].item()))
        return output

    logger = CSVLogger(save_dir=root, name="metrics")
    trainer = Trainer(default_root_dir=root, accelerator="cpu", devices=1, max_epochs=2, logger=logger,
                      log_every_n_steps=1, num_sanity_val_steps=0, enable_checkpointing=False,
                      enable_progress_bar=False, enable_model_summary=False)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(model.post_processor, "update", capture_update)
        patch.setattr(model.post_processor, "finalize", capture_finalize)
        patch.setattr(model, "training_step", capture_step)
        trainer.fit(model, datamodule=data)
    path = os.path.join(root, "model.ckpt")
    trainer.save_checkpoint(path)
    return root, model, data, path, trainer, backbone_before, weight_before, counts, steps


def test_fre_native_learning_frozen_backbone_and_equal_step_epoch_mean(fitted_fre):
    """Change tied weights while preserving backbone buffers and average uneven batches by step."""
    root, model, data, _, trainer, before, weight_before, counts, steps = fitted_fre
    assert counts == {"update": 2 * len(data.val_dataloader()), "finalize": 2}
    assert model.post_processor.is_calibrated
    for name, value in model.model.feature_extractor.state_dict().items():
        assert torch.equal(value, before[name]), name
    assert not torch.equal(model.model.fre_model.weight.detach(), weight_before)
    optimizer = trainer.optimizers[0]
    assert isinstance(optimizer, torch.optim.Adam) and optimizer.defaults["lr"] == 1e-3
    parameters = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert parameters == {id(parameter) for parameter in model.model.fre_model.parameters()}
    assert parameters.isdisjoint(id(parameter) for parameter in model.model.feature_extractor.parameters())
    assert model.trainer_arguments["max_epochs"] == 10
    with open(os.path.join(root, "metrics", "version_0", "metrics.csv"), newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("train_loss_epoch")]
    assert len(rows) == 2
    distinguished = False
    for row in rows:
        epoch = int(float(row["epoch"]))
        values = [(size, loss) for step_epoch, size, loss in steps if step_epoch == epoch]
        assert [size for size, _ in values] == [2, 2, 1]
        simple = sum(loss for _, loss in values) / len(values)
        weighted = sum(size * loss for size, loss in values) / sum(size for size, _ in values)
        assert float(row["train_loss_epoch"]) == pytest.approx(simple, rel=1e-5, abs=1e-7)
        distinguished |= not math.isclose(simple, weighted, rel_tol=1e-5, abs_tol=1e-7)
    assert distinguished, "The fixture must distinguish equal-step and sample-weighted means"


def test_fre_native_checkpoint_restore_and_frozen_evaluation(fitted_fre, tmp_path, monkeypatch):
    """Restore model and calibration and preserve identical standalone validation, test and predict outputs."""
    from lightning.pytorch import Trainer
    from visionlab.anomaly_detection.models.fre.lightning_model import FreLitModel

    _, model, data, path, _, _, _, _, _ = fitted_fre
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["post_processor"] is not None and saved["optimizer_states"]
    assert any(name.startswith("model.fre_model.") for name in saved["state_dict"])
    assert any(name.startswith("model.feature_extractor.") for name in saved["state_dict"])
    restored = FreLitModel(backbone="resnet18", input_dim=16384, latent_dim=7, pre_trained=False, visualizer=False)
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
    predictions = trainer.predict(restored, datamodule=data)
    torch.testing.assert_close(predictions[0]["pred_score"], expected["pred_score"], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(predictions[0]["anomaly_map"], expected["anomaly_map"], rtol=1e-5, atol=1e-6)
    for name, value in restored.post_processor.state_dict().items():
        torch.testing.assert_close(value, before[name], equal_nan=True)


def consume_fre_export(root):
    """Restore teacher-inclusive pure weights while denying all reads except the export file."""
    from visionlab.inference.predictor import Predictor

    io_utils.block_network()
    path = os.path.join(root, "export", "model.pt")
    expected = torch.load(os.path.join(root, "expected.pt"), map_location="cpu", weights_only=True)
    exported = io_utils.load_export(path)
    assert exported["model_name"] == "fre" and exported["model_args"]["pre_trained"] is False
    assert all(type(value) in (str, int, float, bool, list) for value in exported["model_args"].values())
    assert {"fre_model.weight", "fre_model.encoder_bias", "fre_model.decoder_bias"} <= exported["model_state"].keys()
    assert any(name.startswith("feature_extractor.") for name in exported["model_state"])

    def audit_open(event, args):
        if event == "open":
            opened, mode, flags = args
            assert isinstance(opened, (str, bytes)) and os.path.abspath(opened) == path, opened
            assert not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)

    sys.addaudithook(audit_open)
    predictor = Predictor.from_export(path)
    for name, value in predictor.model.state_dict().items():
        assert torch.equal(value, exported["model_state"][name]), name
    for name, value in predictor.predict(expected["images"]).items():
        torch.testing.assert_close(value, expected["outputs"][name], rtol=1e-5, atol=1e-6)


def test_fre_native_export_in_isolated_process(fitted_fre):
    """Match native inference without Lightning, reference imports, paths.toml or backbone-file access."""
    root, model, data, _, _, _, _, _, _ = fitted_fre
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
    """Reject reference and training frameworks before loading pure FRE inference."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("anomalib", "lightning", "pytorch_lightning"):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, Block())
sys.path.insert(0, sys.argv[2])
runpy.run_path(sys.argv[1])["consume_fre_export"](sys.argv[3])
assert not {"anomalib", "lightning", "pytorch_lightning"} & set(sys.modules)
'''
    result = subprocess.run([sys.executable, "-c", script, os.path.abspath(__file__), os.path.join(PROJECT_ROOT, "src"), root],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
