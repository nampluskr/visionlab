# tests/03_padim/test_padim_lightning.py: pytest native PaDiM fitting, checkpoint restoration and pure exports

import inspect
import json
import os
import runpy
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from visionlab.anomaly_detection.common.pre_processor import IMAGENET_MEAN, IMAGENET_STD
from visionlab.utils import io as io_utils

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_padim_assets(root):
    """Reuse the local ResNet weight fixture without altering import paths."""
    helpers = runpy.run_path(os.path.join(PROJECT_ROOT, "tests", "01_stfpm", "test_lightning.py"))
    return helpers["make_assets"](root)


def make_padim_data():
    """Build fixed small RGB batches and paired labels and masks for a native DataModule."""
    from lightning.pytorch import LightningDataModule

    class TinyData(LightningDataModule):
        """Supply small train and evaluation datasets without filesystem assets."""

        def __init__(self):
            super().__init__()
            self.train_data = []
            self.val_data = []
            for index in range(4):
                image = torch.rand(3, 64, 64)
                label = index % 2
                mask = torch.zeros(64, 64, dtype=torch.int64)
                if label:
                    mask[16:32, 16:32] = 1
                self.train_data.append((image, {"label": 0, "mask": torch.zeros_like(mask)}))
                self.val_data.append((image.clone(), {"label": label, "mask": mask,
                                                      "image_path": f"{index}.png", "sample_id": f"fixture/{index}.png"}))

        def train_dataloader(self):
            return torch.utils.data.DataLoader(self.train_data, batch_size=2, num_workers=0)

        def val_dataloader(self):
            return torch.utils.data.DataLoader(self.val_data, batch_size=2, num_workers=0)

        def test_dataloader(self):
            return self.val_dataloader()

        def predict_dataloader(self):
            return self.val_dataloader()

    return TinyData()


@pytest.fixture(scope="module")
def padim_assets(tmp_path_factory):
    """Map a temporary full ResNet state for every native constructor in this module."""
    root = str(tmp_path_factory.mktemp("padim_assets"))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(io_utils, "PATHS_FILE", make_padim_assets(root))
        yield root


def test_padim_signature_and_pretrained_injection(padim_assets, tmp_path, monkeypatch):
    """Keep all eight defaults and inject exact local feature weights only when requested."""
    from visionlab.anomaly_detection.models.padim.lightning_model import PadimLitModel

    expected = {"backbone": "resnet18", "layers": ["layer1", "layer2", "layer3"], "pre_trained": True,
                "n_features": None, "pre_processor": True, "post_processor": True, "evaluator": True, "visualizer": True}
    signature = inspect.signature(PadimLitModel.__init__)
    assert list(signature.parameters) == ["self"] + list(expected)
    assert {name: signature.parameters[name].default for name in expected} == expected
    model = PadimLitModel(n_features=4, visualizer=False)
    source = torch.load(os.path.join(padim_assets, "resnet18.pth"), map_location="cpu", weights_only=True)
    for name, value in model.model.feature_extractor.feature_extractor.state_dict().items():
        assert torch.equal(value, source[name]), name
    assert model.teacher_weights_path == os.path.join(padim_assets, "resnet18.pth")
    assert model.configure_optimizers() is None
    assert model.trainer_arguments == {"max_epochs": 1, "val_check_interval": 1.0, "num_sanity_val_steps": 0}
    assert model.learning_type.value == "one_class"
    monkeypatch.setattr(io_utils, "PATHS_FILE", os.path.join(tmp_path, "absent.toml"))
    with pytest.raises(FileNotFoundError):
        PadimLitModel(n_features=4, visualizer=False)
    mapping = os.path.join(tmp_path, "paths.toml")
    with open(mapping, "w", encoding="utf-8", newline="\n") as stream:
        stream.write('[backbones]\nresnet18 = "missing.pth"\n')
    monkeypatch.setattr(io_utils, "PATHS_FILE", mapping)
    with pytest.raises(FileNotFoundError):
        PadimLitModel(n_features=4, visualizer=False)
    untrained = PadimLitModel(n_features=4, pre_trained=False, visualizer=False)
    assert untrained.teacher_weights_path is None and untrained.embeddings == untrained.stats == []
    assert "_is_fitted" in untrained.state_dict() and not untrained._is_fitted.item()
    io_utils.write_run_info(str(tmp_path), {}, SimpleNamespace(), None, 42)
    with open(os.path.join(tmp_path, "run.json"), encoding="utf-8") as stream:
        assert json.load(stream)["teacher_weights"] is None


@pytest.mark.parametrize("first_hook", ("on_validation_start", "on_train_epoch_end"))
def test_padim_dummy_loss_and_one_time_fitting(first_hook, monkeypatch):
    """Retain embeddings and fit once across both original memory bank hooks."""
    from visionlab.anomaly_detection.models.padim.lightning_model import PadimLitModel

    model = PadimLitModel(pre_trained=False, n_features=4, evaluator=False, visualizer=False)
    seen = []

    def embedding(image):
        seen.append(image.clone())
        return torch.randn(image.shape[0], 4, 2, 2)

    monkeypatch.setattr(model.model, "forward", embedding)
    batch = next(iter(make_padim_data().train_dataloader()))
    loss = model.training_step(batch, 0)
    assert loss.ndim == 0 and loss.item() == 0 and loss.requires_grad
    expected_image = (batch[0] - torch.tensor(IMAGENET_MEAN).reshape(1, 3, 1, 1)) / \
                     torch.tensor(IMAGENET_STD).reshape(1, 3, 1, 1)
    torch.testing.assert_close(seen[0], expected_image)
    calls = []
    original_fit = model.fit

    def fit():
        calls.append(True)
        original_fit()

    monkeypatch.setattr(model, "fit", fit)
    getattr(model, first_hook)()
    model.on_validation_start()
    model.on_validation_start()
    model.on_train_epoch_end()
    assert calls == [True] and model._is_fitted.item()
    assert len(model.embeddings) == 1 and len(model.stats) == 2
    assert model.model.gaussian.mean.shape == (4, 4)


@pytest.fixture(scope="module")
def fitted_padim(tmp_path_factory, padim_assets):
    """Run a single native fitting epoch and save the checkpoint and pure export."""
    from lightning.pytorch import Trainer
    from visionlab.anomaly_detection.models.padim.lightning_model import PadimLitModel

    root = str(tmp_path_factory.mktemp("fitted_padim"))
    data = make_padim_data()
    model = PadimLitModel(n_features=4, visualizer=False)
    trainer = Trainer(default_root_dir=root, accelerator="cpu", devices=1, max_epochs=1, logger=False,
                      num_sanity_val_steps=0, enable_checkpointing=False,
                      enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(model, datamodule=data)
    path = os.path.join(root, "model.ckpt")
    trainer.save_checkpoint(path)
    return root, model, data, path, trainer


def test_padim_fit_and_checkpoint_restore_without_refitting(fitted_padim, tmp_path, monkeypatch):
    """Restore indices, dynamic buffers, fit status and calibration through evaluation and resume."""
    from lightning.pytorch import Trainer
    from visionlab.anomaly_detection.models.padim.lightning_model import PadimLitModel

    root, model, data, path, trainer = fitted_padim
    assert model._is_fitted.item() and model.configure_optimizers() is None
    assert not trainer.lr_scheduler_configs
    assert model.model.gaussian.mean.shape == (4, 256)
    assert model.model.gaussian.inv_covariance.shape == (256, 4, 4)
    assert len(model.embeddings) == 2 and len(model.stats) == 2
    source = torch.load(model.teacher_weights_path, map_location="cpu", weights_only=True)
    for name, value in model.model.feature_extractor.feature_extractor.state_dict().items():
        assert torch.equal(value, source[name]), name
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert {"model.idx", "model.gaussian.mean", "model.gaussian.inv_covariance", "_is_fitted"} <= saved["state_dict"].keys()
    assert saved["state_dict"]["_is_fitted"].item() and saved["post_processor"] is not None
    restored = PadimLitModel(pre_trained=False, n_features=4, visualizer=False)
    restored.load_state_dict(saved["state_dict"], strict=True)
    restored.on_load_checkpoint(saved)
    assert restored._is_fitted.item()
    assert restored.model.gaussian.mean.shape == (4, 256)
    assert restored.model.gaussian.inv_covariance.shape == (256, 4, 4)
    model.eval()
    restored.eval()
    batch = next(iter(data.val_dataloader()))
    with torch.no_grad():
        expected, actual = model(batch[0]), restored(batch[0])
    torch.testing.assert_close(actual.pred_score, expected.pred_score, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual.anomaly_map, expected.anomaly_map, rtol=1e-5, atol=1e-6)
    before = restored.post_processor.state_dict()

    def forbidden():
        raise AssertionError("Checkpoint statistics must not be fitted again")

    monkeypatch.setattr(restored, "fit", forbidden)
    evaluator = Trainer(default_root_dir=str(tmp_path), accelerator="cpu", devices=1, logger=False,
                        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
    evaluator.validate(restored, datamodule=data, ckpt_path=path)
    evaluator.test(restored, datamodule=data)
    evaluator.predict(restored, datamodule=data)
    for name, value in restored.post_processor.state_dict().items():
        torch.testing.assert_close(value, before[name], equal_nan=True)
    for name in ("model.idx", "model.gaussian.mean", "model.gaussian.inv_covariance", "_is_fitted"):
        assert torch.equal(restored.state_dict()[name], saved["state_dict"][name])
    restored.train()
    resumed = Trainer(default_root_dir=os.path.join(tmp_path, "resume"), accelerator="cpu", devices=1,
                      max_epochs=2, logger=False, num_sanity_val_steps=0, enable_checkpointing=False,
                      enable_progress_bar=False, enable_model_summary=False)
    resumed.fit(restored, datamodule=data, ckpt_path=path)
    assert resumed.global_step > saved["global_step"]
    for name in ("model.idx", "model.gaussian.mean", "model.gaussian.inv_covariance", "_is_fitted"):
        assert torch.equal(restored.state_dict()[name], saved["state_dict"][name])


def consume_padim_export(root):
    """Load full pure-model state while blocking all external asset access."""
    from visionlab.inference.predictor import Predictor

    io_utils.block_network()
    path = os.path.join(root, "export", "model.pt")
    expected = torch.load(os.path.join(root, "expected.pt"), map_location="cpu", weights_only=True)
    exported = io_utils.load_export(path)
    assert exported["model_name"] == "padim" and exported["model_args"]["pre_trained"] is False
    assert {"idx", "gaussian.mean", "gaussian.inv_covariance"} <= exported["model_state"].keys()
    assert any(name.startswith("feature_extractor.") for name in exported["model_state"])
    assert all("path" not in name for name in exported["model_args"])

    def audit_open(event, args):
        if event == "open":
            opened, mode, flags = args
            assert isinstance(opened, (str, bytes)) and os.path.abspath(opened) == path, opened
            assert not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)

    sys.addaudithook(audit_open)
    predictor = Predictor.from_export(path)
    outputs = predictor.predict(expected["images"])
    for name, value in predictor.model.state_dict().items():
        assert torch.equal(value, exported["model_state"][name]), name
    for name, value in outputs.items():
        torch.testing.assert_close(value, expected["outputs"][name], rtol=1e-5, atol=1e-6)


def test_padim_export_predictor_in_isolated_process(fitted_padim):
    """Match native predictions without importing Lightning or reading pretrained assets."""
    root, model, data, _, _ = fitted_padim
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
class Block(importlib.abc.MetaPathFinder):
    """Reject training frameworks before loading the pure predictor consumer."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("anomalib", "lightning", "pytorch_lightning"):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, Block())
sys.path.insert(0, sys.argv[2])
runpy.run_path(sys.argv[1])["consume_padim_export"](sys.argv[3])
assert not {"anomalib", "lightning", "pytorch_lightning"} & set(sys.modules)
'''
    result = subprocess.run([sys.executable, "-c", script, os.path.abspath(__file__), os.path.join(PROJECT_ROOT, "src"), root],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_padim_stats_information_and_incompatible_indices(tmp_path, capsys):
    """Record differences only for aligned channel indices and retain an informational report."""
    namespace = runpy.run_path(os.path.join(PROJECT_ROOT, "evaluation", "anomalib", "compare_padim_stats.py"))
    reference_path, native_path = os.path.join(tmp_path, "reference.ckpt"), os.path.join(tmp_path, "native.ckpt")
    reference = {"model.idx": torch.tensor([0, 1]), "model.gaussian.mean": torch.ones(2, 3),
                 "model.gaussian.inv_covariance": torch.eye(2).expand(3, 2, 2).clone()}
    native = {name: value.clone() for name, value in reference.items()}
    native["model.gaussian.mean"] += 0.25
    torch.save({"state_dict": reference}, reference_path)
    torch.save({"state_dict": native}, native_path)
    report_path = os.path.join(tmp_path, "report.json")
    report = namespace["main"](["--reference", reference_path, "--native", native_path, "--out", report_path])
    assert report["informational_only"] and "passed" not in report
    assert report["idx"]["equal"] and report["idx"]["overlapping_channels"] == 2
    assert report["buffers"]["model.gaussian.mean"]["max_absolute_difference"] == 0.25
    assert report["buffers"]["model.gaussian.mean"]["max_relative_difference"] == 0.25
    assert "Informational comparison only" in capsys.readouterr().out
    with open(report_path, encoding="utf-8") as stream:
        assert json.load(stream) == report
    native["model.idx"] = torch.tensor([1, 2])
    torch.save({"state_dict": native}, native_path)
    report = namespace["compare_stats"](reference_path, native_path)
    assert not report["idx"]["equal"] and report["idx"]["overlapping_channels"] == 1
    for values in report["buffers"].values():
        assert values["not_comparable_reason"] == "idx_mismatch"
        assert values["max_absolute_difference"] is values["max_relative_difference"] is None
