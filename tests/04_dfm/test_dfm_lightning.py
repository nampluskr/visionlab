# tests/04_dfm/test_dfm_lightning.py: pytest native DFM PCA fitting, checkpoints and pure exports

import inspect
import json
import os
import runpy
import subprocess
import sys

import pytest
import torch

from visionlab.anomaly_detection.common.pre_processor import IMAGENET_MEAN, IMAGENET_STD
from visionlab.utils import io as io_utils

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def padim_helpers():
    """Reuse the closest native model's synthetic data and local backbone builders."""
    return runpy.run_path(os.path.join(PROJECT_ROOT, "tests", "03_padim", "test_padim_lightning.py"))


@pytest.fixture(scope="module")
def dfm_assets(tmp_path_factory):
    """Map temporary ResNet18 weights for small native DFM integration tests."""
    root = str(tmp_path_factory.mktemp("dfm_assets"))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(io_utils, "PATHS_FILE", padim_helpers()["make_padim_assets"](root))
        yield root


def test_dfm_signature_and_pretrained_injection(dfm_assets, tmp_path, monkeypatch):
    """Keep ten constructor defaults and inject exact local weights only when requested."""
    from visionlab.anomaly_detection.models.dfm.lightning_model import DfmLitModel

    expected = {"backbone": "resnet50", "layer": "layer3", "pre_trained": True, "pooling_kernel_size": 4,
                "pca_level": 0.97, "score_type": "fre", "pre_processor": True, "post_processor": True,
                "evaluator": True, "visualizer": True}
    signature = inspect.signature(DfmLitModel.__init__)
    assert list(signature.parameters) == ["self"] + list(expected)
    assert {name: signature.parameters[name].default for name in expected} == expected
    model = DfmLitModel(backbone="resnet18", pca_level=2, visualizer=False)
    source = torch.load(os.path.join(dfm_assets, "resnet18.pth"), map_location="cpu", weights_only=True)
    for name, value in model.model.feature_extractor.feature_extractor.state_dict().items():
        assert torch.equal(value, source[name]), name
    assert model.teacher_weights_path == os.path.join(dfm_assets, "resnet18.pth")
    assert model.score_type == "fre" and model.configure_optimizers() is None
    assert model.trainer_arguments == {"gradient_clip_val": 0, "max_epochs": 1, "num_sanity_val_steps": 0}
    assert model.learning_type.value == "one_class"
    monkeypatch.setattr(io_utils, "PATHS_FILE", os.path.join(tmp_path, "absent.toml"))
    with pytest.raises(FileNotFoundError):
        DfmLitModel(backbone="resnet18", visualizer=False)
    mapping = os.path.join(tmp_path, "paths.toml")
    with open(mapping, "w", encoding="utf-8", newline="\n") as stream:
        stream.write('[backbones]\nresnet18 = "missing.pth"\n')
    monkeypatch.setattr(io_utils, "PATHS_FILE", mapping)
    with pytest.raises(FileNotFoundError):
        DfmLitModel(backbone="resnet18", visualizer=False)
    untrained = DfmLitModel(backbone="resnet18", pre_trained=False, visualizer=False)
    assert untrained.teacher_weights_path is None and untrained.embeddings == []
    assert "_is_fitted" in untrained.state_dict() and not untrained._is_fitted.item()


@pytest.mark.parametrize("first_hook", ("on_validation_start", "on_train_epoch_end"))
@pytest.mark.parametrize("batch_size", (1, 2))
def test_dfm_squeeze_dummy_loss_and_one_time_fitting(first_hook, batch_size, monkeypatch):
    """Preserve squeeze, in-place PCA fitting, retained embeddings and one-time memory hooks."""
    from visionlab.anomaly_detection.models.dfm.lightning_model import DfmLitModel

    model = DfmLitModel(backbone="resnet18", pre_trained=False, pca_level=2, evaluator=False, visualizer=False)
    seen, collected = [], []

    def get_features(image):
        seen.append(image.clone())
        return torch.randn(image.shape[0], 6)

    monkeypatch.setattr(model.model, "get_features", get_features)
    batch = (torch.rand(batch_size, 3, 64, 64), {})
    for index in range(2):
        loss = model.training_step(batch, index)
        assert loss.ndim == 0 and loss.item() == 0 and loss.requires_grad
        assert model.embeddings[-1].shape == ((6,) if batch_size == 1 else (2, 6))
        collected.append(model.embeddings[-1].clone())
    expected_image = (batch[0] - torch.tensor(IMAGENET_MEAN).reshape(1, 3, 1, 1)) / \
                     torch.tensor(IMAGENET_STD).reshape(1, 3, 1, 1)
    for image in seen:
        torch.testing.assert_close(image, expected_image)
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
    assert len(model.embeddings) == 2
    for actual, expected_embedding in zip(model.embeddings, collected, strict=True):
        assert torch.equal(actual, expected_embedding)
    torch.testing.assert_close(model.model.pca_model.mean, torch.vstack(collected).mean(dim=0))
    assert model.model.pca_model.singular_vectors.shape == (6, 2)


@pytest.fixture(scope="module")
def fitted_dfm(tmp_path_factory, dfm_assets):
    """Fit PCA for one native epoch and save a checkpoint and pure model export."""
    from lightning.pytorch import Trainer
    from visionlab.anomaly_detection.models.dfm.lightning_model import DfmLitModel

    root = str(tmp_path_factory.mktemp("fitted_dfm"))
    data = padim_helpers()["make_padim_data"]()
    model = DfmLitModel(backbone="resnet18", pca_level=2, visualizer=False)
    trainer = Trainer(default_root_dir=root, accelerator="cpu", devices=1, max_epochs=1, logger=False,
                      num_sanity_val_steps=0, enable_checkpointing=False,
                      enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(model, datamodule=data)
    path = os.path.join(root, "model.ckpt")
    trainer.save_checkpoint(path)
    return root, model, data, path, trainer


def test_dfm_fit_checkpoint_restore_and_resume_without_refitting(fitted_dfm, tmp_path, monkeypatch):
    """Restore all PCA buffers and calibration without refitting in standalone evaluation or resume."""
    from lightning.pytorch import Trainer
    from visionlab.anomaly_detection.models.dfm.lightning_model import DfmLitModel

    root, model, data, path, trainer = fitted_dfm
    assert model._is_fitted.item() and model.configure_optimizers() is None
    assert not trainer.lr_scheduler_configs and len(model.embeddings) == 2
    pca = model.model.pca_model
    assert pca.mean.shape == (256,) and pca.singular_vectors.shape == (256, 2) and pca.num_components.item() == 2
    source = torch.load(model.teacher_weights_path, map_location="cpu", weights_only=True)
    for name, value in model.model.feature_extractor.feature_extractor.state_dict().items():
        assert torch.equal(value, source[name]), name
    names = ("model.pca_model.mean", "model.pca_model.singular_vectors", "model.pca_model.num_components", "_is_fitted")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert set(names) <= saved["state_dict"].keys()
    assert saved["state_dict"]["_is_fitted"].item() and saved["post_processor"] is not None
    restored = DfmLitModel(backbone="resnet18", pre_trained=False, pca_level=2, visualizer=False)
    assert restored.model.pca_model.mean.numel() == restored.model.pca_model.singular_vectors.numel() == 0
    restored.load_state_dict(saved["state_dict"], strict=True)
    restored.on_load_checkpoint(saved)
    assert restored._is_fitted.item() and restored.model.pca_model.singular_vectors.shape == (256, 2)
    assert restored.model.pca_model.num_components.item() == 2
    model.eval()
    restored.eval()
    batch = next(iter(data.val_dataloader()))
    with torch.no_grad():
        expected, actual = model(batch[0].clone()), restored(batch[0].clone())
    torch.testing.assert_close(actual.pred_score, expected.pred_score, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual.anomaly_map, expected.anomaly_map, rtol=1e-5, atol=1e-6)
    before = restored.post_processor.state_dict()

    def forbidden():
        raise AssertionError("Checkpoint PCA must not be fitted again")

    monkeypatch.setattr(restored, "fit", forbidden)
    evaluator = Trainer(default_root_dir=str(tmp_path), accelerator="cpu", devices=1, logger=False,
                        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
    evaluator.validate(restored, datamodule=data, ckpt_path=path)
    evaluator.test(restored, datamodule=data)
    evaluator.predict(restored, datamodule=data)
    for name, value in restored.post_processor.state_dict().items():
        torch.testing.assert_close(value, before[name], equal_nan=True)
    for name in names:
        assert torch.equal(restored.state_dict()[name], saved["state_dict"][name])
    restored.train()
    resumed = Trainer(default_root_dir=os.path.join(tmp_path, "resume"), accelerator="cpu", devices=1,
                      max_epochs=2, logger=False, num_sanity_val_steps=0, enable_checkpointing=False,
                      enable_progress_bar=False, enable_model_summary=False)
    resumed.fit(restored, datamodule=data, ckpt_path=path)
    assert resumed.global_step > saved["global_step"]
    for name in names:
        assert torch.equal(restored.state_dict()[name], saved["state_dict"][name])


def consume_dfm_export(root):
    """Restore all pure PCA buffers while denying pretrained files and training imports."""
    from visionlab.inference.predictor import Predictor

    io_utils.block_network()
    path = os.path.join(root, "export", "model.pt")
    expected = torch.load(os.path.join(root, "expected.pt"), map_location="cpu", weights_only=True)
    exported = io_utils.load_export(path)
    assert exported["model_name"] == "dfm" and exported["model_args"]["pre_trained"] is False
    assert {"pca_model.mean", "pca_model.singular_vectors", "pca_model.num_components"} <= exported["model_state"].keys()
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


def test_dfm_export_predictor_in_isolated_process(fitted_dfm):
    """Match native FRE predictions in a fresh process with file and dependency guards."""
    root, model, data, _, _ = fitted_dfm
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
    """Reject training frameworks before loading the pure DFM consumer."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("anomalib", "lightning", "pytorch_lightning"):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, Block())
sys.path.insert(0, sys.argv[2])
runpy.run_path(sys.argv[1])["consume_dfm_export"](sys.argv[3])
assert not {"anomalib", "lightning", "pytorch_lightning"} & set(sys.modules)
'''
    result = subprocess.run([sys.executable, "-c", script, os.path.abspath(__file__), os.path.join(PROJECT_ROOT, "src"), root],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_dfm_stats_subspace_signs_and_component_mismatch(tmp_path, capsys):
    """Compare means and sign-invariant subspaces and reject incompatible component counts."""
    namespace = runpy.run_path(os.path.join(PROJECT_ROOT, "evaluation", "anomalib", "compare_dfm_stats.py"))
    reference_path, native_path = os.path.join(tmp_path, "reference.ckpt"), os.path.join(tmp_path, "native.ckpt")
    reference = {"model.pca_model.mean": torch.ones(4), "model.pca_model.num_components": torch.tensor([2]),
                 "model.pca_model.singular_vectors": torch.eye(4)[:, :2]}
    native = {name: value.clone() for name, value in reference.items()}
    native["model.pca_model.mean"] += 0.25
    native["model.pca_model.singular_vectors"][:, 0] *= -1
    torch.save({"state_dict": reference}, reference_path)
    torch.save({"state_dict": native}, native_path)
    report_path = os.path.join(tmp_path, "report.json")
    report = namespace["main"](["--reference", reference_path, "--native", native_path, "--out", report_path])
    assert report["informational_only"] and "passed" not in report
    assert report["num_components"]["equal"]
    assert report["mean"]["max_absolute_difference"] == report["mean"]["max_relative_difference"] == 0.25
    assert report["singular_vectors"]["subspace_distance"] == 0
    assert report["singular_vectors"]["cross_gram_squared_frobenius"] == 2
    assert "Informational comparison only" in capsys.readouterr().out
    with open(report_path, encoding="utf-8") as stream:
        assert json.load(stream) == report
    native["model.pca_model.num_components"] = torch.tensor([1])
    native["model.pca_model.singular_vectors"] = torch.eye(4)[:, :1]
    torch.save({"state_dict": native}, native_path)
    report = namespace["compare_stats"](reference_path, native_path)
    assert not report["num_components"]["equal"]
    assert report["singular_vectors"]["not_comparable_reason"] == "component_count_mismatch"
    assert report["singular_vectors"]["subspace_distance"] is None
