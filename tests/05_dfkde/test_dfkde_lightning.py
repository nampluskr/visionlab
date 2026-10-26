# tests/05_dfkde/test_dfkde_lightning.py: pytest native image-only DFKDE, checkpoints and pure exports

import inspect
import json
import os
import runpy
import subprocess
import sys

import pytest
import torch

from visionlab.anomaly_detection.common.evaluator import Evaluator
from visionlab.anomaly_detection.common.pre_processor import IMAGENET_MEAN, IMAGENET_STD
from visionlab.anomaly_detection.models.components.classification import FeatureScalingMethod
from visionlab.utils import io as io_utils

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def padim_helpers():
    """Reuse small native data and local ResNet18 fixture builders without running older tests."""
    return runpy.run_path(os.path.join(PROJECT_ROOT, "tests", "03_padim", "test_padim_lightning.py"))


def classifier_state_names():
    """Name all fitted classifier buffers persisted by a native checkpoint."""
    return ["model.classifier." + name for name in (
        "pca_model.mean", "pca_model.singular_vectors", "pca_model.num_components", "max_length",
        "kde_model.bw_transform", "kde_model.dataset", "kde_model.norm")] + ["_is_fitted"]


@pytest.fixture(scope="module")
def dfkde_assets(tmp_path_factory):
    """Map a temporary full ResNet18 state for native constructors."""
    root = str(tmp_path_factory.mktemp("dfkde_assets"))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(io_utils, "PATHS_FILE", padim_helpers()["make_padim_assets"](root))
        yield root


def test_dfkde_signature_pretrained_weights_and_image_evaluator(dfkde_assets, tmp_path, monkeypatch):
    """Preserve ten defaults, optional teacher injection and an image-only default Evaluator."""
    from visionlab.anomaly_detection.models.dfkde.lightning_model import DfkdeLitModel

    expected = {"backbone": "resnet18", "layers": ("layer4",), "pre_trained": True, "n_pca_components": 16,
                "feature_scaling_method": FeatureScalingMethod.SCALE, "max_training_points": 40000,
                "pre_processor": True, "post_processor": True, "evaluator": True, "visualizer": True}
    signature = inspect.signature(DfkdeLitModel.__init__)
    assert list(signature.parameters) == ["self"] + list(expected)
    assert {name: signature.parameters[name].default for name in expected} == expected
    model = DfkdeLitModel(n_pca_components=2, visualizer=False)
    source = torch.load(os.path.join(dfkde_assets, "resnet18.pth"), map_location="cpu", weights_only=True)
    for name, value in model.model.feature_extractor.feature_extractor.state_dict().items():
        assert torch.equal(value, source[name]), name
    assert model.teacher_weights_path == os.path.join(dfkde_assets, "resnet18.pth")
    assert model.configure_optimizers() is None
    assert model.trainer_arguments == {"gradient_clip_val": 0, "max_epochs": 1, "num_sanity_val_steps": 0}
    assert model.learning_type.value == "one_class"
    assert [entry[0] for entry in model.evaluator.val_metrics] == ["image_AUROC"]
    assert [entry[0] for entry in model.evaluator.test_metrics] == ["image_AUROC", "image_F1Score"]
    monkeypatch.setattr(io_utils, "PATHS_FILE", os.path.join(tmp_path, "absent.toml"))
    with pytest.raises(FileNotFoundError):
        DfkdeLitModel(n_pca_components=2, visualizer=False)
    mapping = os.path.join(tmp_path, "paths.toml")
    with open(mapping, "w", encoding="utf-8", newline="\n") as stream:
        stream.write('[backbones]\nresnet18 = "missing.pth"\n')
    monkeypatch.setattr(io_utils, "PATHS_FILE", mapping)
    with pytest.raises(FileNotFoundError):
        DfkdeLitModel(n_pca_components=2, visualizer=False)
    custom = Evaluator()
    untrained = DfkdeLitModel(pre_trained=False, n_pca_components=2, evaluator=custom, visualizer=False)
    assert untrained.teacher_weights_path is None and untrained.embeddings == []
    assert untrained.evaluator is custom and not untrained._is_fitted.item()


@pytest.mark.parametrize("first_hook", ("on_validation_start", "on_train_epoch_end"))
@pytest.mark.parametrize("fit_result", (False, True))
def test_dfkde_one_time_fit_ignores_boolean_and_keeps_embeddings(first_hook, fit_result, monkeypatch):
    """Retain batch dimensions and collected features and ignore the classifier fit return value."""
    from visionlab.anomaly_detection.models.dfkde.lightning_model import DfkdeLitModel

    model = DfkdeLitModel(pre_trained=False, n_pca_components=2, evaluator=False, visualizer=False)
    seen, collected, calls = [], [], []

    def features(image):
        seen.append(image.clone())
        return torch.randn(image.shape[0], 6)

    def fit(embeddings):
        calls.append(embeddings.clone())
        return fit_result

    monkeypatch.setattr(model.model, "forward", features)
    monkeypatch.setattr(model.model.classifier, "fit", fit)
    for batch_size in (1, 2):
        image = torch.rand(batch_size, 3, 64, 64)
        loss = model.training_step((image, {}), 0)
        assert loss.ndim == 0 and loss.item() == 0 and loss.requires_grad
        assert model.embeddings[-1].shape == (batch_size, 6)
        collected.append(model.embeddings[-1].clone())
        expected = (image - torch.tensor(IMAGENET_MEAN).reshape(1, 3, 1, 1)) / \
                   torch.tensor(IMAGENET_STD).reshape(1, 3, 1, 1)
        torch.testing.assert_close(seen[-1], expected)
    getattr(model, first_hook)()
    model.on_validation_start()
    model.on_validation_start()
    model.on_train_epoch_end()
    assert len(calls) == 1 and model._is_fitted.item()
    assert torch.equal(calls[0], torch.vstack(collected)) and len(model.embeddings) == 2
    for actual, expected in zip(model.embeddings, collected, strict=True):
        assert torch.equal(actual, expected)


@pytest.fixture(scope="module")
def fitted_dfkde(tmp_path_factory, dfkde_assets):
    """Fit the real image-only model for one epoch and persist a checkpoint and pure export."""
    from lightning.pytorch import Trainer
    from visionlab.anomaly_detection.models.dfkde.lightning_model import DfkdeLitModel

    root = str(tmp_path_factory.mktemp("fitted_dfkde"))
    data = padim_helpers()["make_padim_data"]()
    model = DfkdeLitModel(n_pca_components=2, visualizer=False)
    trainer = Trainer(default_root_dir=root, accelerator="cpu", devices=1, max_epochs=1, logger=False,
                      num_sanity_val_steps=0, enable_checkpointing=False,
                      enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(model, datamodule=data)
    path = os.path.join(root, "model.ckpt")
    trainer.save_checkpoint(path)
    return root, model, data, path, trainer


def test_dfkde_fit_checkpoint_restore_and_resume_without_refitting(fitted_dfkde, tmp_path, monkeypatch):
    """Restore all density buffers and image calibration across standalone evaluation and resume."""
    from lightning.pytorch import Trainer
    from visionlab.anomaly_detection.models.dfkde.lightning_model import DfkdeLitModel

    root, model, data, path, trainer = fitted_dfkde
    assert model._is_fitted.item() and model.configure_optimizers() is None
    assert not trainer.lr_scheduler_configs and len(model.embeddings) == 2
    classifier = model.model.classifier
    assert classifier.pca_model.mean.shape == (512,) and classifier.pca_model.singular_vectors.shape == (512, 2)
    assert classifier.pca_model.num_components.item() == 2
    assert classifier.kde_model.dataset.shape == (4, 2) and classifier.kde_model.bw_transform.shape == (2, 2)
    assert classifier.kde_model.norm.numel() == classifier.max_length.numel() == 1
    assert torch.isfinite(classifier.kde_model.norm) and torch.isfinite(classifier.max_length)
    source = torch.load(model.teacher_weights_path, map_location="cpu", weights_only=True)
    for name, value in model.model.feature_extractor.feature_extractor.state_dict().items():
        assert torch.equal(value, source[name]), name
    names = classifier_state_names()
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert set(names) <= saved["state_dict"].keys() and saved["post_processor"] is not None
    assert torch.isfinite(saved["post_processor"]["image_threshold"])
    assert torch.isnan(saved["post_processor"]["pixel_threshold"])
    restored = DfkdeLitModel(pre_trained=False, n_pca_components=2, visualizer=False)
    assert restored.model.classifier.pca_model.mean.numel() == restored.model.classifier.kde_model.dataset.numel() == 0
    restored.load_state_dict(saved["state_dict"], strict=True)
    restored.on_load_checkpoint(saved)
    assert restored._is_fitted.item() and restored.model.classifier.kde_model.dataset.shape == (4, 2)
    model.eval()
    restored.eval()
    batch = next(iter(data.val_dataloader()))
    with torch.no_grad():
        expected, actual = model(batch[0].clone()), restored(batch[0].clone())
    assert expected.anomaly_map is actual.anomaly_map is None
    torch.testing.assert_close(actual.pred_score, expected.pred_score, rtol=1e-5, atol=1e-6)
    before = restored.post_processor.state_dict()

    def forbidden(embeddings):
        raise AssertionError("Checkpoint KDE must not be fitted again")

    monkeypatch.setattr(restored.model.classifier, "fit", forbidden)
    evaluator = Trainer(default_root_dir=str(tmp_path), accelerator="cpu", devices=1, logger=False,
                        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
    evaluator.validate(restored, datamodule=data, ckpt_path=path)
    evaluator.test(restored, datamodule=data)
    predictions = evaluator.predict(restored, datamodule=data)
    assert set(restored.last_metrics["val"]) == {"image_AUROC"}
    assert set(restored.last_metrics["test"]) == {"image_AUROC", "image_F1Score"}
    assert all(output["anomaly_map"] is None and output["normalized_anomaly_map"] is None for output in predictions)
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


def consume_dfkde_export(root):
    """Restore image-only predictions and every classifier buffer with external asset access denied."""
    from visionlab.inference.predictor import Predictor

    io_utils.block_network()
    path = os.path.join(root, "export", "model.pt")
    expected = torch.load(os.path.join(root, "expected.pt"), map_location="cpu", weights_only=True)
    exported = io_utils.load_export(path)
    assert exported["model_name"] == "dfkde" and exported["model_args"]["pre_trained"] is False
    assert type(exported["model_args"]["feature_scaling_method"]) is str
    assert exported["model_args"]["feature_scaling_method"] == "scale" and type(exported["model_args"]["layers"]) is list
    keys = {name.removeprefix("model.") for name in classifier_state_names() if name != "_is_fitted"}
    assert keys <= exported["model_state"].keys()
    assert any(name.startswith("feature_extractor.") for name in exported["model_state"])

    def audit_open(event, args):
        if event == "open":
            opened, mode, flags = args
            assert isinstance(opened, (str, bytes)) and os.path.abspath(opened) == path, opened
            assert not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)

    sys.addaudithook(audit_open)
    predictor = Predictor.from_export(path)
    assert predictor.model.classifier.feature_scaling_method is FeatureScalingMethod.SCALE
    outputs = predictor.predict(expected["images"])
    assert outputs["anomaly_map"] is outputs["normalized_anomaly_map"] is outputs["pred_mask"] is None
    for name, value in predictor.model.state_dict().items():
        assert torch.equal(value, exported["model_state"][name]), name
    for name, value in outputs.items():
        if value is None:
            assert expected["outputs"][name] is None
        else:
            torch.testing.assert_close(value, expected["outputs"][name], rtol=1e-5, atol=1e-6)


def test_dfkde_export_predictor_in_isolated_process(fitted_dfkde):
    """Match native image-only predictions without training imports or pretrained asset reads."""
    root, model, data, _, _ = fitted_dfkde
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
    """Reject training frameworks before loading the pure image-only consumer."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("anomalib", "lightning", "pytorch_lightning"):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, Block())
sys.path.insert(0, sys.argv[2])
runpy.run_path(sys.argv[1])["consume_dfkde_export"](sys.argv[3])
assert not {"anomalib", "lightning", "pytorch_lightning"} & set(sys.modules)
'''
    result = subprocess.run([sys.executable, "-c", script, os.path.abspath(__file__), os.path.join(PROJECT_ROOT, "src"), root],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_dfkde_stats_sign_invariance_and_shape_only_kde_buffers(tmp_path, capsys):
    """Record scalar differences and subspaces while keeping sign-dependent KDE buffers informational."""
    namespace = runpy.run_path(os.path.join(PROJECT_ROOT, "evaluation", "anomalib", "compare_dfkde_stats.py"))
    reference_path, native_path = os.path.join(tmp_path, "reference.ckpt"), os.path.join(tmp_path, "native.ckpt")
    prefix = "model.classifier."
    reference = {prefix + "pca_model.mean": torch.ones(4), prefix + "pca_model.num_components": torch.tensor([2]),
                 prefix + "pca_model.singular_vectors": torch.eye(4)[:, :2], prefix + "max_length": torch.tensor(2.0),
                 prefix + "kde_model.norm": torch.tensor(4.0), prefix + "kde_model.bw_transform": torch.eye(2),
                 prefix + "kde_model.dataset": torch.ones(5, 2)}
    native = {name: value.clone() for name, value in reference.items()}
    native[prefix + "pca_model.mean"] += 0.25
    native[prefix + "pca_model.singular_vectors"][:, 0] *= -1
    native[prefix + "kde_model.dataset"] *= -1
    native[prefix + "max_length"] += 0.5
    native[prefix + "kde_model.norm"] += 1
    torch.save({"state_dict": reference}, reference_path)
    torch.save({"state_dict": native}, native_path)
    output = os.path.join(tmp_path, "report.json")
    report = namespace["main"](["--reference", reference_path, "--native", native_path, "--out", output])
    assert report["informational_only"] and "passed" not in report and report["num_components"]["equal"]
    for name in ("pca_model.mean", "max_length", "kde_model.norm"):
        assert report["buffers"][name]["max_relative_difference"] == 0.25
    assert report["buffers"]["pca_model.singular_vectors"]["subspace_distance"] == 0
    for name in ("kde_model.bw_transform", "kde_model.dataset"):
        assert report["buffers"][name]["shape_match"]
        assert report["buffers"][name]["comparison"] == "shape_only_sign_dependent"
        assert "max_absolute_difference" not in report["buffers"][name]
    assert "Informational comparison only" in capsys.readouterr().out
    with open(output, encoding="utf-8") as stream:
        assert json.load(stream) == report
    native[prefix + "pca_model.num_components"] = torch.tensor([1])
    native[prefix + "pca_model.singular_vectors"] = torch.eye(4)[:, :1]
    torch.save({"state_dict": native}, native_path)
    report = namespace["compare_stats"](reference_path, native_path)
    assert not report["num_components"]["equal"]
    assert report["buffers"]["pca_model.singular_vectors"]["not_comparable_reason"] == "component_count_mismatch"
