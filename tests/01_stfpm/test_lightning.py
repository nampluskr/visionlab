# tests/01_stfpm/test_lightning.py: pytest native STFPM training, checkpoint and local teacher contracts

import math
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import timm
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.trainer.states import TrainerFn
from PIL import Image

from visionlab.anomaly_detection.common.evaluator import Evaluator
from visionlab.anomaly_detection.common.post_processor import PostProcessor
from visionlab.anomaly_detection.common.pre_processor import IMAGENET_MEAN, IMAGENET_STD, PreProcessor
from visionlab.anomaly_detection.common.visualizer import ImageVisualizer
from visionlab.anomaly_detection.data.mvtec.lightning_data import MVTecLitData
from visionlab.anomaly_detection.models.stfpm import lightning_model
from visionlab.anomaly_detection.models.stfpm.lightning_model import StfpmLitModel
from visionlab.anomaly_detection.models.stfpm.model import STFPMModel
from visionlab.utils import io as io_utils

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_assets(root):
    """Write a random full ResNet state and a temporary backbone TOML mapping."""
    path = os.path.join(root, "resnet18.pth")
    torch.save(timm.create_model("resnet18", pretrained=False).state_dict(), path)
    paths_file = os.path.join(root, "paths.toml")
    with open(paths_file, "w", encoding="utf-8", newline="\n") as stream:
        stream.write('[backbones]\nresnet18 = "' + path.replace("\\", "/") + '"\n')
    return paths_file


def make_fixture(root, category="bottle"):
    """Write tiny prepared MVTec images with paired masks and both test classes."""
    for split, label, count in (("train", "good", 4), ("test", "good", 4), ("test", "bad", 4)):
        directory = os.path.join(root, category, split, label)
        os.makedirs(directory, exist_ok=True)
        for index in range(count):
            color = (40 + index * 10, 80, 120) if label == "good" else (180, 40 + index * 10, 70)
            with Image.new("RGB", (24, 24), color=color) as image:
                image.save(os.path.join(directory, f"{index:03d}.png"))
            if label == "bad":
                mask_dir = os.path.join(root, category, "ground_truth", "bad")
                os.makedirs(mask_dir, exist_ok=True)
                with Image.new("L", (24, 24), color=0) as mask:
                    mask.paste(255, (8, 8, 16, 16))
                    mask.save(os.path.join(mask_dir, f"{index:03d}_mask.png"))


def make_data(root, category="bottle"):
    """Build the existing deterministic native module on prepared local data."""
    return MVTecLitData(root=root, category=category, train_batch_size=2, eval_batch_size=8,
                       num_workers=0, val_split_mode="from_test", val_split_ratio=0.4, seed=42)


def assert_same_state(left, right):
    """Compare scalar calibration tensors including uncalibrated NaNs."""
    assert set(left) == set(right)
    for name in left:
        torch.testing.assert_close(left[name], right[name], equal_nan=True)


@pytest.fixture(scope="module")
def local_assets(tmp_path_factory):
    """Share one local random teacher file without depending on machine assets."""
    root = str(tmp_path_factory.mktemp("lightning_assets"))
    io_utils.block_network()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(io_utils, "PATHS_FILE", make_assets(root))
        yield root


@pytest.fixture
def data_root(tmp_path, local_assets):
    """Prepare independent image files and a repeatable seed for each integration check."""
    root = os.path.join(tmp_path, "data")
    make_fixture(root)
    seed_everything(42, workers=True)
    return root


@pytest.fixture(scope="module")
def fitted_model(tmp_path_factory, local_assets):
    """Run the two-epoch CPU fixture once and retain its checkpoint and validation counts."""
    root = str(tmp_path_factory.mktemp("fitted_stfpm"))
    make_fixture(os.path.join(root, "data"))
    data = make_data(os.path.join(root, "data"))
    seed_everything(42, workers=True)
    model = StfpmLitModel(visualizer=False)
    checkpoint = ModelCheckpoint(dirpath=os.path.join(root, "weights", "lightning"), filename="model",
                                 monitor=None, save_top_k=1, save_on_train_epoch_end=False, enable_version_counter=False)
    trainer = Trainer(default_root_dir=root, accelerator="cpu", devices=1, max_epochs=2, logger=False,
                      callbacks=[checkpoint], limit_train_batches=1, num_sanity_val_steps=0,
                      enable_progress_bar=False, enable_model_summary=False)
    counts, compute = [], model.evaluator.compute

    def capture(stage):
        if stage == "val":
            metric = model.evaluator.val_metrics[0][1]
            counts.append(metric.positives + metric.negatives)
        return compute(stage)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(model.evaluator, "compute", capture)
        trainer.fit(model, datamodule=data)
    return root, model, trainer, data, checkpoint.best_model_path, counts


def test_constructor_signature_and_component_contracts(monkeypatch):
    """Keep the six-argument signature and strict True, False or component instance contract."""
    import inspect

    expected = {"backbone": "resnet18", "layers": ("layer1", "layer2", "layer3"),
                "pre_processor": True, "post_processor": True, "evaluator": True, "visualizer": True}
    signature = inspect.signature(StfpmLitModel.__init__)
    assert list(signature.parameters) == ["self"] + list(expected)
    assert {name: signature.parameters[name].default for name in expected} == expected
    stub = torch.nn.Module()
    stub.teacher_model = torch.nn.Identity()
    monkeypatch.setattr(lightning_model, "STFPMModel", lambda backbone, layers: stub)
    monkeypatch.setattr(lightning_model, "load_teacher_weights", lambda feature_extractor, backbone: "temporary.pth")
    for name, component_class in (("pre_processor", PreProcessor), ("post_processor", PostProcessor),
                                  ("evaluator", Evaluator), ("visualizer", ImageVisualizer)):
        for value in (True, False, component_class()):
            model = StfpmLitModel(**{name: value})
            actual = getattr(model, name)
            if value is True:
                assert isinstance(actual, component_class)
            else:
                assert actual is (None if value is False else value)
        for value in (None, 1, "invalid", object()):
            with pytest.raises(TypeError):
                StfpmLitModel(**{name: value})


def test_normalization_and_validation_only_calibration(local_assets, monkeypatch):
    """Normalize once across all steps and calibrate only ordinary fit validation."""
    model = StfpmLitModel(visualizer=False)
    image = torch.full((2, 3, 32, 32), 0.4)
    batch = (image, {"label": torch.tensor([0, 1]), "mask": torch.zeros(2, 32, 32, dtype=torch.int64),
                     "image_path": ["a.png", "b.png"], "sample_id": ["bottle/a.png", "bottle/b.png"]})
    seen, logged = [], []
    handle = model.model.register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0].detach().clone()))
    monkeypatch.setattr(model, "log", lambda name, value, **options: logged.append((name, options)))
    try:
        model.train()
        model.training_step(batch, 0)
        assert logged[0][0] == "train_loss" and logged[0][1]["batch_size"] == 1
        model.eval()
        model._trainer = SimpleNamespace(state=SimpleNamespace(fn=TrainerFn.VALIDATING), sanity_checking=False)
        before = model.post_processor.state_dict()
        model.validation_step(batch, 0)
        model.on_validation_epoch_end()
        tested, predicted = model.test_step(batch, 0), model.predict_step(batch, 0)
        assert_same_state(before, model.post_processor.state_dict())
        expected = (image - torch.tensor(IMAGENET_MEAN).reshape(1, 3, 1, 1)) / torch.tensor(IMAGENET_STD).reshape(1, 3, 1, 1)
        assert len(seen) == 4
        for actual in seen:
            torch.testing.assert_close(actual, expected)
        assert torch.equal(image, torch.full_like(image, 0.4))
        score, maps = predicted["pred_score"].clone(), predicted["anomaly_map"].clone()
        model._post_process(predicted)
        assert torch.equal(score, predicted["pred_score"]) and torch.equal(maps, predicted["anomaly_map"])
        assert "normalized_pred_score" in tested
        model._trainer.state.fn, model._trainer.sanity_checking = TrainerFn.FITTING, True
        model.validation_step(batch, 0)
        model.on_validation_epoch_end()
        assert_same_state(before, model.post_processor.state_dict())
        model._trainer.sanity_checking = False
        model.on_validation_epoch_start()
        model.validation_step(batch, 0)
        assert_same_state(before, model.post_processor.state_dict())
        model.on_validation_epoch_end()
        assert model.post_processor.is_calibrated
        assert math.isnan(model.last_metrics["val"]["pixel_AUROC"])
        assert model.metric_reasons["val_pixel_AUROC"] == "no_positive_targets"
    finally:
        handle.remove()


def test_fit_checkpoint_export_restore_resume_and_frozen_evaluation(fitted_model):
    """Preserve independent epochs, full exports, frozen evaluation, checkpoint recovery and resume."""
    root, model, trainer, data, checkpoint, counts = fitted_model
    assert counts == [2, 2] and os.path.basename(checkpoint) == "model.ckpt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert_same_state(saved["post_processor"], model.post_processor.state_dict())
    exported = io_utils.load_export(os.path.join(root, "export", "model.pt"))
    assert exported["format_version"] == 1
    assert any(name.startswith("teacher_model.") for name in exported["model_state"])
    for name, value in model.model.state_dict().items():
        assert torch.equal(exported["model_state"][name], value.cpu())
    pure = STFPMModel(backbone=exported["model_args"]["backbone"], layers=exported["model_args"]["layers"])
    pure.load_state_dict(exported["model_state"], strict=True)
    pure.eval()
    pre = PreProcessor(mean=exported["pre_processor"]["mean"], std=exported["pre_processor"]["std"])
    post = PostProcessor()
    post.load_state_dict(exported["post_processor"])
    image = torch.rand(1, 3, 32, 32)
    model.eval()
    with torch.no_grad():
        expected, actual = model(image), pure(pre(image))
    torch.testing.assert_close(actual.pred_score, expected.pred_score)
    torch.testing.assert_close(actual.anomaly_map, expected.anomaly_map)
    expected_processed = model.post_processor(expected.pred_score, expected.anomaly_map)
    for name, value in post(actual.pred_score, actual.anomaly_map).items():
        torch.testing.assert_close(value, expected_processed[name], equal_nan=True)
    before = model.post_processor.state_dict()
    trainer.validate(model, datamodule=data)
    trainer.test(model, datamodule=data)
    expected_metrics = dict(model.last_metrics["test"])
    trainer.predict(model, datamodule=data)
    assert_same_state(before, model.post_processor.state_dict())
    restored = StfpmLitModel(visualizer=False)
    restored.on_load_checkpoint(saved)
    restored.load_state_dict(saved["state_dict"])
    evaluator = Trainer(default_root_dir=os.path.join(root, "restore"), accelerator="cpu", devices=1, logger=False,
                        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
    evaluator.test(restored, datamodule=data, ckpt_path=checkpoint)
    assert_same_state(before, restored.post_processor.state_dict())
    assert restored.last_metrics["test"] == pytest.approx(expected_metrics, abs=1e-6)
    resumed = StfpmLitModel(visualizer=False)
    resume_trainer = Trainer(default_root_dir=os.path.join(root, "resume"), accelerator="cpu", devices=1,
                             max_epochs=3, logger=False, limit_train_batches=1, num_sanity_val_steps=0,
                             enable_progress_bar=False, enable_model_summary=False)
    resume_trainer.fit(resumed, datamodule=data, ckpt_path=checkpoint)
    assert resume_trainer.global_step > saved["global_step"]


def test_visualizer_category_paths(data_root, tmp_path):
    """Save one PNG per test sample without colliding across category basenames."""
    model = StfpmLitModel(visualizer=True)
    trainer = Trainer(default_root_dir=str(tmp_path), accelerator="cpu", devices=1, logger=False,
                      enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
    expected_count = 0
    for category in ("bottle", "cable"):
        if category != "bottle":
            make_fixture(data_root, category=category)
        data = make_data(data_root, category=category)
        trainer.test(model, datamodule=data)
        expected_count += len(data.test_data)
        first = data.test_data.samples[0]["sample_id"]
        assert os.path.isfile(os.path.join(tmp_path, "images", model.visualizer.file_name(first)))
    files = [name for _, _, names in os.walk(os.path.join(tmp_path, "images")) for name in names if name.endswith(".png")]
    assert len(files) == expected_count


def test_local_teacher_keys_and_missing_assets(local_assets, tmp_path, monkeypatch):
    """Load exact local weights and reject absent assets and invalid feature extractor keys."""
    model = StfpmLitModel(visualizer=False)
    source = torch.load(os.path.join(local_assets, "resnet18.pth"), map_location="cpu", weights_only=True)
    for name, value in model.model.teacher_model.feature_extractor.state_dict().items():
        assert torch.equal(value.cpu(), source[name]), name
    missing = os.path.join(tmp_path, "missing.toml")
    monkeypatch.setattr(io_utils, "PATHS_FILE", missing)
    with pytest.raises(FileNotFoundError):
        StfpmLitModel(visualizer=False)
    for contents in ("[backbones]\n", '[backbones]\nresnet18 = "missing-weights.pth"\n'):
        with open(missing, "w", encoding="utf-8") as stream:
            stream.write(contents)
        with pytest.raises(FileNotFoundError):
            StfpmLitModel(visualizer=False)
    monkeypatch.setattr(io_utils, "read_backbone_path", lambda name: "unused")
    unexpected = dict(source)
    unexpected["layer1.invalid"] = torch.zeros(1)
    for bad in ({name: value for name, value in source.items() if name != "conv1.weight"}, unexpected):
        monkeypatch.setattr(torch, "load", lambda path, map_location, weights_only: bad)
        with pytest.raises(RuntimeError):
            io_utils.load_teacher_weights(model.model.teacher_model, "resnet18")


def test_legacy_teacher_without_batchnorm_counters(local_assets, tmp_path, monkeypatch):
    """Accept legacy BN buffer omissions while rejecting every missing non-counter key."""
    source = torch.load(os.path.join(local_assets, "resnet18.pth"), map_location="cpu", weights_only=True)
    assert any(name.endswith(".num_batches_tracked") for name in source)
    legacy = {name: value.clone() for name, value in source.items() if not name.endswith(".num_batches_tracked")}
    assert any(name.startswith("fc.") for name in legacy) and any(name.startswith("layer4.") for name in legacy)
    for name, value in legacy.items():
        if name.endswith(".running_mean"):
            value.fill_(0.25)
        elif name.endswith(".running_var"):
            value.fill_(1.75)
    path, paths_file = os.path.join(tmp_path, "legacy.pth"), os.path.join(tmp_path, "paths.toml")
    torch.save(legacy, path)
    with open(paths_file, "w", encoding="utf-8", newline="\n") as stream:
        stream.write('[backbones]\nresnet18 = "' + path.replace("\\", "/") + '"\n')
    monkeypatch.setattr(io_utils, "PATHS_FILE", paths_file)
    model = StfpmLitModel(visualizer=False)
    loaded = model.model.teacher_model.feature_extractor.state_dict()
    assert any(name.endswith(".num_batches_tracked") for name in loaded)
    for name, value in loaded.items():
        if name.endswith(".num_batches_tracked"):
            assert value.item() == 0, name
        else:
            assert torch.equal(value.cpu(), legacy[name]), name
    legacy.pop("layer1.0.conv1.weight")
    torch.save(legacy, path)
    with pytest.raises(RuntimeError, match=r"layer1\.0\.conv1\.weight"):
        StfpmLitModel(visualizer=False)


def test_pure_constructor_and_export_loader_without_training_imports(tmp_path):
    """Restore an uncalibrated full export with training imports blocked in a subprocess."""
    pure = STFPMModel(layers=["layer1", "layer2", "layer3"])
    path = os.path.join(tmp_path, "export.pt")
    io_utils.save_export(path, pure, {"backbone": "resnet18", "layers": ["layer1", "layer2", "layer3"]},
                         PreProcessor(mean=IMAGENET_MEAN, std=IMAGENET_STD), PostProcessor())
    script = '''
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    """Reject training frameworks in a fresh export consumer."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("anomalib", "lightning", "pytorch_lightning"):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, Block())
sys.path.insert(0, sys.argv[2])
from visionlab.utils.io import block_network, load_export
block_network()
import torch
from visionlab.inference.predictor import Predictor
payload = load_export(sys.argv[1])
assert any(name.startswith("teacher_model.") for name in payload["model_state"])
predictor = Predictor.from_export(sys.argv[1])
result = predictor.predict(torch.zeros(1, 3, 32, 32))
assert result["pred_label"].shape == (1, 1)
assert result["anomaly_map"].shape == (1, 1, 32, 32)
assert not {"anomalib", "lightning", "pytorch_lightning"} & set(sys.modules)
'''
    result = subprocess.run([sys.executable, "-c", script, path, os.path.join(PROJECT_ROOT, "src")],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
