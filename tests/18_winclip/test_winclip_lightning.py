# tests/18_winclip/test_winclip_lightning.py: Native WinCLIP injection, calibration, checkpoints and pure export

import inspect
import json
import os
import runpy
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from lightning.pytorch import LightningDataModule, Trainer
from lightning.pytorch.trainer.states import TrainerFn

from visionlab.anomaly_detection.common.learning_type import LearningType
from visionlab.anomaly_detection.models.winclip import model as pure_module
from visionlab.anomaly_detection.models.winclip.lightning_model import CLIP_MEAN, CLIP_STD, WinClipLitModel
from visionlab.utils import io as io_utils

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_NAMES = ("pred_score", "anomaly_map", "normalized_pred_score", "normalized_anomaly_map", "pred_label", "pred_mask")


def make_winclip_data():
    """Provide two 256-pixel RGB samples with both image and pixel classes for small-CLIP CPU evaluation."""
    class TinyData(LightningDataModule):
        """Supply an in-memory category and tuple batches without external datasets."""

        def __init__(self):
            super().__init__()
            self.category = "bottle"
            generator = torch.Generator().manual_seed(17)
            self.samples = []
            for index in range(2):
                mask = torch.zeros(256, 256, dtype=torch.int64)
                if index:
                    mask[64:192, 64:192] = 1
                self.samples.append((torch.rand(3, 256, 256, generator=generator), {
                    "label": index, "mask": mask, "image_path": f"{index}.png", "sample_id": f"fixture/{index}.png",
                }))

        def train_dataloader(self):
            return torch.utils.data.DataLoader(self.samples, batch_size=2, num_workers=0)

        def val_dataloader(self):
            return self.train_dataloader()

        def test_dataloader(self):
            return self.train_dataloader()

        def predict_dataloader(self):
            return self.train_dataloader()

    return TinyData()


@pytest.fixture(scope="module")
def winclip_assets(tmp_path_factory):
    """Map a seeded width-64 two-layer CLIP checkpoint to a temporary teacher mapping."""
    root = str(tmp_path_factory.mktemp("winclip_assets"))
    helpers = runpy.run_path(os.path.join(PROJECT_ROOT, "tests", "18_winclip", "test_winclip.py"))
    factory = helpers["make_small_clip"]
    weights = os.path.join(root, "clip.bin")
    torch.save(factory().state_dict(), weights)
    mapping = os.path.join(root, "paths.toml")
    with open(mapping, "w", encoding="utf-8", newline="\n") as stream:
        stream.write('[teachers]\nwinclip_clip = "' + weights.replace("\\", "/") + '"\n')
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pure_module, "CLIP", factory)
        patch.setattr(io_utils, "PATHS_FILE", mapping)
        yield {"root": root, "weights": weights, "mapping": mapping}


def test_winclip_native_signature_teacher_and_contract(winclip_assets):
    """Keep eight original defaults, exact local small-CLIP injection and zero/few-shot contracts."""
    from visionlab.anomaly_detection.common.lightning_visualizer import LitVisualizer

    expected = dict(class_name=None, k_shot=0, scales=(2, 3), few_shot_source=None,
                    pre_processor=True, post_processor=True, evaluator=True, visualizer=True)
    parameters = inspect.signature(WinClipLitModel).parameters
    assert list(parameters) == list(expected)
    assert {name: value.default for name, value in parameters.items()} == expected
    assert all(value.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD for value in parameters.values())
    model = WinClipLitModel(visualizer=False)
    source = torch.load(winclip_assets["weights"], weights_only=True, map_location="cpu")
    assert model.model.clip.state_dict().keys() == source.keys()
    assert all(torch.equal(value, source[key]) for key, value in model.model.clip.state_dict().items())
    assert model.teacher_weights_path == winclip_assets["weights"]
    assert model.model._text_embeddings.numel() == 0 and not model.is_setup
    assert model.trainer_arguments == {} and model.configure_optimizers() is None
    assert model.learning_type == LearningType.ZERO_SHOT and model.configure_callbacks() == []
    assert WinClipLitModel(k_shot=1, visualizer=False).learning_type == LearningType.FEW_SHOT
    assert "training_step" not in WinClipLitModel.__dict__ and "on_fit_end" not in WinClipLitModel.__dict__
    assert "state_dict" not in WinClipLitModel.__dict__ and "load_state_dict" not in WinClipLitModel.__dict__
    torch.testing.assert_close(model.pre_processor.mean.flatten(), torch.tensor(CLIP_MEAN, dtype=torch.float64))
    torch.testing.assert_close(model.pre_processor.std.flatten(), torch.tensor(CLIP_STD, dtype=torch.float64))
    assert model.pre_processor.center_crop is None
    for name in ("pre_processor", "post_processor", "evaluator", "visualizer"):
        with pytest.raises(TypeError):
            WinClipLitModel(**{name: None})
    disabled = WinClipLitModel(pre_processor=False, post_processor=False, evaluator=False, visualizer=False)
    assert disabled.pre_processor is disabled.post_processor is disabled.evaluator is disabled.visualizer is None
    visible = WinClipLitModel()
    assert len(visible.configure_callbacks()) == 1
    assert isinstance(visible.configure_callbacks()[0], LitVisualizer)
    with pytest.raises(NotImplementedError, match="few_shot_source"):
        WinClipLitModel(few_shot_source="images", visualizer=False)


def test_winclip_native_missing_or_incompatible_teacher(winclip_assets, tmp_path, monkeypatch):
    """Reject missing mappings, keys and files and reject a checkpoint with incompatible strict keys."""
    mapping = os.path.join(tmp_path, "paths.toml")
    monkeypatch.setattr(io_utils, "PATHS_FILE", mapping)
    with pytest.raises(FileNotFoundError):
        WinClipLitModel(visualizer=False)
    for content in ('[teachers]\n', '[teachers]\nwinclip_clip = "missing.bin"\n'):
        with open(mapping, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        with pytest.raises(FileNotFoundError):
            WinClipLitModel(visualizer=False)
    weights = os.path.join(tmp_path, "bad.bin")
    torch.save({"wrong_key": torch.ones(1)}, weights)
    with open(mapping, "w", encoding="utf-8", newline="\n") as stream:
        stream.write('[teachers]\nwinclip_clip = "' + weights.replace("\\", "/") + '"\n')
    with pytest.raises(RuntimeError, match="state_dict"):
        WinClipLitModel(visualizer=False)


@pytest.mark.parametrize("explicit,category,expected", (("candle", "bottle", "candle"),
                                                        (None, "bottle", "bottle"), (None, None, "object")))
def test_winclip_native_setup_priority_and_once(winclip_assets, monkeypatch, explicit, category, expected):
    """Resolve class names in original order and collect text on CPU once before device placement."""
    model = WinClipLitModel(class_name=explicit, visualizer=False)
    datamodule = SimpleNamespace() if category is None else SimpleNamespace(category=category)
    model._trainer = SimpleNamespace(datamodule=datamodule)
    calls = []
    setup = model.model.setup

    def capture(class_name, reference_images):
        calls.append((class_name, next(model.model.parameters()).device.type))
        return setup(class_name, reference_images)

    monkeypatch.setattr(model.model, "setup", capture)
    model.setup("validate")
    model.setup("test")
    assert calls == [(expected, "cpu")] and model.class_name == expected and model.is_setup
    assert model.model.text_embeddings.shape == (2, 32) and model.model.text_embeddings.device.type == "cpu"
    model.model.to(dtype=torch.float64)
    assert model.model.text_embeddings.dtype == torch.float64
    model._trainer = None


def test_winclip_native_tuple_reference_collection(winclip_assets):
    """Collect only the requested small synthetic references and normalize tuple-batch images once."""
    model = WinClipLitModel(k_shot=1, visualizer=False)
    data = make_winclip_data()
    images, _ = next(iter(data.train_dataloader()))
    collected = model.collect_reference_images(data.train_dataloader())
    assert collected.shape == (1, 3, 256, 256)
    torch.testing.assert_close(collected, model.pre_processor(images[:1]))


def test_winclip_native_recalibration_and_readonly_test_predict(winclip_assets, monkeypatch):
    """Recalibrate each standalone validate from its own pending batches and never update during test or predict."""
    model = WinClipLitModel(evaluator=False, visualizer=False)
    model._trainer = SimpleNamespace(state=SimpleNamespace(fn=TrainerFn.VALIDATING), sanity_checking=False)
    batch = (torch.zeros(2, 3, 4, 4), {"label": torch.tensor([0, 1]),
                                      "mask": torch.arange(32).reshape(2, 4, 4).remainder(2)})
    offset = [0.0]

    def outputs(batch):
        return {"pred_score": torch.tensor([0.2, 0.8]) + offset[0], "gt_label": batch[1]["label"],
                "anomaly_map": torch.arange(32).reshape(2, 4, 4).float() / 32 + offset[0],
                "gt_mask": batch[1]["mask"]}

    monkeypatch.setattr(model, "_outputs", outputs)
    for value in (0.0, 2.0):
        offset[0] = value
        model.on_validation_epoch_start()
        model.validation_step(batch, 0)
        model.on_validation_epoch_end()
        assert model.post_processor.image_min.item() == pytest.approx(value + 0.2)
        assert model.post_processor.image_max.item() == pytest.approx(value + 0.8)
        assert model.post_processor.pixel_min.item() == pytest.approx(value)
        assert not model.post_processor._pending
    state = model.post_processor.state_dict()
    for stage in (TrainerFn.TESTING, TrainerFn.PREDICTING):
        model.trainer.state.fn = stage
        assert not model._calibrating_validation()
        model.test_step(batch, 0) if stage == TrainerFn.TESTING else model.predict_step(batch, 0)
        assert all(torch.equal(value, state[name]) for name, value in model.post_processor.state_dict().items())
    model.trainer.state.fn = TrainerFn.FITTING
    assert model._calibrating_validation()
    model.trainer.sanity_checking = True
    assert not model._calibrating_validation()
    model._trainer = None


@pytest.fixture(scope="module")
def calibrated_winclip(winclip_assets, tmp_path_factory):
    """Validate width-64 two-layer CLIP on two CPU samples and save one explicit checkpoint for restore tests."""
    root = str(tmp_path_factory.mktemp("winclip_calibrated"))
    data = make_winclip_data()
    model = WinClipLitModel(visualizer=False)
    trainer = Trainer(default_root_dir=root, accelerator="cpu", devices=1, logger=False,
                      enable_progress_bar=False, enable_model_summary=False)
    trainer.validate(model, datamodule=data)
    assert model.post_processor.is_calibrated and "val" in model.last_metrics
    assert trainer.optimizers == []
    assert trainer.checkpoint_callback.best_model_path == trainer.checkpoint_callback.last_model_path == ""
    assert not any(name.endswith(".ckpt") for _, _, files in os.walk(root) for name in files)
    export = os.path.join(root, "export", "model.pt")
    assert os.path.isfile(export)
    checkpoint = os.path.join(root, "explicit.ckpt")
    trainer.save_checkpoint(checkpoint)
    model.eval()
    return {"root": root, "model": model, "trainer": trainer, "data": data,
            "export": export, "checkpoint": checkpoint}


def test_winclip_native_checkpoint_and_evaluation(calibrated_winclip, monkeypatch, tmp_path):
    """Restore CLIP, dynamic text and calibration state; validate and test must preserve raw outputs."""
    fitted = calibrated_winclip
    saved = torch.load(fitted["checkpoint"], weights_only=False, map_location="cpu")
    assert "model.clip.token_embedding.weight" in saved["state_dict"]
    assert saved["state_dict"]["model._text_embeddings"].shape == (2, 32)
    restored = WinClipLitModel.load_from_checkpoint(fitted["checkpoint"], visualizer=False).eval()
    assert all(torch.equal(value, restored.post_processor.state_dict()[key])
               for key, value in fitted["model"].post_processor.state_dict().items())
    batch = next(iter(fitted["data"].test_dataloader()))
    with torch.no_grad():
        expected = fitted["model"].predict_step(batch, 0)
        actual = restored.predict_step(batch, 0)
    for name in OUTPUT_NAMES:
        torch.testing.assert_close(expected[name], actual[name], atol=1e-6, rtol=1e-5)
    observed, tested = [], []
    validation_step = restored.validation_step
    test_step = restored.test_step

    def capture_validation(batch, batch_idx):
        outputs = validation_step(batch, batch_idx)
        observed.append(outputs)
        return outputs

    def capture_test(batch, batch_idx):
        outputs = test_step(batch, batch_idx)
        tested.append(outputs)
        return outputs

    monkeypatch.setattr(restored, "validation_step", capture_validation)
    monkeypatch.setattr(restored, "test_step", capture_test)
    trainer = Trainer(default_root_dir=str(tmp_path), accelerator="cpu", devices=1, logger=False,
                      enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
    trainer.validate(restored, datamodule=fitted["data"])
    assert len(observed) == 1
    for name in ("pred_score", "anomaly_map"):
        torch.testing.assert_close(observed[0][name], expected[name], atol=1e-6, rtol=1e-5)
    before = restored.post_processor.state_dict()
    trainer.test(restored, datamodule=fitted["data"])
    assert len(tested) == 1
    for name in OUTPUT_NAMES:
        torch.testing.assert_close(tested[0][name], expected[name], atol=1e-6, rtol=1e-5)
    trainer.predict(restored, datamodule=fitted["data"])
    assert all(torch.equal(value, restored.post_processor.state_dict()[key]) for key, value in before.items())


def test_winclip_native_export_and_predictor(calibrated_winclip, monkeypatch, tmp_path):
    """Strictly restore a complete small-CLIP export without reading teacher assets or tokenizing again."""
    from visionlab.inference.predictor import Predictor

    fitted = calibrated_winclip
    payload = io_utils.load_export(fitted["export"])
    assert payload["model_args"] == {"scales": [2, 3]}
    assert json.loads(json.dumps(payload["model_args"])) == payload["model_args"]
    assert "clip.token_embedding.weight" in payload["model_state"]
    assert payload["model_state"]["_text_embeddings"].shape == (2, 32)
    monkeypatch.setattr(io_utils, "PATHS_FILE", os.path.join(tmp_path, "absent.toml"))

    def forbidden(*args, **kwargs):
        raise AssertionError("Export restoration must not access pretrained assets or text setup")

    monkeypatch.setattr(io_utils, "load_pretrained_teacher", forbidden)
    monkeypatch.setattr(pure_module, "tokenize", forbidden)
    predictor = Predictor.from_export(fitted["export"])
    assert all(torch.equal(value, payload["model_state"][key]) for key, value in predictor.model.state_dict().items())
    batch = next(iter(fitted["data"].test_dataloader()))
    actual = predictor.predict(batch[0])
    with torch.no_grad():
        expected = fitted["model"].predict_step(batch, 0)
    assert actual["pred_score"].shape == (2,) and actual["anomaly_map"].shape == (2, 256, 256)
    for name in OUTPUT_NAMES:
        torch.testing.assert_close(actual[name], expected[name], atol=1e-6, rtol=1e-5)


def test_winclip_export_isolation_after_shared_dependency_warmup(calibrated_winclip):
    """Block new optional imports after legacy Predictor dependencies warm up; cold torchvision blocking is unsupported."""
    fitted = calibrated_winclip
    batch = next(iter(fitted["data"].test_dataloader()))
    with torch.no_grad():
        expected = fitted["model"].predict_step(batch, 0)
    expected_path = os.path.join(fitted["root"], "expected.pt")
    torch.save({"images": batch[0], "outputs": {key: expected[key] for key in OUTPUT_NAMES}}, expected_path)
    script = '''
import builtins
import importlib.abc
import os
import runpy
import sys
import torch
sys.path.insert(0, sys.argv[1])
# Existing Predictor registrations and torchmetrics load torchvision before the guard, as explicitly allowed.
from visionlab.inference.predictor import Predictor
from visionlab.utils import io as io_utils
from visionlab.anomaly_detection.models.winclip import model as pure_module
helpers = runpy.run_path(sys.argv[2])
pure_module.CLIP = helpers["make_small_clip"]
forbidden = {"anomalib", "lightning", "pytorch_lightning", "open_clip", "ftfy", "timm", "torchvision", "kornia", "regex"}
assert not {"anomalib", "lightning", "pytorch_lightning", "open_clip"} & set(sys.modules)
class Block(importlib.abc.MetaPathFinder):
    """Reject new optional imports after documented shared dependency warmup."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in forbidden:
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, Block())
io_utils.block_network()
io_utils.PATHS_FILE = os.path.join(sys.argv[5], "absent.toml")
original_open = builtins.open
def guarded_open(path, *args, **kwargs):
    """Deny project mappings and backbone files while allowing the export and expected tensors."""
    if isinstance(path, (str, bytes, os.PathLike)):
        name = os.fsdecode(path).replace("\\\\", "/").lower()
        if name.endswith("paths.toml") or name.startswith("d:/backbones"):
            raise AssertionError("Unexpected external asset access")
    return original_open(path, *args, **kwargs)
builtins.open = guarded_open
predictor = Predictor.from_export(sys.argv[3])
expected = torch.load(sys.argv[4], weights_only=True, map_location="cpu")
actual = predictor.predict(expected["images"])
assert actual["pred_score"].shape == (2,) and actual["anomaly_map"].shape == (2, 256, 256)
for name, value in expected["outputs"].items():
    torch.testing.assert_close(actual[name], value, atol=1e-6, rtol=1e-5)
assert not {"anomalib", "lightning", "pytorch_lightning", "open_clip"} & set(sys.modules)
'''
    result = subprocess.run([sys.executable, "-c", script, os.path.join(PROJECT_ROOT, "src"),
                             os.path.join(PROJECT_ROOT, "tests", "18_winclip", "test_winclip.py"),
                             fitted["export"], expected_path, fitted["root"]], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
