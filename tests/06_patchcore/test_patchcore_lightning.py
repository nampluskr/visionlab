# tests/06_patchcore/test_patchcore_lightning.py: pytest cropped PatchCore fitting, checkpoint and export contracts

import inspect
import json
import os
import runpy
import subprocess
import sys

import pytest
import torch
from torchvision.transforms.v2.functional import center_crop

from visionlab.anomaly_detection.common.pre_processor import IMAGENET_MEAN, IMAGENET_STD, PreProcessor
from visionlab.utils import io as io_utils

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def padim_helpers():
    """Reuse small local weight and DataModule builders without running earlier model tests."""
    return runpy.run_path(os.path.join(PROJECT_ROOT, "tests", "03_padim", "test_padim_lightning.py"))


@pytest.fixture(scope="module")
def patchcore_assets(tmp_path_factory):
    """Map one temporary ResNet18 file for small native PatchCore constructors."""
    root = str(tmp_path_factory.mktemp("patchcore_assets"))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(io_utils, "PATHS_FILE", padim_helpers()["make_padim_assets"](root))
        yield root


def test_patchcore_pre_processor_crop_normalization_and_mask_alignment():
    """Keep uncropped behavior identical and crop images and arbitrary spatial tensors consistently."""
    images = torch.rand(2, 3, 256, 256)
    mean, std = torch.tensor(IMAGENET_MEAN).reshape(1, 3, 1, 1), torch.tensor(IMAGENET_STD).reshape(1, 3, 1, 1)
    assert PreProcessor()(images) is images and PreProcessor().crop(images) is images
    assert torch.equal(PreProcessor(mean=IMAGENET_MEAN, std=IMAGENET_STD)(images), (images - mean) / std)
    processor = PreProcessor(mean=IMAGENET_MEAN, std=IMAGENET_STD, center_crop=(224, 224))
    expected = (center_crop(images, [224, 224]) - mean) / std
    assert torch.equal(processor(images), expected) and expected.shape == (2, 3, 224, 224)
    mask = torch.arange(256 * 256).reshape(256, 256).expand(2, -1, -1)
    assert torch.equal(processor.crop(mask), mask[..., 16:240, 16:240])
    odd = torch.rand(1, 3, 255, 257)
    assert torch.equal(PreProcessor(center_crop=(224, 224))(odd), center_crop(odd, [224, 224]))
    assert processor.center_crop == (224, 224)
    for size in ((0, 224), (-1, 224), (224,), (224.0, 224), (True, 224)):
        with pytest.raises(ValueError):
            PreProcessor(center_crop=size)
    with pytest.raises(ValueError, match="exceed"):
        processor(torch.rand(1, 3, 200, 256))
    with pytest.raises(ValueError, match="spatial"):
        processor.crop(torch.ones(10))


def test_patchcore_signature_and_local_pretrained_injection(patchcore_assets, tmp_path, monkeypatch):
    """Preserve nine constructor defaults and inject exactly the requested local backbone."""
    from visionlab.anomaly_detection.models.patchcore.lightning_model import PatchcoreLitModel

    expected = {"backbone": "wide_resnet50_2", "layers": ("layer2", "layer3"), "pre_trained": True,
                "coreset_sampling_ratio": 0.1, "num_neighbors": 9, "pre_processor": True,
                "post_processor": True, "evaluator": True, "visualizer": True}
    signature = inspect.signature(PatchcoreLitModel.__init__)
    assert list(signature.parameters) == ["self"] + list(expected)
    assert {name: signature.parameters[name].default for name in expected} == expected
    model = PatchcoreLitModel(backbone="resnet18", visualizer=False)
    source = torch.load(os.path.join(patchcore_assets, "resnet18.pth"), map_location="cpu", weights_only=True)
    for name, value in model.model.feature_extractor.feature_extractor.state_dict().items():
        assert torch.equal(value, source[name]), name
    assert model.teacher_weights_path == os.path.join(patchcore_assets, "resnet18.pth")
    assert model.pre_processor.center_crop == (224, 224)
    assert model.trainer_arguments == {"gradient_clip_val": 0, "max_epochs": 1, "num_sanity_val_steps": 0}
    assert model.learning_type.value == "one_class" and model.configure_optimizers() is None
    assert [item[0] for item in model.evaluator.test_metrics] == [
        "image_AUROC", "pixel_AUROC", "image_F1Score", "pixel_F1Score"]
    monkeypatch.setattr(io_utils, "PATHS_FILE", os.path.join(tmp_path, "missing.toml"))
    with pytest.raises(FileNotFoundError):
        PatchcoreLitModel(backbone="resnet18", visualizer=False)
    mapping = os.path.join(tmp_path, "paths.toml")
    with open(mapping, "w", encoding="utf-8", newline="\n") as stream:
        stream.write('[backbones]\nresnet18 = "missing.pth"\n')
    monkeypatch.setattr(io_utils, "PATHS_FILE", mapping)
    with pytest.raises(FileNotFoundError):
        PatchcoreLitModel(backbone="resnet18", visualizer=False)
    untrained = PatchcoreLitModel(backbone="resnet18", pre_trained=False, visualizer=False)
    assert untrained.teacher_weights_path is None and untrained.embeddings == [] and not untrained._is_fitted.item()


@pytest.mark.parametrize("first_hook", ("on_validation_start", "on_train_epoch_end"))
def test_patchcore_dummy_loss_preprocessed_features_and_one_time_fit(first_hook, monkeypatch):
    """Retain original embeddings and subsample once across both memory-bank hooks."""
    from visionlab.anomaly_detection.models.patchcore.lightning_model import PatchcoreLitModel

    model = PatchcoreLitModel(backbone="resnet18", pre_trained=False, evaluator=False, visualizer=False)
    calls, seen = [], []

    def features(image):
        seen.append(image.clone())
        return torch.randn(image.shape[0] * 4, 8)

    def subsample(embeddings, ratio):
        calls.append((embeddings.clone(), ratio))

    monkeypatch.setattr(model.model, "forward", features)
    monkeypatch.setattr(model.model, "subsample_embedding", subsample)
    collected = []
    for size in (1, 2):
        image = torch.rand(size, 3, 256, 256)
        loss = model.training_step((image, {}), 0)
        assert loss.ndim == 0 and loss.item() == 0 and loss.requires_grad
        torch.testing.assert_close(seen[-1], model.pre_processor(image))
        assert seen[-1].shape[-2:] == (224, 224)
        collected.append(model.embeddings[-1].clone())
    getattr(model, first_hook)()
    model.on_validation_start()
    model.on_validation_start()
    model.on_train_epoch_end()
    assert len(calls) == 1 and model._is_fitted.item() and len(model.embeddings) == 2
    assert torch.equal(calls[0][0], torch.vstack(collected)) and calls[0][1] == 0.1
    for actual, expected in zip(model.embeddings, collected, strict=True):
        assert torch.equal(actual, expected)


@pytest.fixture(scope="module")
def fitted_patchcore(tmp_path_factory, patchcore_assets):
    """Fit a real small coreset for one epoch using 256 inputs and the default 224 crop."""
    from lightning.pytorch import Trainer
    from visionlab.anomaly_detection.models.patchcore.lightning_model import PatchcoreLitModel

    root = str(tmp_path_factory.mktemp("fitted_patchcore"))
    data = padim_helpers()["make_padim_data"]()
    for dataset in (data.train_data, data.val_data):
        for index, (image, target) in enumerate(dataset):
            image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(256, 256), mode="bilinear",
                                                   align_corners=False)[0]
            target["mask"] = torch.nn.functional.interpolate(target["mask"][None, None].float(),
                                                            size=(256, 256), mode="nearest")[0, 0].long()
            dataset[index] = (image, target)
    model = PatchcoreLitModel(backbone="resnet18", coreset_sampling_ratio=0.005, num_neighbors=3, visualizer=False)
    trainer = Trainer(default_root_dir=root, accelerator="cpu", devices=1, max_epochs=1, logger=False,
                      num_sanity_val_steps=0, enable_checkpointing=False,
                      enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(model, datamodule=data)
    path = os.path.join(root, "model.ckpt")
    trainer.save_checkpoint(path)
    return root, model, data, path, trainer


def test_patchcore_fit_checkpoint_resume_and_aligned_evaluation(fitted_patchcore, tmp_path, monkeypatch):
    """Restore the bank and calibration without refitting in evaluation or checkpoint-based resume."""
    from lightning.pytorch import Trainer
    from visionlab.anomaly_detection.models.patchcore.lightning_model import PatchcoreLitModel

    _, model, data, path, trainer = fitted_patchcore
    count = sum(embedding.shape[0] for embedding in model.embeddings)
    assert model._is_fitted.item() and model.model.memory_bank.shape == (int(count * 0.005), 384)
    assert model.configure_optimizers() is None and not trainer.lr_scheduler_configs
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert {"model.memory_bank", "_is_fitted"} <= saved["state_dict"].keys()
    assert saved["post_processor"] is not None
    restored = PatchcoreLitModel(backbone="resnet18", pre_trained=False, coreset_sampling_ratio=0.005,
                                 num_neighbors=3, visualizer=False)
    assert not restored.model.memory_bank.numel()
    restored.load_state_dict(saved["state_dict"], strict=True)
    restored.on_load_checkpoint(saved)
    model.eval()
    restored.eval()
    batch = next(iter(data.val_dataloader()))
    with torch.no_grad():
        expected, actual = model.predict_step(batch, 0), restored.predict_step(batch, 0)
    for name in ("pred_score", "anomaly_map", "normalized_pred_score", "normalized_anomaly_map", "pred_mask"):
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-5, atol=1e-6)
    assert actual["gt_mask"].shape == (2, 224, 224) and actual["anomaly_map"].shape == (2, 1, 224, 224)
    assert torch.equal(actual["gt_mask"], batch[1]["mask"][..., 16:240, 16:240])
    assert torch.equal(actual["image"], batch[0][..., 16:240, 16:240])
    before = restored.post_processor.state_dict()

    def forbidden(embeddings, ratio):
        raise AssertionError("Checkpoint memory bank must not be fitted again")

    monkeypatch.setattr(restored.model, "subsample_embedding", forbidden)
    evaluator = Trainer(default_root_dir=str(tmp_path), accelerator="cpu", devices=1, logger=False,
                        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
    evaluator.validate(restored, datamodule=data, ckpt_path=path)
    evaluator.test(restored, datamodule=data)
    predictions = evaluator.predict(restored, datamodule=data)
    assert all(output["anomaly_map"].shape[-2:] == output["gt_mask"].shape[-2:] == (224, 224)
               for output in predictions)
    assert "pixel_AUROC" in restored.last_metrics["test"] and "pixel_F1Score" in restored.last_metrics["test"]
    for name, value in restored.post_processor.state_dict().items():
        torch.testing.assert_close(value, before[name], equal_nan=True)
    restored.train()
    resumed = Trainer(default_root_dir=os.path.join(tmp_path, "resume"), accelerator="cpu", devices=1,
                      max_epochs=2, logger=False, num_sanity_val_steps=0, enable_checkpointing=False,
                      enable_progress_bar=False, enable_model_summary=False)
    resumed.fit(restored, datamodule=data, ckpt_path=path)
    assert resumed.global_step > saved["global_step"]
    for name in ("model.memory_bank", "_is_fitted"):
        assert torch.equal(restored.state_dict()[name], saved["state_dict"][name])


def consume_patchcore_export(root):
    """Restore cropped predictions with frameworks and all external asset reads denied."""
    from visionlab.inference.predictor import Predictor

    io_utils.block_network()
    path = os.path.join(root, "export", "model.pt")
    expected = torch.load(os.path.join(root, "expected.pt"), map_location="cpu", weights_only=True)
    exported = io_utils.load_export(path)
    assert exported["model_name"] == "patchcore" and exported["model_args"]["pre_trained"] is False
    assert exported["pre_processor"]["center_crop"] == [224, 224]
    assert "memory_bank" in exported["model_state"]
    assert any(name.startswith("feature_extractor.") for name in exported["model_state"])

    def audit_open(event, args):
        if event == "open":
            opened, mode, flags = args
            assert isinstance(opened, (str, bytes)) and os.path.abspath(opened) == path, opened
            assert not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)

    sys.addaudithook(audit_open)
    predictor = Predictor.from_export(path)
    assert predictor.pre_processor.center_crop == (224, 224)
    outputs = predictor.predict(expected["images"])
    assert outputs["anomaly_map"].shape[-2:] == (224, 224)
    for name, value in predictor.model.state_dict().items():
        assert torch.equal(value, exported["model_state"][name]), name
    for name, value in outputs.items():
        torch.testing.assert_close(value, expected["outputs"][name], rtol=1e-5, atol=1e-6)


def test_patchcore_export_predictor_isolation(fitted_patchcore):
    """Match native cropped outputs independently of training frameworks and pretrained assets."""
    root, model, data, _, _ = fitted_patchcore
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
    """Reject training frameworks before loading the pure cropped consumer."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("anomalib", "lightning", "pytorch_lightning"):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, Block())
sys.path.insert(0, sys.argv[2])
runpy.run_path(sys.argv[1])["consume_patchcore_export"](sys.argv[3])
assert not {"anomalib", "lightning", "pytorch_lightning"} & set(sys.modules)
'''
    result = subprocess.run([sys.executable, "-c", script, os.path.abspath(__file__), os.path.join(PROJECT_ROOT, "src"), root],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_patchcore_legacy_export_crop_default(tmp_path):
    """Read an earlier STFPM export's processing state with the new crop disabled."""
    exported = {"format_version": 1, "model_name": "stfpm",
                "model_args": {"backbone": "resnet18", "layers": ["layer1", "layer2", "layer3"]},
                "model_state": {}, "pre_processor": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
                "post_processor": None}
    legacy = os.path.join(tmp_path, "legacy.pt")
    torch.save(exported, legacy)
    state = io_utils.load_export(legacy)["pre_processor"]
    assert state["center_crop"] is None and state["mean"] == list(IMAGENET_MEAN) and state["std"] == list(IMAGENET_STD)


def test_patchcore_stats_exact_rows_with_duplicates_and_channel_means(tmp_path, capsys):
    """Record bounded-memory multiset overlap independently of row ordering and bank shape."""
    namespace = runpy.run_path(os.path.join(PROJECT_ROOT, "evaluation", "anomalib", "compare_patchcore_stats.py"))
    reference_path, native_path = os.path.join(tmp_path, "reference.ckpt"), os.path.join(tmp_path, "native.ckpt")
    reference = torch.tensor([[1., 2.], [1., 2.], [3., 4.]])
    native = torch.tensor([[3., 4.], [1., 2.]])
    torch.save({"state_dict": {"model.memory_bank": reference}}, reference_path)
    torch.save({"state_dict": {"model.memory_bank": native}}, native_path)
    output = os.path.join(tmp_path, "stats.json")
    report = namespace["main"](["--reference", reference_path, "--native", native_path, "--out", output])
    assert report["informational_only"] and "passed" not in report and not report["shape_match"]
    assert report["exact_rows"]["matched_rows"] == 2
    assert report["exact_rows"]["reference_ratio"] == 2 / 3 and report["exact_rows"]["native_ratio"] == 1
    assert report["channel_mean"]["max_absolute_difference"] == pytest.approx(1 / 3)
    assert report["channel_mean"]["max_relative_difference"] == pytest.approx(0.2)
    assert "Informational comparison only" in capsys.readouterr().out
    with open(output, encoding="utf-8") as stream:
        assert json.load(stream) == report
    torch.save({"state_dict": {"model.memory_bank": reference.flip(0)}}, native_path)
    report = namespace["compare_stats"](reference_path, native_path)
    assert report["shape_match"] and report["exact_rows"]["reference_ratio"] == 1
    assert report["channel_mean"]["max_absolute_difference"] == 0
    torch.save({"state_dict": {"model.memory_bank": torch.ones(2, 3)}}, native_path)
    report = namespace["compare_stats"](reference_path, native_path)
    assert not report["exact_rows"]["comparable"] and report["channel_mean"]["not_comparable_reason"] == "channel_mismatch"
