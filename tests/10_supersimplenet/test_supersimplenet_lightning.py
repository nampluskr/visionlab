# tests/10_supersimplenet/test_supersimplenet_lightning.py: pytest native synthesis training, optimizer groups and independent exports

from collections import Counter
import csv
import inspect
import math
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


def make_ssn_data():
    """Supply three normal 256-pixel training images and labeled evaluation with a batch-one tail."""
    from lightning.pytorch import LightningDataModule

    class TinyData(LightningDataModule):
        """Provide fixed uneven batches for feature synthesis and independent evaluation."""

        def __init__(self):
            super().__init__()
            self.train_data = [(torch.rand(3, 256, 256) * 0.02 + level,
                                {"label": 0, "mask": torch.zeros(256, 256, dtype=torch.int64)})
                               for level in (0.40, 0.44, 0.98)]
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
def ssn_assets(tmp_path_factory):
    """Reuse one temporary local full ResNet18 mapping without executing older model tests."""
    root = str(tmp_path_factory.mktemp("ssn_assets"))
    helpers = runpy.run_path(os.path.join(PROJECT_ROOT, "tests", "03_padim", "test_padim_lightning.py"))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(io_utils, "PATHS_FILE", helpers["make_padim_assets"](root))
        yield root


@pytest.mark.parametrize("supervised", (False, True))
def test_ssn_native_signature_injection_and_supervised_branch(ssn_assets, tmp_path, monkeypatch, supervised):
    """Keep eight defaults, mandatory nested teacher injection and the original supervised flags."""
    from visionlab.anomaly_detection.models.supersimplenet.lightning_model import SupersimplenetLitModel

    expected = {"perlin_threshold": 0.2, "backbone": "wide_resnet50_2", "layers": ["layer2", "layer3"],
                "supervised": False, "pre_processor": True, "post_processor": True, "evaluator": True, "visualizer": True}
    signature = inspect.signature(SupersimplenetLitModel.__init__)
    assert list(signature.parameters) == ["self"] + list(expected)
    assert {name: signature.parameters[name].default for name in expected} == expected
    model = SupersimplenetLitModel(backbone="resnet18", supervised=supervised, visualizer=False)
    source = torch.load(os.path.join(ssn_assets, "resnet18.pth"), map_location="cpu", weights_only=True)
    for name, value in model.model.feature_extractor.feature_extractor.feature_extractor.state_dict().items():
        assert torch.equal(value, source[name]), name
    assert model.teacher_weights_path == os.path.join(ssn_assets, "resnet18.pth")
    assert model.pre_processor.center_crop is None and model.learning_type.value == "one_class"
    assert model.model.segdec.stop_grad is (not supervised)
    assert model.trainer_arguments == {"gradient_clip_val": 1 if supervised else 0, "num_sanity_val_steps": 0}
    assert "max_epochs" not in model.trainer_arguments
    monkeypatch.setattr(io_utils, "PATHS_FILE", os.path.join(tmp_path, "missing.toml"))
    with pytest.raises(FileNotFoundError):
        SupersimplenetLitModel(backbone="resnet18", visualizer=False)
    mapping = os.path.join(tmp_path, "paths.toml")
    with open(mapping, "w", encoding="utf-8", newline="\n") as stream:
        stream.write('[backbones]\nresnet18 = "missing.pth"\n')
    monkeypatch.setattr(io_utils, "PATHS_FILE", mapping)
    with pytest.raises(FileNotFoundError):
        SupersimplenetLitModel(backbone="resnet18", visualizer=False)


@pytest.mark.parametrize("max_epochs", (10, 20))
def test_ssn_native_adamw_groups_and_epoch_scheduler(ssn_assets, max_epochs):
    """Keep ordered parameter groups, inherited adapter decay and epoch-derived Counter milestones."""
    from visionlab.anomaly_detection.models.supersimplenet.lightning_model import SupersimplenetLitModel

    model = SupersimplenetLitModel(backbone="resnet18", visualizer=False)
    model._trainer = SimpleNamespace(max_epochs=max_epochs)
    optimizers, schedulers = model.configure_optimizers()
    assert len(optimizers) == len(schedulers) == 1
    optimizer, scheduler = optimizers[0], schedulers[0]
    assert isinstance(optimizer, torch.optim.AdamW) and len(optimizer.param_groups) == 2
    first, second = optimizer.param_groups
    assert first["lr"] == 1e-4 and first["weight_decay"] == 0.01
    assert second["lr"] == 2e-4 and second["weight_decay"] == 1e-5
    assert {id(value) for value in first["params"]} == {id(value) for value in model.model.adaptor.parameters()}
    assert {id(value) for value in second["params"]} == {id(value) for value in model.model.segdec.parameters()}
    excluded = {id(value) for value in model.model.feature_extractor.parameters()}
    assert excluded.isdisjoint(id(value) for group in optimizer.param_groups for value in group["params"])
    assert isinstance(scheduler, torch.optim.lr_scheduler.MultiStepLR)
    assert scheduler.milestones == Counter([int(max_epochs * 0.8), int(max_epochs * 0.9)]) and scheduler.gamma == 0.4


def test_ssn_native_training_uses_expanded_targets_and_log_contract(ssn_assets, monkeypatch):
    """Normalize once and pass the returned synthetic masks and labels to the keyword loss contract."""
    from visionlab.anomaly_detection.models.supersimplenet.lightning_model import SupersimplenetLitModel

    model = SupersimplenetLitModel(backbone="resnet18", visualizer=False)
    image = torch.rand(2, 3, 256, 256)
    target = {"mask": torch.zeros(2, 256, 256, dtype=torch.int64), "label": torch.zeros(2, dtype=torch.int64)}
    maps, scores = torch.rand(4, 1, 64, 64, requires_grad=True), torch.rand(4, requires_grad=True)
    masks, labels = torch.ones_like(maps), torch.ones(4)
    seen, logged = [], []

    def forward(images, masks, labels):
        seen.append(images.detach().clone())
        assert masks is target["mask"] and labels is target["label"]
        return maps, scores, torch.ones_like(maps), torch.ones(4)

    def loss(pred_map, pred_score, target_mask, target_label):
        assert pred_map is maps and pred_score is scores
        assert torch.equal(target_mask, masks) and torch.equal(target_label, labels)
        return pred_map.mean() + pred_score.mean()

    monkeypatch.setattr(model.model, "forward", forward)
    monkeypatch.setattr(model.loss, "forward", loss)
    monkeypatch.setattr(model, "log", lambda name, value, **options: logged.append((name, value, options)))
    output = model.training_step((image, target), 0)
    assert set(output) == {"loss"} and output["loss"].requires_grad
    expected = (image - torch.tensor(IMAGENET_MEAN).reshape(1, 3, 1, 1)) / torch.tensor(IMAGENET_STD).reshape(1, 3, 1, 1)
    torch.testing.assert_close(seen[0], expected)
    assert logged == [("train_loss", output["loss"].item(),
                       {"on_epoch": True, "prog_bar": True, "logger": True, "batch_size": 1})]


@pytest.fixture(scope="module")
def fitted_ssn(tmp_path_factory, ssn_assets):
    """Fit two real synthesis epochs and capture each step loss, target call and calibration update."""
    from lightning.pytorch import Trainer
    from lightning.pytorch.loggers import CSVLogger
    from visionlab.anomaly_detection.models.supersimplenet.lightning_model import SupersimplenetLitModel

    root = str(tmp_path_factory.mktemp("fitted_ssn"))
    data = make_ssn_data()
    model = SupersimplenetLitModel(backbone="resnet18", visualizer=False)
    before = {name: value.clone() for name, value in model.model.feature_extractor.state_dict().items()}
    heads = {branch: {name: parameter.detach().clone() for name, parameter in getattr(model.model, branch).named_parameters()}
             for branch in ("adaptor", "segdec")}
    counts, steps, forwarded = {"update": 0, "finalize": 0}, [], []
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

    def capture_forward(module, args, kwargs):
        if module.training:
            masks, labels = kwargs["masks"], kwargs["labels"]
            assert masks.dtype == torch.int64 and not masks.any() and not labels.any()
            forwarded.append((masks.shape, labels.shape))

    logger = CSVLogger(save_dir=root, name="metrics")
    trainer = Trainer(default_root_dir=root, accelerator="cpu", devices=1, max_epochs=2, logger=logger,
                      log_every_n_steps=1, num_sanity_val_steps=0, enable_checkpointing=False,
                      enable_progress_bar=False, enable_model_summary=False)
    handle = model.model.register_forward_pre_hook(capture_forward, with_kwargs=True)
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(model.post_processor, "update", capture_update)
            patch.setattr(model.post_processor, "finalize", capture_finalize)
            patch.setattr(model, "training_step", capture_step)
            trainer.fit(model, datamodule=data)
    finally:
        handle.remove()
    path = os.path.join(root, "model.ckpt")
    trainer.save_checkpoint(path)
    return root, model, data, path, trainer, before, heads, counts, steps, forwarded


def test_ssn_native_learning_frozen_features_and_step_mean(fitted_ssn):
    """Train both heads while preserving the backbone and average uneven synthetic steps equally."""
    root, model, data, _, trainer, before, heads, counts, steps, forwarded = fitted_ssn
    assert counts == {"update": 2 * len(data.val_dataloader()), "finalize": 2} and model.post_processor.is_calibrated
    for name, value in model.model.feature_extractor.state_dict().items():
        assert torch.equal(value, before[name]), name
    for branch, state in heads.items():
        assert any(not torch.equal(parameter.detach(), state[name])
                   for name, parameter in getattr(model.model, branch).named_parameters())
    assert len(forwarded) == len(steps) == 4
    assert [(tuple(masks), tuple(labels)) for masks, labels in forwarded] == [
        ((2, 256, 256), (2,)), ((1, 256, 256), (1,)),
        ((2, 256, 256), (2,)), ((1, 256, 256), (1,))]
    assert trainer.lr_scheduler_configs[0].interval == "epoch"
    with open(os.path.join(root, "metrics", "version_0", "metrics.csv"), newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("train_loss_epoch")]
    assert len(rows) == 2
    distinguished = False
    for row in rows:
        values = [(size, loss) for epoch, size, loss in steps if epoch == int(float(row["epoch"]))]
        assert [size for size, _ in values] == [2, 1]
        simple = sum(loss for _, loss in values) / len(values)
        weighted = sum(size * loss for size, loss in values) / sum(size for size, _ in values)
        assert float(row["train_loss_epoch"]) == pytest.approx(simple, rel=1e-5, abs=1e-7)
        distinguished |= not math.isclose(simple, weighted, rel_tol=1e-5, abs_tol=1e-7)
    assert distinguished, "The fixture must distinguish step and sample weighted means"


def test_ssn_native_checkpoint_frozen_evaluation_and_scalar_visualizer(fitted_ssn, tmp_path, monkeypatch):
    """Restore the checkpoint, preserve calibration and visualize a scalar score without changing outputs."""
    from lightning.pytorch import Trainer
    from visionlab.anomaly_detection.common.visualizer import ImageVisualizer
    from visionlab.anomaly_detection.models.supersimplenet.lightning_model import SupersimplenetLitModel

    _, model, data, path, _, _, _, _, _, _ = fitted_ssn
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["optimizer_states"] and saved["lr_schedulers"] and saved["post_processor"] is not None
    restored = SupersimplenetLitModel(backbone="resnet18", visualizer=False)
    restored.load_state_dict(saved["state_dict"], strict=True)
    restored.on_load_checkpoint(saved)
    model.eval()
    restored.eval()
    batch = list(data.val_dataloader())[-1]
    with torch.no_grad():
        expected, actual = model.predict_step(batch, 0), restored.predict_step(batch, 0)
    assert actual["pred_score"].ndim == 0 and actual["anomaly_map"].shape == (1, 1, 256, 256)
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
    torch.testing.assert_close(predictions[-1]["pred_score"], expected["pred_score"], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(predictions[-1]["anomaly_map"], expected["anomaly_map"], rtol=1e-5, atol=1e-6)
    for name, value in restored.post_processor.state_dict().items():
        torch.testing.assert_close(value, before[name], equal_nan=True)
    restored.visualizer = ImageVisualizer()
    restored.configure_callbacks()[0].on_predict_batch_end(
        SimpleNamespace(is_global_zero=True, default_root_dir=str(tmp_path)), restored, actual, batch, 0,
    )
    assert os.path.isfile(os.path.join(tmp_path, "images", "fixture", "2.png"))
    assert actual["pred_score"].ndim == 0


def consume_ssn_export(root):
    """Restore all pure weights and compare both score shapes while denying external asset reads."""
    from visionlab.inference.predictor import Predictor

    io_utils.block_network()
    path = os.path.join(root, "export", "model.pt")
    expected = torch.load(os.path.join(root, "expected.pt"), map_location="cpu", weights_only=True)
    exported = io_utils.load_export(path)
    assert exported["model_name"] == "supersimplenet"
    args = exported["model_args"]
    assert set(args) == {"perlin_threshold", "backbone", "layers", "stop_grad"}
    assert type(args["perlin_threshold"]) in (int, float) and type(args["backbone"]) is str
    assert type(args["layers"]) is list and all(type(layer) is str for layer in args["layers"])
    assert type(args["stop_grad"]) is bool and args["stop_grad"]
    assert all(any(name.startswith(prefix) for name in exported["model_state"])
               for prefix in ("feature_extractor.", "adaptor.", "segdec."))

    def audit_open(event, args):
        if event == "open":
            opened, mode, flags = args
            assert isinstance(opened, (str, bytes)) and os.path.abspath(opened) == path, opened
            assert not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)

    sys.addaudithook(audit_open)
    predictor = Predictor.from_export(path)
    for name, value in predictor.model.state_dict().items():
        assert torch.equal(value, exported["model_state"][name]), name
    for batch in expected["batches"]:
        for name, value in predictor.predict(batch["images"]).items():
            torch.testing.assert_close(value, batch["outputs"][name], rtol=1e-5, atol=1e-6)


def test_ssn_native_export_in_isolated_process(fitted_ssn):
    """Match native inference without reference frameworks, Lightning, paths.toml or backbone file access."""
    root, model, data, _, _, _, _, _, _, _ = fitted_ssn
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
import timm
class Block(importlib.abc.MetaPathFinder):
    """Reject reference and training frameworks before importing pure SuperSimpleNet inference."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("anomalib", "lightning", "pytorch_lightning"):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, Block())
sys.path.insert(0, sys.argv[2])
runpy.run_path(sys.argv[1])["consume_ssn_export"](sys.argv[3])
assert not {"anomalib", "lightning", "pytorch_lightning"} & set(sys.modules)
'''
    result = subprocess.run([sys.executable, "-c", script, os.path.abspath(__file__), os.path.join(PROJECT_ROOT, "src"), root],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
