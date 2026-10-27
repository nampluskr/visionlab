# tests/11_draem/test_draem_lightning.py: pytest native DRAEM training, calibration, checkpoints and pure exports

from collections import Counter
import csv
import inspect
import math
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from visionlab.anomaly_detection.common.perlin import PerlinAnomalyGenerator
from visionlab.anomaly_detection.models.draem.model import DraemModel
from visionlab.anomaly_detection.models.draem.loss import DraemLoss
from visionlab.utils import io as io_utils

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_draem_data():
    """Provide three normal training samples and labeled evaluation with a batch-one tail."""
    from lightning.pytorch import LightningDataModule

    class TinyData(LightningDataModule):
        """Provide fixed 64-pixel synthetic batches without external assets."""

        def __init__(self):
            super().__init__()
            self.train_data = [
                (torch.rand(3, 64, 64) * 0.02 + level,
                 {"label": 0, "mask": torch.zeros(64, 64, dtype=torch.int64)})
                for level in (0.10, 0.15, 0.90)
            ]
            self.val_data = []
            for index in range(3):
                mask = torch.zeros(64, 64, dtype=torch.int64)
                if index % 2:
                    mask[16:32, 16:32] = 1
                self.val_data.append((torch.rand(3, 64, 64), {
                    "label": index % 2, "mask": mask, "image_path": f"{index}.png",
                    "sample_id": f"fixture/{index}.png",
                }))

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
def native_draem():
    """Reuse an untrained full-width Lightning model for nonmutating contract checks."""
    from visionlab.anomaly_detection.models.draem.lightning_model import DraemLitModel

    return DraemLitModel(visualizer=False)


def test_draem_native_signature_and_default_components(native_draem):
    """Keep eight constructor defaults and identity preprocessing with image and pixel metrics."""
    from visionlab.anomaly_detection.models.draem.lightning_model import DraemLitModel

    expected = {"enable_sspcab": False, "sspcab_lambda": 0.1, "anomaly_source_path": None,
                "beta": (0.1, 1.0), "pre_processor": True, "post_processor": True,
                "evaluator": True, "visualizer": True}
    signature = inspect.signature(DraemLitModel)
    assert list(signature.parameters) == list(expected)
    assert {name: parameter.default for name, parameter in signature.parameters.items()} == expected
    assert native_draem.teacher_weights_path is None and native_draem.learning_type.value == "one_class"
    assert native_draem.augmenter.blend_factor == (0.1, 1.0)
    image = torch.rand(2, 3, 64, 64)
    assert native_draem.pre_processor.mean is native_draem.pre_processor.std is None
    assert native_draem.pre_processor(image) is image
    assert [entry[0] for entry in native_draem.evaluator.test_metrics] == [
        "image_AUROC", "pixel_AUROC", "image_F1Score", "pixel_F1Score",
    ]
    assert not hasattr(native_draem, "sspcab_loss") and not hasattr(native_draem, "sspcab_activations")
    encoder = native_draem.model.reconstructive_subnetwork.encoder
    assert not encoder.mp4._forward_hooks and not encoder.block5._forward_hooks


@pytest.mark.parametrize("beta", ([0.1, 1.0], (0.2, 0.8), 0.4))
def test_draem_native_beta_conversion_and_offline_constructor(monkeypatch, beta):
    """Convert only lists and construct without any file or network access."""
    from visionlab.anomaly_detection.models.draem.lightning_model import DraemLitModel

    def forbidden(*args, **kwargs):
        raise AssertionError("DRAEM construction must not read files or connect to a network")

    with monkeypatch.context() as patch:
        for target in ("builtins.open", "io.open", "os.open", "torch.load", "socket.socket.connect",
                       "socket.socket.connect_ex", "socket.create_connection"):
            patch.setattr(target, forbidden)
        model = DraemLitModel(beta=beta, visualizer=False)
    assert model.teacher_weights_path is None
    assert model.augmenter.blend_factor == (tuple(beta) if isinstance(beta, list) else beta)
    assert type(model.augmenter.blend_factor) is (tuple if isinstance(beta, (tuple, list)) else float)


def test_draem_native_initialization_order():
    """Match direct augmenter, pure model and loss construction including the following CPU RNG draw."""
    from visionlab.anomaly_detection.models.draem.lightning_model import DraemLitModel

    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(42)
        augmenter = PerlinAnomalyGenerator(blend_factor=(0.1, 1.0))
        pure = DraemModel()
        criterion = DraemLoss()
        following = torch.rand(4)
        torch.manual_seed(42)
        native = DraemLitModel(visualizer=False)
        assert torch.equal(torch.rand(4), following)
    assert isinstance(native.loss, type(criterion))
    assert native.augmenter.augmenters.transforms[4].threshold == augmenter.augmenters.transforms[4].threshold
    for name, value in pure.state_dict().items():
        assert torch.equal(value, native.model.state_dict()[name]), name


def test_draem_native_adam_and_fixed_milestones(native_draem):
    """Optimize only model parameters in one Adam group with unchanged reference milestones."""
    optimizers, schedulers = native_draem.configure_optimizers()
    assert len(optimizers) == len(schedulers) == 1
    optimizer, scheduler = optimizers[0], schedulers[0]
    assert isinstance(optimizer, torch.optim.Adam) and len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == 1e-4
    assert {id(value) for value in optimizer.param_groups[0]["params"]} == {
        id(value) for value in native_draem.model.parameters()
    }
    assert isinstance(scheduler, torch.optim.lr_scheduler.MultiStepLR)
    assert scheduler.milestones == Counter([400, 600]) and scheduler.gamma == 0.1
    assert native_draem.trainer_arguments == {"gradient_clip_val": 0, "num_sanity_val_steps": 0}
    assert "max_epochs" not in native_draem.trainer_arguments


@pytest.mark.parametrize("sspcab", (False, True))
def test_draem_native_training_inputs_log_and_sspcab(monkeypatch, sspcab):
    """Send augmented images to the model and original images to loss, adding the hooked MSE only when enabled."""
    from visionlab.anomaly_detection.models.draem.lightning_model import DraemLitModel

    model = DraemLitModel(enable_sspcab=sspcab, sspcab_lambda=0.3, visualizer=False)
    image = torch.rand(2, 3, 64, 64)
    augmented = image * 0.5
    mask = torch.ones(2, 1, 64, 64)
    seen, logged = {}, []
    original_forward, original_loss = model.model.forward, model.loss.forward

    def forward(value):
        """Capture model input while executing the real subnetwork hooks."""
        assert value is augmented
        seen["model"] = True
        return original_forward(value)

    def loss(input_image, reconstruction, anomaly_mask, prediction):
        """Capture the unaugmented reconstruction target and retain the unmodified base loss."""
        assert input_image is image and anomaly_mask is mask
        result = original_loss(input_image, reconstruction, anomaly_mask, prediction)
        seen["loss"] = result.detach().clone()
        return result

    monkeypatch.setattr(model.augmenter, "forward", lambda value: (augmented, mask))
    monkeypatch.setattr(model.model, "forward", forward)
    monkeypatch.setattr(model.loss, "forward", loss)
    monkeypatch.setattr(model, "log", lambda name, value, **options: logged.append((name, value, options)))
    with torch.no_grad():
        result = model.training_step((image, {"label": torch.zeros(2), "mask": torch.zeros(2, 64, 64)}), 0)
    expected = seen["loss"]
    assert seen["model"]
    if sspcab:
        assert set(model.sspcab_activations) == {"input", "output"}
        assert model.sspcab_activations["input"].shape == model.sspcab_activations["output"].shape == (2, 1024, 4, 4)
        expected += 0.3 * torch.nn.MSELoss()(model.sspcab_activations["input"], model.sspcab_activations["output"])
    else:
        assert not hasattr(model, "sspcab_loss")
    torch.testing.assert_close(result["loss"], expected)
    assert logged == [("train_loss", result["loss"].item(),
                       {"on_epoch": True, "prog_bar": True, "logger": True, "batch_size": 1})]


@pytest.fixture(scope="module")
def fitted_draem(tmp_path_factory):
    """Fit two full-width epochs and retain uneven step losses, calibration counts and a checkpoint."""
    from lightning.pytorch import Trainer
    from lightning.pytorch.loggers import CSVLogger
    from visionlab.anomaly_detection.models.draem.lightning_model import DraemLitModel

    root = str(tmp_path_factory.mktemp("fitted_draem"))
    data = make_draem_data()
    model = DraemLitModel(visualizer=False)
    parameter = next(model.model.parameters())
    before = parameter.detach().clone()
    counts, steps = {"update": 0, "finalize": 0}, []
    update, finalize, step = model.post_processor.update, model.post_processor.finalize, model.training_step

    def capture_update(outputs):
        """Count only fit validation updates."""
        counts["update"] += 1
        return update(outputs)

    def capture_finalize():
        """Count only fit validation finalization."""
        counts["finalize"] += 1
        return finalize()

    def capture_step(batch, batch_idx):
        """Record losses and verify the normal synthetic training targets."""
        assert not batch[1]["mask"].any() and not batch[1]["label"].any()
        result = step(batch, batch_idx)
        steps.append((model.current_epoch, batch[0].shape[0], result["loss"].item()))
        return result

    trainer = Trainer(default_root_dir=root, accelerator="cpu", devices=1, max_epochs=2,
                      logger=CSVLogger(save_dir=root, name="metrics"), log_every_n_steps=1,
                      num_sanity_val_steps=0, enable_checkpointing=False,
                      enable_progress_bar=False, enable_model_summary=False)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(model.post_processor, "update", capture_update)
        patch.setattr(model.post_processor, "finalize", capture_finalize)
        patch.setattr(model, "training_step", capture_step)
        trainer.fit(model, datamodule=data)
    checkpoint = os.path.join(root, "model.ckpt")
    trainer.save_checkpoint(checkpoint)
    return root, model, data, checkpoint, trainer, before, counts, steps


def test_draem_native_learning_and_epoch_step_mean(fitted_draem):
    """Update real weights and average the uneven training steps with equal weight."""
    root, model, data, _, trainer, before, counts, steps = fitted_draem
    assert not torch.equal(next(model.model.parameters()).detach(), before)
    assert counts == {"update": 2 * len(data.val_dataloader()), "finalize": 2}
    assert model.post_processor.is_calibrated and trainer.lr_scheduler_configs[0].interval == "epoch"
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
    assert distinguished, "Synthetic steps must distinguish step and sample weighted means"


def test_draem_native_checkpoint_and_frozen_evaluation(fitted_draem, tmp_path, monkeypatch):
    """Restore checkpoints, freeze standalone calibration and save a batch-one three-dimensional map."""
    from lightning.pytorch import Trainer
    from visionlab.anomaly_detection.common.visualizer import ImageVisualizer
    from visionlab.anomaly_detection.models.draem.lightning_model import DraemLitModel

    _, model, data, checkpoint, _, _, _, _ = fitted_draem
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["optimizer_states"] and saved["lr_schedulers"] and saved["post_processor"] is not None
    restored = DraemLitModel.load_from_checkpoint(checkpoint, visualizer=False)
    model.eval()
    restored.eval()
    batch = list(data.val_dataloader())[-1]
    with torch.no_grad():
        expected, actual = model.predict_step(batch, 0), restored.predict_step(batch, 0)
    assert actual["pred_score"].shape == (1,) and actual["anomaly_map"].shape == (1, 64, 64)
    for name in ("pred_score", "anomaly_map", "normalized_pred_score", "normalized_anomaly_map", "pred_label", "pred_mask"):
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-5, atol=1e-6)
    before = restored.post_processor.state_dict()

    def forbidden(*args, **kwargs):
        raise AssertionError("Standalone evaluation must not change calibration")

    monkeypatch.setattr(restored.post_processor, "update", forbidden)
    monkeypatch.setattr(restored.post_processor, "finalize", forbidden)
    validation_step, test_step = restored.validation_step, restored.test_step
    observed = {}

    def capture_validation(batch, batch_idx):
        """Retain the final raw validation outputs for checkpoint equivalence."""
        outputs = validation_step(batch, batch_idx)
        observed["val"] = outputs
        return outputs

    def capture_test(batch, batch_idx):
        """Retain the final test outputs for checkpoint equivalence."""
        outputs = test_step(batch, batch_idx)
        observed["test"] = outputs
        return outputs

    monkeypatch.setattr(restored, "validation_step", capture_validation)
    monkeypatch.setattr(restored, "test_step", capture_test)
    trainer = Trainer(default_root_dir=str(tmp_path), accelerator="cpu", devices=1, logger=False,
                      enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
    validated = trainer.validate(restored, datamodule=data, ckpt_path=checkpoint)
    tested = trainer.test(restored, datamodule=data, ckpt_path=checkpoint)
    assert "val_image_AUROC" in validated[0] and "image_AUROC" in tested[0]
    predictions = trainer.predict(restored, datamodule=data)
    for name in ("pred_score", "anomaly_map"):
        torch.testing.assert_close(predictions[-1][name], expected[name], rtol=1e-5, atol=1e-6)
        for stage in ("val", "test"):
            torch.testing.assert_close(observed[stage][name], expected[name], rtol=1e-5, atol=1e-6)
    for name, value in restored.post_processor.state_dict().items():
        torch.testing.assert_close(value, before[name], equal_nan=True)
    restored.visualizer = ImageVisualizer()
    restored.configure_callbacks()[0].on_predict_batch_end(
        SimpleNamespace(is_global_zero=True, default_root_dir=str(tmp_path)), restored, actual, batch, 0,
    )
    assert os.path.isfile(os.path.join(tmp_path, "images", "fixture", "2.png"))
    assert actual["anomaly_map"].shape == (1, 64, 64)


def consume_draem_export(root):
    """Load pure exports while allowing only the export file to be opened after imports."""
    from visionlab.inference.predictor import Predictor

    io_utils.block_network()
    path = os.path.join(root, "export", "model.pt")
    expected = torch.load(os.path.join(root, "expected.pt"), map_location="cpu", weights_only=True)
    exported = io_utils.load_export(path)
    assert exported["model_name"] == "draem" and exported["model_args"] == {"sspcab": False}
    assert type(exported["model_args"]["sspcab"]) is bool
    assert exported["pre_processor"] == {"mean": None, "std": None, "center_crop": None}
    assert not any(name.startswith(("augmenter.", "loss.")) for name in exported["model_state"])

    def audit_open(event, args):
        """Deny reads of paths.toml, backbone files and all other external files."""
        if event == "open":
            opened, mode, flags = args
            assert isinstance(opened, (str, bytes)) and os.path.abspath(opened) == path, opened
            assert not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)

    sys.addaudithook(audit_open)
    predictor = Predictor.from_export(path)
    assert predictor.pre_processor.mean is predictor.pre_processor.std is None
    for batch in expected["batches"]:
        assert predictor.pre_processor(batch["images"]) is batch["images"]
        actual = predictor.predict(batch["images"])
        assert actual["anomaly_map"].shape == (batch["images"].shape[0], 64, 64)
        for name, value in actual.items():
            torch.testing.assert_close(value, batch["outputs"][name], rtol=1e-5, atol=1e-6)


def test_draem_native_export_in_isolated_process(fitted_draem):
    """Match native inference without Lightning, anomalib, paths.toml or local pretrained files."""
    root, model, data, _, _, _, _, _ = fitted_draem
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
class Block(importlib.abc.MetaPathFinder):
    """Reject reference and training frameworks before importing the pure export consumer."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("anomalib", "lightning", "pytorch_lightning"):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, Block())
runpy.run_path(sys.argv[1])["consume_draem_export"](sys.argv[2])
assert not {"anomalib", "lightning", "pytorch_lightning"} & set(sys.modules)
'''
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.path.join(PROJECT_ROOT, "src")
    result = subprocess.run([sys.executable, "-c", script, os.path.abspath(__file__), root],
                            env=environment, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
