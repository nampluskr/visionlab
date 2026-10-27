# tests/13_cflow/test_cflow_lightning.py: pytest native CFLOW manual optimization, local weights and isolated exports

import csv
import inspect
import math
import os
import subprocess
import sys

import pytest
import timm
import torch

from visionlab.anomaly_detection.common.pre_processor import IMAGENET_MEAN, IMAGENET_STD
from visionlab.utils import io as io_utils

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LAYERS = ["layer1", "layer2", "layer3"]
OUTPUT_NAMES = ("pred_score", "anomaly_map", "normalized_pred_score", "normalized_anomaly_map", "pred_label", "pred_mask")


def make_cflow_model(pre_trained=True):
    """Create a small native CFLOW with real resnet18 features and two conditional flow blocks."""
    from visionlab.anomaly_detection.models.cflow.lightning_model import CflowLitModel

    return CflowLitModel(backbone="resnet18", layers=LAYERS, pre_trained=pre_trained,
                         condition_vector=16, coupling_blocks=2, fiber_batch_size=16, visualizer=False)


def make_cflow_data():
    """Provide uneven small training batches and labeled evaluation batches."""
    from lightning.pytorch import LightningDataModule

    class TinyData(LightningDataModule):
        """Supply deterministic synthetic 64-pixel images without external datasets."""

        def __init__(self):
            super().__init__()
            generator = torch.Generator().manual_seed(17)
            self.train_data = [(torch.rand(3, 64, 64, generator=generator) * 0.02 + level, {"label": 0})
                               for level in (0.40, 0.42, 0.98)]
            self.val_data = []
            for index in range(3):
                mask = torch.zeros(64, 64, dtype=torch.int64)
                if index % 2:
                    mask[16:32, 16:32] = 1
                self.val_data.append((torch.rand(3, 64, 64, generator=generator),
                                      {"label": index % 2, "mask": mask, "image_path": f"{index}.png",
                                       "sample_id": f"fixture/{index}.png"}))

        def train_dataloader(self):
            return torch.utils.data.DataLoader(self.train_data, batch_size=2, shuffle=False, num_workers=0)

        def val_dataloader(self):
            return torch.utils.data.DataLoader(self.val_data, batch_size=2, shuffle=False, num_workers=0)

        def test_dataloader(self):
            return self.val_dataloader()

    return TinyData()


@pytest.fixture(scope="module")
def cflow_assets(tmp_path_factory):
    """Map a temporary full timm backbone state through paths.toml."""
    root = str(tmp_path_factory.mktemp("cflow_assets"))
    weights = os.path.join(root, "resnet18.pth")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        torch.save(timm.create_model("resnet18", pretrained=False).state_dict(), weights)
    mapping = os.path.join(root, "paths.toml")
    with open(mapping, "w", encoding="utf-8", newline="\n") as stream:
        stream.write('[backbones]\nresnet18 = "' + weights.replace("\\", "/") + '"\n')
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(io_utils, "PATHS_FILE", mapping)
        yield root


def test_cflow_native_signature_weights_and_optimizer(cflow_assets):
    """Preserve all fourteen defaults, inject the encoder and optimize only decoder parameters."""
    from visionlab.anomaly_detection.models.cflow.lightning_model import CflowLitModel

    expected = {"backbone": "wide_resnet50_2", "layers": ("layer2", "layer3", "layer4"), "pre_trained": True,
                "fiber_batch_size": 64, "decoder": "freia-cflow", "condition_vector": 128,
                "coupling_blocks": 8, "clamp_alpha": 1.9, "permute_soft": False, "lr": 0.0001,
                "pre_processor": True, "post_processor": True, "evaluator": True, "visualizer": True}
    signature = inspect.signature(CflowLitModel)
    assert list(signature.parameters) == list(expected)
    assert {name: parameter.default for name, parameter in signature.parameters.items()} == expected
    assert all(parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD for parameter in signature.parameters.values())
    model = make_cflow_model()
    source = torch.load(os.path.join(cflow_assets, "resnet18.pth"), map_location="cpu", weights_only=True)
    for name, value in model.model.encoder.feature_extractor.state_dict().items():
        assert torch.equal(value, source[name]), name
    assert model.teacher_weights_path == os.path.join(cflow_assets, "resnet18.pth")
    assert model.automatic_optimization is False
    assert model.pre_processor.center_crop is None
    torch.testing.assert_close(model.pre_processor.mean.flatten(), torch.tensor(IMAGENET_MEAN, dtype=torch.float64))
    torch.testing.assert_close(model.pre_processor.std.flatten(), torch.tensor(IMAGENET_STD, dtype=torch.float64))
    assert model.learning_type.value == "one_class"
    assert model.trainer_arguments == {"gradient_clip_val": 0, "num_sanity_val_steps": 0}
    assert [name for name, _, _, _ in model.evaluator.val_metrics] == ["image_AUROC", "pixel_AUROC"]
    assert [name for name, _, _, _ in model.evaluator.test_metrics] == [
        "image_AUROC", "pixel_AUROC", "image_F1Score", "pixel_F1Score",
    ]
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam) and len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == 1e-4
    parameters = optimizer.param_groups[0]["params"]
    assert [id(value) for value in parameters] == [id(value) for value in model.model.decoders.parameters()]
    assert not {id(value) for value in parameters} & {id(value) for value in model.model.encoder.parameters()}


def test_cflow_native_missing_weights_and_opt_out(tmp_path, monkeypatch):
    """Fail on missing local mapping or weights and bypass injection only when explicitly disabled."""
    from visionlab.anomaly_detection.models.cflow import lightning_model as module

    monkeypatch.setattr(io_utils, "PATHS_FILE", os.path.join(tmp_path, "missing.toml"))
    with pytest.raises(FileNotFoundError):
        make_cflow_model()
    mapping = os.path.join(tmp_path, "paths.toml")
    with open(mapping, "w", encoding="utf-8", newline="\n") as stream:
        stream.write('[backbones]\nresnet18 = "missing.pth"\n')
    monkeypatch.setattr(io_utils, "PATHS_FILE", mapping)
    with pytest.raises(FileNotFoundError):
        make_cflow_model()

    def forbidden(*args, **kwargs):
        """Reject injection when pretrained loading is disabled."""
        raise AssertionError("pre_trained=False must not load weights")

    monkeypatch.setattr(module, "load_teacher_weights", forbidden)
    assert make_cflow_model(pre_trained=False).teacher_weights_path is None


def test_cflow_native_components_and_callbacks():
    """Accept native component instances, reject invalid values and retain the standard visualizer callback."""
    from lightning.pytorch.trainer.states import TrainerFn
    from types import SimpleNamespace
    from visionlab.anomaly_detection.common.evaluator import Evaluator
    from visionlab.anomaly_detection.common.lightning_visualizer import LitVisualizer
    from visionlab.anomaly_detection.common.post_processor import PostProcessor
    from visionlab.anomaly_detection.common.pre_processor import PreProcessor
    from visionlab.anomaly_detection.common.visualizer import ImageVisualizer
    from visionlab.anomaly_detection.models.cflow.lightning_model import CflowLitModel

    components = (PreProcessor(), PostProcessor(), Evaluator.default(), ImageVisualizer())
    model = CflowLitModel(backbone="resnet18", layers=LAYERS, pre_trained=False, condition_vector=16,
                          coupling_blocks=2, fiber_batch_size=16, pre_processor=components[0],
                          post_processor=components[1], evaluator=components[2], visualizer=components[3])
    assert (model.pre_processor, model.post_processor, model.evaluator, model.visualizer) == components
    assert isinstance(model.configure_callbacks()[0], LitVisualizer)
    for component_type in (PreProcessor, PostProcessor, Evaluator, ImageVisualizer):
        assert model._resolve(False, component_type, lambda: None) is None
        with pytest.raises(TypeError):
            model._resolve(None, component_type, lambda: None)
    assert make_cflow_model(pre_trained=False).configure_callbacks() == []
    assert not model._fit_validation()
    model._trainer = SimpleNamespace(state=SimpleNamespace(fn=TrainerFn.FITTING), sanity_checking=False)
    assert model._fit_validation()
    model._trainer.sanity_checking = True
    assert not model._fit_validation()


def test_cflow_native_initialization_input_and_log(monkeypatch):
    """Keep pure initialization RNG order, normalize once and preserve the exact logging options."""
    from visionlab.anomaly_detection.models.cflow.model import CflowModel

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        pure = CflowModel(backbone="resnet18", layers=LAYERS, pre_trained=False, fiber_batch_size=16,
                          condition_vector=16, coupling_blocks=2)
        following = torch.rand(4)
        torch.manual_seed(42)
        native = make_cflow_model(pre_trained=False)
        assert torch.equal(torch.rand(4), following)
    for name, value in pure.state_dict().items():
        assert torch.equal(value, native.model.state_dict()[name]), name
    seen, logged = [], []
    encoder_forward = native.model.encoder.forward

    def capture_input(images):
        """Record the single normalized input and the original gradient context."""
        assert torch.is_grad_enabled()
        seen.append(images.detach().clone())
        return encoder_forward(images)

    monkeypatch.setattr(native.model.encoder, "forward", capture_input)
    monkeypatch.setattr(native, "optimizers", native.configure_optimizers)
    monkeypatch.setattr(native, "manual_backward", lambda loss: loss.backward())
    monkeypatch.setattr(native, "log", lambda name, value, **options: logged.append((name, value, options)))
    images = torch.rand(2, 3, 64, 64)
    result = native.training_step((images, {}), 0)
    assert torch.allclose(seen[0], native.pre_processor(images)) and len(seen) == 1
    assert result["loss"].shape == (1,) and result["loss"].dtype == torch.float64
    assert logged == [("train_loss", result["loss"].item(),
                       {"on_epoch": True, "prog_bar": True, "logger": True, "batch_size": 1})]
    native.model.fiber_batch_size = 100000
    with pytest.raises(ValueError, match="Make sure we have enough fibers, otherwise decrease N or batch-size!"):
        native.training_step((images, {}), 0)


def fit_cflow(root):
    """Fit exactly one epoch and record optimizer steps, calibration calls and batch losses."""
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.loggers import CSVLogger

    seed_everything(42, workers=True)
    data = make_cflow_data()
    model = make_cflow_model()
    before = {name: value.detach().clone() for name, value in model.model.encoder.named_parameters()}
    flow_before = model.model.decoders[0][0].subnet[0].weight.detach().clone()
    counts, steps = {"update": 0, "finalize": 0, "optimizer": 0}, []
    optimizer = model.configure_optimizers()
    optimizer_step = optimizer.step
    update, finalize, training_step = model.post_processor.update, model.post_processor.finalize, model.training_step

    def count_optimizer(*args, **kwargs):
        """Count each real Adam update inside the manual fiber loop."""
        counts["optimizer"] += 1
        return optimizer_step(*args, **kwargs)

    def capture_update(outputs):
        """Count fit validation calibration updates."""
        counts["update"] += 1
        return update(outputs)

    def capture_finalize():
        """Count fit validation finalization."""
        counts["finalize"] += 1
        return finalize()

    def capture_step(batch, batch_idx):
        """Record optimizer steps and aggregate loss separately for each uneven image batch."""
        start = counts["optimizer"]
        result = training_step(batch, batch_idx)
        steps.append((batch[0].shape[0], result["loss"].item(), counts["optimizer"] - start))
        return result

    trainer = Trainer(default_root_dir=root, accelerator="cpu", devices=1, max_epochs=1,
                      logger=CSVLogger(save_dir=root, name="metrics"), log_every_n_steps=1,
                      num_sanity_val_steps=0, gradient_clip_val=0, enable_checkpointing=False,
                      enable_progress_bar=False, enable_model_summary=False, deterministic=True)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(optimizer, "step", count_optimizer)
        patch.setattr(model, "configure_optimizers", lambda: optimizer)
        patch.setattr(model.post_processor, "update", capture_update)
        patch.setattr(model.post_processor, "finalize", capture_finalize)
        patch.setattr(model, "training_step", capture_step)
        trainer.fit(model, datamodule=data)
    path = os.path.join(root, "model.ckpt")
    trainer.save_checkpoint(path)
    return root, model, data, path, trainer, before, flow_before, counts, steps


@pytest.fixture(scope="module")
def fitted_cflow(tmp_path_factory, cflow_assets):
    """Retain one trained model and its complete one-epoch artifacts."""
    return fit_cflow(str(tmp_path_factory.mktemp("fitted_cflow")))


def test_cflow_native_learning_steps_and_epoch_mean(fitted_cflow):
    """Update only decoder parameters and average losses per image batch rather than sample."""
    root, model, data, _, trainer, before, flow_before, counts, steps = fitted_cflow
    assert not torch.equal(model.model.decoders[0][0].subnet[0].weight.detach(), flow_before)
    for name, value in model.model.encoder.named_parameters():
        assert not value.requires_grad and torch.equal(value, before[name]), name
    assert counts["update"] == len(data.val_dataloader()) and counts["finalize"] == 1
    assert model.post_processor.is_calibrated
    assert len(trainer.optimizers) == 1 and not trainer.lr_scheduler_configs
    assert [size for size, _, _ in steps] == [2, 1]
    for size, _, updates in steps:
        assert updates == sum(size * spatial * spatial // 16 for spatial in (16, 8, 4))
    assert counts["optimizer"] == sum(updates for _, _, updates in steps)
    with open(os.path.join(root, "metrics", "version_0", "metrics.csv"), newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("train_loss_epoch")]
    assert len(rows) == 1
    simple = sum(loss for _, loss, _ in steps) / len(steps)
    weighted = sum(size * loss for size, loss, _ in steps) / sum(size for size, _, _ in steps)
    assert float(rows[0]["train_loss_epoch"]) == pytest.approx(simple, rel=1e-5, abs=1e-7)
    assert not math.isclose(simple, weighted, rel_tol=1e-5, abs_tol=1e-7)


def test_cflow_native_seeded_training(fitted_cflow, tmp_path):
    """Repeat the one-epoch training with the same CPU seed and match the complete model state."""
    repeated = fit_cflow(str(tmp_path))[1]
    expected = fitted_cflow[1].model.state_dict()
    for name, value in repeated.model.state_dict().items():
        assert torch.equal(value, expected[name]), name


def test_cflow_native_checkpoint_and_frozen_evaluation(fitted_cflow, tmp_path, monkeypatch):
    """Restore checkpoint predictions and optimizer state while standalone evaluation leaves calibration frozen."""
    from lightning.pytorch import Trainer
    from visionlab.anomaly_detection.models.cflow.lightning_model import CflowLitModel

    _, model, data, path, _, _, _, _, _ = fitted_cflow
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["optimizer_states"][0]["state"] and not saved["lr_schedulers"]
    assert saved["post_processor"] is not None
    for prefix in ("model.encoder.", "model.decoders."):
        assert any(name.startswith(prefix) for name in saved["state_dict"])
    restored = CflowLitModel.load_from_checkpoint(path, pre_trained=False, visualizer=False)
    model.eval()
    restored.eval()
    batch = list(data.val_dataloader())[-1]
    with torch.no_grad():
        expected, actual = model.predict_step(batch, 0), restored.predict_step(batch, 0)
    assert actual["pred_score"].shape == (1,) and actual["anomaly_map"].shape == (1, 64, 64)
    for name in OUTPUT_NAMES:
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-5, atol=1e-6)
    before = restored.post_processor.state_dict()

    def forbidden(*args, **kwargs):
        """Reject calibration changes outside fit validation."""
        raise AssertionError("Standalone evaluation must not update calibration")

    monkeypatch.setattr(restored.post_processor, "update", forbidden)
    monkeypatch.setattr(restored.post_processor, "finalize", forbidden)
    observed = {}
    validation_step, test_step = restored.validation_step, restored.test_step

    def capture_validation(batch, batch_idx):
        """Retain raw validation output for the last batch."""
        observed["val"] = validation_step(batch, batch_idx)
        return observed["val"]

    def capture_test(batch, batch_idx):
        """Retain processed test output for the last batch."""
        observed["test"] = test_step(batch, batch_idx)
        return observed["test"]

    monkeypatch.setattr(restored, "validation_step", capture_validation)
    monkeypatch.setattr(restored, "test_step", capture_test)
    trainer = Trainer(default_root_dir=str(tmp_path), accelerator="cpu", devices=1, logger=False,
                      enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
    assert "val_image_AUROC" in trainer.validate(restored, datamodule=data, ckpt_path=path)[0]
    assert "image_AUROC" in trainer.test(restored, datamodule=data, ckpt_path=path)[0]
    for stage in ("val", "test"):
        for name in ("pred_score", "anomaly_map"):
            torch.testing.assert_close(observed[stage][name], expected[name], rtol=1e-5, atol=1e-6)
    for name in OUTPUT_NAMES:
        torch.testing.assert_close(observed["test"][name], expected[name], rtol=1e-5, atol=1e-6)
    for name, value in restored.post_processor.state_dict().items():
        torch.testing.assert_close(value, before[name], equal_nan=True)
    resumed = make_cflow_model(pre_trained=False)
    resume_trainer = Trainer(default_root_dir=os.path.join(tmp_path, "resume"), accelerator="cpu", devices=1,
                             max_epochs=1, logger=False, num_sanity_val_steps=0, gradient_clip_val=0,
                             enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
    resume_trainer.fit(resumed, datamodule=data, ckpt_path=path)
    optimizer_state = resume_trainer.optimizers[0].state_dict()
    assert optimizer_state["param_groups"] == saved["optimizer_states"][0]["param_groups"]
    assert optimizer_state["state"].keys() == saved["optimizer_states"][0]["state"].keys()
    for key, state in optimizer_state["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, saved["optimizer_states"][0]["state"][key][name])


def consume_cflow_export(root):
    """Restore only the export under file guards and match native evaluation without training frameworks."""
    from visionlab.inference.predictor import Predictor
    from visionlab.anomaly_detection.models.cflow.model import CflowModel

    io_utils.block_network()
    path = os.path.join(root, "export", "model.pt")
    expected = torch.load(os.path.join(root, "expected.pt"), map_location="cpu", weights_only=True)
    exported = io_utils.load_export(path)
    assert exported["model_name"] == "cflow"
    assert exported["model_args"] == {
        "backbone": "resnet18", "layers": LAYERS, "pre_trained": False, "fiber_batch_size": 16,
        "decoder": "freia-cflow", "condition_vector": 16, "coupling_blocks": 2,
        "clamp_alpha": 1.9, "permute_soft": False,
    }
    assert all(type(value) in (str, int, float, bool, list) for value in exported["model_args"].values())
    assert all(type(value) is str for value in exported["model_args"]["layers"])
    for prefix in ("encoder.", "decoders."):
        assert any(name.startswith(prefix) for name in exported["model_state"])

    def audit_open(event, args):
        """Deny all reads except the export, including paths.toml and local backbone assets."""
        if event == "open":
            opened, mode, flags = args
            assert isinstance(opened, (str, bytes)) and os.path.abspath(opened) == path, opened
            assert not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)

    sys.addaudithook(audit_open)
    predictor = Predictor.from_export(path)
    assert isinstance(predictor.model, CflowModel)
    for name, value in predictor.model.state_dict().items():
        assert torch.equal(value, exported["model_state"][name]), name
    for batch in expected["batches"]:
        actual = predictor.predict(batch["images"])
        assert actual["anomaly_map"].shape == (len(batch["images"]), 64, 64)
        assert actual["pred_score"].shape == (len(batch["images"]),)
        for name in OUTPUT_NAMES:
            torch.testing.assert_close(actual[name], batch["outputs"][name], rtol=1e-5, atol=1e-6)


def test_cflow_native_isolated_export(fitted_cflow):
    """Match native and pure predictions with Lightning, anomalib and external asset access blocked."""
    root, model, data, _, _, _, _, _, _ = fitted_cflow
    model.eval()
    expected = []
    for batch in data.val_dataloader():
        with torch.no_grad():
            outputs = model.predict_step(batch, 0)
        expected.append({"images": batch[0], "outputs": {name: outputs[name] for name in OUTPUT_NAMES}})
    torch.save({"batches": expected}, os.path.join(root, "expected.pt"))
    script = '''
import importlib.abc
import runpy
import sys
class Block(importlib.abc.MetaPathFinder):
    """Reject reference and training frameworks before consumer imports."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("anomalib", "lightning", "pytorch_lightning"):
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, Block())
runpy.run_path(sys.argv[1])["consume_cflow_export"](sys.argv[2])
assert "FrEIA" in sys.modules
assert not {"anomalib", "lightning", "pytorch_lightning"} & {name.split(".")[0] for name in sys.modules}
'''
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.path.join(PROJECT_ROOT, "src")
    result = subprocess.run([sys.executable, "-c", script, os.path.abspath(__file__), root],
                            env=environment, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_cflow_predictor_lazy_freia_dependency(fitted_cflow):
    """Import Predictor without FrEIA and require FrEIA only when restoring a CFLOW export."""
    path = os.path.join(fitted_cflow[0], "export", "model.pt")
    script = '''
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    """Reject FrEIA and training frameworks before importing Predictor."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("FrEIA", "anomalib", "lightning", "pytorch_lightning"):
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, Block())
from visionlab.inference.predictor import Predictor
from visionlab.utils.io import block_network
block_network()
assert "visionlab.anomaly_detection.models.fastflow.model" not in sys.modules
assert "visionlab.anomaly_detection.models.cflow.model" not in sys.modules
try:
    Predictor.from_export(sys.argv[1])
except ModuleNotFoundError as error:
    assert error.name.split(".")[0] == "FrEIA", error
else:
    raise AssertionError("CFLOW unexpectedly restored without FrEIA")
assert not {"FrEIA", "anomalib", "lightning", "pytorch_lightning"} & {name.split(".")[0] for name in sys.modules}
'''
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.path.join(PROJECT_ROOT, "src")
    result = subprocess.run([sys.executable, "-c", script, path], env=environment,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
