# tests/12_fastflow/test_fastflow_lightning.py: pytest native FastFlow learning, local weights and isolated exports

import csv
import inspect
import math
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import timm
import torch

from visionlab.anomaly_detection.common.pre_processor import IMAGENET_MEAN, IMAGENET_STD
from visionlab.utils import io as io_utils

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_NAMES = ("pred_score", "anomaly_map", "normalized_pred_score", "normalized_anomaly_map", "pred_label", "pred_mask")


def make_fastflow_data():
    """Supply uneven 256-pixel batches and both labels for validation and testing."""
    from lightning.pytorch import LightningDataModule

    class TinyData(LightningDataModule):
        """Provide three normal training images and three labeled evaluation images."""

        def __init__(self):
            super().__init__()
            generator = torch.Generator().manual_seed(17)
            self.train_data = [(torch.rand(3, 256, 256, generator=generator) * 0.02 + level, {"label": 0})
                               for level in (0.40, 0.42, 0.98)]
            self.val_data = []
            for index in range(3):
                mask = torch.zeros(256, 256, dtype=torch.int64)
                if index % 2:
                    mask[64:128, 64:128] = 1
                self.val_data.append((torch.rand(3, 256, 256, generator=generator),
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
def fastflow_assets(tmp_path_factory):
    """Map a full timm ResNet18 state dict to a temporary local backbone file."""
    root = str(tmp_path_factory.mktemp("fastflow_assets"))
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


def test_fastflow_native_signature_local_weights_and_input_size(fastflow_assets):
    """Keep nine defaults and inject the exact local backbone into the fixed-size pure model."""
    from visionlab.anomaly_detection.models.fastflow.lightning_model import FASTFLOW_INPUT_SIZE, FastflowLitModel

    expected = {"backbone": "resnet18", "pre_trained": True, "flow_steps": 8, "conv3x3_only": False,
                "hidden_ratio": 1.0, "pre_processor": True, "post_processor": True, "evaluator": True, "visualizer": True}
    signature = inspect.signature(FastflowLitModel.__init__)
    assert list(signature.parameters) == ["self"] + list(expected)
    assert {name: signature.parameters[name].default for name in expected} == expected
    assert all(parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD for parameter in signature.parameters.values())
    model = FastflowLitModel(flow_steps=2, visualizer=False)
    source = torch.load(os.path.join(fastflow_assets, "resnet18.pth"), map_location="cpu", weights_only=True)
    for name, value in model.model.feature_extractor.state_dict().items():
        assert torch.equal(value, source[name]), name
    assert model.teacher_weights_path == os.path.join(fastflow_assets, "resnet18.pth")
    assert model.model.input_size == FASTFLOW_INPUT_SIZE == (256, 256)
    assert model.model.anomaly_map_generator.input_size == (256, 256)
    for norm, channel, scale in zip(model.model.norms, model.model.feature_extractor.feature_info.channels(),
                                     model.model.feature_extractor.feature_info.reduction(), strict=True):
        assert tuple(norm.normalized_shape) == (channel, 256 // scale, 256 // scale)
    assert model.pre_processor.center_crop is None
    torch.testing.assert_close(model.pre_processor.mean.flatten(), torch.tensor(IMAGENET_MEAN, dtype=torch.float64))
    torch.testing.assert_close(model.pre_processor.std.flatten(), torch.tensor(IMAGENET_STD, dtype=torch.float64))
    assert model.learning_type.value == "one_class"
    assert model.trainer_arguments == {"gradient_clip_val": 0, "num_sanity_val_steps": 0}
    assert "max_epochs" not in model.trainer_arguments
    assert [name for name, _, _, _ in model.evaluator.val_metrics] == ["image_AUROC", "pixel_AUROC"]
    assert [name for name, _, _, _ in model.evaluator.test_metrics] == [
        "image_AUROC", "pixel_AUROC", "image_F1Score", "pixel_F1Score",
    ]


def test_fastflow_native_missing_weights_and_opt_out(tmp_path, monkeypatch):
    """Reject missing mappings and files while pre_trained=False performs no local loading."""
    from visionlab.anomaly_detection.models.fastflow import lightning_model as module

    monkeypatch.setattr(io_utils, "PATHS_FILE", os.path.join(tmp_path, "missing.toml"))
    with pytest.raises(FileNotFoundError):
        module.FastflowLitModel(flow_steps=2, visualizer=False)
    mapping = os.path.join(tmp_path, "paths.toml")
    with open(mapping, "w", encoding="utf-8", newline="\n") as stream:
        stream.write('[backbones]\nresnet18 = "missing.pth"\n')
    monkeypatch.setattr(io_utils, "PATHS_FILE", mapping)
    with pytest.raises(FileNotFoundError):
        module.FastflowLitModel(flow_steps=2, visualizer=False)

    def forbidden(*args, **kwargs):
        """Reject any loader invocation when pretrained injection is disabled."""
        raise AssertionError("pre_trained=False must not load weights")

    with monkeypatch.context() as patch:
        patch.setattr(module, "load_teacher_weights", forbidden)
        model = module.FastflowLitModel(pre_trained=False, flow_steps=2, visualizer=False)
    assert model.teacher_weights_path is None
    # Avoid allocating the unverified large transformer when testing a missing local mapping.
    monkeypatch.setattr(module, "FastflowModel",
                        lambda input_size, backbone, pre_trained, flow_steps, conv3x3_only, hidden_ratio: torch.nn.Identity())
    with pytest.raises(FileNotFoundError, match="cait_m48_448"):
        module.FastflowLitModel(backbone="cait_m48_448", flow_steps=2, visualizer=False)


def test_fastflow_native_initialization_and_training_log(monkeypatch):
    """Keep pure-model initialization order, normalize once and log each step with equal epoch weight."""
    from visionlab.anomaly_detection.models.fastflow.lightning_model import FastflowLitModel
    from visionlab.anomaly_detection.models.fastflow.model import FastflowModel

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        pure = FastflowModel(input_size=(256, 256), backbone="resnet18", pre_trained=False, flow_steps=2)
        following = torch.rand(4)
        torch.manual_seed(42)
        native = FastflowLitModel(pre_trained=False, flow_steps=2, visualizer=False)
        assert torch.equal(torch.rand(4), following)
    for name, value in pure.state_dict().items():
        assert torch.equal(value, native.model.state_dict()[name]), name
    seen, logged = [], []
    forward = native.model.forward

    def capture_input(image):
        """Retain the normalized input while computing the actual flow loss."""
        seen.append(image.detach().clone())
        return forward(image)

    monkeypatch.setattr(native.model, "forward", capture_input)
    monkeypatch.setattr(native, "log", lambda name, value, **options: logged.append((name, value, options)))
    image = torch.rand(2, 3, 256, 256)
    output = native.training_step((image, {}), 0)
    expected = (image - torch.tensor(IMAGENET_MEAN).reshape(1, 3, 1, 1)) / torch.tensor(IMAGENET_STD).reshape(1, 3, 1, 1)
    torch.testing.assert_close(seen[0], expected)
    assert set(output) == {"loss"} and output["loss"].requires_grad
    assert logged == [("train_loss", output["loss"].item(),
                       {"on_epoch": True, "prog_bar": True, "logger": True, "batch_size": 1})]


@pytest.fixture(scope="module")
def fitted_fastflow(tmp_path_factory, fastflow_assets):
    """Fit two epochs and retain initial weights, calibration counts, step losses and checkpoint."""
    from lightning.pytorch import Trainer
    from lightning.pytorch.loggers import CSVLogger
    from visionlab.anomaly_detection.models.fastflow.lightning_model import FastflowLitModel

    root = str(tmp_path_factory.mktemp("fitted_fastflow"))
    data = make_fastflow_data()
    model = FastflowLitModel(flow_steps=2, visualizer=False)
    backbone_before = {name: value.clone() for name, value in model.model.feature_extractor.state_dict().items()}
    flow_before = model.model.fast_flow_blocks[0][0].subnet[1].weight.detach().clone()
    norm_before = model.model.norms[0].weight.detach().clone()
    counts, steps = {"update": 0, "finalize": 0}, []
    update, finalize, training_step = model.post_processor.update, model.post_processor.finalize, model.training_step

    def capture_update(outputs):
        """Count fit validation updates."""
        counts["update"] += 1
        return update(outputs)

    def capture_finalize():
        """Count fit validation finalization."""
        counts["finalize"] += 1
        return finalize()

    def capture_step(batch, batch_idx):
        """Record actual pre-update losses and the uneven batch sizes."""
        result = training_step(batch, batch_idx)
        steps.append((model.current_epoch, batch[0].shape[0], result["loss"].item()))
        return result

    trainer = Trainer(default_root_dir=root, accelerator="cpu", devices=1, max_epochs=2,
                      logger=CSVLogger(save_dir=root, name="metrics"), log_every_n_steps=1,
                      num_sanity_val_steps=0, gradient_clip_val=0, enable_checkpointing=False,
                      enable_progress_bar=False, enable_model_summary=False)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(model.post_processor, "update", capture_update)
        patch.setattr(model.post_processor, "finalize", capture_finalize)
        patch.setattr(model, "training_step", capture_step)
        trainer.fit(model, datamodule=data)
    path = os.path.join(root, "model.ckpt")
    trainer.save_checkpoint(path)
    return root, model, data, path, trainer, backbone_before, flow_before, norm_before, counts, steps


def test_fastflow_native_learning_optimizer_and_epoch_mean(fitted_fastflow):
    """Update flow and LayerNorm weights, preserve the backbone and average step losses equally."""
    root, model, data, _, trainer, before, flow_before, norm_before, counts, steps = fitted_fastflow
    assert not torch.equal(model.model.fast_flow_blocks[0][0].subnet[1].weight.detach(), flow_before)
    assert not torch.equal(model.model.norms[0].weight.detach(), norm_before)
    for name, value in model.model.feature_extractor.state_dict().items():
        assert torch.equal(value, before[name]), name
    assert counts == {"update": 2 * len(data.val_dataloader()), "finalize": 2}
    assert model.post_processor.is_calibrated
    assert len(trainer.optimizers) == 1 and not trainer.lr_scheduler_configs
    optimizer = trainer.optimizers[0]
    assert isinstance(optimizer, torch.optim.Adam) and len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == 1e-3 and optimizer.param_groups[0]["weight_decay"] == 1e-5
    assert {id(value) for value in optimizer.param_groups[0]["params"]} == {
        id(value) for value in model.model.parameters()
    }
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


def test_fastflow_native_checkpoint_and_frozen_evaluation(fitted_fastflow, tmp_path, monkeypatch):
    """Restore model and calibration and reproduce validation, test and prediction without calibration updates."""
    from lightning.pytorch import Trainer
    from visionlab.anomaly_detection.models.fastflow.lightning_model import FastflowLitModel

    _, model, data, path, _, _, _, _, _, _ = fitted_fastflow
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["optimizer_states"] and not saved["lr_schedulers"] and saved["post_processor"] is not None
    for prefix in ("model.feature_extractor.", "model.norms.", "model.fast_flow_blocks."):
        assert any(name.startswith(prefix) for name in saved["state_dict"])
    restored = FastflowLitModel.load_from_checkpoint(path, pre_trained=False, visualizer=False)
    model.eval()
    restored.eval()
    batch = list(data.val_dataloader())[-1]
    with torch.no_grad():
        expected, actual = model.predict_step(batch, 0), restored.predict_step(batch, 0)
    assert actual["pred_score"].shape == (1, 1) and actual["anomaly_map"].shape == (1, 1, 256, 256)
    for name in OUTPUT_NAMES:
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-5, atol=1e-6)
    before = restored.post_processor.state_dict()

    def forbidden(*args, **kwargs):
        """Reject standalone calibration updates and finalization."""
        raise AssertionError("Standalone evaluation must not update calibration")

    monkeypatch.setattr(restored.post_processor, "update", forbidden)
    monkeypatch.setattr(restored.post_processor, "finalize", forbidden)
    validation_step, test_step = restored.validation_step, restored.test_step
    observed = {}

    def capture_validation(batch, batch_idx):
        """Retain the final raw validation outputs."""
        outputs = validation_step(batch, batch_idx)
        observed["val"] = outputs
        return outputs

    def capture_test(batch, batch_idx):
        """Retain the final post-processed test outputs."""
        outputs = test_step(batch, batch_idx)
        observed["test"] = outputs
        return outputs

    monkeypatch.setattr(restored, "validation_step", capture_validation)
    monkeypatch.setattr(restored, "test_step", capture_test)
    trainer = Trainer(default_root_dir=str(tmp_path), accelerator="cpu", devices=1, logger=False,
                      enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
    validated = trainer.validate(restored, datamodule=data, ckpt_path=path)
    tested = trainer.test(restored, datamodule=data, ckpt_path=path)
    assert "val_image_AUROC" in validated[0] and "image_AUROC" in tested[0]
    predictions = trainer.predict(restored, datamodule=data)
    for name in ("pred_score", "anomaly_map"):
        for outputs in (predictions[-1], observed["val"], observed["test"]):
            torch.testing.assert_close(outputs[name], expected[name], rtol=1e-5, atol=1e-6)
    for name in OUTPUT_NAMES:
        torch.testing.assert_close(observed["test"][name], expected[name], rtol=1e-5, atol=1e-6)
    for name, value in restored.post_processor.state_dict().items():
        torch.testing.assert_close(value, before[name], equal_nan=True)


def test_fastflow_native_component_resolution_and_callbacks():
    """Use standard components, permit explicit instances and disable each component without changing model shapes."""
    from visionlab.anomaly_detection.common.evaluator import Evaluator
    from visionlab.anomaly_detection.common.lightning_visualizer import LitVisualizer
    from visionlab.anomaly_detection.common.post_processor import PostProcessor
    from visionlab.anomaly_detection.common.pre_processor import PreProcessor
    from visionlab.anomaly_detection.common.visualizer import ImageVisualizer
    from visionlab.anomaly_detection.models.fastflow.lightning_model import FastflowLitModel

    components = (PreProcessor(), PostProcessor(), Evaluator.default(), ImageVisualizer())
    model = FastflowLitModel(pre_trained=False, flow_steps=2, pre_processor=components[0],
                             post_processor=components[1], evaluator=components[2], visualizer=components[3])
    assert (model.pre_processor, model.post_processor, model.evaluator, model.visualizer) == components
    assert isinstance(model.configure_callbacks()[0], LitVisualizer)
    disabled = FastflowLitModel(pre_trained=False, flow_steps=2, pre_processor=False,
                                post_processor=False, evaluator=False, visualizer=False)
    assert disabled.pre_processor is disabled.post_processor is disabled.evaluator is disabled.visualizer is None
    assert disabled.configure_callbacks() == []
    for component in ("pre_processor", "post_processor", "evaluator", "visualizer"):
        with pytest.raises(TypeError):
            FastflowLitModel._resolve(None, type(getattr(model, component)), lambda: None)
    model._trainer = SimpleNamespace(state=SimpleNamespace(fn=None), sanity_checking=False)
    assert not model._fit_validation()
    from lightning.pytorch.trainer.states import TrainerFn

    model._trainer.state.fn = TrainerFn.FITTING
    assert model._fit_validation()
    model._trainer.sanity_checking = True
    assert not model._fit_validation()


def consume_fastflow_export(root):
    """Restore a pure export with only the artifact readable after imports and match all native outputs."""
    from visionlab.inference.predictor import Predictor
    from visionlab.anomaly_detection.models.fastflow.model import FastflowModel

    io_utils.block_network()
    path = os.path.join(root, "export", "model.pt")
    expected = torch.load(os.path.join(root, "expected.pt"), map_location="cpu", weights_only=True)
    exported = io_utils.load_export(path)
    assert exported["model_name"] == "fastflow"
    assert exported["model_args"] == {"input_size": [256, 256], "backbone": "resnet18", "pre_trained": False,
                                      "flow_steps": 2, "conv3x3_only": False, "hidden_ratio": 1.0}
    assert all(type(value) in (str, int, float, bool, list) for value in exported["model_args"].values())
    assert all(type(value) is int for value in exported["model_args"]["input_size"])
    for prefix in ("feature_extractor.", "norms.", "fast_flow_blocks."):
        assert any(name.startswith(prefix) for name in exported["model_state"])

    def audit_open(event, args):
        """Deny paths.toml, backbone-file reads and every file other than the artifact."""
        if event == "open":
            opened, mode, flags = args
            assert isinstance(opened, (str, bytes)) and os.path.abspath(opened) == path, opened
            assert not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)

    sys.addaudithook(audit_open)
    predictor = Predictor.from_export(path)
    assert isinstance(predictor.model, FastflowModel)
    for name, value in predictor.model.state_dict().items():
        assert torch.equal(value, exported["model_state"][name]), name
    for batch in expected["batches"]:
        actual = predictor.predict(batch["images"])
        assert actual["pred_score"].shape == (batch["images"].shape[0], 1)
        assert actual["anomaly_map"].shape == (batch["images"].shape[0], 1, 256, 256)
        for name, value in actual.items():
            torch.testing.assert_close(value, batch["outputs"][name], rtol=1e-5, atol=1e-6)


def test_fastflow_native_export_in_isolated_process(fitted_fastflow):
    """Match native inference with FrEIA allowed and Lightning, anomalib and external asset reads blocked."""
    root, model, data, _, _, _, _, _, _, _ = fitted_fastflow
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
    """Reject training and reference frameworks before any pure consumer imports."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("anomalib", "lightning", "pytorch_lightning"):
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, Block())
runpy.run_path(sys.argv[1])["consume_fastflow_export"](sys.argv[2])
assert "FrEIA" in sys.modules
assert not {"anomalib", "lightning", "pytorch_lightning"} & {name.split(".")[0] for name in sys.modules}
'''
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.path.join(PROJECT_ROOT, "src")
    result = subprocess.run([sys.executable, "-c", script, os.path.abspath(__file__), root],
                            env=environment, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_fastflow_predictor_lazy_freia_dependency(fitted_fastflow, tmp_path):
    """Import Predictor and restore a small existing-model export without FrEIA, then reject FastFlow."""
    _, model, _, _, _, _, _, _, _, _ = fitted_fastflow
    from visionlab.anomaly_detection.models.dfm.model import DFMModel

    existing = os.path.join(tmp_path, "dfm.pt")
    # This checks the existing export loader without fitting or evaluating an earlier-phase model.
    pure = DFMModel(backbone="resnet18", layer="layer4", pre_trained=False)
    io_utils.save_export(path=existing, model=pure, model_name="dfm",
                         model_args={"backbone": "resnet18", "layer": "layer4", "pre_trained": False},
                         pre_processor=model.pre_processor, post_processor=None)
    fastflow_path = os.path.join(fitted_fastflow[0], "export", "model.pt")
    script = '''
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    """Reject FrEIA and both training frameworks for other-model deployment."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("FrEIA", "anomalib", "lightning", "pytorch_lightning"):
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, Block())
from visionlab.inference.predictor import Predictor
from visionlab.utils.io import block_network
block_network()
predictor = Predictor.from_export(sys.argv[1])
assert type(predictor.model).__name__ == "DFMModel"
assert not any(name.split(".")[0] == "FrEIA" for name in sys.modules)
try:
    Predictor.from_export(sys.argv[2])
except ModuleNotFoundError as error:
    assert error.name.split(".")[0] == "FrEIA", error
else:
    raise AssertionError("FastFlow unexpectedly restored without FrEIA")
assert not {"FrEIA", "anomalib", "lightning", "pytorch_lightning"} & {name.split(".")[0] for name in sys.modules}
'''
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.path.join(PROJECT_ROOT, "src")
    result = subprocess.run([sys.executable, "-c", script, existing, fastflow_path],
                            env=environment, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
