# tests/14_uflow/test_uflow_lightning.py: pytest native UFlow learning, local weights and isolated exports

import csv
import inspect
import os
import runpy
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


def make_uflow_model():
    """Build a small native UFlow with the required 256-pixel input and local ResNet18 weights."""
    from visionlab.anomaly_detection.models.uflow.lightning_model import UflowLitModel

    return UflowLitModel(backbone="resnet18", flow_steps=1, visualizer=False)


@pytest.fixture(scope="module")
def uflow_assets(tmp_path_factory):
    """Map a full timm ResNet18 state dict to a temporary local backbone file."""
    root = str(tmp_path_factory.mktemp("uflow_assets"))
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


def test_uflow_native_signature_weights_and_input_size(uflow_assets):
    """Keep nine original defaults and inject local weights into the nested timm backbone."""
    from visionlab.anomaly_detection.models.uflow.lightning_model import UFLOW_INPUT_SIZE, UflowLitModel
    from visionlab.anomaly_detection.models.uflow.model import UflowModel

    expected = {"backbone": "mcait", "flow_steps": 4, "affine_clamp": 2.0,
                "affine_subnet_channels_ratio": 1.0, "permute_soft": False,
                "pre_processor": True, "post_processor": True, "evaluator": True, "visualizer": True}
    parameters = inspect.signature(UflowLitModel).parameters
    assert len(parameters) == 9 and list(parameters) == list(expected)
    assert {name: parameter.default for name, parameter in parameters.items()} == expected
    assert all(parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD for parameter in parameters.values())
    model = make_uflow_model()
    path = os.path.join(uflow_assets, "resnet18.pth")
    source = torch.load(path, map_location="cpu", weights_only=True)
    for name, value in model.model.feature_extractor.feature_extractor.state_dict().items():
        assert torch.equal(value, source[name]), name
    assert model.teacher_weights_path == path
    assert inspect.signature(UflowModel).parameters["input_size"].default == (448, 448)
    assert model.model.input_size == model.model.anomaly_map_generator.input_size == UFLOW_INPUT_SIZE == (256, 256)
    assert model.pre_processor.center_crop is None
    torch.testing.assert_close(model.pre_processor.mean.flatten(), torch.tensor(IMAGENET_MEAN, dtype=torch.float64))
    torch.testing.assert_close(model.pre_processor.std.flatten(), torch.tensor(IMAGENET_STD, dtype=torch.float64))
    assert model.learning_type.value == "one_class"
    assert model.trainer_arguments == {"num_sanity_val_steps": 0} and "max_epochs" not in model.trainer_arguments
    assert [name for name, _, _, _ in model.evaluator.val_metrics] == ["image_AUROC", "pixel_AUROC"]
    assert [name for name, _, _, _ in model.evaluator.test_metrics] == [
        "image_AUROC", "pixel_AUROC", "image_F1Score", "pixel_F1Score",
    ]


def test_uflow_native_missing_weights(tmp_path, monkeypatch):
    """Fail when the local mapping, backbone entry or referenced weight file is absent."""
    mapping = os.path.join(tmp_path, "paths.toml")
    monkeypatch.setattr(io_utils, "PATHS_FILE", mapping)
    with pytest.raises(FileNotFoundError):
        make_uflow_model()
    for content in ('[backbones]\n', '[backbones]\nresnet18 = "missing.pth"\n'):
        with open(mapping, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        with pytest.raises(FileNotFoundError):
            make_uflow_model()


def test_uflow_native_optimizer_and_scheduler(uflow_assets):
    """Optimize every native parameter in one Adam group and step the original LinearLR by epoch."""
    model = make_uflow_model()
    assert model.automatic_optimization is True
    optimizers, schedulers = model.configure_optimizers()
    assert len(optimizers) == len(schedulers) == 1
    optimizer, scheduler = optimizers[0], schedulers[0]
    assert isinstance(optimizer, torch.optim.Adam) and len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert group["lr"] == group["initial_lr"] == 1e-3 and group["weight_decay"] == 1e-5
    assert [id(value) for value in group["params"]] == [id(value) for value in model.parameters()]
    assert {id(value) for value in model.model.feature_extractor.feature_extractor.parameters()} <= {
        id(value) for value in group["params"] if not value.requires_grad
    }
    assert isinstance(scheduler, torch.optim.lr_scheduler.LinearLR) and scheduler.optimizer is optimizer
    assert scheduler.start_factor == 1.0 and scheduler.end_factor == 0.4 and scheduler.total_iters == 25000


def test_uflow_native_initialization_normalization_and_log(uflow_assets, monkeypatch):
    """Preserve pure initialization RNG order, normalize once and log only step loss with original options."""
    from visionlab.anomaly_detection.models.uflow.model import UflowModel

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        pure = UflowModel(input_size=(256, 256), flow_steps=1, backbone="resnet18")
        following = torch.rand(4)
        torch.manual_seed(42)
        native = make_uflow_model()
        assert torch.equal(torch.rand(4), following)
    for component in ("flow", "feature_extractor.feature_normalizations"):
        expected = pure.get_submodule(component).state_dict()
        for name, value in native.model.get_submodule(component).state_dict().items():
            assert torch.equal(value, expected[name]), name
    seen, logged = [], []
    forward = native.model.forward

    def capture_input(image):
        seen.append(image.detach().clone())
        return forward(image)

    monkeypatch.setattr(native.model, "forward", capture_input)
    monkeypatch.setattr(native, "log_dict", lambda values, **options: logged.append((values, options)))
    image = torch.rand(2, 3, 256, 256)
    output = native.training_step((image, {}), 0)
    expected = (image - torch.tensor(IMAGENET_MEAN).reshape(1, 3, 1, 1)) / torch.tensor(IMAGENET_STD).reshape(1, 3, 1, 1)
    assert len(seen) == 1
    torch.testing.assert_close(seen[0], expected)
    assert set(output) == {"loss"} and output["loss"].requires_grad
    assert logged == [({"loss": output["loss"]},
                       {"on_step": True, "on_epoch": False, "prog_bar": False, "logger": True})]


def test_uflow_native_components_and_callbacks(uflow_assets):
    """Reuse the native processing, evaluation and visualizer contracts including fit-only calibration."""
    from lightning.pytorch.trainer.states import TrainerFn
    from visionlab.anomaly_detection.common.evaluator import Evaluator
    from visionlab.anomaly_detection.common.lightning_visualizer import LitVisualizer
    from visionlab.anomaly_detection.common.post_processor import PostProcessor
    from visionlab.anomaly_detection.common.pre_processor import PreProcessor
    from visionlab.anomaly_detection.common.visualizer import ImageVisualizer
    from visionlab.anomaly_detection.models.uflow.lightning_model import UflowLitModel

    components = (PreProcessor(), PostProcessor(), Evaluator.default(), ImageVisualizer())
    model = UflowLitModel(backbone="resnet18", flow_steps=1, pre_processor=components[0],
                         post_processor=components[1], evaluator=components[2], visualizer=components[3])
    assert (model.pre_processor, model.post_processor, model.evaluator, model.visualizer) == components
    assert isinstance(model.configure_callbacks()[0], LitVisualizer)
    disabled = UflowLitModel(backbone="resnet18", flow_steps=1, pre_processor=False,
                            post_processor=False, evaluator=False, visualizer=False)
    assert disabled.pre_processor is disabled.post_processor is disabled.evaluator is disabled.visualizer is None
    assert disabled.configure_callbacks() == []
    for component_type in (PreProcessor, PostProcessor, Evaluator, ImageVisualizer):
        with pytest.raises(TypeError):
            model._resolve(None, component_type, lambda: None)
    assert not model._fit_validation()
    model._trainer = SimpleNamespace(state=SimpleNamespace(fn=TrainerFn.FITTING), sanity_checking=False)
    assert model._fit_validation()
    model._trainer.sanity_checking = True
    assert not model._fit_validation()


@pytest.fixture(scope="module")
def fitted_uflow(tmp_path_factory, uflow_assets):
    """Fit one epoch using the established tiny 256-pixel dataset and retain checkpoint and calibration counts."""
    from lightning.pytorch import Trainer
    from lightning.pytorch.loggers import CSVLogger

    helpers = runpy.run_path(os.path.join(PROJECT_ROOT, "tests", "12_fastflow", "test_fastflow_lightning.py"))
    data = helpers["make_fastflow_data"]()
    root = str(tmp_path_factory.mktemp("fitted_uflow"))
    model = make_uflow_model()
    before = {name: value.clone() for name, value in model.model.feature_extractor.feature_extractor.state_dict().items()}
    flow_before = next(model.model.flow.parameters()).detach().clone()
    counts, steps = {"update": 0, "finalize": 0}, []
    update, finalize, training_step = model.post_processor.update, model.post_processor.finalize, model.training_step

    def capture_update(outputs):
        counts["update"] += 1
        return update(outputs)

    def capture_finalize():
        counts["finalize"] += 1
        return finalize()

    def capture_step(batch, batch_idx):
        result = training_step(batch, batch_idx)
        steps.append(result["loss"].item())
        return result

    trainer = Trainer(default_root_dir=root, accelerator="cpu", devices=1, max_epochs=1,
                      logger=CSVLogger(save_dir=root, name="metrics"), log_every_n_steps=1,
                      num_sanity_val_steps=0, enable_checkpointing=False,
                      enable_progress_bar=False, enable_model_summary=False)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(model.post_processor, "update", capture_update)
        patch.setattr(model.post_processor, "finalize", capture_finalize)
        patch.setattr(model, "training_step", capture_step)
        trainer.fit(model, datamodule=data)
    path = os.path.join(root, "model.ckpt")
    trainer.save_checkpoint(path)
    return root, model, data, path, trainer, before, flow_before, counts, steps


def test_uflow_native_learning_and_step_logging(fitted_uflow):
    """Update flow weights, retain the frozen backbone and record loss at each optimizer step only."""
    root, model, data, _, trainer, before, flow_before, counts, steps = fitted_uflow
    assert not torch.equal(next(model.model.flow.parameters()).detach(), flow_before)
    for name, value in model.model.feature_extractor.feature_extractor.state_dict().items():
        assert torch.equal(value, before[name]), name
    assert counts == {"update": len(data.val_dataloader()), "finalize": 1}
    assert model.post_processor.is_calibrated
    assert len(trainer.optimizers) == len(trainer.lr_scheduler_configs) == 1
    assert trainer.lr_scheduler_configs[0].interval == "epoch"
    assert trainer.lr_scheduler_configs[0].scheduler.last_epoch == 1
    with open(os.path.join(root, "metrics", "version_0", "metrics.csv"), newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert "loss" in reader.fieldnames
        assert "train_loss_epoch" not in reader.fieldnames and "loss_epoch" not in reader.fieldnames
        losses = [float(row["loss"]) for row in reader if row.get("loss")]
    assert losses == pytest.approx(steps, rel=1e-5, abs=1e-6)


def test_uflow_native_checkpoint_and_frozen_evaluation(fitted_uflow, tmp_path, monkeypatch):
    """Restore checkpoint outputs and keep standalone validation and test calibration unchanged."""
    from lightning.pytorch import Trainer
    from visionlab.anomaly_detection.models.uflow.lightning_model import UflowLitModel

    _, model, data, path, _, _, _, _, _ = fitted_uflow
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["optimizer_states"][0]["state"] and saved["lr_schedulers"] and saved["post_processor"] is not None
    for prefix in ("model.feature_extractor.", "model.flow."):
        assert any(name.startswith(prefix) for name in saved["state_dict"])
    restored = UflowLitModel.load_from_checkpoint(path, visualizer=False)
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
        raise AssertionError("Standalone evaluation must not update calibration")

    monkeypatch.setattr(restored.post_processor, "update", forbidden)
    monkeypatch.setattr(restored.post_processor, "finalize", forbidden)
    observed = {}
    validation_step, test_step = restored.validation_step, restored.test_step

    def capture_validation(batch, batch_idx):
        observed["val"] = validation_step(batch, batch_idx)
        return observed["val"]

    def capture_test(batch, batch_idx):
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


def consume_uflow_export(root):
    """Restore only the export under file guards and match native predictions without training frameworks."""
    from visionlab.inference.predictor import Predictor
    from visionlab.anomaly_detection.models.uflow.model import UflowModel

    io_utils.block_network()
    path = os.path.join(root, "export", "model.pt")
    expected = torch.load(os.path.join(root, "expected.pt"), map_location="cpu", weights_only=True)
    exported = io_utils.load_export(path)
    assert exported["model_name"] == "uflow"
    assert exported["model_args"] == {
        "backbone": "resnet18", "flow_steps": 1, "affine_clamp": 2.0,
        "affine_subnet_channels_ratio": 1.0, "permute_soft": False, "input_size": [256, 256],
    }
    assert all(type(value) in (str, int, float, bool, list) for value in exported["model_args"].values())
    assert all(type(value) is int for value in exported["model_args"]["input_size"])
    for prefix in ("feature_extractor.feature_extractor.", "feature_extractor.feature_normalizations.", "flow."):
        assert any(name.startswith(prefix) for name in exported["model_state"])

    def audit_open(event, args):
        if event == "open":
            opened, mode, flags = args
            assert isinstance(opened, (str, bytes)) and os.path.abspath(opened) == path, opened
            assert not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)

    sys.addaudithook(audit_open)
    predictor = Predictor.from_export(path)
    assert isinstance(predictor.model, UflowModel) and predictor.model.input_size == (256, 256)
    assert isinstance(predictor.model.input_size, tuple)
    for name, value in predictor.model.state_dict().items():
        assert torch.equal(value, exported["model_state"][name]), name
    for batch in expected["batches"]:
        actual = predictor.predict(batch["images"])
        assert actual["pred_score"].shape == (len(batch["images"]), 1)
        assert actual["anomaly_map"].shape == (len(batch["images"]), 1, 256, 256)
        for name in OUTPUT_NAMES:
            torch.testing.assert_close(actual[name], batch["outputs"][name], rtol=1e-5, atol=1e-6)


def test_uflow_native_isolated_export(fitted_uflow):
    """Match native and pure predictions with Lightning, anomalib and external asset reads blocked."""
    root, model, data, _, _, _, _, _, _ = fitted_uflow
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
    """Reject training and reference frameworks before pure consumer imports."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("anomalib", "lightning", "pytorch_lightning"):
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, Block())
runpy.run_path(sys.argv[1])["consume_uflow_export"](sys.argv[2])
assert "FrEIA" in sys.modules
assert not {"anomalib", "lightning", "pytorch_lightning"} & {name.split(".")[0] for name in sys.modules}
'''
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.path.join(PROJECT_ROOT, "src")
    result = subprocess.run([sys.executable, "-c", script, os.path.abspath(__file__), root],
                            env=environment, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_uflow_predictor_lazy_freia_dependency(fitted_uflow):
    """Import Predictor without FrEIA and require it only when restoring the UFlow export."""
    path = os.path.join(fitted_uflow[0], "export", "model.pt")
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
for model in ("fastflow", "cflow", "uflow"):
    assert f"visionlab.anomaly_detection.models.{model}.model" not in sys.modules
try:
    Predictor.from_export(sys.argv[1])
except ModuleNotFoundError as error:
    assert error.name.split(".")[0] == "FrEIA", error
else:
    raise AssertionError("UFlow unexpectedly restored without FrEIA")
assert not {"FrEIA", "anomalib", "lightning", "pytorch_lightning"} & {name.split(".")[0] for name in sys.modules}
'''
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.path.join(PROJECT_ROOT, "src")
    result = subprocess.run([sys.executable, "-c", script, path], env=environment,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
