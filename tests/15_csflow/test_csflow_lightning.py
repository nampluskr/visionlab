# tests/15_csflow/test_csflow_lightning.py: pytest native CSFlow learning, local teachers and isolated exports

import csv
import inspect
import os
import runpy
import subprocess
import sys

import pytest
import torch
from torchvision.models import efficientnet_b5

from visionlab.anomaly_detection.common.pre_processor import IMAGENET_MEAN, IMAGENET_STD
from visionlab.utils import io as io_utils

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_NAMES = ("pred_score", "anomaly_map", "normalized_pred_score", "normalized_anomaly_map", "pred_label", "pred_mask")


def make_csflow_model():
    """Build native CSFlow with two small coupling blocks and the required 256-pixel input."""
    from visionlab.anomaly_detection.models.csflow.lightning_model import CsflowLitModel

    return CsflowLitModel(cross_conv_hidden_channels=16, n_coupling_blocks=2, visualizer=False)


@pytest.fixture(scope="module")
def csflow_assets(tmp_path_factory):
    """Map a full torchvision EfficientNet-B5 state dict to a temporary local teacher file."""
    root = str(tmp_path_factory.mktemp("csflow_assets"))
    weights = os.path.join(root, "efficientnet_b5.pth")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        torch.save(efficientnet_b5(weights=None).state_dict(), weights)
    mapping = os.path.join(root, "paths.toml")
    with open(mapping, "w", encoding="utf-8", newline="\n") as stream:
        stream.write('[teachers]\ncsflow_efficientnet_b5 = "' + weights.replace("\\", "/") + '"\n')
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(io_utils, "PATHS_FILE", mapping)
        yield root


def test_csflow_native_signature_weights_and_input_size(csflow_assets):
    """Keep eight defaults and inject the complete local backbone including shared FX buffers."""
    from visionlab.anomaly_detection.models.csflow.lightning_model import CSFLOW_INPUT_SIZE, CsflowLitModel

    expected = {"cross_conv_hidden_channels": 1024, "n_coupling_blocks": 4, "clamp": 3, "num_channels": 3,
                "pre_processor": True, "post_processor": True, "evaluator": True, "visualizer": True}
    parameters = inspect.signature(CsflowLitModel).parameters
    assert len(parameters) == 8 and list(parameters) == list(expected)
    assert {name: parameter.default for name, parameter in parameters.items()} == expected
    assert all(parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD for parameter in parameters.values())
    model = make_csflow_model()
    path = os.path.join(csflow_assets, "efficientnet_b5.pth")
    source = torch.load(path, map_location="cpu", weights_only=True)
    extractor = model.model.feature_extractor.feature_extractor
    assert extractor.backbone.state_dict().keys() == source.keys()
    for name, value in extractor.backbone.state_dict().items():
        assert torch.equal(value, source[name]), name
    for name, value in extractor.feature_extractor.state_dict().items():
        assert torch.equal(value, source[name]), name
    assert model.teacher_weights_path == path
    assert model.model.input_dims == (3, *CSFLOW_INPUT_SIZE) == (3, 256, 256)
    assert not model.model.feature_extractor.training and extractor.requires_grad is False
    assert all(parameter.requires_grad for parameter in extractor.backbone.parameters())
    assert model.pre_processor.center_crop is None
    torch.testing.assert_close(model.pre_processor.mean.flatten(), torch.tensor(IMAGENET_MEAN, dtype=torch.float64))
    torch.testing.assert_close(model.pre_processor.std.flatten(), torch.tensor(IMAGENET_STD, dtype=torch.float64))
    assert model.learning_type.value == "one_class"
    assert [name for name, _, _, _ in model.evaluator.val_metrics] == ["image_AUROC", "pixel_AUROC"]
    assert [name for name, _, _, _ in model.evaluator.test_metrics] == [
        "image_AUROC", "pixel_AUROC", "image_F1Score", "pixel_F1Score",
    ]


def test_csflow_native_missing_and_invalid_teacher(tmp_path, monkeypatch):
    """Reject missing mappings, teacher keys, files and incompatible state dictionaries."""
    mapping = os.path.join(tmp_path, "paths.toml")
    monkeypatch.setattr(io_utils, "PATHS_FILE", mapping)
    with pytest.raises(FileNotFoundError):
        make_csflow_model()
    for content in ('[teachers]\n', '[teachers]\ncsflow_efficientnet_b5 = "missing.pth"\n'):
        with open(mapping, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        with pytest.raises(FileNotFoundError):
            make_csflow_model()
    invalid = os.path.join(tmp_path, "invalid.pth")
    torch.save({"wrong_key": torch.ones(1)}, invalid)
    with open(mapping, "w", encoding="utf-8", newline="\n") as stream:
        stream.write('[teachers]\ncsflow_efficientnet_b5 = "' + invalid.replace("\\", "/") + '"\n')
    with pytest.raises(RuntimeError, match="state_dict"):
        make_csflow_model()


def test_csflow_native_optimizer_and_trainer_arguments(csflow_assets):
    """Use automatic Adam over every parameter with the original hyperparameters and no scheduler."""
    model = make_csflow_model()
    assert model.automatic_optimization is True
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam) and len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert group["lr"] == 2e-4 and group["eps"] == 1e-4 and group["weight_decay"] == 1e-5
    assert group["betas"] == (0.5, 0.9)
    assert [id(parameter) for parameter in group["params"]] == [id(parameter) for parameter in model.parameters()]
    assert model.trainer_arguments == {"gradient_clip_val": 1, "num_sanity_val_steps": 0}
    assert "max_epochs" not in model.trainer_arguments


def test_csflow_native_initialization_normalization_and_logging(csflow_assets, monkeypatch):
    """Preserve flow initialization order, normalize once and count each training step once in the epoch mean."""
    from visionlab.anomaly_detection.models.csflow.model import CsFlowModel

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        pure = CsFlowModel(input_size=(256, 256), cross_conv_hidden_channels=16, n_coupling_blocks=2)
        following = torch.rand(4)
        torch.manual_seed(42)
        native = make_csflow_model()
        assert torch.equal(torch.rand(4), following)
    expected_state = pure.graph.state_dict()
    assert all(torch.equal(value, expected_state[name]) for name, value in native.model.graph.state_dict().items())
    seen, logged = [], []
    forward = native.model.forward

    def capture_input(image):
        seen.append(image.detach().clone())
        return forward(image)

    monkeypatch.setattr(native.model, "forward", capture_input)
    monkeypatch.setattr(native, "log", lambda name, value, **options: logged.append((name, value, options)))
    images = torch.rand(2, 3, 256, 256)
    result = native.training_step((images, {}), 0)
    normalized = (images - torch.tensor(IMAGENET_MEAN).reshape(1, 3, 1, 1)) / torch.tensor(IMAGENET_STD).reshape(1, 3, 1, 1)
    assert len(seen) == 1
    torch.testing.assert_close(seen[0], normalized)
    assert set(result) == {"loss"} and result["loss"].requires_grad
    assert logged == [("train_loss", result["loss"].item(),
                       {"on_epoch": True, "prog_bar": True, "logger": True, "batch_size": 1})]


def test_csflow_native_components_and_callbacks(csflow_assets):
    """Use shared processing objects and the standard visualizer without changing score or map axes."""
    from visionlab.anomaly_detection.common.evaluator import Evaluator
    from visionlab.anomaly_detection.common.lightning_visualizer import LitVisualizer
    from visionlab.anomaly_detection.common.post_processor import PostProcessor
    from visionlab.anomaly_detection.common.pre_processor import PreProcessor
    from visionlab.anomaly_detection.common.visualizer import ImageVisualizer
    from visionlab.anomaly_detection.models.csflow.lightning_model import CsflowLitModel

    components = (PreProcessor(), PostProcessor(), Evaluator.default(), ImageVisualizer())
    model = CsflowLitModel(cross_conv_hidden_channels=16, n_coupling_blocks=2,
                          pre_processor=components[0], post_processor=components[1],
                          evaluator=components[2], visualizer=components[3])
    assert (model.pre_processor, model.post_processor, model.evaluator, model.visualizer) == components
    assert isinstance(model.configure_callbacks()[0], LitVisualizer)
    disabled = CsflowLitModel(cross_conv_hidden_channels=16, n_coupling_blocks=2,
                             pre_processor=False, post_processor=False, evaluator=False, visualizer=False)
    assert disabled.pre_processor is disabled.post_processor is disabled.evaluator is disabled.visualizer is None
    assert disabled.configure_callbacks() == []
    for component_type in (PreProcessor, PostProcessor, Evaluator, ImageVisualizer):
        with pytest.raises(TypeError):
            model._resolve(None, component_type, lambda: None)


@pytest.fixture(scope="module")
def fitted_csflow(tmp_path_factory, csflow_assets):
    """Fit two epochs on uneven tiny batches and observe clipping immediately before Adam steps."""
    from lightning.pytorch import Callback, Trainer
    from lightning.pytorch.loggers import CSVLogger

    class ObserveClipping(Callback):
        """Force a known norm above one before Trainer applies its configured clipping."""

        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            gradients = [parameter.grad for parameter in pl_module.parameters() if parameter.grad is not None]
            norm = torch.linalg.vector_norm(torch.stack([gradient.norm() for gradient in gradients]))
            assert torch.isfinite(norm) and norm > 0
            # Ensure this test exercises clipping even when naturally occurring gradients are small.
            for gradient in gradients:
                gradient.mul_(2 / norm)

    helpers = runpy.run_path(os.path.join(PROJECT_ROOT, "tests", "12_fastflow", "test_fastflow_lightning.py"))
    data = helpers["make_fastflow_data"]()
    root = str(tmp_path_factory.mktemp("fitted_csflow"))
    model = make_csflow_model()
    gamma_before = {name: value.detach().clone() for name, value in model.model.graph.named_parameters() if ".gamma" in name}
    counts, steps, clipped_norms = {"update": 0, "finalize": 0}, [], []
    update, finalize, training_step = model.post_processor.update, model.post_processor.finalize, model.training_step
    configure_clipping = model.configure_gradient_clipping

    def capture_clipping(optimizer, gradient_clip_val, gradient_clip_algorithm):
        configure_clipping(optimizer, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm)
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        clipped_norms.append(torch.linalg.vector_norm(torch.stack([gradient.norm() for gradient in gradients])).item())

    def capture_update(outputs):
        counts["update"] += 1
        return update(outputs)

    def capture_finalize():
        counts["finalize"] += 1
        return finalize()

    def capture_training_step(batch, batch_idx):
        result = training_step(batch, batch_idx)
        steps.append((model.current_epoch, result["loss"].item()))
        return result

    trainer = Trainer(default_root_dir=root, accelerator="cpu", devices=1, max_epochs=2,
                      gradient_clip_val=1, num_sanity_val_steps=0, callbacks=[ObserveClipping()],
                      logger=CSVLogger(save_dir=root, name="metrics"), log_every_n_steps=1,
                      enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(model.post_processor, "update", capture_update)
        patch.setattr(model.post_processor, "finalize", capture_finalize)
        patch.setattr(model, "training_step", capture_training_step)
        patch.setattr(model, "configure_gradient_clipping", capture_clipping)
        trainer.fit(model, datamodule=data)
    path = os.path.join(root, "model.ckpt")
    trainer.save_checkpoint(path)
    return root, model, data, path, trainer, gamma_before, counts, steps, clipped_norms


def test_csflow_native_learning_epoch_mean_and_clipping(fitted_csflow, csflow_assets):
    """Update gamma, retain backbone parameters and buffers, and log the unweighted mean of step losses."""
    root, model, data, _, trainer, gamma_before, counts, steps, clipped_norms = fitted_csflow
    parameters = dict(model.model.graph.named_parameters())
    assert any(not torch.equal(parameters[name], value) for name, value in gamma_before.items())
    source = torch.load(os.path.join(csflow_assets, "efficientnet_b5.pth"), map_location="cpu", weights_only=True)
    extractor = model.model.feature_extractor.feature_extractor
    for name, value in extractor.backbone.state_dict().items():
        assert torch.equal(value, source[name]), name
    for name, value in extractor.feature_extractor.state_dict().items():
        assert torch.equal(value, source[name]), name
    assert all(parameter.grad is None for parameter in extractor.backbone.parameters())
    assert counts == {"update": 2 * len(data.val_dataloader()), "finalize": 2}
    assert model.post_processor.is_calibrated
    assert len(trainer.optimizers) == 1 and not trainer.lr_scheduler_configs
    assert trainer.gradient_clip_val == 1
    assert len(clipped_norms) == len(steps) == 2 * len(data.train_dataloader())
    assert all(0 < norm <= 1 + 1e-6 for norm in clipped_norms)
    with open(os.path.join(root, "metrics", "version_0", "metrics.csv"), newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert "train_loss_epoch" in reader.fieldnames and "train_loss_step" in reader.fieldnames
        epochs = {int(row["epoch"]): float(row["train_loss_epoch"]) for row in reader if row.get("train_loss_epoch")}
    assert set(epochs) == {0, 1}
    for epoch, actual in epochs.items():
        values = [loss for step_epoch, loss in steps if step_epoch == epoch]
        assert actual == pytest.approx(sum(values) / len(values), rel=1e-5, abs=1e-6)


def test_csflow_native_checkpoint_and_frozen_evaluation(fitted_csflow, tmp_path, monkeypatch):
    """Restore checkpoint predictions and prevent standalone validation or test from updating calibration."""
    from lightning.pytorch import Trainer
    from visionlab.anomaly_detection.models.csflow.lightning_model import CsflowLitModel

    _, model, data, path, _, _, _, _, _ = fitted_csflow
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["optimizer_states"][0]["state"] and not saved["lr_schedulers"] and saved["post_processor"] is not None
    for prefix in ("model.feature_extractor.feature_extractor.backbone.",
                   "model.feature_extractor.feature_extractor.feature_extractor.", "model.graph."):
        assert any(name.startswith(prefix) for name in saved["state_dict"])
    restored = CsflowLitModel.load_from_checkpoint(path, visualizer=False)
    model.eval()
    restored.eval()
    batch = list(data.val_dataloader())[-1]
    with torch.no_grad():
        expected, actual = model.predict_step(batch, 0), restored.predict_step(batch, 0)
    assert actual["pred_score"].shape == (1,) and actual["anomaly_map"].shape == (1, 1, 256, 256)
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


def consume_csflow_export(root):
    """Restore the export under file guards and compare native predictions without training frameworks."""
    from visionlab.inference.predictor import Predictor
    from visionlab.anomaly_detection.models.csflow.model import CsFlowModel

    io_utils.block_network()
    path = os.path.join(root, "export", "model.pt")
    expected = torch.load(os.path.join(root, "expected.pt"), map_location="cpu", weights_only=True)
    exported = io_utils.load_export(path)
    assert exported["model_name"] == "csflow"
    assert exported["model_args"] == {"cross_conv_hidden_channels": 16, "n_coupling_blocks": 2,
                                      "clamp": 3, "num_channels": 3, "input_size": [256, 256]}
    assert all(type(value) in (int, list) for value in exported["model_args"].values())
    assert all(type(value) is int for value in exported["model_args"]["input_size"])
    for prefix in ("feature_extractor.feature_extractor.backbone.",
                   "feature_extractor.feature_extractor.feature_extractor.", "graph."):
        assert any(name.startswith(prefix) for name in exported["model_state"])

    def audit_open(event, args):
        if event == "open":
            opened, mode, flags = args
            assert isinstance(opened, (str, bytes)) and os.path.abspath(opened) == path, opened
            assert not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)

    sys.addaudithook(audit_open)
    predictor = Predictor.from_export(path)
    assert isinstance(predictor.model, CsFlowModel) and predictor.model.input_dims == (3, 256, 256)
    assert isinstance(predictor.model.feature_extractor.input_size, tuple)
    for name, value in predictor.model.state_dict().items():
        assert torch.equal(value, exported["model_state"][name]), name
    for batch in expected["batches"]:
        actual = predictor.predict(batch["images"])
        assert actual["pred_score"].shape == (len(batch["images"]),)
        assert actual["anomaly_map"].shape == (len(batch["images"]), 1, 256, 256)
        for name in OUTPUT_NAMES:
            torch.testing.assert_close(actual[name], batch["outputs"][name], rtol=1e-5, atol=1e-6)


def test_csflow_native_isolated_export(fitted_csflow):
    """Match native and pure inference while blocking Lightning, anomalib and all external asset reads."""
    root, model, data, _, _, _, _, _, _ = fitted_csflow
    model.eval()
    batches = []
    for batch in data.val_dataloader():
        with torch.no_grad():
            outputs = model.predict_step(batch, 0)
        batches.append({"images": batch[0], "outputs": {name: outputs[name] for name in OUTPUT_NAMES}})
    torch.save({"batches": batches}, os.path.join(root, "expected.pt"))
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
runpy.run_path(sys.argv[1])["consume_csflow_export"](sys.argv[2])
assert "FrEIA" in sys.modules
assert not {"anomalib", "lightning", "pytorch_lightning"} & {name.split(".")[0] for name in sys.modules}
'''
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.path.join(PROJECT_ROOT, "src")
    result = subprocess.run([sys.executable, "-c", script, os.path.abspath(__file__), root],
                            env=environment, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_csflow_predictor_lazy_freia_dependency(fitted_csflow):
    """Import Predictor without FrEIA and require it only when restoring the CSFlow export."""
    path = os.path.join(fitted_csflow[0], "export", "model.pt")
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
for model in ("fastflow", "cflow", "uflow", "csflow"):
    assert f"visionlab.anomaly_detection.models.{model}.model" not in sys.modules
try:
    Predictor.from_export(sys.argv[1])
except ModuleNotFoundError as error:
    assert error.name.split(".")[0] == "FrEIA", error
else:
    raise AssertionError("CSFlow unexpectedly restored without FrEIA")
assert not {"FrEIA", "anomalib", "lightning", "pytorch_lightning"} & {name.split(".")[0] for name in sys.modules}
'''
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.path.join(PROJECT_ROOT, "src")
    result = subprocess.run([sys.executable, "-c", script, path], env=environment,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
