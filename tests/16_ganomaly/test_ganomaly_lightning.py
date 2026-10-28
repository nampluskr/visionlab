# tests/16_ganomaly/test_ganomaly_lightning.py: pytest native GANomaly manual optimization, raw metrics and isolated exports
"""Use 256x256 inputs with n_features=4, latent_vec_size=16 and batches of at most two."""

import csv
import inspect
import os
import subprocess
import sys

import pytest
import torch

from visionlab.anomaly_detection.common.pre_processor import IMAGENET_MEAN, IMAGENET_STD
from visionlab.utils import io as io_utils

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_NAMES = ("pred_score", "anomaly_map", "normalized_pred_score", "normalized_anomaly_map", "pred_label", "pred_mask")


def make_ganomaly_model():
    """Build a small GANomaly while retaining the required native input size."""
    from visionlab.anomaly_detection.models.ganomaly.lightning_model import GanomalyLitModel

    return GanomalyLitModel(batch_size=2, n_features=4, latent_vec_size=16, visualizer=False)


def make_ganomaly_data():
    """Provide deterministic uneven training batches and labeled image-only evaluation data."""
    from lightning.pytorch import LightningDataModule

    class TinyData(LightningDataModule):
        """Supply synthetic 256-pixel RGB images without dataset or weight files."""

        def __init__(self):
            super().__init__()
            generator = torch.Generator().manual_seed(17)
            self.train_data = [(torch.rand(3, 256, 256, generator=generator) * 0.02 + level, {"label": 0})
                               for level in (0.40, 0.42, 0.98)]
            self.val_data = [(torch.rand(3, 256, 256, generator=generator) * 0.2 + level,
                              {"label": index % 2, "image_path": f"{index}.png", "sample_id": f"fixture/{index}.png"})
                             for index, level in enumerate((0.1, 0.7, 0.2))]

        def train_dataloader(self):
            return torch.utils.data.DataLoader(self.train_data, batch_size=2, shuffle=False, num_workers=0)

        def val_dataloader(self):
            return torch.utils.data.DataLoader(self.val_data, batch_size=2, shuffle=False, num_workers=0)

        def test_dataloader(self):
            return self.val_dataloader()

    return TinyData()


def test_ganomaly_native_signature_input_and_image_evaluator():
    """Keep all fifteen constructor defaults, constant input size and image-only metric fields."""
    from visionlab.anomaly_detection.models.ganomaly.lightning_model import GANOMALY_INPUT_SIZE, GanomalyLitModel

    expected = {"batch_size": 32, "n_features": 64, "latent_vec_size": 100, "extra_layers": 0,
                "add_final_conv_layer": True, "wadv": 1, "wcon": 50, "wenc": 1, "lr": 0.0002,
                "beta1": 0.5, "beta2": 0.999, "pre_processor": True, "post_processor": True,
                "evaluator": True, "visualizer": True}
    parameters = inspect.signature(GanomalyLitModel).parameters
    assert len(parameters) == 15 and list(parameters) == list(expected)
    assert {name: value.default for name, value in parameters.items()} == expected
    assert all(value.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD for value in parameters.values())
    assert isinstance(inspect.getattr_static(GanomalyLitModel, "configure_evaluator"), staticmethod)
    model = make_ganomaly_model()
    assert GANOMALY_INPUT_SIZE == (256, 256)
    assert model.model.generator.encoder1.final_conv_layer.in_channels == \
        model.model.generator.decoder.latent_input[0].out_channels == model.n_features * 2**5 == 128
    assert torch.equal(model.real_label, torch.ones(2)) and torch.equal(model.fake_label, torch.zeros(2))
    assert model.min_scores.item() == float("inf") and model.max_scores.item() == float("-inf")
    assert model.learning_type.value == "one_class" and model.teacher_weights_path is None
    assert model.pre_processor.center_crop is None
    torch.testing.assert_close(model.pre_processor.mean.flatten(), torch.tensor(IMAGENET_MEAN, dtype=torch.float64))
    torch.testing.assert_close(model.pre_processor.std.flatten(), torch.tensor(IMAGENET_STD, dtype=torch.float64))
    assert [entry[0] for entry in model.evaluator.val_metrics] == ["image_AUROC"]
    assert [entry[0] for entry in model.evaluator.test_metrics] == ["image_AUROC", "image_F1Score"]


def test_ganomaly_native_no_assets_and_same_initialization(tmp_path, monkeypatch):
    """Consume the same RNG as the pure model without reading paths.toml, weights or network resources."""
    from visionlab.anomaly_detection.models.ganomaly.lightning_model import GanomalyLitModel
    from visionlab.anomaly_detection.models.ganomaly.model import GanomalyModel

    monkeypatch.setattr(io_utils, "PATHS_FILE", os.path.join(tmp_path, "absent.toml"))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        pure = GanomalyModel(input_size=(256, 256), num_input_channels=3, n_features=4, latent_vec_size=16)
        following = torch.rand(4)
        torch.manual_seed(42)

        def forbidden(*args, **kwargs):
            raise AssertionError("GANomaly construction must not read assets or access the network")

        with pytest.MonkeyPatch.context() as patch:
            for target in ("builtins.open", "io.open", "os.open", "torch.load", "socket.socket.connect",
                           "socket.socket.connect_ex", "socket.create_connection"):
                patch.setattr(target, forbidden)
            native = GanomalyLitModel(batch_size=2, n_features=4, latent_vec_size=16, visualizer=False)
        assert torch.equal(torch.rand(4), following)
    assert pure.state_dict().keys() == native.model.state_dict().keys()
    assert all(torch.equal(value, native.model.state_dict()[name]) for name, value in pure.state_dict().items())


def test_ganomaly_native_optimizer_order_and_trainer_arguments():
    """Return disjoint discriminator and generator Adam groups in original order without a scheduler."""
    model = make_ganomaly_model()
    assert model.automatic_optimization is False
    optimizers = model.configure_optimizers()
    assert isinstance(optimizers, list) and len(optimizers) == 2
    groups = []
    for optimizer, network in zip(optimizers, (model.model.discriminator, model.model.generator), strict=True):
        assert isinstance(optimizer, torch.optim.Adam) and len(optimizer.param_groups) == 1
        group = optimizer.param_groups[0]
        assert group["lr"] == 2e-4 and group["betas"] == (0.5, 0.999)
        assert [id(value) for value in group["params"]] == [id(value) for value in network.parameters()]
        groups.append({id(value) for value in group["params"]})
    assert not groups[0] & groups[1]
    assert model.trainer_arguments == {"gradient_clip_val": 0, "num_sanity_val_steps": 0}
    assert "max_epochs" not in model.trainer_arguments


def test_ganomaly_native_single_normalization_and_log_options(monkeypatch):
    """Normalize the training input once and retain the exact backward and logging contracts."""
    model = make_ganomaly_model()
    seen, logged, retained = [], [], []
    forward = model.model.forward

    def capture_input(image):
        seen.append(image.detach().clone())
        return forward(image)

    def backward(loss, retain_graph=False):
        retained.append(retain_graph)
        loss.backward(retain_graph=retain_graph)

    optimizers = model.configure_optimizers()
    monkeypatch.setattr(model.model, "forward", capture_input)
    monkeypatch.setattr(model, "optimizers", lambda: optimizers)
    monkeypatch.setattr(model, "manual_backward", backward)
    monkeypatch.setattr(model, "log_dict", lambda values, **options: logged.append((values, options)))
    images = torch.rand(2, 3, 256, 256)
    result = model.training_step((images, {}), 0)
    assert len(seen) == 1 and retained == [True, False]
    torch.testing.assert_close(seen[0], model.pre_processor(images))
    assert set(result) == {"generator_loss", "discriminator_loss"}
    assert all(value.requires_grad and torch.isfinite(value) for value in result.values())
    assert logged == [({name: value.item() for name, value in result.items()},
                       {"on_epoch": True, "prog_bar": True, "logger": True, "batch_size": 1})]


def test_ganomaly_native_components_and_running_normalization():
    """Accept native components and preserve the unguarded running min/max normalization formula."""
    from lightning.pytorch.trainer.states import TrainerFn
    from types import SimpleNamespace
    from visionlab.anomaly_detection.common.evaluator import Evaluator
    from visionlab.anomaly_detection.common.lightning_visualizer import LitVisualizer
    from visionlab.anomaly_detection.common.post_processor import PostProcessor
    from visionlab.anomaly_detection.common.pre_processor import PreProcessor
    from visionlab.anomaly_detection.common.visualizer import ImageVisualizer
    from visionlab.anomaly_detection.models.ganomaly.lightning_model import GanomalyLitModel

    components = (PreProcessor(), PostProcessor(), Evaluator(), ImageVisualizer())
    model = GanomalyLitModel(n_features=4, latent_vec_size=16, pre_processor=components[0],
                            post_processor=components[1], evaluator=components[2], visualizer=components[3])
    assert (model.pre_processor, model.post_processor, model.evaluator, model.visualizer) == components
    assert isinstance(model.configure_callbacks()[0], LitVisualizer)
    disabled = GanomalyLitModel(n_features=4, latent_vec_size=16, pre_processor=False,
                               post_processor=False, evaluator=False, visualizer=False)
    assert disabled.pre_processor is disabled.post_processor is disabled.evaluator is disabled.visualizer is None
    assert disabled.configure_callbacks() == []
    for component_type in (PreProcessor, PostProcessor, Evaluator, ImageVisualizer):
        with pytest.raises(TypeError):
            model._resolve(None, component_type, lambda: None)
    model.min_scores, model.max_scores = torch.tensor(2.0), torch.tensor(6.0)
    scores = torch.tensor([1.0, 2.0, 4.0, 6.0, 7.0])
    expected = torch.tensor([-0.25, 0.0, 0.5, 1.0, 1.25])
    assert torch.equal(model._normalize(scores), expected)
    for hook in (model.on_validation_batch_end, model.on_test_batch_end):
        output = {"pred_score": scores.clone()}
        hook(output, None, 0)
        assert torch.equal(output["pred_score"], expected)
    for hook in (model.on_validation_start, model.on_test_start):
        model.min_scores, model.max_scores = torch.tensor(2.0), torch.tensor(6.0)
        hook()
        assert model.min_scores.item() == float("inf") and model.max_scores.item() == float("-inf")
    assert not model._fit_validation()
    model._trainer = SimpleNamespace(state=SimpleNamespace(fn=TrainerFn.FITTING), sanity_checking=False)
    assert model._fit_validation()
    model._trainer.sanity_checking = True
    assert not model._fit_validation()


@pytest.fixture(scope="module")
def fitted_ganomaly(tmp_path_factory):
    """Fit two epochs while recording the real optimizer order, uneven batch losses and calibration calls."""
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.loggers import CSVLogger

    seed_everything(42, workers=True)
    root = str(tmp_path_factory.mktemp("fitted_ganomaly"))
    data, model = make_ganomaly_data(), make_ganomaly_model()
    before = {name: value.detach().clone() for name, value in model.model.named_parameters()}
    counts, steps, events = {"update": 0, "finalize": 0}, [], []
    optimizers = model.configure_optimizers()
    d_step, g_step = optimizers[0].step, optimizers[1].step
    d_forward, d_loss = model.model.discriminator.forward, model.discriminator_loss.forward
    training_step = model.training_step
    update, finalize = model.post_processor.update, model.post_processor.finalize

    def capture_d_step(*args, **kwargs):
        events.append("D step")
        return d_step(*args, **kwargs)

    def capture_g_step(*args, **kwargs):
        events.append("G step")
        return g_step(*args, **kwargs)

    def capture_d_forward(image):
        events.append("D forward")
        return d_forward(image)

    def capture_d_loss(pred_real, pred_fake):
        events.append("D loss")
        return d_loss(pred_real, pred_fake)

    def capture_training_step(batch, batch_idx):
        start = len(events)
        result = training_step(batch, batch_idx)
        steps.append((model.current_epoch, len(batch[0]), {name: value.item() for name, value in result.items()},
                      events[start:]))
        return result

    def capture_update(outputs):
        counts["update"] += 1
        return update(outputs)

    def capture_finalize():
        counts["finalize"] += 1
        return finalize()

    trainer = Trainer(default_root_dir=root, accelerator="cpu", devices=1, max_epochs=2,
                      gradient_clip_val=0, num_sanity_val_steps=0,
                      logger=CSVLogger(save_dir=root, name="metrics"), log_every_n_steps=1,
                      enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                      deterministic=True)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(optimizers[0], "step", capture_d_step)
        patch.setattr(optimizers[1], "step", capture_g_step)
        patch.setattr(model, "configure_optimizers", lambda: optimizers)
        patch.setattr(model.model.discriminator, "forward", capture_d_forward)
        patch.setattr(model.discriminator_loss, "forward", capture_d_loss)
        patch.setattr(model, "training_step", capture_training_step)
        patch.setattr(model.post_processor, "update", capture_update)
        patch.setattr(model.post_processor, "finalize", capture_finalize)
        trainer.fit(model, datamodule=data)
    path = os.path.join(root, "model.ckpt")
    trainer.save_checkpoint(path)
    return root, model, data, path, trainer, before, counts, steps


def test_ganomaly_native_learning_optimizer_order_and_epoch_means(fitted_ganomaly):
    """Update both networks once per batch and average each epoch's losses by step rather than sample count."""
    root, model, data, _, trainer, before, counts, steps = fitted_ganomaly
    for prefix in ("generator.", "discriminator."):
        assert any(not torch.equal(value, before[name]) for name, value in model.model.named_parameters()
                   if name.startswith(prefix))
    assert len(trainer.optimizers) == 2 and not trainer.lr_scheduler_configs
    assert trainer.gradient_clip_val == 0 and model.automatic_optimization is False
    assert counts == {"update": 2 * len(data.val_dataloader()), "finalize": 2}
    assert all(torch.isfinite(getattr(model.post_processor, name))
               for name in ("image_min", "image_max", "image_threshold"))
    assert [size for _, size, _, _ in steps] == [2, 1, 2, 1]
    assert all(events == ["D forward", "D forward", "G step", "D forward", "D loss", "D step"]
               for _, _, _, events in steps)
    with open(os.path.join(root, "metrics", "version_0", "metrics.csv"), newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert all(name + suffix in reader.fieldnames for name in ("generator_loss", "discriminator_loss")
                   for suffix in ("_epoch", "_step"))
        rows = list(reader)
    for name in ("generator_loss", "discriminator_loss"):
        epochs = {int(row["epoch"]): float(row[name + "_epoch"]) for row in rows if row.get(name + "_epoch")}
        assert set(epochs) == {0, 1}
        for epoch, actual in epochs.items():
            values = [losses[name] for step_epoch, _, losses, _ in steps if step_epoch == epoch]
            assert actual == pytest.approx(sum(values) / len(values), rel=1e-5, abs=1e-6)


def test_ganomaly_native_checkpoint_frozen_processing_and_raw_metrics(fitted_ganomaly, tmp_path, monkeypatch):
    """Check saved optimizers, restore predictions and keep standalone raw metrics independent of batch hooks."""
    from lightning.pytorch import Trainer
    from torchmetrics.classification import BinaryAUROC
    from visionlab.anomaly_detection.models.ganomaly.lightning_model import GanomalyLitModel

    _, model, data, path, _, _, _, _ = fitted_ganomaly
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert len(saved["optimizer_states"]) == 2 and all(state["state"] for state in saved["optimizer_states"])
    assert not saved["lr_schedulers"] and saved["post_processor"] is not None
    assert any(name.startswith("model.generator.") for name in saved["state_dict"])
    assert any(name.startswith("model.discriminator.") for name in saved["state_dict"])
    restored = GanomalyLitModel.load_from_checkpoint(path, visualizer=False)
    model.eval()
    restored.eval()
    expected = []
    with torch.no_grad():
        for batch in data.val_dataloader():
            left, right = model.predict_step(batch, 0), restored.predict_step(batch, 0)
            for name in OUTPUT_NAMES:
                if left[name] is None:
                    assert right[name] is None
                else:
                    torch.testing.assert_close(left[name], right[name], rtol=1e-5, atol=1e-6)
            expected.append(right)
    before = restored.post_processor.state_dict()

    def forbidden(*args, **kwargs):
        raise AssertionError("Standalone evaluation must not update calibration")

    observed = {"val": [], "test": []}
    validation_step, test_step = restored.validation_step, restored.test_step

    def capture_validation(batch, batch_idx):
        output = validation_step(batch, batch_idx)
        observed["val"].append(output["pred_score"].clone())
        return output

    def capture_test(batch, batch_idx):
        output = test_step(batch, batch_idx)
        observed["test"].append(output["pred_score"].clone())
        return output

    normalized = []
    batch_end = restored.on_test_batch_end

    def capture_batch_end(outputs, batch, batch_idx, dataloader_idx=0):
        raw = outputs["pred_score"].clone()
        formula = restored._normalize(raw)
        batch_end(outputs, batch, batch_idx, dataloader_idx)
        torch.testing.assert_close(outputs["pred_score"], formula, equal_nan=True)
        normalized.append(outputs["pred_score"].clone())

    monkeypatch.setattr(restored.post_processor, "update", forbidden)
    monkeypatch.setattr(restored.post_processor, "finalize", forbidden)
    monkeypatch.setattr(restored, "validation_step", capture_validation)
    monkeypatch.setattr(restored, "test_step", capture_test)
    monkeypatch.setattr(restored, "on_test_batch_end", capture_batch_end)
    trainer = Trainer(default_root_dir=str(tmp_path), accelerator="cpu", devices=1, logger=False,
                      enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
    assert "val_image_AUROC" in trainer.validate(restored, datamodule=data, ckpt_path=path)[0]
    metrics = trainer.test(restored, datamodule=data, ckpt_path=path)[0]
    labels = torch.cat([batch[1]["label"] for batch in data.test_dataloader()])
    raw = torch.cat([output["pred_score"] for output in expected])
    assert metrics["image_AUROC"] == pytest.approx(BinaryAUROC()(raw, labels).item(), abs=1e-6)
    assert set(metrics) == {"image_AUROC", "image_F1Score"}
    for stage in ("val", "test"):
        for actual, output in zip(observed[stage], expected, strict=True):
            torch.testing.assert_close(actual, output["pred_score"], rtol=1e-5, atol=1e-6)
    assert any(not torch.allclose(left, right) for left, right in zip(observed["test"], normalized, strict=True))
    for name, value in restored.post_processor.state_dict().items():
        torch.testing.assert_close(value, before[name], equal_nan=True)


def consume_ganomaly_export(root):
    """Restore both networks from the export under strict file guards without Lightning or reference imports."""
    from visionlab.inference.predictor import Predictor
    from visionlab.anomaly_detection.models.ganomaly.model import GanomalyModel

    io_utils.block_network()
    path = os.path.join(root, "export", "model.pt")
    expected = torch.load(os.path.join(root, "expected.pt"), map_location="cpu", weights_only=True)
    exported = io_utils.load_export(path)
    assert exported["model_name"] == "ganomaly"
    assert exported["model_args"] == {"input_size": [256, 256], "num_input_channels": 3, "n_features": 4,
                                      "latent_vec_size": 16, "extra_layers": 0, "add_final_conv_layer": True}
    assert all(type(value) in (int, bool, list) for value in exported["model_args"].values())
    assert all(type(value) is int for value in exported["model_args"]["input_size"])
    assert all(any(name.startswith(prefix) for name in exported["model_state"])
               for prefix in ("generator.", "discriminator."))

    def audit_open(event, args):
        if event == "open":
            opened, mode, flags = args
            assert isinstance(opened, (str, bytes)) and os.path.abspath(opened) == path, opened
            assert not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)

    sys.addaudithook(audit_open)
    predictor = Predictor.from_export(path)
    assert isinstance(predictor.model, GanomalyModel)
    assert predictor.model.generator.encoder1.final_conv_layer.in_channels == \
        predictor.model.generator.decoder.latent_input[0].out_channels == exported["model_args"]["n_features"] * 2**5 == 128
    for name, value in predictor.model.state_dict().items():
        assert torch.equal(value, exported["model_state"][name]), name
    for batch in expected["batches"]:
        actual = predictor.predict(batch["images"])
        assert actual["pred_score"].shape == (len(batch["images"]),) and actual["anomaly_map"] is None
        for name in OUTPUT_NAMES:
            if batch["outputs"][name] is None:
                assert actual[name] is None
            else:
                torch.testing.assert_close(actual[name], batch["outputs"][name], rtol=1e-5, atol=1e-6)


def test_ganomaly_native_isolated_export(fitted_ganomaly):
    """Match exported inference with native raw predictions while rejecting frameworks, assets and FrEIA."""
    root, model, data, _, _, _, _, _ = fitted_ganomaly
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
blocked = {"anomalib", "lightning", "pytorch_lightning", "FrEIA"}
class Block(importlib.abc.MetaPathFinder):
    """Reject training and reference frameworks before importing the pure consumer."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, Block())
runpy.run_path(sys.argv[1])["consume_ganomaly_export"](sys.argv[2])
assert not blocked & {name.split(".")[0] for name in sys.modules}
'''
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.path.join(PROJECT_ROOT, "src")
    result = subprocess.run([sys.executable, "-c", script, os.path.abspath(__file__), root],
                            env=environment, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
