# tests/17_dsr/test_dsr_lightning.py: pytest native DSR manual stages, VQ injection, checkpoints and pure exports
"""Use default-width DSR with 64x64 CPU fixtures, batches of at most two and test-only CUDA transfer patches."""

import csv
import inspect
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from visionlab.anomaly_detection.models.dsr.model import DsrModel
from visionlab.utils import io as io_utils

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_NAMES = ("pred_score", "anomaly_map", "normalized_pred_score", "normalized_anomaly_map", "pred_label", "pred_mask")
SECOND_STAGE_GROUPS = ("image_reconstruction_network", "subspace_restriction_module_hi",
                       "subspace_restriction_module_lo", "anomaly_detection_module")


def make_dsr_model(upsampling_train_ratio=0.7):
    """Construct a native default-width model using the fixture's local VQ mapping."""
    from visionlab.anomaly_detection.models.dsr.lightning_model import DsrLitModel

    return DsrLitModel(upsampling_train_ratio=upsampling_train_ratio, visualizer=False)


def make_dsr_data():
    """Provide three fixed RGB samples with uneven batches and both image and pixel labels."""
    from lightning.pytorch import LightningDataModule

    class TinyData(LightningDataModule):
        """Supply 64-pixel test fixtures while production datasets retain their 256-pixel resize."""

        def __init__(self):
            super().__init__()
            generator = torch.Generator().manual_seed(17)
            self.train_data = [(torch.rand(3, 64, 64, generator=generator) * 0.02 + level, {"label": 0})
                               for level in (0.1, 0.2, 0.8)]
            self.val_data = []
            for index in range(3):
                mask = torch.zeros(64, 64, dtype=torch.int64)
                if index % 2:
                    mask[16:40, 16:40] = 1
                self.val_data.append((torch.rand(3, 64, 64, generator=generator), {
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
def dsr_assets(tmp_path_factory):
    """Save a complete synthetic discrete state including EMA buffers and map its teacher key locally."""
    root = str(tmp_path_factory.mktemp("dsr_assets"))
    path = os.path.join(root, "vq.pckl")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        pure = DsrModel()
        pure.discrete_latent_model._vq_vae_top._ema_cluster_size.fill_(3)
        pure.discrete_latent_model._vq_vae_bot._ema_cluster_size.fill_(7)
        torch.save(pure.discrete_latent_model.state_dict(), path)
        del pure
    mapping = os.path.join(root, "paths.toml")
    with open(mapping, "w", encoding="utf-8", newline="\n") as stream:
        stream.write('[teachers]\ndsr_vq = "' + path.replace("\\", "/") + '"\n')
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(io_utils, "PATHS_FILE", mapping)
        yield root


def test_dsr_native_signature_teacher_and_components(dsr_assets):
    """Keep six defaults, strict VQ injection, frozen parameters and identity image preprocessing."""
    from visionlab.anomaly_detection.models.dsr.lightning_model import DsrLitModel

    expected = {"latent_anomaly_strength": 0.2, "upsampling_train_ratio": 0.7,
                "pre_processor": True, "post_processor": True, "evaluator": True, "visualizer": True}
    parameters = inspect.signature(DsrLitModel).parameters
    assert len(parameters) == 6 and list(parameters) == list(expected)
    assert {name: value.default for name, value in parameters.items()} == expected
    assert all(value.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD for value in parameters.values())
    model = make_dsr_model()
    path = os.path.join(dsr_assets, "vq.pckl")
    source = torch.load(path, map_location="cpu", weights_only=True)
    actual = model.model.discrete_latent_model.state_dict()
    assert len(actual) == 40 and actual.keys() == source.keys()
    assert all(torch.equal(value, source[name]) for name, value in actual.items())
    assert model.teacher_weights_path == path
    assert all(not value.requires_grad for value in model.model.discrete_latent_model.parameters())
    assert model.pre_processor.mean is model.pre_processor.std is model.pre_processor.center_crop is None
    image = torch.rand(2, 3, 64, 64)
    assert model.pre_processor(image) is image
    assert not hasattr(model, "second_phase")
    assert model.learning_type.value == "one_class" and model.configure_callbacks() == []
    assert [entry[0] for entry in model.evaluator.val_metrics] == ["image_AUROC", "pixel_AUROC"]
    assert [entry[0] for entry in model.evaluator.test_metrics] == [
        "image_AUROC", "pixel_AUROC", "image_F1Score", "pixel_F1Score",
    ]


def test_dsr_native_missing_and_incompatible_teacher(tmp_path, monkeypatch):
    """Reject absent mappings, missing teacher keys or files and invalid strict state dictionaries."""
    mapping = os.path.join(tmp_path, "paths.toml")
    monkeypatch.setattr(io_utils, "PATHS_FILE", mapping)
    with pytest.raises(FileNotFoundError):
        make_dsr_model()
    for content in ('[teachers]\n', '[teachers]\ndsr_vq = "missing.pckl"\n'):
        with open(mapping, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        with pytest.raises(FileNotFoundError):
            make_dsr_model()
    path = os.path.join(tmp_path, "invalid.pckl")
    torch.save({"wrong_key": torch.ones(1)}, path)
    with open(mapping, "w", encoding="utf-8", newline="\n") as stream:
        stream.write('[teachers]\ndsr_vq = "' + path.replace("\\", "/") + '"\n')
    with pytest.raises(RuntimeError, match="state_dict"):
        make_dsr_model()


def test_dsr_native_seed_and_default_perlin_order(dsr_assets):
    """Consume the default Perlin constructor RNG before model creation and inject VQ without random draws."""
    from visionlab.anomaly_detection.common.perlin import PerlinAnomalyGenerator
    from visionlab.anomaly_detection.models.dsr.anomaly_generator import DsrAnomalyGenerator

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        DsrAnomalyGenerator()
        perlin = PerlinAnomalyGenerator()
        pure = DsrModel(0.2)
        following = torch.get_rng_state()
        source = torch.load(os.path.join(dsr_assets, "vq.pckl"), map_location="cpu", weights_only=True)
        pure.discrete_latent_model.load_state_dict(source, strict=True)
        torch.manual_seed(42)
        model = make_dsr_model()
        assert torch.equal(torch.get_rng_state(), following)
    assert all(torch.equal(value, pure.state_dict()[name]) for name, value in model.model.state_dict().items())
    assert model.perlin_generator.probability == perlin.probability == 0.5
    assert model.perlin_generator.blend_factor == perlin.blend_factor == (0.2, 1.0)
    assert model.perlin_generator.anomaly_source_paths == []
    assert model.perlin_generator.augmenters.transforms[4].threshold == perlin.augmenters.transforms[4].threshold


@pytest.mark.parametrize("epochs,ratio,max_steps,batches,second_phase", (
    (10, 0.7, -1, 2, 7), (2, 0.5, -1, 2, 1), (5, 0.7, -1, 3, 3), (10, 0.7, 120, 6, 14),
))
def test_dsr_native_optimizers_and_stage_calculation(dsr_assets, epochs, ratio, max_steps, batches, second_phase):
    """Retain original epoch arithmetic, hi-before-lo Adam ordering and the unstepped StepLR configuration."""
    model = make_dsr_model(upsampling_train_ratio=ratio)
    model._trainer = SimpleNamespace(max_steps=max_steps, max_epochs=epochs,
                                     datamodule=SimpleNamespace(train_dataloader=lambda: [None] * batches))
    configuration = model.configure_optimizers()
    assert model.automatic_optimization is False and isinstance(configuration, tuple) and len(configuration) == 2
    first, second = configuration
    assert set(first) == {"optimizer", "lr_scheduler"} and set(second) == {"optimizer"}
    expected_groups = [value for name in SECOND_STAGE_GROUPS for value in getattr(model.model, name).parameters()]
    for optimizer, expected in ((first["optimizer"], expected_groups),
                                (second["optimizer"], list(model.model.upsampling_module.parameters()))):
        assert isinstance(optimizer, torch.optim.Adam) and len(optimizer.param_groups) == 1
        group = optimizer.param_groups[0]
        assert group["lr"] == 2e-4 and group["betas"] == (0.9, 0.999) and group["weight_decay"] == 0
        assert [id(value) for value in group["params"]] == [id(value) for value in expected]
    scheduler = first["lr_scheduler"]
    assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
    assert scheduler.optimizer is first["optimizer"] and scheduler.gamma == 0.1
    assert model.second_phase == second_phase and scheduler.step_size == int(0.8 * second_phase)
    assert model.trainer_arguments == {"num_sanity_val_steps": 0}
    model._trainer = None


def test_dsr_native_component_flags_and_callbacks(dsr_assets):
    """Accept shared components, reject invalid values and configure the standard visualization callback."""
    from visionlab.anomaly_detection.common.evaluator import Evaluator
    from visionlab.anomaly_detection.common.lightning_visualizer import LitVisualizer
    from visionlab.anomaly_detection.common.post_processor import PostProcessor
    from visionlab.anomaly_detection.common.pre_processor import PreProcessor
    from visionlab.anomaly_detection.common.visualizer import ImageVisualizer
    from visionlab.anomaly_detection.models.dsr.lightning_model import DsrLitModel

    components = (PreProcessor(), PostProcessor(), Evaluator.default(), ImageVisualizer())
    model = DsrLitModel(pre_processor=components[0], post_processor=components[1],
                        evaluator=components[2], visualizer=components[3])
    assert (model.pre_processor, model.post_processor, model.evaluator, model.visualizer) == components
    assert isinstance(model.configure_callbacks()[0], LitVisualizer)
    del model
    disabled = DsrLitModel(pre_processor=False, post_processor=False, evaluator=False, visualizer=False)
    assert disabled.pre_processor is disabled.post_processor is disabled.evaluator is disabled.visualizer is None
    assert disabled.configure_callbacks() == []
    for cls in (PreProcessor, PostProcessor, Evaluator, ImageVisualizer):
        with pytest.raises(TypeError):
            disabled._resolve(None, cls, lambda: None)


@pytest.fixture(scope="module")
def fitted_dsr(tmp_path_factory, dsr_assets):
    """Fit two tiny epochs with deterministic nonempty anomaly masks and observe actual optimizer steps."""
    from lightning.pytorch import Callback, Trainer, seed_everything
    from lightning.pytorch.loggers import CSVLogger
    from visionlab.anomaly_detection.models.dsr import lightning_model as native_module

    class ObserveStages(Callback):
        """Compare complete group states across each epoch, including frozen VQ buffers."""

        def on_train_epoch_start(self, trainer, pl_module):
            self.before = {name: {key: value.detach().clone() for key, value in group.state_dict().items()}
                           for name, group in pl_module.model.named_children()}

        def on_train_epoch_end(self, trainer, pl_module):
            changes[pl_module.current_epoch] = {
                name: any(not torch.equal(value, self.before[name][key]) for key, value in group.state_dict().items())
                for name, group in pl_module.model.named_children()
            }
            self.before = None

    seed_everything(42, workers=True)
    root = str(tmp_path_factory.mktemp("fitted_dsr"))
    data, model = make_dsr_data(), make_dsr_model(upsampling_train_ratio=0.5)
    counts, steps, events, changes, messages, logs = {"update": 0, "finalize": 0}, [], [], {}, [], []
    model._trainer = SimpleNamespace(max_steps=-1, max_epochs=2, datamodule=data)
    configuration = model.configure_optimizers()
    model._trainer = None
    optimizers = [entry["optimizer"] for entry in configuration]
    scheduler_before = configuration[0]["lr_scheduler"].state_dict()
    original_steps = [optimizer.step for optimizer in optimizers]
    update, finalize, training_step, log = (model.post_processor.update, model.post_processor.finalize,
                                           model.training_step, model.log)
    logger_info = native_module.logger.info

    def first_step(*args, **kwargs):
        events.append((model.current_epoch, "ph1"))
        return original_steps[0](*args, **kwargs)

    def second_step(*args, **kwargs):
        events.append((model.current_epoch, "ph2"))
        return original_steps[1](*args, **kwargs)

    def masks_for(image):
        mask = torch.zeros(len(image), 1, 64, 64, device=image.device)
        mask[:, :, 16:40, 16:40] = 1
        return mask

    def augment(image):
        mask = masks_for(image)
        return image * (1 - mask) + 0.9 * mask, mask

    def capture_step(batch, batch_idx):
        result = training_step(batch, batch_idx)
        steps.append((model.current_epoch, len(batch[0]), result["loss"].item()))
        return result

    def capture_update(outputs):
        counts["update"] += 1
        return update(outputs)

    def capture_finalize():
        counts["finalize"] += 1
        return finalize()

    def capture_log(name, value, **options):
        if name == "train_loss":
            logs.append((value, options))
        return log(name, value, **options)

    def capture_message(message):
        messages.append((model.current_epoch, message))
        return logger_info(message)

    trainer = Trainer(default_root_dir=root, accelerator="cpu", devices=1, max_epochs=2,
                      num_sanity_val_steps=0, callbacks=[ObserveStages()], limit_train_batches=2, limit_val_batches=2,
                      logger=CSVLogger(save_dir=root, name="metrics"), log_every_n_steps=1,
                      enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.Tensor, "cuda", lambda self: self)
        patch.setattr(torch.cuda, "is_available", lambda: False)
        patch.setattr(optimizers[0], "step", first_step)
        patch.setattr(optimizers[1], "step", second_step)
        patch.setattr(model, "configure_optimizers", lambda: configuration)
        patch.setattr(model.quantized_anomaly_generator, "augment_batch", masks_for)
        patch.setattr(model.perlin_generator, "forward", augment)
        patch.setattr(model, "training_step", capture_step)
        patch.setattr(model, "log", capture_log)
        patch.setattr(model.post_processor, "update", capture_update)
        patch.setattr(model.post_processor, "finalize", capture_finalize)
        patch.setattr(native_module.logger, "info", capture_message)
        trainer.fit(model, datamodule=data)
    path = os.path.join(root, "model.ckpt")
    trainer.save_checkpoint(path)
    return {"root": root, "model": model, "data": data, "trainer": trainer, "path": path,
            "counts": counts, "steps": steps, "events": events, "changes": changes, "messages": messages,
            "logs": logs, "scheduler_before": scheduler_before}


def test_dsr_native_learning_stage_weights_and_epoch_mean(fitted_dsr):
    """Step only the active optimizer, update its groups and log the unweighted mean on uneven batches."""
    fitted = fitted_dsr
    assert fitted["events"] == [(0, "ph1"), (0, "ph1"), (1, "ph2"), (1, "ph2")]
    for epoch, changes in fitted["changes"].items():
        expected = set(SECOND_STAGE_GROUPS) if epoch == 0 else {"upsampling_module"}
        assert {name for name, changed in changes.items() if changed} == expected
    assert fitted["messages"] == [(1, "Now training upsampling module.")]
    assert fitted["counts"] == {"update": 4, "finalize": 2}
    assert fitted["model"].post_processor.is_calibrated
    assert fitted["model"].second_phase == 1
    trainer = fitted["trainer"]
    assert len(trainer.optimizers) == 2 and len(trainer.lr_scheduler_configs) == 1
    assert trainer.lr_scheduler_configs[0].scheduler.state_dict() == fitted["scheduler_before"]
    assert all(group["lr"] == 2e-4 for optimizer in trainer.optimizers for group in optimizer.param_groups)
    assert len(fitted["logs"]) == 4
    assert all(options == {"on_epoch": True, "prog_bar": True, "logger": True, "batch_size": 1}
               for _, options in fitted["logs"])
    with open(os.path.join(fitted["root"], "metrics", "version_0", "metrics.csv"), newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert "train_loss_epoch" in reader.fieldnames and "train_loss_step" in reader.fieldnames
        epochs = {int(row["epoch"]): float(row["train_loss_epoch"]) for row in reader if row.get("train_loss_epoch")}
    assert set(epochs) == {0, 1}
    for epoch, actual in epochs.items():
        values = [loss for step_epoch, _, loss in fitted["steps"] if step_epoch == epoch]
        assert actual == pytest.approx(sum(values) / len(values), rel=1e-5, abs=1e-6)


def test_dsr_native_checkpoint_and_frozen_evaluation(fitted_dsr, tmp_path, monkeypatch):
    """Restore both optimizer states and processing state without recalibrating standalone validation or test."""
    from lightning.pytorch import Trainer
    from visionlab.anomaly_detection.models.dsr.lightning_model import DsrLitModel

    fitted = fitted_dsr
    saved = torch.load(fitted["path"], map_location="cpu", weights_only=False)
    assert len(saved["optimizer_states"]) == 2 and all(state["state"] for state in saved["optimizer_states"])
    assert len(saved["lr_schedulers"]) == 1 and saved["lr_schedulers"][0] == fitted["scheduler_before"]
    assert "second_phase" not in saved and "second_phase" not in saved["hyper_parameters"]
    assert any(name.startswith("model.discrete_latent_model.") for name in saved["state_dict"])
    restored = DsrLitModel.load_from_checkpoint(fitted["path"], visualizer=False)
    assert not hasattr(restored, "second_phase")
    restored._trainer = SimpleNamespace(max_steps=-1, max_epochs=2, datamodule=fitted["data"])
    configuration = restored.configure_optimizers()
    restored._trainer = None
    for entry, state in zip(configuration, saved["optimizer_states"], strict=True):
        entry["optimizer"].load_state_dict(state)
        actual = entry["optimizer"].state_dict()
        assert actual["param_groups"] == state["param_groups"] and actual["state"].keys() == state["state"].keys()
        for key, values in state["state"].items():
            assert all(torch.equal(actual["state"][key][name], value) for name, value in values.items())
    configuration[0]["lr_scheduler"].load_state_dict(saved["lr_schedulers"][0])
    assert configuration[0]["lr_scheduler"].state_dict() == saved["lr_schedulers"][0]
    del configuration, saved
    fitted["model"].eval()
    restored.eval()
    batches = list(fitted["data"].val_dataloader())
    with torch.no_grad():
        expected = [fitted["model"].predict_step(batch, 0) for batch in batches]
        actual = [restored.predict_step(batch, 0) for batch in batches]
    for left, right in zip(expected, actual, strict=True):
        for name in OUTPUT_NAMES:
            torch.testing.assert_close(left[name], right[name], rtol=1e-5, atol=1e-6)
    assert actual[-1]["pred_score"].shape == (1,) and actual[-1]["anomaly_map"].shape == (1, 64, 64)
    before = restored.post_processor.state_dict()

    def forbidden(*args, **kwargs):
        raise AssertionError("Standalone evaluation must not update calibration")

    observed = {"val": [], "test": []}
    validation_step, test_step = restored.validation_step, restored.test_step

    def capture_validation(batch, batch_idx):
        output = validation_step(batch, batch_idx)
        observed["val"].append(output)
        return output

    def capture_test(batch, batch_idx):
        output = test_step(batch, batch_idx)
        observed["test"].append(output)
        return output

    monkeypatch.setattr(restored.post_processor, "update", forbidden)
    monkeypatch.setattr(restored.post_processor, "finalize", forbidden)
    monkeypatch.setattr(restored, "validation_step", capture_validation)
    monkeypatch.setattr(restored, "test_step", capture_test)
    trainer = Trainer(default_root_dir=str(tmp_path), accelerator="cpu", devices=1, logger=False,
                      enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
    assert "val_image_AUROC" in trainer.validate(restored, datamodule=fitted["data"], ckpt_path=fitted["path"])[0]
    assert "image_AUROC" in trainer.test(restored, datamodule=fitted["data"], ckpt_path=fitted["path"])[0]
    for stage in ("val", "test"):
        assert len(observed[stage]) == len(expected)
        for left, right in zip(expected, observed[stage], strict=True):
            for name in (("pred_score", "anomaly_map") if stage == "val" else OUTPUT_NAMES):
                torch.testing.assert_close(left[name], right[name], rtol=1e-5, atol=1e-6)
    for name, value in restored.post_processor.state_dict().items():
        torch.testing.assert_close(value, before[name], equal_nan=True)


def consume_dsr_export(root):
    """Restore all pure model state while allowing only the export file after imports finish."""
    from visionlab.inference.predictor import Predictor

    io_utils.block_network()
    path = os.path.join(root, "export", "model.pt")
    expected = torch.load(os.path.join(root, "expected.pt"), map_location="cpu", weights_only=True)
    exported = io_utils.load_export(path)
    assert exported["model_name"] == "dsr" and exported["model_args"] == {"latent_anomaly_strength": 0.2}
    assert all(type(value) is float for value in exported["model_args"].values())
    assert "discrete_latent_model._vq_vae_top._ema_cluster_size" in exported["model_state"]
    assert "discrete_latent_model._vq_vae_bot._ema_cluster_size" in exported["model_state"]

    def audit_open(event, args):
        if event == "open":
            opened, mode, flags = args
            assert isinstance(opened, (str, bytes)) and os.path.abspath(opened) == path, opened
            assert not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)

    sys.addaudithook(audit_open)
    predictor = Predictor.from_export(path)
    assert isinstance(predictor.model, DsrModel)
    assert predictor.pre_processor.mean is predictor.pre_processor.std is None
    assert predictor.model.state_dict().keys() == exported["model_state"].keys()
    for name, value in predictor.model.state_dict().items():
        assert torch.equal(value, exported["model_state"][name]), name
    for batch in expected["batches"]:
        actual = predictor.predict(batch["images"])
        assert actual["pred_score"].shape == (len(batch["images"]),)
        assert actual["anomaly_map"].shape == (len(batch["images"]), 64, 64)
        for name in OUTPUT_NAMES:
            torch.testing.assert_close(actual[name], batch["outputs"][name], rtol=1e-5, atol=1e-6)


def test_dsr_native_isolated_export(fitted_dsr):
    """Match native predictions without training frameworks or asset reads; retain Predictor's existing kornia imports."""
    fitted = fitted_dsr
    fitted["model"].eval()
    batches = []
    for batch in fitted["data"].val_dataloader():
        with torch.no_grad():
            outputs = fitted["model"].predict_step(batch, 0)
        batches.append({"images": batch[0], "outputs": {name: outputs[name] for name in OUTPUT_NAMES}})
    torch.save({"batches": batches}, os.path.join(fitted["root"], "expected.pt"))
    script = '''
import importlib.abc
import runpy
import sys
blocked = {"anomalib", "lightning", "pytorch_lightning", "FrEIA"}
class Block(importlib.abc.MetaPathFinder):
    """Reject training frameworks before importing the pure consumer."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, Block())
runpy.run_path(sys.argv[1])["consume_dsr_export"](sys.argv[2])
assert not blocked & {name.split(".")[0] for name in sys.modules}
'''
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.path.join(PROJECT_ROOT, "src")
    result = subprocess.run([sys.executable, "-c", script, os.path.abspath(__file__), fitted["root"]],
                            env=environment, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
