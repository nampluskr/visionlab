# tests/02_efficient_ad/test_efficient_ad.py: pytest pure EfficientAD shapes, state and dependency boundaries

import os
import subprocess
import sys

import pytest
import torch

from visionlab.anomaly_detection.common.batch import InferenceBatch
from visionlab.anomaly_detection.models.efficient_ad import EfficientAdModel, EfficientAdModelSize

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module", params=(EfficientAdModelSize.S, EfficientAdModelSize.M))
def model_size(request):
    """Cover both original network sizes."""
    return request.param


@pytest.fixture(scope="module", params=(False, True))
def model(model_size, request):
    """Construct the original 384-channel model with both padding settings."""
    if request.param:
        return EfficientAdModel(teacher_out_channels=384, model_size=model_size, padding=True)
    return EfficientAdModel(teacher_out_channels=384, model_size=model_size)


def test_eval_shapes(model):
    """Preserve FR-4 score and map shapes with default map padding."""
    model.eval()
    assert model.pad_maps is True
    with torch.no_grad():
        prediction = model(torch.rand(2, 3, 256, 256))
    assert isinstance(prediction, InferenceBatch)
    assert prediction.pred_score.shape == (2, 1)
    assert prediction.anomaly_map.shape == (2, 1, 256, 256)
    assert torch.isfinite(prediction.anomaly_map).all()
    torch.testing.assert_close(prediction.pred_score, prediction.anomaly_map.amax(dim=(-2, -1)))


def test_training_losses(model):
    """Return three finite differentiable scalar losses for training batch one."""
    model.train()
    losses = model(batch=torch.rand(1, 3, 256, 256), batch_imagenet=torch.rand(1, 3, 256, 256))
    assert isinstance(losses, tuple) and len(losses) == 3
    loss_st, loss_ae, loss_stae = losses
    for loss in (loss_st, loss_ae, loss_stae):
        assert loss.ndim == 0 and torch.isfinite(loss) and loss.requires_grad


def test_is_set(model):
    """Distinguish initial zero statistics from populated parameter dictionaries."""
    assert not model.is_set(model.mean_std)
    assert not model.is_set(model.quantiles)
    with torch.no_grad():
        model.mean_std["std"].fill_(1)
        model.quantiles["qb_st"].fill_(1)
    try:
        assert model.is_set(model.mean_std)
        assert model.is_set(model.quantiles)
    finally:
        with torch.no_grad():
            model.mean_std["std"].zero_()
            model.quantiles["qb_st"].zero_()


def constructor_checks():
    """Check constructor I/O, complete strict restoration and forward in a fresh consumer."""
    def forbidden(*args, **kwargs):
        raise AssertionError("Forbidden constructor file or network access")

    for model_size in EfficientAdModelSize:
        with pytest.MonkeyPatch.context() as patch:
            for target in ("builtins.open", "io.open", "os.open", "torch.load", "socket.socket.connect",
                           "socket.socket.connect_ex", "socket.create_connection"):
                patch.setattr(target, forbidden)
            original = EfficientAdModel(teacher_out_channels=384, model_size=model_size).eval()
            assert not original.is_set(original.mean_std)
            assert not original.is_set(original.quantiles)
            with torch.no_grad():
                original.mean_std["mean"].fill_(0.25)
                original.mean_std["std"].fill_(1.5)
                original.quantiles["qa_st"].fill_(0.1)
                original.quantiles["qa_ae"].fill_(0.2)
                original.quantiles["qb_st"].fill_(0.9)
                original.quantiles["qb_ae"].fill_(0.8)
            state = original.state_dict()
            for prefix in ("teacher.", "student.", "ae.", "mean_std.", "quantiles."):
                assert any(name.startswith(prefix) for name in state), prefix
            restored = EfficientAdModel(teacher_out_channels=384, model_size=model_size).eval()
            result = restored.load_state_dict(state, strict=True)
            assert result.missing_keys == result.unexpected_keys == []
            assert all(torch.equal(value, state[name]) for name, value in restored.state_dict().items())
        batch = torch.rand(1, 3, 256, 256)
        with torch.no_grad():
            expected = original(batch)
            actual = restored(batch)
        assert actual.pred_score.shape == (1, 1)
        assert actual.anomaly_map.shape == (1, 1, 256, 256)
        torch.testing.assert_close(actual.pred_score, expected.pred_score)
        torch.testing.assert_close(actual.anomaly_map, expected.anomaly_map)
        del original, restored, state


def test_constructor_state_and_forward_without_frameworks():
    """Check FR-22 and NFR-1/NFR-2 with blocked imports before model import."""
    script = '''
import importlib.abc
import runpy
import sys
class Block(importlib.abc.MetaPathFinder):
    """Reject reference and training dependencies in a fresh interpreter."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("anomalib", "lightning", "pytorch_lightning"):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, Block())
sys.path.insert(0, sys.argv[2])
from visionlab.anomaly_detection.models.efficient_ad.model import EfficientAdModel
runpy.run_path(sys.argv[1])["constructor_checks"]()
assert not {"anomalib", "lightning", "pytorch_lightning"} & set(sys.modules)
'''
    result = subprocess.run([sys.executable, "-c", script, os.path.abspath(__file__), os.path.join(PROJECT_ROOT, "src")],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_original_headers():
    """Keep NFR-5 copyright and SPDX headers in the model package."""
    root = os.path.join(PROJECT_ROOT, "src", "visionlab", "anomaly_detection", "models", "efficient_ad")
    for name in ("model.py", "__init__.py", "lightning_model.py"):
        with open(os.path.join(root, name), encoding="utf-8") as stream:
            source = stream.read()
        assert "# Copyright (C) 2023-2024 Intel Corporation" in source
        assert "# SPDX-License-Identifier: Apache-2.0" in source


def make_efficient_assets(root, teacher_out_channels=8):
    """Write a temporary teacher mapping and a two-image Imagenette class."""
    from PIL import Image

    teacher = EfficientAdModel(teacher_out_channels=teacher_out_channels).teacher
    teacher_file = os.path.join(root, "teacher.pth")
    torch.save(teacher.state_dict(), teacher_file)
    paths_file = os.path.join(root, "paths.toml")
    with open(paths_file, "w", encoding="utf-8", newline="\n") as stream:
        stream.write('[teachers]\nefficient_ad_small = "' + teacher_file.replace("\\", "/") + '"\n')
    imagenet_dir = os.path.join(root, "imagenette")
    os.makedirs(os.path.join(imagenet_dir, "class_a"))
    for index in range(2):
        with Image.new("RGB", (24, 24), color=(50 + index * 40, 80, 120)) as image:
            image.save(os.path.join(imagenet_dir, "class_a", f"{index}.png"))
    return paths_file, imagenet_dir, teacher_file


class TinyImages(torch.utils.data.Dataset):
    """Provide two 256-pixel samples with labels and masks for native integration."""

    def __init__(self, training=False):
        self.images = torch.rand(2, 3, 256, 256)
        self.training = training

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        label = 0 if self.training else index
        mask = torch.zeros(256, 256, dtype=torch.int64)
        if label:
            mask[32:64, 32:64] = 1
        return self.images[index], {"label": label, "mask": mask,
                                    "image_path": f"{index}.png", "sample_id": f"fixture/{index}.png"}


def make_tiny_data():
    """Build a native batch-one DataModule without importing Lightning in pure consumers."""
    from lightning.pytorch import LightningDataModule

    class TinyData(LightningDataModule):
        """Supply fixed synthetic train, validation, test and predict loaders."""

        def __init__(self):
            super().__init__()
            self.train_batch_size = 1
            self.train_data = TinyImages(training=True)
            self.val_data = TinyImages()

        def train_dataloader(self):
            return torch.utils.data.DataLoader(self.train_data, batch_size=self.train_batch_size, num_workers=0)

        def val_dataloader(self):
            return torch.utils.data.DataLoader(self.val_data, batch_size=2, num_workers=0)

        def test_dataloader(self):
            return self.val_dataloader()

        def predict_dataloader(self):
            return self.val_dataloader()

    return TinyData()


@pytest.fixture(scope="module")
def efficient_assets(tmp_path_factory):
    """Share local teacher weights while keeping machine paths inaccessible to tests."""
    from visionlab.utils import io as io_utils

    root = str(tmp_path_factory.mktemp("efficient_assets"))
    paths_file, imagenet_dir, teacher_file = make_efficient_assets(root)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(io_utils, "PATHS_FILE", paths_file)
        yield root, imagenet_dir, teacher_file


@pytest.fixture
def native_model(efficient_assets):
    """Construct the actual Lightning wrapper with a smaller teacher output width."""
    from visionlab.anomaly_detection.models.efficient_ad.lightning_model import EfficientAdLitModel

    return EfficientAdLitModel(imagenet_dir=efficient_assets[1], teacher_out_channels=8, visualizer=False)


def attach_trainer(model, data, fitting=True):
    """Attach only the trainer fields consumed by individual hook checks."""
    from types import SimpleNamespace
    from lightning.pytorch.trainer.states import TrainerFn

    model._trainer = SimpleNamespace(datamodule=data, train_dataloader=data.train_dataloader(),
                                     state=SimpleNamespace(fn=TrainerFn.FITTING if fitting else TrainerFn.VALIDATING),
                                     sanity_checking=False, max_epochs=10, max_steps=-1)


def test_native_signature_teacher_and_missing_assets(native_model, efficient_assets, tmp_path, monkeypatch):
    """Preserve eleven constructor defaults and inject only exact required teacher weights."""
    import inspect
    from visionlab.anomaly_detection.models.efficient_ad.lightning_model import EfficientAdLitModel
    from visionlab.utils import io as io_utils

    expected = {"imagenet_dir": "./datasets/imagenette", "teacher_out_channels": 384,
                "model_size": EfficientAdModelSize.S, "lr": 0.0001, "weight_decay": 0.00001,
                "padding": False, "pad_maps": True, "pre_processor": True, "post_processor": True,
                "evaluator": True, "visualizer": True}
    signature = inspect.signature(EfficientAdLitModel.__init__)
    assert list(signature.parameters) == ["self"] + list(expected)
    assert {name: signature.parameters[name].default for name in expected} == expected
    assert native_model.pre_processor.mean is native_model.pre_processor.std is None
    assert native_model.batch_size == 1 and native_model.model_size == EfficientAdModelSize.S
    source = torch.load(efficient_assets[2], map_location="cpu", weights_only=True)
    assert native_model.teacher_weights_path == efficient_assets[2]
    for name, value in native_model.model.teacher.state_dict().items():
        assert torch.equal(value, source[name])
    mapping = os.path.join(tmp_path, "paths.toml")
    monkeypatch.setattr(io_utils, "PATHS_FILE", mapping)
    for contents in ("[teachers]\n", '[teachers]\nefficient_ad_small = "missing.pth"\n'):
        with open(mapping, "w", encoding="utf-8") as stream:
            stream.write(contents)
        with pytest.raises(FileNotFoundError):
            EfficientAdLitModel(teacher_out_channels=8, visualizer=False)
    monkeypatch.setattr(io_utils, "PATHS_FILE", os.path.join(tmp_path, "absent.toml"))
    with pytest.raises(FileNotFoundError):
        EfficientAdLitModel(teacher_out_channels=8, visualizer=False)
    monkeypatch.setattr(io_utils, "PATHS_FILE", os.path.join(efficient_assets[0], "paths.toml"))
    with pytest.raises(RuntimeError):
        io_utils.load_pretrained_teacher(torch.nn.Linear(1, 1), "efficient_ad_small")


def test_train_start_guards_statistics_and_resume(native_model, monkeypatch):
    """Reject invalid training inputs and preserve restored teacher statistics at restart."""
    from visionlab.anomaly_detection.common.pre_processor import PreProcessor
    from visionlab.anomaly_detection.models.efficient_ad import lightning_model

    data = make_tiny_data()
    data.train_batch_size = 2
    attach_trainer(native_model, data)
    with pytest.raises(ValueError, match="train_batch_size for EfficientAd should be 1"):
        native_model.on_train_start()
    data.train_batch_size = 1
    attach_trainer(native_model, data)
    native_model.pre_processor = PreProcessor(mean=[0, 0, 0], std=[1, 1, 1])
    with pytest.raises(ValueError, match="should not contain Normalize"):
        native_model.on_train_start()
    native_model.pre_processor = PreProcessor()
    computed = native_model.teacher_channel_mean_std(data.train_dataloader())
    assert computed["mean"].shape == computed["std"].shape == (1, 8, 1, 1)
    calls = []

    def calculate(loader):
        calls.append(loader)
        return computed

    monkeypatch.setattr(native_model, "teacher_channel_mean_std", calculate)
    native_model.on_train_start()
    assert len(calls) == 1
    for name, value in computed.items():
        assert torch.equal(native_model.model.mean_std[name], value)
    teacher = {name: value.clone() for name, value in native_model.model.teacher.state_dict().items()}

    def forbidden(module, key):
        raise AssertionError("Teacher must not reload at train start")

    monkeypatch.setattr(lightning_model, "load_pretrained_teacher", forbidden)
    native_model.on_train_start()
    assert len(calls) == 1
    assert all(torch.equal(value, teacher[name]) for name, value in native_model.model.teacher.state_dict().items())
    native_model.imagenet_dir = os.path.join(native_model.imagenet_dir, "missing")
    with pytest.raises(FileNotFoundError):
        native_model.prepare_imagenette_data((256, 256))


def test_penalty_iterator_and_epoch_scheduler(native_model, monkeypatch):
    """Use penalty batches, restart exhausted iteration and retain the epoch scheduler contract."""
    data = make_tiny_data()
    attach_trainer(native_model, data)
    native_model.prepare_imagenette_data((256, 256))
    assert native_model.imagenet_loader.batch_size == 1 and native_model.imagenet_loader.pin_memory
    seen, logged = [], []

    def losses(batch, batch_imagenet):
        seen.append((batch.clone(), batch_imagenet.clone()))
        return tuple(torch.tensor(value, requires_grad=True) for value in (1.0, 2.0, 3.0))

    monkeypatch.setattr(native_model.model, "forward", losses)
    monkeypatch.setattr(native_model, "log", lambda name, value, **options: logged.append((name, value, options)))
    batch = next(iter(data.train_dataloader()))
    for _ in range(3):
        assert native_model.training_step(batch, 0)["loss"].item() == 6
    assert len(seen) == 3
    for image, penalty in seen:
        assert torch.equal(image, batch[0])
        assert penalty.shape == (1, 3, 256, 256)
        assert not torch.equal(penalty, image)
    assert [name for name, _, _ in logged[:4]] == ["train_st", "train_ae", "train_stae", "train_loss"]
    assert all(options["on_epoch"] and options["batch_size"] == 1 for _, _, options in logged)
    configured = native_model.configure_optimizers()
    scheduler = configured["lr_scheduler"]
    assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
    assert scheduler.step_size == int(0.95 * 10 * len(data.train_dataloader()))
    assert scheduler.gamma == 0.1 and "interval" not in configured
    optimizer = configured["optimizer"]
    assert isinstance(optimizer, torch.optim.Adam)
    expected_parameters = list(native_model.model.student.parameters()) + list(native_model.model.ae.parameters())
    assert {id(value) for value in optimizer.param_groups[0]["params"]} == {id(value) for value in expected_parameters}
    assert optimizer.defaults["lr"] == native_model.lr
    assert optimizer.defaults["weight_decay"] == native_model.weight_decay


def test_quantiles_fit_only_and_unbatched_normal_images(native_model, monkeypatch):
    """Use only normal unbatched images and update quantiles exclusively in fit validation."""
    from lightning.pytorch.trainer.states import TrainerFn

    data = make_tiny_data()
    attach_trainer(native_model, data)
    seen = []

    def maps(image, normalize):
        seen.append(image.clone())
        assert image.shape == (3, 256, 256) and normalize is False
        values = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        return values, values + 10

    monkeypatch.setattr(native_model.model, "get_maps", maps)
    native_model.on_validation_start()
    assert len(seen) == 1 and torch.equal(seen[0], data.val_data.images[0])
    assert native_model.model.is_set(native_model.model.quantiles)
    torch.testing.assert_close(native_model.model.quantiles["qa_st"], torch.tensor(13.5))
    before = {name: value.clone() for name, value in native_model.model.quantiles.items()}
    for stage in (TrainerFn.VALIDATING, TrainerFn.TESTING, TrainerFn.PREDICTING):
        native_model._trainer.state.fn = stage
        native_model.on_validation_start()
    native_model._trainer.state.fn = TrainerFn.FITTING
    native_model._trainer.sanity_checking = True
    native_model.on_validation_start()
    assert len(seen) == 1
    assert all(torch.equal(value, before[name]) for name, value in native_model.model.quantiles.items())


def test_no_external_normalization_in_evaluation(native_model, monkeypatch):
    """Keep validation, test and prediction inputs unchanged before the internally normalized model."""
    native_model.eval()
    batch = next(iter(make_tiny_data().val_dataloader()))
    seen = []

    def prediction(image):
        seen.append(image.clone())
        maps = image.mean(dim=1, keepdim=True)
        return InferenceBatch(pred_score=maps.amax(dim=(-2, -1)), anomaly_map=maps)

    monkeypatch.setattr(native_model.model, "forward", prediction)
    native_model.validation_step(batch, 0)
    native_model.test_step(batch, 0)
    native_model.predict_step(batch, 0)
    assert len(seen) == 3 and all(torch.equal(value, batch[0]) for value in seen)


def test_export_rejects_unknown_models_and_missing_teacher_keys(native_model, tmp_path):
    """Reject invalid model names and incomplete weights while retaining tensor-only exports."""
    from visionlab.inference.predictor import Predictor
    from visionlab.utils import io as io_utils

    path = os.path.join(tmp_path, "model.pt")
    io_utils.save_export(path, native_model.model, "efficient_ad", native_model.model_args,
                         native_model.pre_processor, native_model.post_processor)
    payload = io_utils.load_export(path)
    payload["model_name"] = "unknown"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="Unsupported model export format"):
        Predictor.from_export(path)
    payload["model_name"] = "efficient_ad"
    del payload["model_state"][next(name for name in payload["model_state"] if name.startswith("teacher."))]
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        Predictor.from_export(path)


@pytest.fixture(scope="module")
def fitted_efficient(tmp_path_factory, efficient_assets):
    """Fit the real model once and save a native checkpoint and pure export."""
    from lightning.pytorch import Trainer
    from visionlab.anomaly_detection.models.efficient_ad.lightning_model import EfficientAdLitModel

    root = str(tmp_path_factory.mktemp("fitted_efficient"))
    data = make_tiny_data()
    model = EfficientAdLitModel(imagenet_dir=efficient_assets[1], teacher_out_channels=8, visualizer=False)
    trainer = Trainer(default_root_dir=root, accelerator="cpu", devices=1, max_epochs=1, logger=False,
                      limit_train_batches=1, num_sanity_val_steps=0, enable_progress_bar=False,
                      enable_model_summary=False, enable_checkpointing=False)
    trainer.fit(model, datamodule=data)
    checkpoint = os.path.join(root, "model.ckpt")
    trainer.save_checkpoint(checkpoint)
    return root, model, data, checkpoint, trainer


def test_native_fit_checkpoint_restore_resume_and_evaluation(fitted_efficient, efficient_assets, tmp_path, monkeypatch):
    """Restore all learned states and preserve calibration in standalone evaluation and resume."""
    from lightning.pytorch import Trainer
    from visionlab.anomaly_detection.models.efficient_ad.lightning_model import EfficientAdLitModel
    from visionlab.utils import io as io_utils

    root, model, data, checkpoint, trainer = fitted_efficient
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for prefix in ("model.teacher.", "model.student.", "model.ae.", "model.mean_std.", "model.quantiles."):
        assert any(name.startswith(prefix) for name in saved["state_dict"])
    assert saved["post_processor"] is not None and model.post_processor.is_calibrated
    assert trainer.lr_scheduler_configs[0].interval == "epoch"
    assert model.trainer_arguments == {"num_sanity_val_steps": 0}
    exported = io_utils.load_export(os.path.join(root, "export", "model.pt"))
    assert exported["model_name"] == "efficient_ad" and exported["model_args"]["model_size"] == "small"
    assert all(torch.equal(value.cpu(), exported["model_state"][name]) for name, value in model.model.state_dict().items())
    restored = EfficientAdLitModel(imagenet_dir=efficient_assets[1], teacher_out_channels=8, visualizer=False)
    restored.on_load_checkpoint(saved)
    restored.load_state_dict(saved["state_dict"], strict=True)
    for name, value in saved["state_dict"].items():
        assert torch.equal(restored.state_dict()[name], value)
    model.eval()
    restored.eval()
    with torch.no_grad():
        image = next(iter(data.val_dataloader()))[0]
        expected, actual = model(image), restored(image)
    torch.testing.assert_close(actual.pred_score, expected.pred_score)
    torch.testing.assert_close(actual.anomaly_map, expected.anomaly_map)
    before = {name: value.clone() for name, value in restored.model.quantiles.items()}
    post_before = restored.post_processor.state_dict()
    evaluator = Trainer(default_root_dir=str(tmp_path), accelerator="cpu", devices=1, logger=False,
                        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
    evaluator.validate(restored, datamodule=data, ckpt_path=checkpoint)
    evaluator.test(restored, datamodule=data)
    evaluator.predict(restored, datamodule=data)
    assert all(torch.equal(value, before[name]) for name, value in restored.model.quantiles.items())
    for name, value in restored.post_processor.state_dict().items():
        torch.testing.assert_close(value, post_before[name], equal_nan=True)

    def forbidden(loader):
        raise AssertionError("Restored mean and std must not be recomputed")

    monkeypatch.setattr(restored, "teacher_channel_mean_std", forbidden)
    resumed = Trainer(default_root_dir=os.path.join(tmp_path, "resume"), accelerator="cpu", devices=1,
                      max_epochs=2, logger=False, limit_train_batches=1, num_sanity_val_steps=0,
                      enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
    restored.train()
    resumed.fit(restored, datamodule=data, ckpt_path=checkpoint)
    assert resumed.global_step > saved["global_step"]
    for name, value in restored.state_dict().items():
        if name.startswith(("model.teacher.", "model.mean_std.")):
            assert torch.equal(value, saved["state_dict"][name])


def consume_efficient_export(root):
    """Restore pure exports under dependency and file-access guards in a fresh interpreter."""
    from visionlab.inference.predictor import Predictor
    from visionlab.utils import io as io_utils

    io_utils.block_network()
    path = os.path.join(root, "export", "model.pt")
    expected = torch.load(os.path.join(root, "expected.pt"), weights_only=True)
    payload = io_utils.load_export(path)
    for prefix in ("teacher.", "mean_std.", "quantiles."):
        assert any(name.startswith(prefix) for name in payload["model_state"])

    def audit_open(event, args):
        if event == "open":
            opened, mode, flags = args
            assert isinstance(opened, (str, bytes)) and os.path.abspath(opened) == path, opened
            assert not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)

    sys.addaudithook(audit_open)
    predictor = Predictor.from_export(path)
    actual = predictor.predict(expected["images"])
    for name, value in actual.items():
        torch.testing.assert_close(value, expected["outputs"][name], rtol=1e-5, atol=1e-6)
    for name, value in predictor.model.state_dict().items():
        assert torch.equal(value, payload["model_state"][name])


def test_native_export_predictor_without_assets_or_frameworks(fitted_efficient):
    """Match the Lightning prediction path without loading pretrained files or training packages."""
    root, model, data, _, _ = fitted_efficient
    model.eval()
    batch = next(iter(data.val_dataloader()))
    with torch.no_grad():
        expected = model.predict_step(batch, 0)
    names = ("pred_score", "anomaly_map", "normalized_pred_score", "normalized_anomaly_map", "pred_label", "pred_mask")
    torch.save({"images": batch[0], "outputs": {name: expected[name] for name in names}}, os.path.join(root, "expected.pt"))
    script = '''
import importlib.abc
import runpy
import sys
class Block(importlib.abc.MetaPathFinder):
    """Reject training dependencies before importing the export consumer."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("anomalib", "lightning", "pytorch_lightning"):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, Block())
sys.path.insert(0, sys.argv[2])
runpy.run_path(sys.argv[1])["consume_efficient_export"](sys.argv[3])
assert not {"anomalib", "lightning", "pytorch_lightning"} & set(sys.modules)
'''
    result = subprocess.run([sys.executable, "-c", script, os.path.abspath(__file__),
                             os.path.join(PROJECT_ROOT, "src"), root], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
