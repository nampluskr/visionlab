# tests/01_stfpm/test_model.py: pytest pure STFPM, offline construction and tiler contracts

import os
import subprocess
import sys

import pytest
import torch
from torch.nn import functional
from torchvision.transforms import RandomCrop

from visionlab.anomaly_detection.common.batch import InferenceBatch
from visionlab.anomaly_detection.models.components.feature_extractors import TimmFeatureExtractor
from visionlab.anomaly_detection.models.components.tiling.tiler import ImageUpscaleMode, StrideSizeError, Tiler
from visionlab.anomaly_detection.models.stfpm import STFPMModel
from visionlab.anomaly_detection.models.stfpm import model as stfpm_model
from visionlab.anomaly_detection.models.stfpm.loss import STFPMLoss

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def shared_model():
    """Construct one pure model for shape, feature and tiling checks."""
    return STFPMModel(layers=["layer1", "layer2", "layer3"])


@pytest.fixture
def model(shared_model):
    """Restore evaluation mode and remove tiling before each independent check."""
    shared_model.eval()
    shared_model.tiler = None
    shared_model.zero_grad(set_to_none=True)
    return shared_model


def constructor_checks():
    """Exercise forbidden constructor I/O and strict restoration in the isolated consumer."""
    calls = []
    omitted = object()

    def factory(backbone, pre_trained, layers, requires_grad=omitted):
        call = {"pre_trained": pre_trained}
        if requires_grad is not omitted:
            call["requires_grad"] = requires_grad
        calls.append(call)
        return TimmFeatureExtractor(backbone=backbone, pre_trained=pre_trained, layers=layers,
                                    requires_grad=False if requires_grad is omitted else requires_grad)

    def forbidden(*args, **kwargs):
        raise AssertionError("Forbidden constructor file or network access")

    with pytest.MonkeyPatch.context() as patch:
        for target in ("builtins.open", "io.open", "os.open", "torch.load", "socket.socket.connect",
                       "socket.socket.connect_ex", "socket.create_connection"):
            patch.setattr(target, forbidden)
        patch.setattr(stfpm_model, "TimmFeatureExtractor", factory)
        original = STFPMModel(layers=["layer1", "layer2", "layer3"])
        assert not original.teacher_model.training
        assert calls == [{"pre_trained": False}, {"pre_trained": False, "requires_grad": True}]
        state = original.state_dict()
        assert any(name.startswith("teacher_model.") for name in state)
        restored = STFPMModel(layers=["layer1", "layer2", "layer3"])
        result = restored.load_state_dict(state, strict=True)
        assert result.missing_keys == result.unexpected_keys == []
        assert all(torch.equal(value, state[name]) for name, value in restored.state_dict().items())


def test_constructor_and_state_dict_without_external_io():
    """Check FR-22 and NFR-1/NFR-2 without cached framework imports."""
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
runpy.run_path(sys.argv[1])["constructor_checks"]()
assert not {"anomalib", "lightning", "pytorch_lightning"} & set(sys.modules)
'''
    result = subprocess.run([sys.executable, "-c", script, os.path.abspath(__file__), os.path.join(PROJECT_ROOT, "src")],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_eval_shapes(model):
    """Preserve FR-4 score and map shapes and their reduction relationship."""
    with torch.no_grad():
        prediction = model(torch.randn(2, 3, 64, 80))
    assert isinstance(prediction, InferenceBatch)
    assert prediction.pred_score.shape == (2, 1) and prediction.anomaly_map.shape == (2, 1, 64, 80)
    torch.testing.assert_close(prediction.pred_score, prediction.anomaly_map.amax(dim=(-2, -1)))


def test_training_features_and_loss(model):
    """Keep teacher inference frozen and student features differentiable through the loss."""
    model.train()
    teacher, student = model(torch.randn(2, 3, 64, 64))
    assert list(teacher) == list(student) == ["layer1", "layer2", "layer3"]
    assert not model.teacher_model.feature_extractor.training and model.student_model.feature_extractor.training
    assert all(not value.requires_grad for value in model.teacher_model.parameters())
    assert all(value.requires_grad for value in model.student_model.parameters())
    for layer in teacher:
        assert teacher[layer].shape == student[layer].shape
        assert not teacher[layer].requires_grad and student[layer].requires_grad
    loss = STFPMLoss()(teacher_features=teacher, student_features=student)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert all(value.grad is None for value in model.teacher_model.parameters())
    assert any(value.grad is not None for value in model.student_model.parameters())


def test_tiled_model_resolution(model):
    """Restore a tiled model anomaly map to the full 1024 input resolution."""
    model.tiler = Tiler(tile_size=256, stride=256)
    with torch.no_grad():
        prediction = model(torch.randn(1, 3, 1024, 1024))
    assert prediction.pred_score.shape == (1, 1) and prediction.anomaly_map.shape == (1, 1, 1024, 1024)


def test_original_headers_and_framework_independence():
    """Keep every copied model and feature extractor copyright and SPDX header."""
    root = os.path.join(PROJECT_ROOT, "src", "visionlab", "anomaly_detection", "models")
    copied = [os.path.join("stfpm", name) for name in ("model.py", "anomaly_map.py", "loss.py", "__init__.py")]
    copied += [os.path.join("components", "feature_extractors", name) for name in ("timm.py", "utils.py", "__init__.py")]
    for name in copied:
        with open(os.path.join(root, name), encoding="utf-8") as stream:
            source = stream.read()
        assert "# Copyright (C) 2022-2025 Intel Corporation" in source
        assert "# SPDX-License-Identifier: Apache-2.0" in source


def test_1024_round_trip_and_tile_order():
    """Preserve sixteen tiles per image, tile order and exact reconstruction in both modes."""
    image = torch.arange(2 * 3 * 1024 * 1024, dtype=torch.float32).reshape(2, 3, 1024, 1024)
    for mode in ImageUpscaleMode:
        tiler = Tiler(tile_size=256, stride=256, mode=mode)
        tiles = tiler.tile(image)
        assert tiles.shape == (32, 3, 256, 256)
        assert torch.equal(tiles[1], image[0, :, :256, 256:512])
        assert torch.equal(tiles[16], image[1, :, :256, :256])
        assert torch.equal(tiler.untile(tiles), image)


def test_overlap_padding_and_interpolation():
    """Match overlapping reconstruction with padding and interpolation at uneven sizes."""
    image = torch.arange(17 * 19, dtype=torch.float32).reshape(1, 1, 17, 19)
    for mode in ImageUpscaleMode:
        tiler = Tiler(tile_size=8, stride=4, mode=mode)
        tiles = tiler.tile(image)
        assert tiles.shape == (16, 1, 8, 8)
        expected = image if mode == ImageUpscaleMode.PADDING else functional.interpolate(
            functional.interpolate(image, size=(20, 20)), size=(17, 19))
        assert torch.equal(tiler.untile(tiles), expected)


def test_scaled_feature_reconstruction():
    """Preserve reconstruction of spatially downsampled feature tiles."""
    image = torch.arange(16 * 16, dtype=torch.float32).reshape(1, 1, 16, 16)
    for mode in ImageUpscaleMode:
        tiler = Tiler(tile_size=8, stride=8, mode=mode)
        features = functional.avg_pool2d(tiler.tile(image), kernel_size=2)
        expected = functional.avg_pool2d(image, kernel_size=2)
        if mode == ImageUpscaleMode.INTERPOLATION:
            expected = functional.interpolate(expected, size=(16, 16))
        assert torch.equal(tiler.untile(features), expected)


def test_border_removal_and_default_stride():
    """Keep discarded borders zero and use tile dimensions as the default stride."""
    image = torch.arange(12 * 12, dtype=torch.float32).reshape(1, 1, 12, 12)
    tiler = Tiler(tile_size=8, stride=4, remove_border_count=1)
    expected = torch.zeros_like(image)
    expected[:, :, 1:-1, 1:-1] = image[:, :, 1:-1, 1:-1]
    assert torch.equal(tiler.untile(tiler.tile(image)), expected)
    tiler = Tiler(tile_size=(4, 6))
    assert (tiler.stride_h, tiler.stride_w) == (4, 6) and not tiler.overlapping
    assert torch.equal(tiler.untile(tiler.tile(image[0])), image)


def test_random_tile_rng_and_invalid_arguments():
    """Keep random crop RNG consumption and invalid argument exceptions unchanged."""
    image = torch.arange(16 * 16, dtype=torch.float32).reshape(1, 1, 16, 16)
    with torch.random.fork_rng():
        torch.manual_seed(42)
        expected = torch.vstack([RandomCrop(8)(image) for _ in range(4)])
        torch.manual_seed(42)
        assert torch.equal(Tiler(tile_size=8, stride=8).tile(image, use_random_tiling=True), expected)
    with pytest.raises(StrideSizeError):
        Tiler(tile_size=8, stride=9)
    with pytest.raises(ValueError):
        Tiler(tile_size=8, stride=8, mode="unknown")
    with pytest.raises(ValueError):
        Tiler(tile_size=32, stride=32).tile(image)
    with pytest.raises(TypeError):
        Tiler(tile_size=object(), stride=8)
