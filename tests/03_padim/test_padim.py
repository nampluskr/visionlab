# tests/03_padim/test_padim.py: pytest pure PaDiM, Gaussian statistics and dependency boundaries

import os
import random
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch.nn import functional

from visionlab.anomaly_detection.common.batch import InferenceBatch
from visionlab.anomaly_detection.models.components.feature_extractors import TimmFeatureExtractor
from visionlab.anomaly_detection.models.components.filters import GaussianBlur2d
from visionlab.anomaly_detection.models.components.filters.blur import compute_kernel_size
from visionlab.anomaly_detection.models.components.stats import MultiVariateGaussian
from visionlab.anomaly_detection.models.components.tiling.tiler import Tiler
from visionlab.anomaly_detection.models.padim import PadimModel
from visionlab.anomaly_detection.models.padim import model as padim_model
from visionlab.anomaly_detection.models.padim.anomaly_map import AnomalyMapGenerator

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def default_model():
    """Share an offline ResNet model for default dimensions and hierarchical embeddings."""
    return PadimModel(pre_trained=False)


@pytest.fixture
def small_model():
    """Use four selected channels to keep small-image covariance checks inexpensive."""
    return PadimModel(pre_trained=False, n_features=4)


def test_constructor_defaults_and_invalid_feature_counts(default_model, monkeypatch):
    """Preserve ResNet defaults, zero fallback and invalid feature-count errors."""
    assert default_model.n_features == 100 and default_model.n_features_original == 448

    def extractor(backbone, layers, pre_trained):
        assert pre_trained is False
        return default_model.feature_extractor

    monkeypatch.setattr(padim_model, "TimmFeatureExtractor", extractor)
    assert PadimModel(pre_trained=False, n_features=0).n_features == 100
    for n_features in (-1, 449):
        with pytest.raises(ValueError, match="0 < n_features <= 448"):
            PadimModel(pre_trained=False, n_features=n_features)
    with pytest.raises(ValueError, match="n_features must be specified"):
        PadimModel(backbone="resnet34", pre_trained=False)


def test_idx_buffer_and_python_random_repeatability(default_model, monkeypatch):
    """Keep unique persisted indices and the original Python random sampling sequence."""
    idx = default_model.idx
    assert idx.shape == (100,) and idx.dtype == torch.int64
    assert idx.min() >= 0 and idx.max() < 448 and idx.unique().numel() == 100
    assert torch.equal(default_model.state_dict()["idx"], idx)

    def extractor(backbone, layers, pre_trained):
        return default_model.feature_extractor

    monkeypatch.setattr(padim_model, "TimmFeatureExtractor", extractor)
    previous = random.getstate()
    try:
        random.seed(0)
        first = PadimModel(pre_trained=False)
        random.seed(0)
        second = PadimModel(pre_trained=False)
        assert torch.equal(first.idx, second.idx)
        random.seed(0)
        assert torch.equal(first.idx, torch.tensor(random.sample(range(448), 100)))
    finally:
        random.setstate(previous)


def test_generate_embedding_nearest_layers_and_idx(default_model):
    """Merge real ResNet feature maps at 64 pixels and select the persisted channels."""
    with torch.no_grad():
        features = default_model.feature_extractor(torch.rand(2, 3, 256, 256))
        actual = default_model.generate_embedding(features)
    assert [features[layer].shape[1] for layer in default_model.layers] == [64, 128, 256]
    merged = torch.cat([features["layer1"],
                        functional.interpolate(features["layer2"], size=(64, 64), mode="nearest"),
                        functional.interpolate(features["layer3"], size=(64, 64), mode="nearest")], dim=1)
    expected = merged.index_select(1, default_model.idx)
    assert actual.shape == (2, 100, 64, 64)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_gaussian_numpy_reference_and_dynamic_buffer_restore():
    """Match NumPy covariance with both regularizers and strictly restore initially empty buffers."""
    embedding = torch.randn(8, 3, 2, 3, generator=torch.Generator().manual_seed(7))
    gaussian = MultiVariateGaussian()
    mean, inv_covariance = gaussian.fit(embedding)
    vectors = embedding.numpy().reshape(8, 3, 6)
    expected_mean = vectors.mean(axis=0)
    expected_inverse = np.stack([
        np.linalg.inv(np.cov(vectors[:, :, patch], rowvar=False) + 0.01 * np.eye(3) + 1e-5 * np.eye(3))
        for patch in range(6)
    ])
    assert mean.shape == (3, 6) and inv_covariance.shape == (6, 3, 3)
    np.testing.assert_allclose(mean.numpy(), expected_mean, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(inv_covariance.numpy(), expected_inverse, rtol=1e-5, atol=1e-6)
    restored = MultiVariateGaussian()
    assert restored.mean.numel() == restored.inv_covariance.numel() == 0
    result = restored.load_state_dict(gaussian.state_dict(), strict=True)
    assert result.missing_keys == result.unexpected_keys == []
    assert restored.mean.shape == mean.shape and restored.inv_covariance.shape == inv_covariance.shape
    assert torch.equal(restored.mean, mean) and torch.equal(restored.inv_covariance, inv_covariance)


def test_gaussian_blur_normalization_reflect_and_shape():
    """Preserve normalized kernels and constant multichannel images with reflect padding."""
    blur = GaussianBlur2d(sigma=1.0, channels=3)
    assert compute_kernel_size(1.0) == 9 and compute_kernel_size(4.0) == 33
    assert blur.border_type == "reflect"
    torch.testing.assert_close(blur.kernel.sum(dim=(-2, -1)), torch.ones(3, 1))
    image = torch.full((2, 3, 24, 28), 2.5)
    output = blur(image)
    assert output.shape == image.shape
    torch.testing.assert_close(output, image)


def test_mahalanobis_distance_upsampling_and_anomaly_map():
    """Match patch-wise Mahalanobis distances and the bilinear-plus-blur map pipeline."""
    embedding = torch.randn(2, 3, 4, 5)
    gaussian = MultiVariateGaussian()
    mean, inv_covariance = gaussian.fit(torch.randn(8, 3, 4, 5))
    generator = AnomalyMapGenerator()
    actual = generator.compute_distance(embedding, [mean, inv_covariance])
    expected = torch.empty(2, 20)
    flattened = embedding.reshape(2, 3, 20)
    for batch in range(2):
        for patch in range(20):
            delta = flattened[batch, :, patch] - mean[:, patch]
            expected[batch, patch] = torch.dot(delta, inv_covariance[patch] @ delta).clamp(0).sqrt()
    assert actual.shape == (2, 1, 4, 5)
    torch.testing.assert_close(actual, expected.reshape(2, 1, 4, 5), rtol=1e-5, atol=1e-6)
    upsampled = generator.up_sample(actual, (256, 256))
    assert upsampled.shape == (2, 1, 256, 256)
    torch.testing.assert_close(upsampled, functional.interpolate(actual, size=(256, 256), mode="bilinear",
                                                                align_corners=False))
    output = generator(embedding=embedding, mean=mean, inv_covariance=inv_covariance, image_size=(256, 256))
    assert output.shape == (2, 1, 256, 256)
    torch.testing.assert_close(output, generator.blur(upsampled))


def test_model_train_eval_and_strict_fitted_state_restore(small_model):
    """Restore feature weights, indices and Gaussian buffer sizes with identical pure predictions."""
    images = torch.rand(2, 3, 64, 64)
    small_model.train()
    embedding = small_model(images)
    assert isinstance(embedding, torch.Tensor) and embedding.shape == (2, 4, 16, 16)
    assert not embedding.requires_grad
    small_model.gaussian.fit(torch.randn(8, 4, 16, 16))
    small_model.eval()
    with torch.no_grad():
        expected = small_model(images)
    assert isinstance(expected, InferenceBatch)
    assert expected.pred_score.shape == (2, 1) and expected.anomaly_map.shape == (2, 1, 64, 64)
    torch.testing.assert_close(expected.pred_score, expected.anomaly_map.amax(dim=(-2, -1)))
    restored = PadimModel(pre_trained=False, n_features=4).eval()
    assert restored.gaussian.mean.numel() == restored.gaussian.inv_covariance.numel() == 0
    result = restored.load_state_dict(small_model.state_dict(), strict=True)
    assert result.missing_keys == result.unexpected_keys == []
    assert restored.gaussian.mean.shape == (4, 256) and restored.gaussian.inv_covariance.shape == (256, 4, 4)
    for name, value in restored.state_dict().items():
        assert torch.equal(value, small_model.state_dict()[name]), name
    with torch.no_grad():
        actual = restored(images)
    torch.testing.assert_close(actual.pred_score, expected.pred_score, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual.anomaly_map, expected.anomaly_map, rtol=1e-5, atol=1e-6)


def test_optional_tiling_small_input(small_model):
    """Reuse the existing Tiler for embeddings and restore the input map resolution."""
    small_model.tiler = Tiler(tile_size=32, stride=32)
    images = torch.rand(2, 3, 64, 64)
    small_model.train()
    embedding = small_model(images)
    assert embedding.shape == (2, 4, 16, 16)
    small_model.gaussian.fit(torch.randn(8, 4, 16, 16))
    small_model.eval()
    with torch.no_grad():
        prediction = small_model(images)
    assert prediction.anomaly_map.shape == (2, 1, 64, 64)
    torch.testing.assert_close(prediction.pred_score, prediction.anomaly_map.amax(dim=(-2, -1)))


def constructor_checks():
    """Check both pretrained arguments, strict restoration and forward under constructor I/O guards."""
    calls = []

    def extractor(backbone, layers, pre_trained):
        calls.append(pre_trained)
        return TimmFeatureExtractor(backbone=backbone, layers=layers, pre_trained=pre_trained)

    def forbidden(*args, **kwargs):
        raise AssertionError("Forbidden constructor file or network access")

    with pytest.MonkeyPatch.context() as patch:
        for target in ("builtins.open", "io.open", "os.open", "torch.load", "socket.socket.connect",
                       "socket.socket.connect_ex", "socket.create_connection"):
            patch.setattr(target, forbidden)
        patch.setattr(padim_model, "TimmFeatureExtractor", extractor)
        original = PadimModel(n_features=4)
        restored = PadimModel(pre_trained=False, n_features=4)
        assert calls == [False, False]
        assert not original.feature_extractor.training
        state = original.state_dict()
        assert any(name.startswith("feature_extractor.") for name in state)
        assert {"idx", "gaussian.mean", "gaussian.inv_covariance"} <= state.keys()
        result = restored.load_state_dict(state, strict=True)
        assert result.missing_keys == result.unexpected_keys == []
    restored.train()
    embedding = restored(torch.rand(2, 3, 64, 64))
    restored.gaussian.fit(torch.randn(8, *embedding.shape[1:]))
    restored.eval()
    with torch.no_grad():
        prediction = restored(torch.rand(2, 3, 64, 64))
    assert prediction.pred_score.shape == (2, 1) and prediction.anomaly_map.shape == (2, 1, 64, 64)


def test_constructor_and_forward_without_frameworks_or_external_io():
    """Check FR-22 and NFR-1/NFR-2 before importing PaDiM in a fresh process."""
    script = '''
import importlib.abc
import runpy
import sys
class Block(importlib.abc.MetaPathFinder):
    """Reject reference and training frameworks before importing pure components."""
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


def test_original_copyright_and_spdx_headers():
    """Retain original copyright and SPDX markers in every copied file."""
    root = os.path.join(PROJECT_ROOT, "src", "visionlab", "anomaly_detection", "models")
    copied = [os.path.join("padim", name) for name in ("__init__.py", "model.py", "anomaly_map.py")]
    for group, name in (("base", "dynamic_buffer.py"), ("filters", "blur.py"), ("stats", "multi_variate_gaussian.py")):
        copied.extend(os.path.join("components", group, file_name) for file_name in ("__init__.py", name))
    for name in copied:
        with open(os.path.join(root, name), encoding="utf-8") as stream:
            source = stream.read()
        assert "# Copyright (C) 2022-2025 Intel Corporation" in source
        assert "# SPDX-License-Identifier: Apache-2.0" in source
