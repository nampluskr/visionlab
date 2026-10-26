# tests/04_dfm/test_dfm.py: pytest pure DFM, original PCA side effects and Gaussian scoring

import inspect
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch.nn import functional

from visionlab.anomaly_detection.common.batch import InferenceBatch
from visionlab.anomaly_detection.models.components.dimensionality_reduction import PCA
from visionlab.anomaly_detection.models.components.feature_extractors import TimmFeatureExtractor
from visionlab.anomaly_detection.models.dfm import DFMModel
from visionlab.anomaly_detection.models.dfm import model as dfm_model
from visionlab.anomaly_detection.models.dfm.model import SingleClassGaussian

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def gaussian_reference(dataset):
    """Build NumPy Gaussian parameters without invoking the original unsupported mean signature."""
    values = dataset.numpy().astype(np.float64)
    mean = values.mean(axis=1)
    centered = (values - mean[:, None]) / np.sqrt(values.shape[1])
    u_mat, sigma, _ = np.linalg.svd(centered, full_matrices=False)
    return mean, u_mat, sigma


def set_gaussian_buffers(gaussian, dataset):
    """Populate synthetic Gaussian buffers for scoring tests while leaving fit unchanged."""
    mean, u_mat, sigma = gaussian_reference(dataset)
    gaussian.mean_vec = torch.from_numpy(mean).to(dataset)
    gaussian.u_mat = torch.from_numpy(u_mat).to(dataset)
    gaussian.sigma_mat = torch.from_numpy(sigma).to(dataset)


@pytest.mark.parametrize("components", (2, 3))
def test_pca_integer_numpy_subspace_and_inplace_fit(components):
    """Match double-SVD subspaces and preserve in-place mean subtraction during fit."""
    original = torch.randn(12, 5, generator=torch.Generator().manual_seed(42)) + 2
    dataset = original.clone()
    expected_mean = original.numpy().mean(axis=0)
    _, _, v_h = np.linalg.svd((original.numpy() - expected_mean).astype(np.float64), full_matrices=False)
    expected_vectors = v_h.T[:, :components]
    pca = PCA(n_components=components)
    pointer = dataset.data_ptr()
    pca.fit(dataset)
    assert dataset.data_ptr() == pointer and not torch.equal(dataset, original)
    torch.testing.assert_close(dataset, original - original.mean(dim=0))
    assert pca.num_components.item() == components and pca.singular_vectors.shape == (5, components)
    np.testing.assert_allclose(pca.mean.numpy(), expected_mean, rtol=1e-5, atol=1e-6)
    actual_vectors = pca.singular_vectors.numpy()
    np.testing.assert_allclose(actual_vectors @ actual_vectors.T, expected_vectors @ expected_vectors.T,
                               rtol=1e-5, atol=1e-6)
    assert "singular_values" not in pca.state_dict()


def test_pca_ratio_uses_first_crossing_index_without_increment():
    """Retain the original variance-ratio component index without adding one."""
    original = torch.randn(20, 5, generator=torch.Generator().manual_seed(7))
    centered = original.numpy() - original.numpy().mean(axis=0)
    singular_values = np.linalg.svd(centered.astype(np.float64), full_matrices=False)[1]
    ratios = np.cumsum(singular_values**2) / np.sum(singular_values**2)
    expected_index = np.flatnonzero(ratios >= 0.97)[0]
    pca = PCA(n_components=0.97)
    pca.fit(original.clone())
    assert pca.num_components.item() == expected_index
    assert pca.singular_vectors.shape == (5, expected_index)
    assert pca.singular_values.shape == (expected_index,)


def test_pca_transform_side_effect_reconstruction_and_dynamic_restore():
    """Center transform inputs in place and restore only the original persistent buffers."""
    pca = PCA(n_components=3)
    pca.fit(torch.randn(12, 5) + 2)
    features = torch.randn(4, 5) + 3
    original = features.clone()
    pointer = features.data_ptr()
    transformed = pca.transform(features)
    assert features.data_ptr() == pointer
    torch.testing.assert_close(features, original - pca.mean)
    vectors = pca.singular_vectors.numpy().astype(np.float64)
    expected = (original.numpy() - pca.mean.numpy()) @ vectors @ vectors.T
    reconstructed = pca.inverse_transform(transformed)
    np.testing.assert_allclose(reconstructed.numpy(), expected, rtol=1e-5, atol=1e-6)
    restored = PCA(n_components=3)
    assert restored.mean.numel() == restored.singular_vectors.numel() == restored.num_components.numel() == 0
    result = restored.load_state_dict(pca.state_dict(), strict=True)
    assert result.missing_keys == result.unexpected_keys == []
    assert restored.mean.shape == (5,) and restored.singular_vectors.shape == (5, 3)
    assert restored.num_components.shape == (1,) and not hasattr(restored, "singular_values")
    for name, value in pca.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value)
    torch.testing.assert_close(restored.transform(original.clone()), transformed)


def test_gaussian_original_fit_rejects_device_keyword():
    """Expose the unmodified torch.mean device keyword error instead of hiding a source defect."""
    with pytest.raises(TypeError, match="mean"):
        SingleClassGaussian().fit(torch.randn(3, 12))


def test_gaussian_synthetic_buffers_numpy_nll_and_restore():
    """Check Gaussian NLL and dynamic buffer restoration using independently constructed parameters."""
    dataset = torch.randn(3, 12, generator=torch.Generator().manual_seed(19)) + 2
    gaussian = SingleClassGaussian()
    set_gaussian_buffers(gaussian, dataset)
    assert gaussian.mean_vec.shape == (3,) and gaussian.u_mat.shape == (3, 3) and gaussian.sigma_mat.shape == (3,)
    features = torch.randn(4, 3)
    mean, u_mat, sigma = gaussian_reference(dataset)
    projected = (features.numpy() - mean) @ (u_mat / sigma)
    expected = np.square(projected).sum(axis=1) + 2 * np.log(sigma).sum()
    np.testing.assert_allclose(gaussian.score_samples(features).numpy(), expected, rtol=1e-5, atol=1e-6)
    restored = SingleClassGaussian()
    assert all(value.numel() == 0 for value in restored.buffers())
    result = restored.load_state_dict(gaussian.state_dict(), strict=True)
    assert result.missing_keys == result.unexpected_keys == []
    for name, value in gaussian.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value)
    torch.testing.assert_close(restored.score_samples(features), gaussian.score_samples(features))


@pytest.fixture
def model():
    """Construct an offline ResNet18 DFM with two PCA components for small-image tests."""
    return DFMModel(backbone="resnet18", layer="layer3", pre_trained=False, n_comps=2)


def test_dfm_constructor_signature_and_defaults(model):
    """Preserve all six constructor parameters and the original model defaults."""
    signature = inspect.signature(DFMModel.__init__)
    assert list(signature.parameters) == ["self", "backbone", "layer", "pre_trained", "pooling_kernel_size",
                                          "n_comps", "score_type"]
    assert signature.parameters["backbone"].default is inspect.Parameter.empty
    assert signature.parameters["layer"].default is inspect.Parameter.empty
    assert {name: signature.parameters[name].default for name in
            ("pre_trained", "pooling_kernel_size", "n_comps", "score_type")} == {
                "pre_trained": True, "pooling_kernel_size": 4, "n_comps": 0.97, "score_type": "fre"}
    assert model.pooling_kernel_size == 4 and model.score_type == "fre"


@pytest.mark.parametrize("kernel", (1, 2, 4))
def test_dfm_get_features_train_eval_and_pooling(model, kernel):
    """Return flat training features and evaluation shapes with optional average pooling."""
    images = torch.rand(2, 3, 64, 64)
    model.pooling_kernel_size = kernel
    with torch.no_grad():
        raw = model.feature_extractor(images)["layer3"]
    pooled = raw if kernel == 1 else functional.avg_pool2d(raw, kernel_size=kernel)
    model.train()
    features = model.get_features(images)
    assert isinstance(features, torch.Tensor)
    assert features.shape == (2, 256 * (4 // kernel) * (4 // kernel))
    assert not features.requires_grad
    torch.testing.assert_close(features, pooled.reshape(2, -1))
    model.eval()
    actual_features, shapes = model.get_features(images)
    assert shapes == pooled.shape
    torch.testing.assert_close(actual_features, features)


def test_dfm_fre_reconstruction_map_and_strict_state_restore(model):
    """Preserve the centered-feature reconstruction side effect and fitted pure predictions."""
    images = torch.rand(2, 3, 64, 64)
    model.eval()
    features, shapes = model.get_features(images)
    model.fit(torch.randn(8, features.shape[1]) + 0.5)
    centered = features - model.pca_model.mean
    projected = centered @ model.pca_model.singular_vectors
    reconstructed = projected @ model.pca_model.singular_vectors.T
    error = (centered - reconstructed).square()
    expected_score = error.sum(dim=1)
    expected_map = functional.interpolate(error.reshape(shapes).sum(dim=1, keepdim=True),
                                           size=(64, 64), mode="bilinear", align_corners=False)
    with torch.no_grad():
        prediction = model(images)
    assert isinstance(prediction, InferenceBatch)
    assert prediction.pred_score.shape == (2,) and prediction.anomaly_map.shape == (2, 1, 64, 64)
    torch.testing.assert_close(prediction.pred_score, expected_score, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(prediction.anomaly_map, expected_map, rtol=1e-5, atol=1e-6)
    restored = DFMModel(backbone="resnet18", layer="layer3", pre_trained=False, n_comps=2).eval()
    assert restored.pca_model.mean.numel() == restored.pca_model.singular_vectors.numel() == 0
    result = restored.load_state_dict(model.state_dict(), strict=True)
    assert result.missing_keys == result.unexpected_keys == []
    assert restored.pca_model.mean.shape == (features.shape[1],)
    assert restored.pca_model.singular_vectors.shape == (features.shape[1], 2)
    with torch.no_grad():
        actual = restored(images)
    torch.testing.assert_close(actual.pred_score, prediction.pred_score, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual.anomaly_map, prediction.anomaly_map, rtol=1e-5, atol=1e-6)


def test_dfm_nll_synthetic_scoring_and_unsupported_score_type(model):
    """Exercise NLL from synthetic Gaussian state while retaining the original fitting failure."""
    model.score_type = "nll"
    model.eval()
    images = torch.rand(2, 3, 64, 64)
    features, shapes = model.get_features(images)
    training = torch.randn(8, features.shape[1]) + 0.5
    with pytest.raises(TypeError, match="mean"):
        model.fit(training.clone())
    model.pca_model.fit(training)
    reduced = model.pca_model.transform(training)
    set_gaussian_buffers(model.gaussian_model, reduced.T)
    projected = (features - model.pca_model.mean) @ model.pca_model.singular_vectors
    expected = model.gaussian_model.score_samples(projected)
    with torch.no_grad():
        prediction = model(images)
    assert isinstance(prediction, InferenceBatch)
    assert prediction.anomaly_map is None and prediction.pred_score.shape == (2,)
    torch.testing.assert_close(prediction.pred_score, expected, rtol=1e-5, atol=1e-6)
    restored = DFMModel(backbone="resnet18", layer="layer3", pre_trained=False, n_comps=2, score_type="nll").eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        actual = restored(images)
    torch.testing.assert_close(actual.pred_score, prediction.pred_score, rtol=1e-5, atol=1e-6)
    assert actual.anomaly_map is None
    model.score_type = "invalid"
    with pytest.raises(ValueError, match="unsupported score type"):
        model.score(features.clone(), shapes)


def constructor_checks():
    """Check both pretrained arguments, strict restoration and pure forward under isolated guards."""
    calls = []

    def extractor(backbone, pre_trained, layers):
        calls.append(pre_trained)
        return TimmFeatureExtractor(backbone=backbone, pre_trained=pre_trained, layers=layers)

    def forbidden(*args, **kwargs):
        raise AssertionError("Forbidden constructor file or network access")

    with pytest.MonkeyPatch.context() as patch:
        for target in ("builtins.open", "io.open", "os.open", "torch.load", "socket.socket.connect",
                       "socket.socket.connect_ex", "socket.create_connection"):
            patch.setattr(target, forbidden)
        patch.setattr(dfm_model, "TimmFeatureExtractor", extractor)
        original = DFMModel(backbone="resnet18", layer="layer3", n_comps=2)
        restored = DFMModel(backbone="resnet18", layer="layer3", pre_trained=False, n_comps=2)
        assert calls == [False, False] and not original.feature_extractor.training
        state = original.state_dict()
        assert any(name.startswith("feature_extractor.") for name in state)
        result = restored.load_state_dict(state, strict=True)
        assert result.missing_keys == result.unexpected_keys == []
    restored.eval()
    restored.fit(torch.randn(8, 256))
    with torch.no_grad():
        prediction = restored(torch.rand(2, 3, 64, 64))
    assert prediction.pred_score.shape == (2,) and prediction.anomaly_map.shape == (2, 1, 64, 64)


def test_constructor_and_forward_without_frameworks_or_external_io():
    """Check FR-22 and NFR-1/NFR-2 before importing DFM in a fresh interpreter."""
    script = '''
import importlib.abc
import runpy
import sys
class Block(importlib.abc.MetaPathFinder):
    """Reject reference and training frameworks before loading the pure consumer."""
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
    """Retain original copyright and SPDX markers in all four copied files."""
    root = os.path.join(PROJECT_ROOT, "src", "visionlab", "anomaly_detection", "models")
    files = [os.path.join("dfm", name) for name in ("__init__.py", "model.py")]
    files += [os.path.join("components", "dimensionality_reduction", name) for name in ("__init__.py", "pca.py")]
    for name in files:
        with open(os.path.join(root, name), encoding="utf-8") as stream:
            source = stream.read()
        assert "# Copyright (C) 2022-2025 Intel Corporation" in source
        assert "# SPDX-License-Identifier: Apache-2.0" in source
