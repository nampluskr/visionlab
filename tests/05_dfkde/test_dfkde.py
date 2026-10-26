# tests/05_dfkde/test_dfkde.py: pytest pure DFKDE, original KDE equations and dynamic classifier state

import inspect
import math
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from visionlab.anomaly_detection.common.batch import InferenceBatch
from visionlab.anomaly_detection.models.components.classification import FeatureScalingMethod, KDEClassifier
from visionlab.anomaly_detection.models.components.classification import kde_classifier
from visionlab.anomaly_detection.models.components.feature_extractors import TimmFeatureExtractor
from visionlab.anomaly_detection.models.components.stats import GaussianKDE
from visionlab.anomaly_detection.models.dfkde import DfkdeModel
from visionlab.anomaly_detection.models.dfkde import model as dfkde_model

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_gaussian_kde_numpy_density_inplace_covariance_and_restore():
    """Match centered training data, Cholesky bandwidth and Gaussian density against NumPy."""
    original = torch.randn(16, 2, generator=torch.Generator().manual_seed(7)) + torch.tensor([1.0, -0.5])
    values = original.numpy().astype(np.float64)
    centered = values - values.mean(axis=0)
    covariance = centered.T @ centered / (len(values) - 1)
    factor = len(values) ** (-1 / (values.shape[1] + 4))
    bandwidth = np.linalg.cholesky(np.linalg.inv(covariance) / factor**2)
    normalization = np.diag(bandwidth).prod() * (2 * math.pi) ** (-values.shape[1] / 2)
    transformed_training = centered @ bandwidth
    covariance_input = original.T.clone()
    pointer = covariance_input.data_ptr()
    actual_covariance = GaussianKDE.cov(covariance_input)
    assert covariance_input.data_ptr() == pointer
    np.testing.assert_allclose(covariance_input.numpy(), centered.T, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(actual_covariance.numpy(), covariance, rtol=1e-5, atol=1e-6)
    dataset = original.clone()
    kde = GaussianKDE()
    kde.fit(dataset)
    np.testing.assert_allclose(dataset.numpy(), centered, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(kde.bw_transform.numpy(), bandwidth, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(kde.dataset.numpy(), transformed_training, rtol=1e-5, atol=1e-6)
    assert kde.norm.item() == pytest.approx(normalization, rel=1e-5, abs=1e-6)
    queries = torch.tensor([[0.0, 0.0], [1.0, 0.2], [-0.5, 0.7]])
    transformed_queries = queries.numpy() @ bandwidth
    expected = np.array([np.exp(-np.square(transformed_training - query).sum(axis=1) / 2).mean() * normalization
                         for query in transformed_queries])
    np.testing.assert_allclose(kde(queries).numpy(), expected, rtol=1e-5, atol=1e-6)
    restored = GaussianKDE()
    assert all(value.numel() == 0 for value in restored.buffers())
    result = restored.load_state_dict(kde.state_dict(), strict=True)
    assert result.missing_keys == result.unexpected_keys == []
    assert restored.bw_transform.shape == (2, 2) and restored.dataset.shape == (16, 2) and restored.norm.shape == ()
    for name, value in kde.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value)
    torch.testing.assert_close(restored(queries), kde(queries), rtol=1e-5, atol=1e-6)


def test_feature_scaling_enum_and_inplace_preprocessing():
    """Keep both scaling values, scalar maximum length and in-place feature division."""
    assert [method.value for method in FeatureScalingMethod] == ["norm", "scale"]
    original = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    for method in FeatureScalingMethod:
        classifier = KDEClassifier(n_pca_components=2, feature_scaling_method=method)
        assert classifier.max_length.shape == ()
        features = original.clone()
        pointer = features.data_ptr()
        scaled, maximum = classifier.pre_process(features)
        assert scaled.data_ptr() == features.data_ptr() == pointer
        assert maximum.item() == 5
        if method == FeatureScalingMethod.SCALE:
            torch.testing.assert_close(scaled, original / 5)
            explicit, _ = classifier.pre_process(original.clone(), max_length=torch.tensor(10.0))
            torch.testing.assert_close(explicit, original / 10)
        else:
            torch.testing.assert_close(torch.linalg.norm(scaled, dim=1), torch.ones(2))
            torch.testing.assert_close(scaled, original / torch.linalg.norm(original, dim=1)[:, None])


@pytest.mark.parametrize("method", (FeatureScalingMethod.SCALE, FeatureScalingMethod.NORM))
def test_classifier_fit_side_effects_scaling_and_strict_restore(method):
    """Fit PCA in place and restore every PCA, KDE and scalar scaling buffer."""
    original = torch.randn(32, 6, generator=torch.Generator().manual_seed(42)) + 2
    embeddings = original.clone()
    classifier = KDEClassifier(n_pca_components=2, feature_scaling_method=method)
    assert classifier.fit(embeddings) is True
    torch.testing.assert_close(embeddings, original - original.mean(dim=0))
    assert not torch.equal(embeddings, original)
    projected = embeddings @ classifier.pca_model.singular_vectors
    maximum = torch.linalg.norm(projected, dim=1).max()
    torch.testing.assert_close(classifier.max_length, maximum)
    scaled = projected / maximum if method == FeatureScalingMethod.SCALE else \
        projected / torch.linalg.norm(projected, dim=1)[:, None]
    expected_training = (scaled - scaled.mean(dim=0)) @ classifier.kde_model.bw_transform
    torch.testing.assert_close(classifier.kde_model.dataset, expected_training, rtol=1e-5, atol=1e-6)
    queries = original[:4].clone()
    expected = classifier(queries.clone())
    assert expected.shape == (4,) and torch.isfinite(expected).all()
    restored = KDEClassifier(n_pca_components=2, feature_scaling_method=method)
    assert restored.pca_model.mean.numel() == restored.kde_model.dataset.numel() == 0
    result = restored.load_state_dict(classifier.state_dict(), strict=True)
    assert result.missing_keys == result.unexpected_keys == []
    assert restored.pca_model.singular_vectors.shape == (6, 2) and restored.kde_model.dataset.shape == (32, 2)
    assert restored.max_length.shape == ()
    for name, value in classifier.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value)
    torch.testing.assert_close(restored(queries.clone()), expected, rtol=1e-5, atol=1e-6)


def test_classifier_insufficient_samples_and_random_subset(monkeypatch):
    """Retain insufficient-sample False and random.sample selection before PCA fitting."""
    classifier = KDEClassifier(n_pca_components=3)
    assert classifier.fit(torch.randn(2, 6)) is False
    assert classifier.pca_model.mean.numel() == classifier.kde_model.dataset.numel() == 0
    original = torch.randn(12, 4, generator=torch.Generator().manual_seed(19))
    selected = [7, 1, 9, 3, 5]
    calls = []

    def sample(population, count):
        calls.append((list(population), count))
        return selected

    monkeypatch.setattr(kde_classifier.random, "sample", sample)
    classifier = KDEClassifier(n_pca_components=2, max_training_points=5)
    embeddings = original.clone()
    assert classifier.fit(embeddings) is True
    assert calls == [(list(range(12)), 5)]
    torch.testing.assert_close(classifier.pca_model.mean, original[selected].mean(dim=0))
    assert torch.equal(embeddings, original) and classifier.kde_model.dataset.shape == (5, 2)


def test_classifier_probability_equation_and_far_point_underflow():
    """Keep the original probability equation and ineffective float32 density offset."""
    scores = torch.tensor([-100.0, -12.0, 0.0, 12.0, 30.0])
    torch.testing.assert_close(KDEClassifier.compute_probabilities(scores), 1 / (1 + torch.exp(0.05 * (scores - 12))))
    training = torch.randn(32, 4, generator=torch.Generator().manual_seed(3))
    classifier = KDEClassifier(n_pca_components=2)
    assert classifier.fit(training.clone()) is True
    near_scores = classifier(training[:3].clone())
    far = classifier.pca_model.mean + 1000 * classifier.pca_model.singular_vectors[:, 0]
    far = far.unsqueeze(0)
    far_scores = classifier(far.clone())
    assert far_scores.shape == (1,) and far_scores.item() > near_scores.max().item()
    density = classifier.compute_kde_scores(far.clone())
    assert density.dtype == torch.float32 and density.item() == 0.0
    assert torch.isneginf(classifier.compute_kde_scores(far.clone(), as_log_likelihood=True)).all()
    assert far_scores.item() == 1.0


@pytest.fixture
def model():
    """Use an offline single-layer ResNet18 with two PCA dimensions for small inputs."""
    return DfkdeModel(backbone="resnet18", layers=["layer4"], pre_trained=False, n_pca_components=2)


def test_dfkde_constructor_signature_and_original_defaults(model):
    """Keep required backbone and layers plus all four original optional defaults."""
    signature = inspect.signature(DfkdeModel.__init__)
    assert list(signature.parameters) == ["self", "backbone", "layers", "pre_trained", "n_pca_components",
                                          "feature_scaling_method", "max_training_points"]
    assert signature.parameters["backbone"].default is inspect.Parameter.empty
    assert signature.parameters["layers"].default is inspect.Parameter.empty
    expected = {"pre_trained": True, "n_pca_components": 16,
                "feature_scaling_method": FeatureScalingMethod.SCALE, "max_training_points": 40000}
    assert {name: signature.parameters[name].default for name in expected} == expected
    assert model.feature_extractor.layers == ["layer4"]


def test_dfkde_single_layer_training_features(model):
    """Return detached 512-channel features during training with one pooled ResNet layer."""
    images = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        expected = model.feature_extractor(images)["layer4"].mean(dim=(-2, -1))
    actual = model.get_features(images)
    assert actual.shape == (2, 512) and not actual.requires_grad
    torch.testing.assert_close(actual, expected)
    model.train()
    features = model(images)
    assert isinstance(features, torch.Tensor) and features.shape == (2, 512)
    torch.testing.assert_close(features, expected)


def test_dfkde_two_equally_wide_layers_concatenate_batches_and_mutate_dict(model, monkeypatch):
    """Concatenate equally wide layer outputs along dim zero and replace their dictionary entries."""
    first, second = torch.randn(2, 8, 4, 4), torch.randn(2, 8, 2, 2)
    outputs = {"first": first, "second": second}

    def extract(images):
        return outputs

    monkeypatch.setattr(model.feature_extractor, "forward", extract)
    actual = model.get_features(torch.rand(2, 3, 64, 64))
    assert actual.shape == (4, 8)
    assert outputs["first"].shape == outputs["second"].shape == (2, 8)
    expected = torch.cat([first.mean(dim=(-2, -1)), second.mean(dim=(-2, -1))], dim=0)
    torch.testing.assert_close(actual, expected)


def test_dfkde_image_only_eval_and_fitted_state_restore(model):
    """Restore feature, PCA, KDE and scaling state for identical image-only predictions."""
    assert model.classifier.fit(torch.randn(24, 512, generator=torch.Generator().manual_seed(11))) is True
    images = torch.rand(2, 3, 64, 64)
    model.eval()
    with torch.no_grad():
        expected = model(images)
    assert isinstance(expected, InferenceBatch) and expected.anomaly_map is None
    assert expected.pred_score.shape == (2,) and torch.isfinite(expected.pred_score).all()
    restored = DfkdeModel(backbone="resnet18", layers=["layer4"], pre_trained=False, n_pca_components=2).eval()
    assert restored.classifier.pca_model.mean.numel() == restored.classifier.kde_model.dataset.numel() == 0
    result = restored.load_state_dict(model.state_dict(), strict=True)
    assert result.missing_keys == result.unexpected_keys == []
    assert restored.classifier.pca_model.singular_vectors.shape == (512, 2)
    assert restored.classifier.kde_model.dataset.shape == (24, 2)
    assert restored.classifier.kde_model.bw_transform.shape == (2, 2)
    assert restored.classifier.kde_model.norm.shape == restored.classifier.max_length.shape == ()
    for name, value in model.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value)
    with torch.no_grad():
        actual = restored(images)
    assert actual.anomaly_map is None
    torch.testing.assert_close(actual.pred_score, expected.pred_score, rtol=1e-5, atol=1e-6)


def constructor_checks():
    """Check both pretrained arguments under I/O guards and infer without training frameworks."""
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
        patch.setattr(dfkde_model, "TimmFeatureExtractor", extractor)
        original = DfkdeModel(backbone="resnet18", layers=["layer4"], n_pca_components=2)
        restored = DfkdeModel(backbone="resnet18", layers=["layer4"], pre_trained=False, n_pca_components=2)
        assert calls == [False, False] and not original.feature_extractor.training
        state = original.state_dict()
        assert any(name.startswith("feature_extractor.") for name in state)
        result = restored.load_state_dict(state, strict=True)
        assert result.missing_keys == result.unexpected_keys == []
    assert restored.classifier.fit(torch.randn(24, 512, generator=torch.Generator().manual_seed(17))) is True
    restored.eval()
    with torch.no_grad():
        prediction = restored(torch.rand(2, 3, 64, 64))
    assert prediction.pred_score.shape == (2,) and prediction.anomaly_map is None


def test_constructor_and_forward_without_frameworks_or_external_io():
    """Check FR-22 and NFR-1/NFR-2 before importing the model in a fresh process."""
    script = '''
import importlib.abc
import runpy
import sys
class Block(importlib.abc.MetaPathFinder):
    """Reject reference and training frameworks before loading the pure model consumer."""
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
    """Retain original copyright and SPDX markers in all copied and extended package files."""
    root = os.path.join(PROJECT_ROOT, "src", "visionlab", "anomaly_detection", "models")
    files = [os.path.join("dfkde", name) for name in ("__init__.py", "model.py")]
    files += [os.path.join("components", "classification", name) for name in ("__init__.py", "kde_classifier.py")]
    files += [os.path.join("components", "stats", name) for name in ("__init__.py", "kde.py")]
    for name in files:
        with open(os.path.join(root, name), encoding="utf-8") as stream:
            source = stream.read()
        assert "# Copyright (C) 2022-2025 Intel Corporation" in source
        assert "# SPDX-License-Identifier: Apache-2.0" in source
