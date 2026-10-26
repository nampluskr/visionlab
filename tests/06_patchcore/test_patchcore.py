# tests/06_patchcore/test_patchcore.py: pytest pure PatchCore, sparse projection and greedy coreset contracts

import inspect
import os
import subprocess
import sys
from contextlib import contextmanager

import numpy as np
import pytest
import torch
from torch.nn import functional

from visionlab.anomaly_detection.common.batch import InferenceBatch
from visionlab.anomaly_detection.models.components.dimensionality_reduction import SparseRandomProjection
from visionlab.anomaly_detection.models.components.dimensionality_reduction.random_projection import NotFittedError
from visionlab.anomaly_detection.models.components.feature_extractors import TimmFeatureExtractor
from visionlab.anomaly_detection.models.components.sampling import KCenterGreedy
from visionlab.anomaly_detection.models.components.sampling import k_center_greedy
from visionlab.anomaly_detection.models.components.tiling.tiler import Tiler
from visionlab.anomaly_detection.models.patchcore import PatchcoreModel
from visionlab.anomaly_detection.models.patchcore import model as patchcore_model
from visionlab.anomaly_detection.models.patchcore.anomaly_map import AnomalyMapGenerator

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@contextmanager
def seeded_rng(seed):
    """Restore NumPy and PyTorch RNG states after deterministic sampling checks."""
    previous = np.random.get_state()
    with torch.random.fork_rng():
        np.random.seed(seed)
        torch.manual_seed(seed)
        try:
            yield
        finally:
            np.random.set_state(previous)


@pytest.mark.parametrize("features", (1, 16))
def test_sparse_projection_dimensions_and_repeatable_matrix(features):
    """Keep the JL dimension formula and both dense and sparse matrix generation branches."""
    embedding = torch.randn(200, features, generator=torch.Generator().manual_seed(7))
    expected = int(4 * np.log(200) / (0.9**2 / 2 - 0.9**3 / 3))
    with seeded_rng(42):
        first = SparseRandomProjection(eps=0.9)
        assert first.fit(embedding) is first
    with seeded_rng(42):
        second = SparseRandomProjection(eps=0.9)
        second.fit(embedding)
    assert first.n_components == expected
    assert first.sparse_random_matrix.shape == (expected, features)
    assert torch.equal(first.sparse_random_matrix, second.sparse_random_matrix)
    projected = first.transform(embedding)
    assert projected.shape == (200, expected)
    torch.testing.assert_close(projected, embedding @ first.sparse_random_matrix.T.float())


def test_projection_original_unset_attribute_and_explicit_none():
    """Preserve the original unset attribute before fit and the explicit None error branch."""
    projector = SparseRandomProjection(eps=0.9)
    assert not hasattr(projector, "sparse_random_matrix")
    with pytest.raises(AttributeError) as error:
        projector.transform(torch.zeros(2, 4))
    assert type(error.value) is AttributeError
    projector.sparse_random_matrix = None
    with pytest.raises(NotFittedError):
        projector.transform(torch.zeros(2, 4))


def test_coreset_shape_uniqueness_and_seed_repeatability():
    """Select an actual greedy coreset reproducibly with both RNG streams fixed."""
    embedding = torch.randn(200, 16, generator=torch.Generator().manual_seed(19))
    with seeded_rng(42):
        sampler = KCenterGreedy(embedding=embedding, sampling_ratio=0.1)
        indices = sampler.select_coreset_idxs()
    assert sampler.coreset_size == 20 and len(indices) == len(set(indices)) == 20
    with seeded_rng(42):
        repeated = KCenterGreedy(embedding=embedding, sampling_ratio=0.1).sample_coreset()
    assert repeated.shape == (20, 16)
    assert torch.equal(repeated, embedding[indices])


def test_coreset_farthest_points_and_outlier_in_projected_space(monkeypatch):
    """Check each selected point against independent distances to all previous projected centers."""
    embedding = torch.randn(200, 16, generator=torch.Generator().manual_seed(11))
    embedding[-1] = 100
    calls = []

    def first_index(high, size):
        calls.append((high, size))
        return torch.tensor([0])

    monkeypatch.setattr(k_center_greedy.torch, "randint", first_index)
    with seeded_rng(42):
        sampler = KCenterGreedy(embedding=embedding, sampling_ratio=0.1)
        indices = sampler.select_coreset_idxs()
    assert calls == [(200, (1,))] and 199 in indices
    centers = [0]
    selected = []
    for index in indices:
        differences = sampler.features[:, None, :] - sampler.features[centers][None, :, :] + 1e-6
        nearest = torch.linalg.vector_norm(differences, dim=-1).amin(dim=1)
        nearest[selected] = 0
        torch.testing.assert_close(nearest[index], nearest.max(), rtol=1e-5, atol=1e-6)
        centers.append(index)
        selected.append(index)
    assert sampler.min_distances[indices[-1]].item() == 0


def test_coreset_original_distance_and_selected_index_errors(monkeypatch):
    """Retain errors for missing distances and a newly chosen preselected index."""
    sampler = KCenterGreedy(embedding=torch.randn(20, 4), sampling_ratio=0.1)
    with pytest.raises(TypeError):
        sampler.get_new_idx()
    monkeypatch.setattr(sampler, "get_new_idx", lambda: 3)
    with seeded_rng(42), pytest.raises(ValueError, match="New indices should not be in selected indices"):
        sampler.select_coreset_idxs(selected_idxs=[3])


@pytest.fixture
def model():
    """Construct an offline two-layer ResNet18 PatchCore model for small input checks."""
    return PatchcoreModel(layers=("layer2", "layer3"), backbone="resnet18", pre_trained=False, num_neighbors=3)


def test_patchcore_constructor_signature_and_pooler(model):
    """Preserve four constructor arguments, empty memory and original local averaging."""
    signature = inspect.signature(PatchcoreModel.__init__)
    assert list(signature.parameters) == ["self", "layers", "backbone", "pre_trained", "num_neighbors"]
    assert signature.parameters["layers"].default is inspect.Parameter.empty
    expected = {"backbone": "wide_resnet50_2", "pre_trained": True, "num_neighbors": 9}
    assert {name: signature.parameters[name].default for name in expected} == expected
    assert model.tiler is None and model.memory_bank.numel() == 0
    assert model.feature_pooler.kernel_size == 3 and model.feature_pooler.stride == model.feature_pooler.padding == 1


def test_patchcore_bilinear_embedding_and_flatten_order(model):
    """Merge actual pooled feature layers by bilinear interpolation and flatten in patch order."""
    with torch.no_grad():
        raw = model.feature_extractor(torch.rand(2, 3, 64, 80))
    features = {name: model.feature_pooler(value) for name, value in raw.items()}
    expected = torch.cat([features["layer2"], functional.interpolate(features["layer3"],
                         size=features["layer2"].shape[-2:], mode="bilinear")], dim=1)
    actual = model.generate_embedding(features)
    assert actual.shape == (2, 128 + 256, 8, 10)
    torch.testing.assert_close(actual, expected)
    embedding = torch.arange(2 * 3 * 2 * 4).reshape(2, 3, 2, 4)
    flattened = model.reshape_embedding(embedding)
    reference = torch.stack([embedding[batch, :, row, column]
                             for batch in range(2) for row in range(2) for column in range(4)])
    assert flattened.shape == (16, 3)
    assert torch.equal(flattened, reference)


def test_euclidean_dist_and_neighbor_branches(model):
    """Match independent distances and both nearest-neighbor output shapes without changing inputs."""
    x, y = torch.randn(7, 5), torch.randn(11, 5)
    original_x, original_y = x.clone(), y.clone()
    torch.testing.assert_close(model.euclidean_dist(x, y), torch.cdist(x, y), rtol=1e-5, atol=1e-6)
    assert torch.equal(x, original_x) and torch.equal(y, original_y)
    model.memory_bank = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0], [6.0, 7.0]])
    query = torch.tensor([[0.2, 0.7], [4.2, 2.4]])
    distances = torch.cdist(query, model.memory_bank)
    for neighbors in (1, 3):
        scores, indices = model.nearest_neighbors(query, n_neighbors=neighbors)
        expected_scores, expected_indices = distances.min(dim=1) if neighbors == 1 else \
            distances.topk(k=neighbors, largest=False, dim=1)
        assert scores.shape == ((2,) if neighbors == 1 else (2, 3))
        torch.testing.assert_close(scores, expected_scores, rtol=1e-5, atol=1e-6)
        assert torch.equal(indices, expected_indices)


@pytest.mark.parametrize("neighbors", (2, 8))
def test_anomaly_score_maximum_and_original_support_weights(model, neighbors):
    """Compare the maximum-only branch and the original first-support softmax weight formula."""
    bank = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0], [6.0, 7.0]])
    model.memory_bank = bank
    embeddings = torch.tensor([[0.2, 0.1], [3.5, 1.0], [7.0, 9.0], [1.0, 1.0], [2.4, 0.0], [0.0, 8.0]])
    patch_scores, locations = torch.cdist(embeddings, bank).min(dim=1)
    patch_scores, locations = patch_scores.reshape(2, 3), locations.reshape(2, 3)
    model.num_neighbors = 1
    torch.testing.assert_close(model.compute_anomaly_score(patch_scores, locations, embeddings), patch_scores.amax(dim=1))
    model.num_neighbors = neighbors
    expected = []
    for batch in range(2):
        patch = int(patch_scores[batch].argmax())
        query = embeddings[batch * 3 + patch]
        anchor = bank[locations[batch, patch]]
        support = torch.linalg.vector_norm(bank - anchor, dim=1).topk(k=min(neighbors, len(bank)), largest=False).indices
        distances = torch.linalg.vector_norm(bank[support] - query, dim=1)
        weight = 1 - distances[0].exp() / distances.exp().sum()
        expected.append(weight * patch_scores[batch, patch])
    torch.testing.assert_close(model.compute_anomaly_score(patch_scores, locations, embeddings),
                               torch.stack(expected), rtol=1e-5, atol=1e-6)


def test_anomaly_map_nearest_interpolation_optional_size_and_constant():
    """Retain nearest upsampling, optional no-resize behavior and constant-preserving blur."""
    generator = AnomalyMapGenerator()
    scores = torch.randn(2, 1, 24, 28)
    actual = generator(scores, image_size=(48, 56))
    expected = generator.blur(functional.interpolate(scores, size=(48, 56), mode="nearest"))
    assert actual.shape == (2, 1, 48, 56)
    torch.testing.assert_close(actual, expected)
    unchanged_size = generator(scores, image_size=None)
    assert unchanged_size.shape == scores.shape
    torch.testing.assert_close(unchanged_size, generator.blur(scores))
    constant = torch.full_like(scores, 2.5)
    torch.testing.assert_close(generator(constant), constant)
    torch.testing.assert_close(generator(constant, (48, 56)), torch.full((2, 1, 48, 56), 2.5))


def test_patchcore_training_memory_fit_eval_and_strict_restore(model):
    """Fit a small actual coreset and restore dynamic memory with identical nonsquare predictions."""
    images = torch.rand(2, 3, 64, 80)
    model.train()
    embedding = model(images)
    assert isinstance(embedding, torch.Tensor) and embedding.shape == (160, 384) and not embedding.requires_grad
    with seeded_rng(42):
        model.subsample_embedding(embedding, sampling_ratio=0.1)
    assert model.memory_bank.shape == (16, 384)
    model.eval()
    with torch.no_grad():
        expected = model(images)
    assert isinstance(expected, InferenceBatch)
    assert expected.pred_score.shape == (2,) and expected.anomaly_map.shape == (2, 1, 64, 80)
    assert torch.isfinite(expected.pred_score).all() and torch.isfinite(expected.anomaly_map).all()
    restored = PatchcoreModel(layers=("layer2", "layer3"), backbone="resnet18", pre_trained=False, num_neighbors=3).eval()
    assert restored.memory_bank.numel() == 0
    result = restored.load_state_dict(model.state_dict(), strict=True)
    assert result.missing_keys == result.unexpected_keys == [] and restored.memory_bank.shape == (16, 384)
    for name, value in model.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value)
    with torch.no_grad():
        actual = restored(images)
    torch.testing.assert_close(actual.pred_score, expected.pred_score, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual.anomaly_map, expected.anomaly_map, rtol=1e-5, atol=1e-6)


def test_patchcore_optional_tiler_branch(model):
    """Keep tile and untile paths while preserving full input map size."""
    model.tiler = Tiler(tile_size=32, stride=32)
    images = torch.rand(1, 3, 64, 64)
    model.train()
    assert model(images).shape == (64, 384)
    model.memory_bank = torch.randn(16, 384)
    model.eval()
    with torch.no_grad():
        prediction = model(images)
    assert prediction.pred_score.shape == (1,) and prediction.anomaly_map.shape == (1, 1, 64, 64)


def constructor_checks():
    """Construct under I/O guards and strictly restore inference memory without sklearn or frameworks."""
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
        patch.setattr(patchcore_model, "TimmFeatureExtractor", extractor)
        original = PatchcoreModel(layers=("layer2", "layer3"), backbone="resnet18", num_neighbors=3)
        restored = PatchcoreModel(layers=("layer2", "layer3"), backbone="resnet18", pre_trained=False, num_neighbors=3)
        assert calls == [False, False] and not original.feature_extractor.training
        assert any(name.startswith("feature_extractor.") for name in original.state_dict())
    original.memory_bank = torch.randn(16, 384)
    result = restored.load_state_dict(original.state_dict(), strict=True)
    assert result.missing_keys == result.unexpected_keys == []
    restored.eval()
    with torch.no_grad():
        prediction = restored(torch.rand(2, 3, 64, 80))
    assert prediction.pred_score.shape == (2,) and prediction.anomaly_map.shape == (2, 1, 64, 80)


def test_isolated_import_inference_and_fitting_dependency():
    """Import and infer without sklearn while requiring it only for sparse projection fitting."""
    script = '''
import importlib.abc
import runpy
import sys
import torch
import timm
assert not any(name.split(".")[0] == "sklearn" for name in sys.modules), (
    "External torch/timm initialization imported sklearn before visionlab or its import guard"
)
modules_before = set(sys.modules)
class Block(importlib.abc.MetaPathFinder):
    """Reject training frameworks and sklearn after external library initialization."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("anomalib", "lightning", "pytorch_lightning", "sklearn"):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, Block())
sys.path.insert(0, sys.argv[2])
from visionlab.anomaly_detection.models.components.dimensionality_reduction.random_projection import SparseRandomProjection
namespace = runpy.run_path(sys.argv[1])
namespace["constructor_checks"]()
try:
    SparseRandomProjection(eps=0.9).fit(torch.randn(200, 16))
except ModuleNotFoundError as error:
    assert str(error).startswith("sklearn"), error
else:
    raise AssertionError("Sparse fitting must require sklearn")
assert not {"anomalib", "lightning", "pytorch_lightning"} & set(sys.modules)
assert not any(name.split(".")[0] == "sklearn" for name in set(sys.modules) - modules_before), (
    "The visionlab pure import or inference path newly imported sklearn"
)
'''
    result = subprocess.run([sys.executable, "-c", script, os.path.abspath(__file__), os.path.join(PROJECT_ROOT, "src")],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_original_copyright_and_spdx_headers():
    """Retain original copyright and SPDX markers in all seven copied or extended source files."""
    root = os.path.join(PROJECT_ROOT, "src", "visionlab", "anomaly_detection", "models")
    paths = [os.path.join("patchcore", name) for name in ("__init__.py", "model.py", "anomaly_map.py")]
    paths += [os.path.join("components", "sampling", name) for name in ("__init__.py", "k_center_greedy.py")]
    paths += [os.path.join("components", "dimensionality_reduction", name) for name in ("__init__.py", "random_projection.py")]
    for path in paths:
        with open(os.path.join(root, path), encoding="utf-8") as stream:
            source = stream.read()
        assert "# Copyright (C) 2022-2025 Intel Corporation" in source
        assert "# SPDX-License-Identifier: Apache-2.0" in source
