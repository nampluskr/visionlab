# tests/08_cfa/test_cfa.py: pytest pure CFA, tensor rearrangements and fitting-only sklearn dependency

import inspect
import os
import runpy
import subprocess
import sys

from einops import rearrange
import pytest
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from visionlab.anomaly_detection.common.batch import InferenceBatch
from visionlab.anomaly_detection.models.cfa import CfaModel
from visionlab.anomaly_detection.models.cfa import model as cfa_model
from visionlab.anomaly_detection.models.cfa.anomaly_map import AnomalyMapGenerator
from visionlab.anomaly_detection.models.cfa.loss import CfaLoss
from visionlab.anomaly_detection.models.cfa.model import (
    SUPPORTED_BACKBONES, AddCoords, CoordConv2d, Descriptor, get_feature_extractor, get_return_nodes,
)
from visionlab.anomaly_detection.models.components.filters import GaussianBlur2d

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_model(gamma_c=1):
    """Build a small pure CFA with compressed descriptors and no external weights."""
    return CfaModel(backbone="resnet18", gamma_c=gamma_c, gamma_d=8, num_nearest_neighbors=3,
                    num_hard_negative_features=3, radius=1e-5)


def make_loader():
    """Provide three uneven batches of raw RGB values for centroid initialization."""
    samples = [(torch.rand(3, 64, 64) * (index + 1) / 5, {"label": 0}) for index in range(5)]
    return DataLoader(samples, batch_size=2, shuffle=False, num_workers=0)


@pytest.fixture(scope="module")
def initialized_cfa():
    """Initialize real descriptor centroids once from a small raw tuple loader."""
    model = make_model()
    loader = make_loader()
    with torch.no_grad():
        average = None
        for index, (image, _) in enumerate(loader):
            target = model.descriptor(list(model.feature_extractor(image).values())).mean(dim=0, keepdim=True)
            average = target if average is None else (average * index + target) / (index + 1)
    expected = average.permute(0, 2, 3, 1).reshape(-1, average.shape[1]).permute(1, 0)
    model.initialize_centroid(loader)
    return model, loader, expected


@pytest.mark.parametrize("noncontiguous", (False, True))
@pytest.mark.parametrize("pattern", ("flatten", "transpose", "spatial", "unflatten"))
def test_cfa_torch_rearrangements_match_einops(pattern, noncontiguous):
    """Match all four replacements bitwise for contiguous and strided tensors."""
    if pattern == "transpose":
        value = torch.randn(5, 7)
        if noncontiguous:
            value = value.T
        actual, expected = value.permute(1, 0), rearrange(value, "h w -> w h")
    elif pattern == "unflatten":
        value = torch.randn(2, 24, 3) if not noncontiguous else torch.randn(2, 3, 24).transpose(1, 2)
        actual = value.reshape(2, 4, 6, 3).permute(0, 3, 1, 2)
        expected = rearrange(value, "b (h w) c -> b c h w", h=4, w=6)
    else:
        value = torch.randn(2, 3, 4, 6)
        if noncontiguous:
            value = value.transpose(2, 3)
        batch, channels, height, width = value.shape
        if pattern == "flatten":
            actual = value.permute(0, 2, 3, 1).reshape(-1, channels)
            expected = rearrange(value, "b c h w -> (b h w) c")
        else:
            actual = value.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
            expected = rearrange(value, "b c h w -> b (h w) c")
    assert value.is_contiguous() is (not noncontiguous)
    assert torch.equal(actual, expected)


def test_cfa_return_nodes_and_offline_feature_factory(monkeypatch):
    """Keep all supported node names and use only weights=None for torchvision construction."""
    assert SUPPORTED_BACKBONES == ("vgg19_bn", "resnet18", "wide_resnet50_2", "efficientnet_b5")
    for backbone in ("resnet18", "wide_resnet50_2"):
        assert get_return_nodes(backbone) == ["layer1", "layer2", "layer3"]
    assert get_return_nodes("vgg19_bn") == ["features.25", "features.38", "features.52"]
    with pytest.raises(NotImplementedError, match="EfficientNet feature extractor has not implemented yet"):
        get_return_nodes("efficientnet_b5")
    with pytest.raises(ValueError, match="Backbone unsupported is not supported"):
        get_return_nodes("unsupported")
    calls = []
    factory = torchvision.models.resnet18

    def resnet18(weights):
        calls.append(weights)
        return factory(weights=weights)

    def forbidden(*args, **kwargs):
        raise AssertionError("Torchvision must not load pretrained weights")

    monkeypatch.setattr(torchvision.models, "resnet18", resnet18)
    monkeypatch.setattr(torchvision.models.ResNet18_Weights, "get_state_dict", forbidden)
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", forbidden)
    extractor = get_feature_extractor("resnet18", get_return_nodes("resnet18"))
    assert calls == [None] and not extractor.training
    with torch.no_grad():
        features = extractor(torch.rand(1, 3, 64, 64))
    assert [value.shape[1] for value in features.values()] == [64, 128, 256]


@pytest.mark.parametrize("with_r", (False, True))
def test_cfa_coordinate_grid_and_coordconv(with_r):
    """Preserve signed coordinate channels and the original radius offset in the optional channel."""
    image = torch.rand(2, 3, 4, 6)
    xx = (torch.arange(4).float() / 3 * 2 - 1).reshape(1, 1, 4, 1).expand(2, 1, 4, 6)
    yy = (torch.arange(6).float() / 5 * 2 - 1).reshape(1, 1, 1, 6).expand(2, 1, 4, 6)
    expected = torch.cat([image, xx, yy], dim=1)
    if with_r:
        expected = torch.cat([expected, ((xx - 0.5).square() + (yy - 0.5).square()).sqrt()], dim=1)
    assert torch.equal(AddCoords(with_r=with_r)(image), expected)
    conv = CoordConv2d(in_channels=3, out_channels=5, kernel_size=3, padding=1, with_r=with_r)
    assert conv(image).shape == (2, 5, 4, 6)
    torch.testing.assert_close(conv(image), conv.conv2d(expected))


@pytest.mark.parametrize("backbone,channels", (("resnet18", 448), ("wide_resnet50_2", 1792),
                                             ("vgg19_bn", 1280), ("efficientnet_b5", 568)))
def test_cfa_descriptor_dimensions(backbone, channels):
    """Retain all backbone channel counts and the efficientnet-specific output multiplier."""
    descriptor = Descriptor(gamma_d=4, backbone=backbone)
    features = {"first": torch.rand(2, channels // 4, 8, 8),
                "second": torch.rand(2, channels - channels // 4, 4, 4)}
    output = descriptor(features)
    expected_channels = (2 * channels if backbone == "efficientnet_b5" else channels) // 4
    assert output.shape == (2, expected_channels, 8, 8)
    assert descriptor.layer.in_channels == channels
    with pytest.raises(ValueError, match="Supported backbones"):
        Descriptor(gamma_d=1, backbone="unsupported")


def test_cfa_constructor_contract_and_uninitialized_memory():
    """Keep six required arguments, the scalar bank and the original non-Parameter radius tensor."""
    signature = inspect.signature(CfaModel.__init__)
    names = ["backbone", "gamma_c", "gamma_d", "num_nearest_neighbors", "num_hard_negative_features", "radius"]
    assert list(signature.parameters) == ["self"] + names
    assert all(signature.parameters[name].default is inspect.Parameter.empty for name in names)
    model = make_model()
    assert model.memory_bank.ndim == 0 and model.memory_bank.item() == 0
    assert model.radius.requires_grad and not isinstance(model.radius, nn.Parameter)
    assert model.radius.item() == pytest.approx(1e-5) and "radius" not in model.state_dict()
    with pytest.raises(ValueError, match="Memory bank is not initialized"):
        model(torch.rand(2, 3, 64, 64))


def test_cfa_centroid_raw_tuple_batches_and_original_incremental_mean(initialized_cfa):
    """Average batch means equally on raw images and store transposed descriptor centroids."""
    model, _, expected = initialized_cfa
    assert model.memory_bank.shape == (56, 256)
    torch.testing.assert_close(model.memory_bank, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("resolution,expected", ((16, (16, 16)), ((8, 12), (8, 12)), ([8, 12], None)))
def test_cfa_get_scale_resolution_contract(initialized_cfa, monkeypatch, resolution, expected):
    """Keep original integer and tuple scales and reject other metadata resolution types."""
    model, _, _ = initialized_cfa

    def metadata(feature_extractor, input_size, layers):
        assert input_size == (64, 64) and layers == ["layer1", "layer2", "layer3"]
        return {"layer1": {"resolution": resolution}}

    monkeypatch.setattr(cfa_model, "dryrun_find_featuremap_dims", metadata)
    if expected is None:
        with pytest.raises(TypeError, match="Unknown type"):
            model.get_scale((64, 64))
    else:
        assert model.get_scale((64, 64)) == expected


def test_cfa_compute_distance_matches_direct_squared_distances(monkeypatch, initialized_cfa):
    """Match the original quadratic expression for both spatial and preflattened strided features."""
    model, _, _ = initialized_cfa
    bank = torch.randn(7, 11, dtype=torch.float64)
    monkeypatch.setattr(model, "memory_bank", bank)
    features = torch.randn(2, 7, 4, 6, dtype=torch.float64).transpose(2, 3)
    points = rearrange(features, "b c h w -> b (h w) c")
    expected = (points.unsqueeze(2) - bank.T[None, None]).square().sum(dim=-1)
    torch.testing.assert_close(model.compute_distance(features), expected)
    torch.testing.assert_close(model.compute_distance(points), expected)


def test_cfa_train_eval_squeeze_and_dynamic_bank_restore(initialized_cfa):
    """Keep distance training outputs and squeezed batch-one inference while restoring a fitted bank."""
    model, loader, _ = initialized_cfa
    image = next(iter(loader))[0]
    model.train()
    distance = model(image)
    assert distance.shape == (2, 256, 256) and distance.requires_grad and not model.feature_extractor.training
    model.eval()
    with torch.no_grad():
        expected = model(image)
        single = model(image[:1])
    assert isinstance(expected, InferenceBatch)
    assert expected.anomaly_map.shape == (2, 64, 64) and expected.pred_score.shape == (2,)
    torch.testing.assert_close(expected.pred_score, expected.anomaly_map.amax(dim=(-2, -1)))
    assert single.anomaly_map.shape == (64, 64) and single.pred_score.ndim == 0
    torch.testing.assert_close(single.pred_score, single.anomaly_map.amax(dim=(-2, -1)))
    restored = make_model().eval()
    assert restored.memory_bank.ndim == 0
    result = restored.load_state_dict(model.state_dict(), strict=True)
    assert result.missing_keys == result.unexpected_keys == [] and restored.memory_bank.shape == (56, 256)
    with torch.no_grad():
        actual = restored(image)
    torch.testing.assert_close(actual.pred_score, expected.pred_score, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual.anomaly_map, expected.anomaly_map, rtol=1e-5, atol=1e-6)


def test_cfa_anomaly_scores_maps_and_required_kwargs():
    """Match sqrt, nearest softmin weighting, detached reshaping, interpolation and final blur."""
    generator = AnomalyMapGenerator(num_nearest_neighbors=3)
    distance = (torch.rand(2, 16 * 20, 7) + 0.1).requires_grad_()
    nearest = distance.sqrt().sort(dim=-1).values[..., :3]
    weighted = F.softmin(nearest, dim=-1)[..., 0] * nearest[..., 0]
    expected = rearrange(weighted.unsqueeze(-1), "b (h w) c -> b c h w", h=16, w=20).detach()
    score = generator.compute_score(distance, scale=(16, 20))
    assert not score.requires_grad
    torch.testing.assert_close(score, expected)
    expected_map = GaussianBlur2d(sigma=4)(F.interpolate(expected, size=(64, 80), mode="bilinear", align_corners=False))
    torch.testing.assert_close(generator.compute_anomaly_map(score, image_size=(64, 80)), expected_map)
    torch.testing.assert_close(generator(distance=distance, scale=(16, 20), image_size=(64, 80)), expected_map)
    # Reflect padding requires each spatial dimension to exceed the blur's half-kernel width.
    large_score = torch.rand(2, 2, 24, 24)
    torch.testing.assert_close(generator.compute_anomaly_map(large_score),
                               GaussianBlur2d(sigma=4)(large_score.mean(dim=1, keepdim=True)))
    with pytest.raises(ValueError, match="Expected keys"):
        generator(scale=(16, 20))
    with pytest.raises(ValueError, match="Expected keys"):
        generator(distance=distance)


def test_cfa_loss_matches_reference_radius_and_determinism():
    """Keep the original attraction and hard-negative slices, radius contribution and scaling."""
    distance = torch.tensor([[[0.01, 0.04, 0.09, 0.16, 0.25, 0.36, 0.49],
                              [0.49, 0.01, 0.16, 0.09, 0.64, 0.04, 0.25]]])
    values = []
    for radius in (0.1, 0.6):
        criterion = CfaLoss(num_nearest_neighbors=2, num_hard_negative_features=3, radius=radius)
        nearest = distance.sort(dim=-1).values[..., :5]
        attraction = (nearest[..., :2] - radius ** 2).clamp_min(0).mean()
        repulsion = (radius ** 2 - nearest[..., 3:] - 0.1).clamp_min(0).mean()
        expected = (attraction + repulsion) * 1000
        actual = criterion(distance)
        torch.testing.assert_close(actual, expected)
        assert torch.equal(actual, criterion(distance))
        values.append(actual.detach())
    assert not torch.equal(*values)


def constructor_checks():
    """Build offline under I/O guards and initialize and infer with sklearn imports blocked."""
    calls = []
    factory = torchvision.models.resnet18
    get_feature_extractor("resnet18", get_return_nodes("resnet18"))

    def resnet18(weights):
        calls.append(weights)
        return factory(weights=weights)

    def forbidden(*args, **kwargs):
        raise AssertionError("Forbidden CFA constructor file or network access")

    with pytest.MonkeyPatch.context() as patch:
        for target in ("builtins.open", "io.open", "os.open", "torch.load", "socket.socket.connect",
                       "socket.socket.connect_ex", "socket.create_connection", "torch.hub.load_state_dict_from_url"):
            patch.setattr(target, forbidden)
        patch.setattr(torchvision.models, "resnet18", resnet18)
        model = make_model()
        assert calls == [None] and model.memory_bank.ndim == 0
    model.initialize_centroid(make_loader())
    model.eval()
    with torch.no_grad():
        output = model(torch.rand(2, 3, 64, 64))
    assert output.anomaly_map.shape == (2, 64, 64)
    compressed = make_model(gamma_c=2)
    try:
        compressed.initialize_centroid(make_loader())
    except ModuleNotFoundError as error:
        assert "sklearn" in str(error), error
    else:
        raise AssertionError("gamma_c > 1 must import sklearn")


def test_cfa_isolated_import_offline_inference_and_kmeans_dependency():
    """Check gamma_c=1 without sklearn and gamma_c>1 import failure; KMeans numerics are not tested."""
    script = '''
import importlib.abc
import runpy
import sys
import torch
import torchvision
import timm
assert not any(name.split(".")[0] == "sklearn" for name in sys.modules), "External library warmup imported sklearn"
class Block(importlib.abc.MetaPathFinder):
    """Reject reference, training and sklearn imports after external library warmup."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("anomalib", "lightning", "pytorch_lightning", "sklearn"):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, Block())
sys.path.insert(0, sys.argv[2])
runpy.run_path(sys.argv[1])["constructor_checks"]()
assert not {"anomalib", "lightning", "pytorch_lightning", "sklearn"} & set(sys.modules)
'''
    result = subprocess.run([sys.executable, "-c", script, os.path.abspath(__file__), os.path.join(PROJECT_ROOT, "src")],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_cfa_headers_and_no_source_einops_imports():
    """Preserve all copied licenses and prevent direct einops imports anywhere in visionlab."""
    root = os.path.join(PROJECT_ROOT, "src", "visionlab")
    helpers = runpy.run_path(os.path.join(PROJECT_ROOT, "tests", "common", "test_dependency.py"))
    for path in helpers["python_files"](root):
        assert all(name != "einops" for name, _ in helpers["source_imports"](path)), path
    for name in ("__init__.py", "model.py", "anomaly_map.py", "loss.py"):
        with open(os.path.join(root, "anomaly_detection", "models", "cfa", name), encoding="utf-8") as stream:
            source = stream.read()
        assert "# Copyright (C) 2022-2025 Intel Corporation" in source
        assert "# SPDX-License-Identifier: Apache-2.0" in source


@pytest.mark.parametrize("rule_index", (0, 1))
def test_cfa_shared_sklearn_allowlist_scopes(tmp_path, monkeypatch, rule_index):
    """Reject top-level, nested, aliased, duplicate and mismatched sklearn imports for both exact rules."""
    helpers = runpy.run_path(os.path.join(PROJECT_ROOT, "tests", "common", "test_dependency.py"))
    check = helpers["allowed_sklearn_import_lines"]
    monkeypatch.setitem(check.__globals__, "SOURCE_ROOT", str(tmp_path))
    relative, cls, method, module, name = helpers["SKLEARN_IMPORTS"][rule_index]
    path = os.path.join(tmp_path, relative)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    statement = f"from {module} import {name}"
    valid = f"class {cls}:\n    def {method}(self):\n        if True:\n            {statement}\n"
    invalid = [statement, valid.replace(cls, "OtherClass"), valid.replace(method, "other_method"),
               valid.replace(module, "sklearn.other"), valid.replace(statement, f"from {module} import Other"),
               valid.replace(statement, statement + " as alias"), valid.replace(statement, statement + ", other"),
               valid + f"            {statement}\n",
               f"class {cls}:\n    def {method}(self):\n        def nested():\n            {statement}\n"]
    for source in [valid] + invalid:
        with open(path, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(source)
        assert bool(check(path)) is (source == valid)
