# tests/13_cflow/test_cflow.py: pytest pure CFLOW, tensor replacements and FrEIA dependency contracts

import inspect
import math
import os
import runpy
import subprocess
import sys

import einops
import pytest
import torch
from FrEIA.framework import SequenceINN
from torch import nn
from torch.nn import functional as F

from visionlab.anomaly_detection.common.batch import InferenceBatch
from visionlab.anomaly_detection.models.cflow import CflowModel
from visionlab.anomaly_detection.models.cflow import model as cflow_model
from visionlab.anomaly_detection.models.cflow.anomaly_map import AnomalyMapGenerator
from visionlab.anomaly_detection.models.cflow.utils import cflow_head, get_logp, positional_encoding_2d, subnet_fc
from visionlab.anomaly_detection.models.components.feature_extractors import TimmFeatureExtractor
from visionlab.anomaly_detection.models.components.flow import AllInOneBlock

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE_ROOT = os.path.join(PROJECT_ROOT, "src", "visionlab")
MODEL_ROOT = os.path.join(SOURCE_ROOT, "anomaly_detection", "models")
LAYERS = ["layer1", "layer2", "layer3"]


def make_model(pre_trained=True):
    """Build a small real CFLOW with uneven fiber chunks and no external weights."""
    return CflowModel(backbone="resnet18", layers=LAYERS, pre_trained=pre_trained,
                      fiber_batch_size=13, condition_vector=16, coupling_blocks=2)


@pytest.fixture
def model():
    """Provide an independent small model for each inference contract."""
    return make_model()


@pytest.mark.parametrize("noncontiguous", (False, True))
def test_cflow_torch_transforms_match_einops(noncontiguous):
    """Match batch tiling and BHW flattening bitwise for contiguous and strided inputs."""
    position = torch.randn(1, 16, 3, 5)
    features = torch.randn(2, 7, 3, 5)
    if noncontiguous:
        position = position.transpose(2, 3)
        features = features.transpose(2, 3)
    assert position.is_contiguous() is (not noncontiguous)
    assert features.is_contiguous() is (not noncontiguous)
    repeated = position.repeat(2, 1, 1, 1)
    assert torch.equal(repeated, einops.repeat(position, "b c h w-> (tile b) c h w", tile=2))
    for value in (position, repeated, features):
        actual = value.permute(0, 2, 3, 1).reshape(-1, value.shape[1])
        assert torch.equal(actual, einops.rearrange(value, "b c h w -> (b h w) c"))


def test_cflow_positional_encoding_formula_and_invalid_channels():
    """Match horizontal and vertical sine/cosine channels including the second frequency."""
    actual = positional_encoding_2d(condition_vector=8, height=2, width=3)
    assert actual.shape == (8, 2, 3)
    x = torch.tensor([0.0, 1.0, 2.0])
    y = torch.tensor([0.0, 1.0])
    for offset, frequency in ((0, 1.0), (2, 0.01)):
        assert torch.allclose(actual[offset], torch.sin(x * frequency).repeat(2, 1))
        assert torch.allclose(actual[offset + 1], torch.cos(x * frequency).repeat(2, 1))
        assert torch.allclose(actual[4 + offset], torch.sin(y * frequency)[:, None].repeat(1, 3))
        assert torch.allclose(actual[5 + offset], torch.cos(y * frequency)[:, None].repeat(1, 3))
    with pytest.raises(ValueError, match="Cannot use sin/cos positional encoding"):
        positional_encoding_2d(condition_vector=6, height=2, width=3)


def test_cflow_get_logp_formula():
    """Match a manually evaluated Gaussian log density plus the log determinant."""
    values = torch.tensor([[1.0, 2.0], [-3.0, 4.0]], dtype=torch.float64)
    jacobians = torch.tensor([0.5, -1.0], dtype=torch.float64)
    expected = torch.tensor([-math.log(2 * math.pi) - 2.5 + 0.5,
                             -math.log(2 * math.pi) - 12.5 - 1.0], dtype=torch.float64)
    assert torch.allclose(get_logp(2, values, jacobians), expected)


def test_cflow_head_structure_inverse_and_seed():
    """Keep conditional copied blocks, invertibility and seeded weights including permutations."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        decoder = cflow_head(condition_vector=16, coupling_blocks=2, clamp_alpha=1.9, n_features=8)
        following = torch.rand(4)
        torch.manual_seed(42)
        repeated = cflow_head(condition_vector=16, coupling_blocks=2, clamp_alpha=1.9, n_features=8)
        assert torch.equal(torch.rand(4), following)
        assert isinstance(decoder, SequenceINN) and len(decoder) == 2
        for name, value in decoder.state_dict().items():
            assert torch.equal(value, repeated.state_dict()[name]), name
        for block in decoder:
            assert isinstance(block, AllInOneBlock)
            assert block.conditional and block.condition_channels == 16 and block.clamp == 1.9
            assert [type(layer) for layer in block.subnet] == [nn.Linear, nn.ReLU, nn.Linear]
            assert block.subnet[0].in_features == 20 and block.subnet[0].out_features == 40
            assert block.subnet[2].in_features == 40 and block.subnet[2].out_features == 8
            assert torch.equal(block.w_perm.sum(0), torch.ones(8))
            assert torch.equal(block.w_perm.sum(1), torch.ones(8))
        values, condition = torch.randn(5, 8), torch.randn(5, 16)
        output, logdet = decoder(values, [condition])
        assert output.shape == (5, 8) and logdet.shape == (5,)
        restored, reverse_logdet = decoder(output, [condition], rev=True)
        assert torch.allclose(restored, values, atol=1e-6, rtol=1e-5)
        assert torch.allclose(logdet, -reverse_logdet, atol=1e-6, rtol=1e-5)
    subnet = subnet_fc(dims_in=7, dims_out=10)
    assert subnet(torch.randn(3, 7)).shape == (3, 10)


def test_cflow_constructor_signature_and_structure(model):
    """Retain the nine original parameters, frozen encoder and layer-wise decoder structure."""
    expected = {
        "backbone": inspect.Parameter.empty, "layers": inspect.Parameter.empty,
        "pre_trained": True, "fiber_batch_size": 64, "decoder": "freia-cflow",
        "condition_vector": 128, "coupling_blocks": 8, "clamp_alpha": 1.9, "permute_soft": False,
    }
    parameters = inspect.signature(CflowModel).parameters
    assert list(parameters) == list(expected)
    assert {name: parameter.default for name, parameter in parameters.items()} == expected
    assert all(parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD for parameter in parameters.values())
    assert model.backbone == "resnet18" and model.pool_layers == LAYERS
    assert model.fiber_batch_size == 13 and model.condition_vector == 16 and model.dec_arch == "freia-cflow"
    assert model.pool_dims == [64, 128, 256]
    assert list(model._modules) == ["encoder", "decoders", "anomaly_map_generator"]
    assert len(model.decoders) == len(LAYERS)
    assert all(isinstance(decoder, SequenceINN) and len(decoder) == 2 for decoder in model.decoders)
    assert not model.encoder.training
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert isinstance(model.anomaly_map_generator.distance, nn.PairwiseDistance)


def test_cflow_initialization_order_and_seed():
    """Match encoder then decoder initialization and subsequent CPU random draws."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        actual = make_model()
        following = torch.rand(4)
        torch.manual_seed(42)
        encoder = TimmFeatureExtractor(backbone="resnet18", layers=LAYERS, pre_trained=False).eval()
        decoders = nn.ModuleList([
            cflow_head(condition_vector=16, coupling_blocks=2, clamp_alpha=1.9,
                       n_features=channels, permute_soft=False)
            for channels in encoder.out_dims
        ])
        assert torch.equal(torch.rand(4), following)
        for built, expected in ((actual.encoder, encoder), (actual.decoders, decoders)):
            assert built.state_dict().keys() == expected.state_dict().keys()
            for name, value in built.state_dict().items():
                assert torch.equal(value, expected.state_dict()[name]), name
        torch.manual_seed(42)
        repeated = make_model(pre_trained=False)
        assert torch.equal(torch.rand(4), following)
        for name, value in actual.state_dict().items():
            assert torch.equal(value, repeated.state_dict()[name]), name


@pytest.mark.parametrize("pre_trained", (False, True))
def test_cflow_encoder_always_disables_pretrained(monkeypatch, pre_trained):
    """Pass False to the real feature extractor regardless of the public argument."""
    calls = []

    def feature_extractor(backbone, layers, pre_trained):
        calls.append(pre_trained)
        return TimmFeatureExtractor(backbone=backbone, layers=layers, pre_trained=pre_trained)

    monkeypatch.setattr(cflow_model, "TimmFeatureExtractor", feature_extractor)
    make_model(pre_trained=pre_trained)
    assert calls == [False]


def test_cflow_forward_shapes_modes_and_determinism(model):
    """Keep three-dimensional maps, per-image maxima and the original decoder train reset."""
    model.eval()
    images = torch.rand(2, 3, 64, 64)
    actual = model(images)
    assert isinstance(actual, InferenceBatch)
    assert actual.anomaly_map.shape == (2, 64, 64) and actual.pred_score.shape == (2,)
    assert torch.allclose(actual.pred_score, actual.anomaly_map.amax(dim=(-2, -1)))
    assert not actual.anomaly_map.requires_grad and torch.isfinite(actual.anomaly_map).all()
    assert not model.encoder.training and model.decoders.training
    assert all(decoder.training for decoder in model.decoders)
    repeated = model(images)
    assert torch.equal(actual.anomaly_map, repeated.anomaly_map)
    assert torch.equal(actual.pred_score, repeated.pred_score)
    single = model(images[:1])
    assert single.anomaly_map.shape == (1, 64, 64) and single.pred_score.shape == (1,)


def test_cflow_forward_preserves_batch_dependent_scores(model, monkeypatch):
    """Expose batch-wide normalization using controlled features and identity flow outputs."""
    def features(images):
        return {layer: images[:, :1, ::16, ::16].repeat(1, channels, 1, 1)
                for layer, channels in zip(LAYERS, model.pool_dims, strict=True)}

    def identity(values, condition):
        assert not torch.is_grad_enabled()
        assert condition[0].shape == (len(values), 16)
        return values, torch.zeros(len(values))

    monkeypatch.setattr(model.encoder, "forward", features)
    for decoder in model.decoders:
        monkeypatch.setattr(decoder, "forward", identity)
    images = torch.stack([torch.full((3, 64, 64), 2.0), torch.zeros(3, 64, 64)])
    model.eval()
    alone = model(images[:1])
    together = model(images)
    assert torch.allclose(alone.pred_score, torch.zeros(1), atol=1e-6, rtol=1e-5)
    assert together.pred_score[0] > alone.pred_score[0]
    assert not torch.allclose(alone.anomaly_map[0], together.anomaly_map[0], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("image_size", (None, (5, 7)))
def test_cflow_anomaly_map_formula(image_size):
    """Match batch-wide exp normalization, aligned bilinear interpolation, sum and inversion."""
    distributions = [torch.tensor([-2.0, -1.0, 0.0, 1.0, -3.0, -2.0, -1.0, 0.0], requires_grad=True),
                     torch.tensor([0.0, -2.0, -1.0, 1.0, -4.0, -1.0, 0.0, 2.0], requires_grad=True)]
    heights, widths = [2, 2], [2, 2]
    maps = []
    for distribution in distributions:
        probability = (distribution.detach() - distribution.detach().max()).exp().reshape(2, 1, 2, 2)
        if image_size is not None:
            probability = F.interpolate(probability, size=image_size, mode="bilinear", align_corners=True)
        maps.append(probability[:, 0])
    score = maps[0] + maps[1]
    expected = score.max() - score
    generator = AnomalyMapGenerator(pool_layers=["layer1", "layer2"])
    actual = generator(distribution=distributions, height=heights, width=widths, image_size=image_size)
    assert not actual.requires_grad
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert torch.equal(generator.compute_anomaly_map(distributions, heights, widths, image_size), actual)
    if image_size is None:
        assert torch.equal(generator(distribution=distributions, height=heights, width=widths), actual)


@pytest.mark.parametrize("missing", ("distribution", "height", "width"))
def test_cflow_anomaly_map_missing_keywords(missing):
    """Raise the original KeyError for each absent required keyword."""
    arguments = {"distribution": [torch.zeros(4)], "height": [2], "width": [2]}
    del arguments[missing]
    with pytest.raises(KeyError, match="Expected keys `distribution`, `height` and `width`"):
        AnomalyMapGenerator(pool_layers=["layer1"])(**arguments)


def test_cflow_strict_state_dict_round_trip(model):
    """Strictly restore a fresh model and reproduce both evaluation outputs."""
    restored = make_model().eval()
    result = restored.load_state_dict(model.state_dict(), strict=True)
    assert result.missing_keys == result.unexpected_keys == []
    images = torch.rand(2, 3, 64, 64)
    model.eval()
    expected, actual = model(images), restored(images)
    assert torch.allclose(actual.pred_score, expected.pred_score, atol=1e-6, rtol=1e-5)
    assert torch.allclose(actual.anomaly_map, expected.anomaly_map, atol=1e-6, rtol=1e-5)


def constructor_checks():
    """Construct and strictly restore under file and network guards after isolated imports."""
    def forbidden(*args, **kwargs):
        raise AssertionError("Forbidden CFLOW constructor file or network access")

    with pytest.MonkeyPatch.context() as patch:
        for target in ("builtins.open", "io.open", "os.open", "torch.load", "socket.socket.connect",
                       "socket.socket.connect_ex", "socket.create_connection"):
            patch.setattr(target, forbidden)
        original = make_model(pre_trained=True)
        restored = make_model(pre_trained=False)
        result = restored.load_state_dict(original.state_dict(), strict=True)
        assert result.missing_keys == result.unexpected_keys == []
    restored.eval()
    prediction = restored(torch.rand(2, 3, 64, 64))
    assert prediction.pred_score.shape == (2,) and prediction.anomaly_map.shape == (2, 64, 64)


@pytest.mark.parametrize("block_freia", (False, True))
def test_cflow_isolated_dependencies_and_offline_constructor(block_freia):
    """Require FrEIA for inference while blocking anomalib and both Lightning namespaces."""
    script = '''
import importlib.abc
import runpy
import sys
blocked = {"anomalib", "lightning", "pytorch_lightning"}
if sys.argv[2] == "True":
    blocked.add("FrEIA")
class Block(importlib.abc.MetaPathFinder):
    """Reject forbidden frameworks before any model imports."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, Block())
if "FrEIA" in blocked:
    try:
        from visionlab.anomaly_detection.models.cflow.model import CflowModel
    except ModuleNotFoundError as error:
        assert error.name.split(".")[0] == "FrEIA", error
    else:
        raise AssertionError("Pure CFLOW unexpectedly imported without FrEIA")
else:
    runpy.run_path(sys.argv[1])["constructor_checks"]()
    assert "FrEIA" in sys.modules
assert not blocked & {name.split(".")[0] for name in sys.modules}
'''
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.path.join(PROJECT_ROOT, "src")
    result = subprocess.run([sys.executable, "-c", script, os.path.abspath(__file__), str(block_freia)],
                            env=environment, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_cflow_source_dependency_locations_and_headers():
    """Keep einops out of source, allow only three FrEIA import sites and preserve copied headers."""
    helpers = runpy.run_path(os.path.join(PROJECT_ROOT, "tests", "common", "test_dependency.py"))
    freia_locations = set()
    for path in helpers["python_files"](SOURCE_ROOT):
        imports = helpers["source_imports"](path)
        assert all(name != "einops" for name, _ in imports), path
        if any(name == "FrEIA" for name, _ in imports):
            freia_locations.add(os.path.relpath(path, SOURCE_ROOT).replace(os.sep, "/"))
    assert freia_locations == {
        "anomaly_detection/models/components/flow/all_in_one_block.py",
        "anomaly_detection/models/fastflow/model.py",
        "anomaly_detection/models/cflow/utils.py",
    }
    pure_modules = {module for _, module in helpers["classified_modules"]()["pure"]}
    for name in ("__init__.py", "model.py", "utils.py", "anomaly_map.py"):
        path = os.path.join(MODEL_ROOT, "cflow", name)
        with open(path, encoding="utf-8") as stream:
            source = stream.read()
        assert "# Copyright (C) 2022-2025 Intel Corporation" in source
        assert "# SPDX-License-Identifier: Apache-2.0" in source
        assert all(module not in ("anomalib", "lightning", "pytorch_lightning")
                   for module, _ in helpers["source_imports"](path))
        suffix = "" if name == "__init__.py" else "." + name[:-3]
        assert "visionlab.anomaly_detection.models.cflow" + suffix in pure_modules
