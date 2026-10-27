# tests/14_uflow/test_uflow.py: pytest pure UFlow and FrEIA dependency contracts
"""MCait and wide_resnet50_2 are copied only, without execution: local weights are absent or models are large."""

import inspect
import os
import runpy
import subprocess
import sys

import pytest
import torch
from FrEIA import framework as ff
from FrEIA import modules as fm
from torch import nn
from torch.nn import functional as F

from visionlab.anomaly_detection.common.batch import InferenceBatch
from visionlab.anomaly_detection.models.components.feature_extractors import TimmFeatureExtractor
from visionlab.anomaly_detection.models.components.flow import AllInOneBlock
from visionlab.anomaly_detection.models.uflow import UflowModel
from visionlab.anomaly_detection.models.uflow.anomaly_map import AnomalyMapGenerator
from visionlab.anomaly_detection.models.uflow.feature_extraction import (
    AVAILABLE_EXTRACTORS,
    LayerNormFeatureExtractor,
    get_feature_extractor,
)
from visionlab.anomaly_detection.models.uflow.loss import UFlowLoss
from visionlab.anomaly_detection.models.uflow.model import AffineCouplingSubnet

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE_ROOT = os.path.join(PROJECT_ROOT, "src", "visionlab")
MODEL_ROOT = os.path.join(SOURCE_ROOT, "anomaly_detection", "models")


def make_model():
    """Build a real three-scale UFlow without external weights."""
    return UflowModel(input_size=(256, 256), flow_steps=2, backbone="resnet18")


@pytest.fixture
def model():
    """Provide an independent model for each pure inference contract."""
    return make_model()


@pytest.mark.parametrize("kernel", (1, 3))
@pytest.mark.parametrize("ratio", (0.5, 1.5))
def test_uflow_subnet_shape_and_ratio(kernel, ratio):
    """Preserve spatial dimensions with both kernels and scale the intermediate channels."""
    constructor = AffineCouplingSubnet(kernel_size=kernel, subnet_channels_ratio=ratio)
    assert constructor.kernel_size == kernel and constructor.subnet_channels_ratio == ratio
    subnet = constructor(in_channels=6, out_channels=10)
    assert [type(layer) for layer in subnet] == [nn.Conv2d, nn.ReLU, nn.Conv2d]
    assert subnet[0].in_channels == 6 and subnet[0].out_channels == int(6 * ratio)
    assert subnet[2].in_channels == int(6 * ratio) and subnet[2].out_channels == 10
    assert all(layer.kernel_size == (kernel, kernel) and layer.padding == "same"
               for layer in (subnet[0], subnet[2]))
    assert subnet(torch.randn(2, 6, 5, 7)).shape == (2, 10, 5, 7)


def test_uflow_constructor_signature_and_module_order(model):
    """Retain all six original constructor parameters and the original module creation order."""
    expected = {
        "input_size": (448, 448), "flow_steps": 4, "backbone": "mcait",
        "affine_clamp": 2.0, "affine_subnet_channels_ratio": 1.0, "permute_soft": False,
    }
    parameters = inspect.signature(UflowModel).parameters
    assert list(parameters) == list(expected)
    assert {name: parameter.default for name, parameter in parameters.items()} == expected
    assert all(parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD for parameter in parameters.values())
    assert list(model._modules) == ["feature_extractor", "flow", "anomaly_map_generator"]
    assert model.input_size == (256, 256)
    assert model.affine_clamp == 2.0 and model.affine_subnet_channels_ratio == 1.0
    assert model.permute_soft is False


def test_uflow_feature_extractor_structure_and_pretrained(monkeypatch):
    """Disable pretrained loading and keep the backbone frozen while retaining affine LayerNorms."""
    calls = []
    original = TimmFeatureExtractor.__init__

    def feature_init(self, backbone, layers, pre_trained=True, requires_grad=False):
        calls.append((backbone, layers, pre_trained, requires_grad))
        original(self, backbone, layers, pre_trained=pre_trained, requires_grad=requires_grad)

    monkeypatch.setattr(TimmFeatureExtractor, "__init__", feature_init)
    extractor = get_feature_extractor(backbone="resnet18", input_size=(256, 256))
    assert calls == [("resnet18", ("layer1", "layer2", "layer3"), False, False)]
    assert AVAILABLE_EXTRACTORS == ["mcait", "resnet18", "wide_resnet50_2"]
    assert isinstance(extractor, LayerNormFeatureExtractor) and isinstance(extractor, TimmFeatureExtractor)
    assert not extractor.training and not extractor.requires_grad
    assert extractor.channels == [64, 128, 256] and extractor.scale_factors == [4, 8, 16]
    assert list(extractor.scales) == [0, 1, 2]
    assert all(not parameter.requires_grad for parameter in extractor.feature_extractor.parameters())
    assert all(parameter.requires_grad for parameter in extractor.feature_normalizations.parameters())
    with torch.no_grad():
        features = extractor(torch.rand(2, 3, 256, 256))
    for feature, norm, channels, scale in zip(features, extractor.feature_normalizations,
                                             extractor.channels, extractor.scale_factors, strict=True):
        shape = (channels, 256 // scale, 256 // scale)
        assert feature.shape == (2, *shape)
        assert isinstance(norm, nn.LayerNorm) and tuple(norm.normalized_shape) == shape
        assert norm.elementwise_affine
    with pytest.raises(ValueError, match="Feature extractor must be one of"):
        get_feature_extractor(backbone="unsupported", input_size=(256, 256))


def test_uflow_graph_nodes_and_dimensions(model):
    """Keep the original node names, split sizes, scale ordering and alternating coupling kernels."""
    assert isinstance(model.flow, ff.GraphINN)
    assert [node.name for node in model.flow.in_nodes] == ["cond_64", "cond_128", "cond_256"]
    assert [node.name for node in model.flow.out_nodes] == ["output_scale_1", "output_scale_2", "output_scale_3"]
    nodes = {node.name: node for node in model.flow.node_list}
    assert len(nodes) == len(model.flow.node_list) == 18
    assert set(nodes) == {
        "cond_64", "cond_128", "cond_256", "flow16_step0", "flow16_step1",
        "flow32_step0", "flow32_step1", "flow64_step0", "flow64_step1",
        "split_3", "split_2", "up_3", "up_2", "cat_2", "cat_1",
        "output_scale_1", "output_scale_2", "output_scale_3",
    }
    assert model.flow.global_out_shapes == [(84, 64, 64), (80, 32, 32), (128, 16, 16)]
    for scale, channels in ((3, 256), (2, 160)):
        split = nodes[f"split_{scale}"]
        assert isinstance(split.module, fm.Split)
        assert split.module_args == {"section_sizes": (channels // 8 * 4, channels - channels // 8 * 4), "dim": 0}
        assert split.inputs == [(nodes[f"flow{16 if scale == 3 else 32}_step1"], 0)]
        up = nodes[f"up_{scale}"]
        assert isinstance(up.module, fm.IRevNetUpsampling) and up.inputs == [(split, 0)]
        cat = nodes[f"cat_{scale - 1}"]
        assert isinstance(cat.module, fm.Concat) and cat.module_args == {"dim": 0}
        assert cat.inputs == [(nodes[f"cond_{128 if scale == 3 else 64}"], 0), (up, 0)]
        assert nodes[f"output_scale_{scale}"].inputs == [(split, 1)]
    assert nodes["output_scale_1"].inputs == [(nodes["flow64_step1"], 0)]
    for size in (16, 32, 64):
        for step in range(2):
            node = nodes[f"flow{size}_step{step}"]
            assert isinstance(node.module, AllInOneBlock) and not node.conditions
            assert node.module_args["affine_clamping"] == 2.0 and node.module_args["permute_soft"] is False
            constructor = node.module_args["subnet_constructor"]
            kernel = 3 if step % 2 == 0 else 1
            assert constructor.kernel_size == kernel and constructor.subnet_channels_ratio == 1.0
            assert node.module.subnet[0].kernel_size == (kernel, kernel)


def test_uflow_seeded_initial_state_and_following_rng():
    """Reproduce every initialized tensor including permutations and subsequent random draws."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        first = make_model()
        following = torch.rand(4)
        torch.manual_seed(42)
        repeated = make_model()
        assert torch.equal(torch.rand(4), following)
        assert first.state_dict().keys() == repeated.state_dict().keys()
        for name, value in first.state_dict().items():
            assert torch.equal(value, repeated.state_dict()[name]), name
        permutations = [module.w_perm for module in first.flow.module_list if isinstance(module, AllInOneBlock)]
        assert len(permutations) == 6
        for permutation in permutations:
            matrix = permutation[:, :, 0, 0]
            assert torch.equal(matrix.sum(0), torch.ones(matrix.shape[1]))
            assert torch.equal(matrix.sum(1), torch.ones(matrix.shape[0]))


def test_uflow_forward_modes_gradients_and_inverse(model):
    """Keep tuple latents, invertibility, frozen backbone and trainable flow and normalization parameters."""
    image = torch.rand(2, 3, 256, 256)
    model.train()
    training = model(image)
    assert isinstance(training, tuple) and len(training) == 2
    hidden, logdet = training
    assert isinstance(hidden, tuple) and len(hidden) == 3 and logdet.shape == (2,)
    assert [value.shape for value in hidden] == [(2, 84, 64, 64), (2, 80, 32, 32), (2, 128, 16, 16)]
    assert not model.feature_extractor.feature_extractor.training
    UFlowLoss()(hidden, logdet).backward()
    assert all(parameter.grad is None for parameter in model.feature_extractor.feature_extractor.parameters())
    for component in (model.flow, model.feature_extractor.feature_normalizations):
        gradients = [parameter.grad for parameter in component.parameters() if parameter.requires_grad]
        assert gradients and all(value is not None and torch.isfinite(value).all() for value in gradients)
        assert any(value.abs().sum() > 0 for value in gradients)
    model.eval()
    with torch.no_grad():
        features = model.feature_extractor(image)
        hidden, logdet = model.encode(features)
        restored, reverse_logdet = model.flow(hidden, rev=True)
        for actual, expected in zip(restored, features, strict=True):
            assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
        assert torch.allclose(logdet, -reverse_logdet, atol=1e-6, rtol=1e-5)
        prediction = model(image)
    assert isinstance(prediction, InferenceBatch)
    assert prediction.anomaly_map.shape == (2, 1, 256, 256) and prediction.pred_score.shape == (2, 1)
    assert torch.allclose(prediction.pred_score, prediction.anomaly_map.amax(dim=(-2, -1)),
                          atol=1e-6, rtol=1e-5)
    model.train()
    with torch.no_grad():
        model(image)
    assert not model.feature_extractor.feature_extractor.training


def test_uflow_encode_single_scale_wraps_tensor(model, monkeypatch):
    """Wrap the single-scale GraphINN tensor result without changing its value or log determinant."""
    latent, logdet = torch.randn(2, 4, 2, 2), torch.randn(2)
    features = [torch.randn(2, 4, 2, 2)]

    def single_scale(values, rev):
        assert values is features and rev is False
        return latent, logdet

    monkeypatch.setattr(model.feature_extractor, "scales", range(1))
    monkeypatch.setattr(model.flow, "forward", single_scale)
    actual, actual_logdet = model.encode(features)
    assert isinstance(actual, list) and len(actual) == 1 and actual[0] is latent
    assert actual_logdet is logdet


def test_uflow_anomaly_map_formula_and_list_size():
    """Match one minus the mean of interpolated scale probabilities and remove OmegaConf dependency."""
    hidden = [torch.arange(32).float().reshape(2, 4, 2, 2) / 10,
              torch.tensor([0.5, 1.5, 2.5, 3.5]).reshape(2, 2, 1, 1)]
    generator = AnomalyMapGenerator(input_size=[256, 256])
    assert isinstance(generator.input_size, tuple) and generator.input_size == (256, 256)
    reference = 1 - sum(F.interpolate(torch.exp(-(value * value).mean(dim=1, keepdim=True) / 2),
                                      size=(256, 256), mode="bilinear", align_corners=False)
                        for value in hidden) / len(hidden)
    assert torch.allclose(generator(hidden), reference, atol=1e-6, rtol=1e-5)
    with open(os.path.join(MODEL_ROOT, "uflow", "anomaly_map.py"), encoding="utf-8") as stream:
        source = stream.read()
    assert "omegaconf" not in source and "ListConfig" not in source


def test_uflow_loss_formula_and_determinism():
    """Retain the original scalar sum across both scales and batch before the Jacobian mean."""
    hidden = [torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(2, 2, 1, 1),
              torch.tensor([5.0, 6.0]).reshape(2, 1, 1, 1)]
    logdet = torch.tensor([1.0, 3.0])
    reference = torch.tensor(0.5 * (1 + 4 + 9 + 16 + 25 + 36) - 2)
    criterion = UFlowLoss()
    actual = criterion(hidden, logdet)
    assert actual.shape == () and torch.allclose(actual, reference, atol=1e-6, rtol=1e-5)
    assert torch.equal(actual, criterion(hidden, logdet))


def test_uflow_strict_state_dict_round_trip(model):
    """Strictly restore the backbone, normalization and flow state and reproduce evaluation outputs."""
    restored = make_model().eval()
    result = restored.load_state_dict(model.state_dict(), strict=True)
    assert result.missing_keys == result.unexpected_keys == []
    image = torch.rand(2, 3, 256, 256)
    model.eval()
    with torch.no_grad():
        expected, actual = model(image), restored(image)
    assert torch.allclose(actual.pred_score, expected.pred_score, atol=1e-6, rtol=1e-5)
    assert torch.allclose(actual.anomaly_map, expected.anomaly_map, atol=1e-6, rtol=1e-5)


def constructor_checks():
    """Construct and strictly restore under file and network guards after isolated imports."""
    def forbidden(*args, **kwargs):
        raise AssertionError("Forbidden UFlow constructor file or network access")

    with pytest.MonkeyPatch.context() as patch:
        for target in ("builtins.open", "io.open", "os.open", "torch.load", "socket.socket.connect",
                       "socket.socket.connect_ex", "socket.create_connection"):
            patch.setattr(target, forbidden)
        original = make_model()
        restored = make_model()
        result = restored.load_state_dict(original.state_dict(), strict=True)
        assert result.missing_keys == result.unexpected_keys == []
    restored.eval()
    with torch.no_grad():
        prediction = restored(torch.rand(2, 3, 256, 256))
    assert prediction.pred_score.shape == (2, 1) and prediction.anomaly_map.shape == (2, 1, 256, 256)


@pytest.mark.parametrize("block_freia", (False, True))
def test_uflow_isolated_dependencies_and_offline_constructor(block_freia):
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
        from visionlab.anomaly_detection.models.uflow.model import UflowModel
    except ModuleNotFoundError as error:
        assert error.name.split(".")[0] == "FrEIA", error
    else:
        raise AssertionError("Pure UFlow unexpectedly imported without FrEIA")
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


def test_uflow_source_dependency_locations_and_headers():
    """Allow four FrEIA import sites, classify pure UFlow modules and preserve original SPDX headers."""
    helpers = runpy.run_path(os.path.join(PROJECT_ROOT, "tests", "common", "test_dependency.py"))
    freia_locations = set()
    for path in helpers["python_files"](SOURCE_ROOT):
        if any(name == "FrEIA" for name, _ in helpers["source_imports"](path)):
            freia_locations.add(os.path.relpath(path, SOURCE_ROOT).replace(os.sep, "/"))
    assert freia_locations == {
        "anomaly_detection/models/components/flow/all_in_one_block.py",
        "anomaly_detection/models/fastflow/model.py",
        "anomaly_detection/models/cflow/utils.py",
        "anomaly_detection/models/uflow/model.py",
    }
    pure_modules = {module for _, module in helpers["classified_modules"]()["pure"]}
    for name in ("__init__.py", "model.py", "feature_extraction.py", "anomaly_map.py", "loss.py"):
        path = os.path.join(MODEL_ROOT, "uflow", name)
        with open(path, encoding="utf-8") as stream:
            source = stream.read()
        assert "# Copyright (C) 2023-2024 Intel Corporation" in source
        assert "# SPDX-License-Identifier: Apache-2.0" in source
        assert all(module not in ("anomalib", "lightning", "pytorch_lightning")
                   for module, _ in helpers["source_imports"](path))
        suffix = "" if name == "__init__.py" else "." + name[:-3]
        assert "visionlab.anomaly_detection.models.uflow" + suffix in pure_modules
