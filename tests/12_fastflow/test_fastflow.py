# tests/12_fastflow/test_fastflow.py: pytest pure FastFlow and FrEIA dependency contracts
"""CaiT and DeiT are copied only, without execution: local weights are absent and models are large."""

import ast
import inspect
import os
import subprocess
import sys

import numpy as np
import pytest
import timm
import torch
from FrEIA.framework import SequenceINN
from torch import nn
from torch.nn import functional as F

from visionlab.anomaly_detection.common.batch import InferenceBatch
from visionlab.anomaly_detection.models.components.flow import AllInOneBlock
from visionlab.anomaly_detection.models.fastflow import FastflowModel
from visionlab.anomaly_detection.models.fastflow.anomaly_map import AnomalyMapGenerator
from visionlab.anomaly_detection.models.fastflow.loss import FastflowLoss
from visionlab.anomaly_detection.models.fastflow.model import create_fast_flow_block, subnet_conv_func

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE_ROOT = os.path.join(PROJECT_ROOT, "src", "visionlab")
MODEL_ROOT = os.path.join(SOURCE_ROOT, "anomaly_detection", "models")


@pytest.fixture
def model():
    """Create a small real CNN model without external weights."""
    return FastflowModel(input_size=(64, 64), backbone="resnet18", flow_steps=2)


def test_fastflow_constructor_signatures():
    """Retain every original constructor parameter in order with its original default."""
    for cls, expected in (
        (FastflowModel, {
            "input_size": inspect.Parameter.empty, "backbone": inspect.Parameter.empty,
            "pre_trained": True, "flow_steps": 8, "conv3x3_only": False, "hidden_ratio": 1.0,
        }),
        (AllInOneBlock, {
            "dims_in": inspect.Parameter.empty, "dims_c": None, "subnet_constructor": None,
            "affine_clamping": 2.0, "gin_block": False, "global_affine_init": 1.0,
            "global_affine_type": "SOFTPLUS", "permute_soft": False,
            "learned_householder_permutation": 0, "reverse_permutation": False,
        }),
    ):
        parameters = inspect.signature(cls).parameters
        assert list(parameters) == list(expected)
        assert {name: parameter.default for name, parameter in parameters.items()} == expected
        assert all(parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD for parameter in parameters.values())


def test_fastflow_all_in_one_forward_inverse_and_dimensions():
    """Restore four-dimensional inputs and cancel the forward and reverse log determinants."""
    block = AllInOneBlock(dims_in=[(8, 4, 4)], subnet_constructor=subnet_conv_func(3, 1.0))
    image = torch.randn(2, 8, 4, 4)
    output, logdet = block((image,))
    assert isinstance(output, tuple) and len(output) == 1
    assert output[0].shape == image.shape and logdet.shape == (2,)
    restored, reverse_logdet = block(output, rev=True)
    assert torch.allclose(restored[0], image, atol=1e-6, rtol=1e-5)
    assert torch.allclose(logdet, -reverse_logdet, atol=1e-6, rtol=1e-5)
    assert block.output_dims([(8, 4, 4)]) == [(8, 4, 4)]
    with pytest.raises(ValueError, match="Must be 1D-4D"):
        AllInOneBlock(dims_in=[(8, 2, 2, 2, 2)], subnet_constructor=subnet_conv_func(3, 1.0))
    with pytest.raises(ValueError, match="Dimensions of input and condition"):
        AllInOneBlock(dims_in=[(8, 4, 4)], dims_c=[(2, 3, 3)],
                      subnet_constructor=subnet_conv_func(3, 1.0))


def test_fastflow_all_in_one_seed_and_permutation_rng():
    """Match randperm before subnet initialization and leave the numpy random state untouched."""
    numpy_before = np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        indices = torch.randperm(8)
        expected_subnet = subnet_conv_func(3, 1.0)(4, 8)
        following = torch.rand(4)
        torch.manual_seed(42)
        before = torch.get_rng_state().clone()
        block = AllInOneBlock(dims_in=[(8, 4, 4)], subnet_constructor=subnet_conv_func(3, 1.0))
        assert not torch.equal(before, torch.get_rng_state())
        assert torch.equal(torch.rand(4), following)
        expected_permutation = torch.zeros(8, 8)
        expected_permutation[torch.arange(8), indices] = 1
        assert torch.equal(block.w_perm[:, :, 0, 0], expected_permutation)
        for name, value in expected_subnet.state_dict().items():
            assert torch.equal(value, block.subnet.state_dict()[name]), name
        torch.manual_seed(42)
        repeated = AllInOneBlock(dims_in=[(8, 4, 4)], subnet_constructor=subnet_conv_func(3, 1.0))
        for name, value in block.state_dict().items():
            assert torch.equal(value, repeated.state_dict()[name]), name
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    assert np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]


@pytest.mark.parametrize("conv3x3_only", (False, True))
def test_fastflow_sequence_inn_kernels_and_hidden_ratio(conv3x3_only):
    """Use the copied blocks in SequenceINN with alternating kernels and scaled subnet channels."""
    flow = create_fast_flow_block(input_dimensions=[8, 4, 4], conv3x3_only=conv3x3_only,
                                  hidden_ratio=1.5, flow_steps=3)
    assert isinstance(flow, SequenceINN) and len(flow) == 3
    for index, block in enumerate(flow):
        assert isinstance(block, AllInOneBlock)
        convolutions = [layer for layer in block.subnet if isinstance(layer, nn.Conv2d)]
        kernel = 3 if conv3x3_only or index % 2 == 0 else 1
        assert convolutions[0].kernel_size == (kernel, kernel)
        assert convolutions[0].in_channels == 4 and convolutions[0].out_channels == 6
        assert convolutions[1].out_channels == 8
        assert block.clamp == 2.0


@pytest.mark.parametrize("kernel", (1, 3))
def test_fastflow_subnet_padding_preserves_size(kernel):
    """Preserve both spatial dimensions through both manually padded convolutions."""
    subnet = subnet_conv_func(kernel_size=kernel, hidden_ratio=1.5)(4, 8)
    assert isinstance(subnet[0], nn.ZeroPad2d) and isinstance(subnet[3], nn.ZeroPad2d)
    assert subnet(torch.randn(2, 4, 5, 7)).shape == (2, 8, 5, 7)


def test_fastflow_cnn_structure(model):
    """Keep three feature stages, affine LayerNorms and frozen backbone parameters."""
    channels = model.feature_extractor.feature_info.channels()
    scales = model.feature_extractor.feature_info.reduction()
    assert channels == [64, 128, 256] and scales == [4, 8, 16]
    assert len(model.norms) == len(model.fast_flow_blocks) == 3
    assert list(model._modules) == ["feature_extractor", "norms", "fast_flow_blocks", "anomaly_map_generator"]
    model.feature_extractor.eval()
    with torch.no_grad():
        features = model.feature_extractor(torch.rand(2, 3, 64, 64))
    for feature, norm, channel, scale in zip(features, model.norms, channels, scales, strict=True):
        shape = (channel, 64 // scale, 64 // scale)
        assert feature.shape == (2, *shape)
        assert isinstance(norm, nn.LayerNorm) and tuple(norm.normalized_shape) == shape
        assert norm.elementwise_affine
    assert all(not parameter.requires_grad for parameter in model.feature_extractor.parameters())
    with pytest.raises(ValueError) as error:
        FastflowModel(input_size=(64, 64), backbone="unsupported")
    assert str(error.value) == (
        "Backbone unsupported is not supported. List of available backbones are "
        "[cait_m48_448, deit_base_distilled_patch16_384, resnet18, wide_resnet50_2]."
    )


def test_fastflow_initialization_order_and_seed():
    """Match backbone, norms and flows initialized in the original order and following RNG draws."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        first = FastflowModel(input_size=(64, 64), backbone="resnet18", flow_steps=2)
        following = torch.rand(4)
        torch.manual_seed(42)
        backbone = timm.create_model("resnet18", pretrained=False, features_only=True, out_indices=[1, 2, 3])
        channels = backbone.feature_info.channels()
        scales = backbone.feature_info.reduction()
        norms = nn.ModuleList([
            nn.LayerNorm([channel, 64 // scale, 64 // scale], elementwise_affine=True)
            for channel, scale in zip(channels, scales, strict=True)
        ])
        flows = nn.ModuleList([
            create_fast_flow_block(input_dimensions=[channel, 64 // scale, 64 // scale],
                                   conv3x3_only=False, hidden_ratio=1.0, flow_steps=2)
            for channel, scale in zip(channels, scales, strict=True)
        ])
        assert torch.equal(torch.rand(4), following)
        for actual, expected in ((first.feature_extractor, backbone), (first.norms, norms),
                                 (first.fast_flow_blocks, flows)):
            assert actual.state_dict().keys() == expected.state_dict().keys()
            for name, value in actual.state_dict().items():
                assert torch.equal(value, expected.state_dict()[name]), name
        torch.manual_seed(42)
        repeated = FastflowModel(input_size=(64, 64), backbone="resnet18", flow_steps=2)
        assert torch.equal(torch.rand(4), following)
        for name, value in first.state_dict().items():
            assert torch.equal(value, repeated.state_dict()[name]), name


def test_fastflow_training_eval_and_gradient_contract(model):
    """Keep training lists, evaluation batch shapes and flow gradients while freezing the backbone."""
    image = torch.rand(2, 3, 64, 64)
    model.train()
    training = model(image)
    assert isinstance(training, tuple) and len(training) == 2
    hidden, jacobians = training
    assert len(hidden) == len(jacobians) == 3
    channels = model.feature_extractor.feature_info.channels()
    scales = model.feature_extractor.feature_info.reduction()
    for value, jacobian, channel, scale in zip(hidden, jacobians, channels, scales, strict=True):
        assert value.shape == (2, channel, 64 // scale, 64 // scale) and jacobian.shape == (2,)
    assert not model.feature_extractor.training
    FastflowLoss()(hidden, jacobians).backward()
    assert all(parameter.grad is None for parameter in model.feature_extractor.parameters())
    for block in model.fast_flow_blocks:
        gradients = [parameter.grad for parameter in block.parameters() if parameter.requires_grad]
        assert gradients and all(value is not None and torch.isfinite(value).all() for value in gradients)
        assert any(value.abs().sum() > 0 for value in gradients)
    model.eval()
    with torch.no_grad():
        prediction = model(image)
    assert isinstance(prediction, InferenceBatch)
    assert prediction.anomaly_map.shape == (2, 1, 64, 64) and prediction.pred_score.shape == (2, 1)
    assert torch.allclose(prediction.pred_score, prediction.anomaly_map.amax(dim=(-2, -1)),
                          atol=1e-6, rtol=1e-5)
    model.train()
    with torch.no_grad():
        model(image)
    assert not model.feature_extractor.training


def test_fastflow_anomaly_map_formula_and_list_input():
    """Match the manually calculated negative probabilities and bilinear average."""
    hidden = [torch.arange(32).float().reshape(2, 4, 2, 2) / 10,
              torch.tensor([0.5, 1.5, 2.5, 3.5]).reshape(2, 2, 1, 1)]
    generator = AnomalyMapGenerator(input_size=[7, 9])
    assert generator.input_size == (7, 9)
    reference = sum(F.interpolate(-torch.exp(-(value * value).mean(dim=1, keepdim=True) / 2),
                                  size=(7, 9), mode="bilinear", align_corners=False) for value in hidden) / 2
    assert torch.allclose(generator(hidden), reference, atol=1e-6, rtol=1e-5)
    with open(os.path.join(MODEL_ROOT, "fastflow", "anomaly_map.py"), encoding="utf-8") as stream:
        source = stream.read()
    assert "omegaconf" not in source and "ListConfig" not in source


def test_fastflow_loss_formula_determinism_and_jacobian_direction():
    """Sum scale-wise negative log likelihood and reduce loss when Jacobians increase."""
    hidden = [torch.arange(32).float().reshape(2, 4, 2, 2) / 10,
              torch.tensor([0.5, 1.5, 2.5, 3.5]).reshape(2, 2, 1, 1)]
    jacobians = [torch.tensor([1.0, 2.0]), torch.tensor([-0.5, 0.5])]
    reference = sum(((value * value).flatten(1).sum(dim=1) / 2 - jacobian).mean()
                    for value, jacobian in zip(hidden, jacobians, strict=True))
    criterion = FastflowLoss()
    actual = criterion(hidden, jacobians)
    assert actual.shape == ()
    assert torch.allclose(actual, reference, atol=1e-6, rtol=1e-5)
    assert torch.equal(actual, criterion(hidden, jacobians))
    reduced = criterion(hidden, [jacobian + 1 for jacobian in jacobians])
    assert reduced < actual and torch.allclose(reduced, actual - 2)


def test_fastflow_strict_state_dict_round_trip(model):
    """Restore a fresh model strictly and reproduce both evaluation outputs."""
    restored = FastflowModel(input_size=(64, 64), backbone="resnet18", flow_steps=2).eval()
    result = restored.load_state_dict(model.state_dict(), strict=True)
    assert result.missing_keys == result.unexpected_keys == []
    model.eval()
    image = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        expected, actual = model(image), restored(image)
    assert torch.allclose(actual.pred_score, expected.pred_score, atol=1e-6, rtol=1e-5)
    assert torch.allclose(actual.anomaly_map, expected.anomaly_map, atol=1e-6, rtol=1e-5)


def constructor_checks():
    """Construct and strictly restore without file or network access after guarded imports."""
    def forbidden(*args, **kwargs):
        raise AssertionError("Forbidden FastFlow constructor file or network access")

    with pytest.MonkeyPatch.context() as patch:
        for target in ("builtins.open", "io.open", "os.open", "torch.load", "socket.socket.connect",
                       "socket.socket.connect_ex", "socket.create_connection"):
            patch.setattr(target, forbidden)
        original = FastflowModel(input_size=(64, 64), backbone="resnet18", pre_trained=True, flow_steps=2)
        restored = FastflowModel(input_size=(64, 64), backbone="resnet18", pre_trained=False, flow_steps=2)
        result = restored.load_state_dict(original.state_dict(), strict=True)
        assert result.missing_keys == result.unexpected_keys == []
    restored.eval()
    with torch.no_grad():
        prediction = restored(torch.rand(2, 3, 64, 64))
    assert prediction.pred_score.shape == (2, 1) and prediction.anomaly_map.shape == (2, 1, 64, 64)


@pytest.mark.parametrize("block_freia", (False, True))
def test_fastflow_isolated_dependency_and_offline_constructor(block_freia):
    """Require FrEIA for pure inference while blocking anomalib and both Lightning namespaces."""
    script = '''
import importlib.abc
import runpy
import sys
blocked = {"anomalib", "lightning", "pytorch_lightning"}
if sys.argv[2] == "True":
    blocked.add("FrEIA")
class Block(importlib.abc.MetaPathFinder):
    """Reject forbidden dependencies before any model imports."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, Block())
if "FrEIA" in blocked:
    try:
        from visionlab.anomaly_detection.models.fastflow.model import FastflowModel
    except ModuleNotFoundError as error:
        assert error.name.split(".")[0] == "FrEIA", error
    else:
        raise AssertionError("Pure FastFlow unexpectedly imported without FrEIA")
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


def test_fastflow_freia_import_locations():
    """Limit actual FrEIA imports to the pure flow models and copied common flow block."""
    locations = set()
    for directory, _, names in os.walk(SOURCE_ROOT):
        for name in names:
            if not name.endswith(".py"):
                continue
            path = os.path.join(directory, name)
            with open(path, encoding="utf-8") as stream:
                tree = ast.parse(stream.read(), filename=path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    modules = [node.module or ""]
                else:
                    continue
                if any(module.split(".")[0] == "FrEIA" for module in modules):
                    locations.add(os.path.relpath(path, SOURCE_ROOT).replace(os.sep, "/"))
    assert locations == {
        "anomaly_detection/models/fastflow/model.py",
        "anomaly_detection/models/components/flow/all_in_one_block.py",
        "anomaly_detection/models/cflow/utils.py",
    }


def test_fastflow_original_spdx_headers():
    """Preserve both upstream flow licenses and all FastFlow Apache headers."""
    for relative in ("components/flow/__init__.py", "components/flow/all_in_one_block.py",
                     "fastflow/__init__.py", "fastflow/model.py", "fastflow/loss.py", "fastflow/anomaly_map.py"):
        with open(os.path.join(MODEL_ROOT, relative), encoding="utf-8") as stream:
            source = stream.read()
        assert "# SPDX-License-Identifier: Apache-2.0" in source, relative
        assert "# Copyright (C) 2022-2025 Intel Corporation" in source, relative
        if relative == "components/flow/all_in_one_block.py":
            assert "# SPDX-License-Identifier: MIT" in source
            assert "# Copyright (c) https://github.com/vislearn/FrEIA" in source
