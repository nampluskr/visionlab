# tests/15_csflow/test_csflow.py: pytest pure CSFlow equations and dependency contracts
"""Use hidden channels 16 and two blocks; the default configuration has about 275 million parameters."""

import inspect
import os
import runpy
import subprocess
import sys

import numpy as np
import pytest
import torch
from FrEIA.framework import GraphINN
from torch import nn
from torch.nn import functional as F

from visionlab.anomaly_detection.common.batch import InferenceBatch
from visionlab.anomaly_detection.models.csflow import CsFlowModel
from visionlab.anomaly_detection.models.csflow import model as csflow_model
from visionlab.anomaly_detection.models.csflow.anomaly_map import AnomalyMapGenerator, AnomalyMapMode
from visionlab.anomaly_detection.models.csflow.loss import CsFlowLoss
from visionlab.anomaly_detection.models.csflow.model import (
    CrossConvolutions,
    CrossScaleFlow,
    ParallelGlowCouplingLayer,
    ParallelPermute,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE_ROOT = os.path.join(PROJECT_ROOT, "src", "visionlab")
MODEL_ROOT = os.path.join(SOURCE_ROOT, "anomaly_detection", "models", "csflow")


def make_model():
    """Build a small real CSFlow and reset only test BatchNorm statistics after the uninitialized dry-run."""
    model = CsFlowModel(input_size=(256, 256), cross_conv_hidden_channels=16, n_coupling_blocks=2)
    # The original train-mode torch.empty dry-run can leave non-finite running statistics.
    for module in model.feature_extractor.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.reset_running_stats()
    return model


@pytest.fixture
def model():
    """Provide an independent small model without pretrained assets."""
    return make_model()


@pytest.mark.parametrize("kernel", (3, 5))
def test_csflow_cross_convolutions_shapes_and_zero_gamma(kernel):
    """Retain all three spatial scales and zero outputs with the original gamma initialization."""
    block = CrossConvolutions(in_channels=4, channels=6, channels_hidden=16, kernel_size=kernel)
    inputs = [torch.randn(2, 4, size, size) for size in (8, 4, 2)]
    assert block.use_gamma is True
    assert all(torch.equal(gamma, torch.zeros(1)) for gamma in (block.gamma0, block.gamma1, block.gamma2))
    outputs = block(*inputs)
    assert [value.shape for value in outputs] == [(2, 6, size, size) for size in (8, 4, 2)]
    assert all(torch.count_nonzero(value) == 0 for value in outputs)


def test_csflow_parallel_permute_seed_inverse_state_and_forward():
    """Use a local numpy generator and retain ordinary unregistered permutation lists."""
    dims = [(8, size, size) for size in (8, 4, 2)]
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1)
            np.random.seed(1)
            first = ParallelPermute(dims_in=dims, seed=7)
            torch.manual_seed(999)
            np.random.seed(999)
            second = ParallelPermute(dims_in=dims, seed=7)
    finally:
        np.random.set_state(numpy_state)
    assert isinstance(first.perm, list) and isinstance(first.perm_inv, list)
    assert not first.state_dict() and not list(first.buffers())
    inputs = [torch.randn(2, *shape) for shape in dims]
    outputs, jacobian = first(inputs, rev=False)
    assert jacobian == 0.0
    for index, (perm, inverse) in enumerate(zip(first.perm, first.perm_inv, strict=True)):
        assert perm.dtype == inverse.dtype == torch.int64
        assert torch.equal(perm, second.perm[index])
        assert torch.equal(perm, torch.tensor(np.random.default_rng(7).permutation(8)))
        assert torch.equal(perm[inverse], torch.arange(8))
        assert torch.equal(inverse[perm], torch.arange(8))
        assert torch.equal(outputs[index], inputs[index][:, perm])


@pytest.mark.parametrize("gamma", (0.0, 0.25))
def test_csflow_coupling_forward_formula_and_scalar_jacobian(gamma):
    """Match forward affine equations and the scalar sum over batch and scales without testing reverse."""
    block = ParallelGlowCouplingLayer(
        dims_in=[(8, size, size) for size in (8, 4, 2)],
        subnet_args={"channels_hidden": 16, "kernel_size": 3}, clamp=3,
    )
    for module in (block.cross_convolution1, block.cross_convolution2):
        for parameter in (module.gamma0, module.gamma1, module.gamma2):
            with torch.no_grad():
                parameter.fill_(gamma)
    inputs = [torch.randn(2, 8, size, size) for size in (8, 4, 2)]
    first = [value[:, :4] for value in inputs]
    second = [value[:, 4:] for value in inputs]
    scales2, shifts2 = zip(*(value.chunk(2, dim=1) for value in block.cross_convolution2(*second)))
    logs2 = [3 * 0.636 * torch.atan(value / 3) for value in scales2]
    transformed1 = [torch.exp(log) * value + shift
                    for log, value, shift in zip(logs2, first, shifts2, strict=True)]
    scales1, shifts1 = zip(*(value.chunk(2, dim=1) for value in block.cross_convolution1(*transformed1)))
    logs1 = [3 * 0.636 * torch.atan(value / 3) for value in scales1]
    transformed2 = [torch.exp(log) * value + shift
                    for log, value, shift in zip(logs1, second, shifts1, strict=True)]
    reference = [torch.cat((a, b), dim=1).clamp(-1e6, 1e6)
                 for a, b in zip(transformed1, transformed2, strict=True)]
    reference_jacobian = sum(value.sum() for value in logs1 + logs2)
    outputs, jacobian = block(inputs, rev=False)
    assert isinstance(outputs, list) and len(outputs) == 3 and jacobian.shape == ()
    for actual, expected in zip(outputs, reference, strict=True):
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert torch.allclose(jacobian, reference_jacobian, atol=1e-6, rtol=1e-5)
    if gamma == 0:
        assert jacobian == 0
        assert all(torch.equal(actual, expected) for actual, expected in zip(outputs, inputs, strict=True))
    else:
        assert any(not torch.allclose(actual, original) for actual, original in zip(outputs, inputs, strict=True))


def test_csflow_constructor_signature_and_graph(model):
    """Preserve constructor order, nested GraphINN nodes, dimensions, kernels and creation order."""
    expected = {"input_size": inspect.Parameter.empty, "cross_conv_hidden_channels": inspect.Parameter.empty,
                "n_coupling_blocks": 4, "clamp": 3, "num_channels": 3}
    parameters = inspect.signature(CsFlowModel).parameters
    assert list(parameters) == list(expected)
    assert {name: parameter.default for name, parameter in parameters.items()} == expected
    assert all(parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD for parameter in parameters.values())
    assert list(model._modules) == ["feature_extractor", "graph", "anomaly_map_generator"]
    assert model.input_dims == (3, 256, 256) and model.clamp == 3 and model.cross_conv_hidden_channels == 16
    assert isinstance(model.graph, CrossScaleFlow) and model.graph.kernel_sizes == [3, 5]
    graph = model.graph.graph
    assert isinstance(graph, GraphINN)
    assert [node.name for node in graph.in_nodes] == ["input", "input2", "input3"]
    assert [node.name for node in graph.out_nodes] == ["output_end0", "output_end1", "output_end2"]
    shapes = [(304, size, size) for size in (8, 4, 2)]
    assert [node.dims for node in graph.in_nodes] == shapes and graph.global_out_shapes == shapes
    nodes = {node.name: node for node in graph.node_list}
    assert len(nodes) == len(graph.node_list) == 10
    assert set(nodes) == {"input", "input2", "input3", "permute_0", "permute_1", "fc1_0", "fc1_1",
                          "output_end0", "output_end1", "output_end2"}
    for step, kernel in enumerate((3, 5)):
        permute, coupling = nodes[f"permute_{step}"], nodes[f"fc1_{step}"]
        assert isinstance(permute.module, ParallelPermute) and permute.module_args == {"seed": step}
        assert isinstance(coupling.module, ParallelGlowCouplingLayer)
        assert coupling.module_args == {"clamp": 3, "subnet_args": {"channels_hidden": 16, "kernel_size": kernel}}
        previous = [(node, 0) for node in graph.in_nodes] if step == 0 else [(nodes["fc1_0"], i) for i in range(3)]
        assert permute.inputs == previous
        assert coupling.inputs == [(permute, i) for i in range(3)]
        for subnet in (coupling.module.cross_convolution1, coupling.module.cross_convolution2):
            assert subnet.conv_scale0_0.kernel_size == (kernel, kernel)
    assert all(node.inputs == [(nodes["fc1_1"], index)] for index, node in enumerate(graph.out_nodes))


def test_csflow_same_seed_initial_parameters():
    """Compare initialized parameters independently of uninitialized dry-run BatchNorm buffers."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        first = make_model()
        torch.manual_seed(42)
        second = make_model()
    first_parameters, second_parameters = dict(first.named_parameters()), dict(second.named_parameters())
    assert first_parameters.keys() == second_parameters.keys()
    assert all(torch.equal(value, second_parameters[name]) for name, value in first_parameters.items())


def test_csflow_backbone_configuration_and_shared_parameters(monkeypatch):
    """Request weights=None and preserve shared FX parameters and the no-grad configuration."""
    calls = []
    original = csflow_model.efficientnet_b5

    def build_backbone(weights):
        calls.append(weights)
        return original(weights=weights)

    monkeypatch.setattr(csflow_model, "efficientnet_b5", build_backbone)
    model = make_model()
    assert calls == [None]
    assert not model.feature_extractor.training
    extractor = model.feature_extractor.feature_extractor
    assert extractor.requires_grad is False
    assert extractor.layers == ["features.6.8"] and extractor.out_dims == [304]
    original_parameters = dict(extractor.backbone.named_parameters())
    extracted_parameters = dict(extractor.feature_extractor.named_parameters())
    assert extracted_parameters
    assert all(parameter is original_parameters[name] for name, parameter in extracted_parameters.items())
    state = extractor.state_dict()
    assert all("backbone." + name in state and "feature_extractor." + name in state
               for name in extracted_parameters)
    with open(os.path.join(MODEL_ROOT, "model.py"), encoding="utf-8") as stream:
        source = stream.read()
    assert "EfficientNet_B5_Weights" not in source
    assert "backbone=efficientnet_b5(weights=None)" in source


def test_csflow_features_and_initial_training_forward(model):
    """Return the original GraphINN tuple and batch Jacobian with identity couplings and seeded permutations."""
    images = torch.rand(2, 3, 256, 256)
    features = model.feature_extractor(images)
    shapes = [(2, 304, size, size) for size in (8, 4, 2)]
    assert [value.shape for value in features] == shapes
    assert all(not value.requires_grad for value in features)
    assert model.training
    latents, jacobians = model(images)
    assert isinstance(latents, tuple) and len(latents) == 3
    assert [value.shape for value in latents] == shapes
    assert jacobians.shape == (2,) and torch.equal(jacobians, torch.zeros(2))
    expected = features
    for step in range(2):
        expected, _ = ParallelPermute(dims_in=[tuple(value.shape[1:]) for value in features], seed=step)(expected)
    assert all(torch.equal(actual, reference) for actual, reference in zip(latents, expected, strict=True))


def test_csflow_eval_score_formula_shapes_and_batch_independence(model):
    """Return unsqueezed maps and per-image means that agree for single and paired batches."""
    model.eval()
    images = torch.rand(2, 3, 256, 256)
    with torch.no_grad():
        latents, _ = model.graph(model.feature_extractor(images))
        expected_score = (torch.cat([value.flatten(1) for value in latents], dim=1).square() / 2).mean(dim=1)
        actual, alone = model(images), model(images[:1])
    assert isinstance(actual, InferenceBatch)
    assert actual.pred_score.shape == (2,) and actual.anomaly_map.shape == (2, 1, 256, 256)
    assert alone.pred_score.shape == (1,) and alone.anomaly_map.shape == (1, 1, 256, 256)
    assert torch.isfinite(actual.pred_score).all() and torch.isfinite(actual.anomaly_map).all()
    assert torch.allclose(actual.pred_score, expected_score, atol=1e-6, rtol=1e-5)
    assert torch.allclose(alone.pred_score, actual.pred_score[:1], atol=1e-6, rtol=1e-5)


def test_csflow_training_gradients_only_reach_flow(model):
    """Keep backbone gradients absent and allow nonzero flow gamma gradients at initialization."""
    model.train()
    latents, jacobians = model(torch.rand(2, 3, 256, 256))
    CsFlowLoss()(latents, jacobians).backward()
    assert all(parameter.grad is None for parameter in model.feature_extractor.parameters())
    gammas = [parameter for name, parameter in model.graph.named_parameters() if ".gamma" in name]
    assert gammas and any(parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
                          for parameter in gammas)


@pytest.mark.parametrize("mode", (AnomalyMapMode.ALL, AnomalyMapMode.MAX))
@pytest.mark.parametrize("input_dims", ((3, 256, 256), (7, 19, 23)))
def test_csflow_anomaly_map_modes_and_input_dims(mode, input_dims):
    """Match products of interpolated channel means or the largest-scale mean only."""
    inputs = [torch.arange(2 * 4 * size * size).reshape(2, 4, size, size).float() / 100
              for size in (8, 4, 2)]
    maps = [F.interpolate(value.square().mean(dim=1, keepdim=True), size=input_dims[1:],
                          mode="bilinear", align_corners=False) for value in inputs]
    expected = maps[0] * maps[1] * maps[2] if mode == AnomalyMapMode.ALL else maps[0]
    generator = AnomalyMapGenerator(input_dims=input_dims, mode=mode)
    actual = generator(inputs)
    assert generator.input_dims == input_dims and generator.mode == mode
    assert actual.shape == (2, 1, *input_dims[1:]) and actual.device == inputs[0].device
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("jacobians", (torch.tensor(2.0), torch.tensor([1.0, 3.0])))
def test_csflow_loss_formula_and_determinism(jacobians):
    """Subtract the original Jacobian before batch averaging and dimension normalization."""
    latents = [torch.tensor([[1.0, 2.0], [3.0, 4.0]]).reshape(2, 2, 1, 1),
               torch.tensor([5.0, 6.0]).reshape(2, 1, 1, 1)]
    expected = ((torch.tensor([15.0, 30.5]) - jacobians).sum() / 2) / 3
    criterion = CsFlowLoss()
    actual = criterion(latents, jacobians)
    assert isinstance(inspect.getattr_static(CsFlowLoss, "forward"), staticmethod)
    assert actual.shape == () and torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert torch.equal(actual, criterion(latents, jacobians))


def test_csflow_strict_state_dict_round_trip(model):
    """Restore both backbone paths and flow state while reconstructing unregistered seeded permutations."""
    with torch.no_grad():
        for name, parameter in model.graph.named_parameters():
            if ".gamma" in name:
                parameter.fill_(0.125)
    restored = make_model().eval()
    result = restored.load_state_dict(model.state_dict(), strict=True)
    assert result.missing_keys == result.unexpected_keys == []
    model.eval()
    images = torch.rand(2, 3, 256, 256)
    with torch.no_grad():
        expected, actual = model(images), restored(images)
    assert torch.allclose(actual.pred_score, expected.pred_score, atol=1e-6, rtol=1e-5)
    assert torch.allclose(actual.anomaly_map, expected.anomaly_map, atol=1e-6, rtol=1e-5)


def constructor_checks():
    """Construct and strictly restore under file and network guards after isolated imports."""
    def forbidden(*args, **kwargs):
        raise AssertionError("Forbidden CSFlow constructor file or network access")

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
    assert prediction.pred_score.shape == (2,) and prediction.anomaly_map.shape == (2, 1, 256, 256)
    assert torch.isfinite(prediction.pred_score).all() and torch.isfinite(prediction.anomaly_map).all()


@pytest.mark.parametrize("block_freia", (False, True))
def test_csflow_isolated_dependencies_and_offline_constructor(block_freia):
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
        from visionlab.anomaly_detection.models.csflow.model import CsFlowModel
    except ModuleNotFoundError as error:
        assert error.name.split(".")[0] == "FrEIA", error
    else:
        raise AssertionError("Pure CSFlow unexpectedly imported without FrEIA")
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


def test_csflow_pure_modules_and_original_headers():
    """Classify pure modules, reject training framework imports and preserve original license headers."""
    helpers = runpy.run_path(os.path.join(PROJECT_ROOT, "tests", "common", "test_dependency.py"))
    pure_modules = {module for _, module in helpers["classified_modules"]()["pure"]}
    for name in ("__init__.py", "model.py", "anomaly_map.py", "loss.py"):
        path = os.path.join(MODEL_ROOT, name)
        with open(path, encoding="utf-8") as stream:
            source = stream.read()
        assert "# Copyright (C) 2022-2025 Intel Corporation" in source
        assert "# SPDX-License-Identifier: Apache-2.0" in source
        if name == "model.py":
            assert "# Copyright (c) 2021 marco-rudolph" in source
            assert "# SPDX-License-Identifier: MIT" in source
        assert all(module not in ("anomalib", "lightning", "pytorch_lightning")
                   for module, _ in helpers["source_imports"](path))
        suffix = "" if name == "__init__.py" else "." + name[:-3]
        assert "visionlab.anomaly_detection.models.csflow" + suffix in pure_modules
