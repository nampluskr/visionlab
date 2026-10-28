# tests/17_dsr/test_dsr.py: pytest pure DSR architecture, random order, gradients and dependency contracts
"""Use default DsrModel parameters, 64x64 CPU inputs and batch 2; patch cuda only in second-stage tests."""

import inspect
import os
import runpy
import subprocess
import sys

import pytest
import torch
from torch import nn
from torchvision.transforms import v2

from visionlab.anomaly_detection.common.batch import InferenceBatch
from visionlab.anomaly_detection.common.perlin import generate_perlin_noise
from visionlab.anomaly_detection.models.dsr import DsrModel
from visionlab.anomaly_detection.models.dsr import anomaly_generator as generator_module
from visionlab.anomaly_detection.models.dsr import model as model_module
from visionlab.anomaly_detection.models.dsr.anomaly_generator import DsrAnomalyGenerator

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_ROOT = os.path.join(PROJECT_ROOT, "src", "visionlab", "anomaly_detection", "models", "dsr")
PARAMETER_COUNTS = {
    "discrete_latent_model": 4021379,
    "image_reconstruction_network": 9418371,
    "subspace_restriction_module_lo": 7818240,
    "subspace_restriction_module_hi": 7818240,
    "anomaly_detection_module": 5610690,
    "upsampling_module": 5611842,
}


@pytest.fixture
def model():
    """Provide an independent default-width model without loading pretrained assets."""
    return DsrModel()


@pytest.fixture
def cpu_cuda(monkeypatch):
    """Keep the original CPU RNG calls while bypassing only hardcoded CUDA transfers, as authorized."""
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)


def second_stage_loss(output, images):
    """Apply the original Lightning second-stage argument order without importing Lightning."""
    from visionlab.anomaly_detection.models.dsr.loss import DsrSecondStageLoss

    return DsrSecondStageLoss()(
        output["recon_feat_hi"], output["recon_feat_lo"], output["embedding_bot"], output["embedding_top"],
        images, output["obj_spec_image"], output["anomaly_map"], output["true_anomaly_map"],
    )


def test_dsr_constructor_signatures():
    """Preserve all four constructor names, positional order and defaults."""
    from visionlab.anomaly_detection.models.dsr.loss import DsrSecondStageLoss, DsrThirdStageLoss

    expected_model = {"latent_anomaly_strength": 0.2, "embedding_dim": 128, "num_embeddings": 4096,
                      "num_hiddens": 128, "num_residual_layers": 2, "num_residual_hiddens": 64}
    for cls, expected in ((DsrModel, expected_model), (DsrAnomalyGenerator, {"p_anomalous": 0.5}),
                          (DsrSecondStageLoss, {}), (DsrThirdStageLoss, {})):
        parameters = inspect.signature(cls).parameters
        assert list(parameters) == list(expected)
        assert {name: value.default for name, value in parameters.items()} == expected
        assert all(value.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD for value in parameters.values())


def test_dsr_group_counts_order_freezing_and_state_prefixes(model):
    """Match layer-by-layer counts, module registration, VQ buffers and parameter freezing."""
    assert list(model._modules) == list(PARAMETER_COUNTS)
    assert model.image_dim == 3 and model.anomaly_map_dim == 2 and model.latent_anomaly_strength == 0.2
    assert not hasattr(model, "load_pretrained_discrete_model_weights")
    assert {name: sum(value.numel() for value in module.parameters())
            for name, module in model.named_children()} == PARAMETER_COUNTS
    assert sum(value.numel() for value in model.parameters()) == 40298762
    assert {key.split(".")[0] for key in model.state_dict()} == set(PARAMETER_COUNTS)
    for name, value in model.named_parameters():
        assert value.requires_grad == (not name.startswith("discrete_latent_model.")), name
    buffers = dict(model.named_buffers())
    assert set(buffers) == {"discrete_latent_model._vq_vae_top._ema_cluster_size",
                            "discrete_latent_model._vq_vae_bot._ema_cluster_size"}
    assert sum(value.numel() for value in buffers.values()) == 8192
    assert not any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules())


def test_dsr_seed_and_sequential_component_initialization():
    """Match repeated seeds and independently build the original six groups in creation order."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        original = DsrModel()
        following = torch.get_rng_state()
        torch.manual_seed(42)
        repeated = DsrModel()
        assert torch.equal(following, torch.get_rng_state())
        assert all(torch.equal(value, repeated.state_dict()[name])
                   for name, value in original.state_dict().items())
        del repeated
        constructors = (
            (model_module.DiscreteLatentModel, (128, 2, 64, 4096, 128)),
            (model_module.ImageReconstructionNetwork, (256, 128, 2, 64)),
            (model_module.SubspaceRestrictionModule, (128,)),
            (model_module.SubspaceRestrictionModule, (128,)),
            (model_module.AnomalyDetectionModule, (6, 2, 64)),
            (model_module.UpsamplingModule, (8, 2, 64)),
        )
        torch.manual_seed(42)
        for name, (cls, args) in zip(PARAMETER_COUNTS, constructors):
            component = cls(*args)
            state = getattr(original, name).state_dict()
            assert state.keys() == component.state_dict().keys()
            assert all(torch.equal(value, state[key]) for key, value in component.state_dict().items())
            del component
        assert torch.equal(following, torch.get_rng_state())


@pytest.mark.parametrize("batch_size", (1, 2))
def test_dsr_evaluation_shapes_and_original_score_formula(model, batch_size):
    """Keep class-one maps without a channel axis and score pooling before the upsampling network."""
    image = torch.rand(batch_size, 3, 64, 64)
    captured = {}

    def capture_detection(module, args, output):
        captured["detection"] = output.detach()

    def capture_upsampling(module, args, output):
        captured["upsampling"] = output.detach()

    detection_hook = model.anomaly_detection_module.register_forward_hook(capture_detection)
    upsampling_hook = model.upsampling_module.register_forward_hook(capture_upsampling)
    try:
        with torch.no_grad():
            output = model.eval()(image)
    finally:
        detection_hook.remove()
        upsampling_hook.remove()
    assert isinstance(output, InferenceBatch)
    assert output.pred_score.shape == (batch_size,) and output.anomaly_map.shape == (batch_size, 64, 64)
    expected_map = torch.softmax(captured["upsampling"], dim=1)[:, 1]
    averaged = nn.functional.avg_pool2d(torch.softmax(captured["detection"], dim=1)[:, 1:], 21, 1, 10)
    expected_score = averaged.amax(dim=(2, 3)).reshape(batch_size)
    assert torch.equal(output.anomaly_map, expected_map) and torch.equal(output.pred_score, expected_score)
    with pytest.raises(RuntimeError, match="^There should not be an anomaly map to generate when not training$"):
        model(image, torch.zeros(batch_size, 1, 64, 64))


def test_dsr_second_stage_shapes_rng_order_and_gradients(model, cpu_cuda, monkeypatch):
    """Keep two CPU strength draws before latent sampling and train exactly the four second-stage groups."""
    images = torch.rand(2, 3, 64, 64)
    masks = torch.zeros(2, 1, 64, 64)
    masks[:, :, 16:40, 8:48] = 1
    calls = []
    original_rand, original_randint = torch.rand, torch.randint

    def record_rand(*args, **kwargs):
        calls.append(("rand", args, kwargs))
        return original_rand(*args, **kwargs)

    def record_randint(*args, **kwargs):
        calls.append(("randint", args, kwargs))
        return original_randint(*args, **kwargs)

    monkeypatch.setattr(torch, "rand", record_rand)
    monkeypatch.setattr(torch, "randint", record_randint)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        output = model.train()(images, masks)
        following = torch.get_rng_state()
        first_calls = calls.copy()
        calls.clear()
        torch.manual_seed(17)
        with torch.no_grad():
            repeated = model(images, masks)
        assert torch.equal(following, torch.get_rng_state()) and calls == first_calls
    assert [name for name, _, _ in first_calls] == ["rand", "rand"] + ["randint"] * 8
    assert first_calls[:2] == [("rand", (2,), {}), ("rand", (2,), {})]
    assert all(kwargs.get("device", "cpu") == "cpu" for _, _, kwargs in first_calls)
    assert first_calls[6:8] == [("randint", (0, 2, (2, 1, 1, 1)), {})] * 2
    expected = {"recon_feat_hi": (2, 128, 16, 16), "recon_feat_lo": (2, 128, 8, 8),
                "embedding_bot": (2, 128, 16, 16), "embedding_top": (2, 128, 8, 8),
                "obj_spec_image": (2, 3, 64, 64), "anomaly_map": (2, 2, 64, 64),
                "true_anomaly_map": (2, 1, 64, 64)}
    assert output.keys() == expected.keys()
    for name, shape in expected.items():
        assert output[name].shape == shape and torch.equal(output[name], repeated[name]), name
    assert torch.allclose(output["anomaly_map"].sum(dim=1), torch.ones(2, 64, 64))
    assert not output["embedding_bot"].requires_grad and not output["embedding_top"].requires_grad
    model.zero_grad(set_to_none=True)
    loss = second_stage_loss(output, images)
    assert torch.isfinite(loss)
    loss.backward()
    for name, component in model.named_children():
        gradients = [parameter.grad for parameter in component.parameters()]
        expected_gradient = name not in ("discrete_latent_model", "upsampling_module")
        assert any(value is not None for value in gradients) == expected_gradient, name
        assert all(torch.isfinite(value).all() for value in gradients if value is not None), name


def test_dsr_third_stage_shapes_and_gradient_isolation(model):
    """Return two-class probabilities and backpropagate through only the upsampling module."""
    from visionlab.anomaly_detection.models.dsr.loss import DsrThirdStageLoss

    images = torch.rand(2, 3, 64, 64)
    masks = torch.arange(2 * 64 * 64).remainder(2).reshape(2, 1, 64, 64).float()
    output = model.train()(images)
    assert set(output) == {"anomaly_map"} and output["anomaly_map"].shape == (2, 2, 64, 64)
    assert torch.allclose(output["anomaly_map"].sum(dim=1), torch.ones(2, 64, 64))
    model.zero_grad(set_to_none=True)
    loss = DsrThirdStageLoss()(output["anomaly_map"], masks)
    assert torch.isfinite(loss)
    loss.backward()
    for name, component in model.named_children():
        gradients = [parameter.grad for parameter in component.parameters()]
        assert any(value is not None for value in gradients) == (name == "upsampling_module"), name
        assert all(torch.isfinite(value).all() for value in gradients if value is not None), name


def test_dsr_vector_quantization_distance_argmin_and_straight_through():
    """Compare explicit squared distances and require identity input gradients through quantization."""
    quantizer = model_module.VectorQuantizer(num_embeddings=3, embedding_dim=2)
    with torch.no_grad():
        quantizer.embedding.weight.copy_(torch.tensor([[0., 0.], [1., 2.], [-2., -1.]]))
    inputs = torch.tensor([[[[0.1, 1.1]], [[0.2, 1.9]]]], requires_grad=True)
    flattened = inputs.permute(0, 2, 3, 1).reshape(-1, 2)
    distances = (flattened[:, None] - quantizer.embedding.weight[None]).square().sum(dim=-1)
    expected = quantizer.embedding.weight[distances.argmin(dim=1)].reshape(1, 1, 2, 2).permute(0, 3, 1, 2)
    actual = quantizer(inputs)
    assert torch.allclose(actual, expected)
    actual.sum().backward()
    assert torch.equal(inputs.grad, torch.ones_like(inputs))
    assert all(value.grad is None for value in quantizer.parameters())


def test_dsr_generator_shapes_seed_and_original_random_pipeline(monkeypatch):
    """Match scale-x, scale-y, Perlin RNG, affine rotation and strict half-threshold in order."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    generator = DsrAnomalyGenerator()
    assert generator.p_anomalous == 0.5 and isinstance(generator.rot, v2.RandomAffine)
    assert tuple(generator.rot.degrees) == (-90, 90)
    parameters = inspect.signature(generate_perlin_noise).parameters
    assert list(parameters) == ["height", "width", "scale", "device"]
    assert parameters["height"].default is parameters["width"].default is inspect.Parameter.empty
    assert parameters["scale"].default is parameters["device"].default is None
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        actual = generator.generate_anomaly(64, 64)
        following = torch.get_rng_state()
        torch.manual_seed(42)
        scalex = int(2 ** torch.randint(0, 6, (1,)).item())
        scaley = int(2 ** torch.randint(0, 6, (1,)).item())
        noise = generate_perlin_noise(64, 64, scale=(scalex, scaley))
        expected = (generator.rot(noise.unsqueeze(0)).squeeze(0) > 0.5).float().unsqueeze(0)
        assert torch.equal(actual, expected) and torch.equal(following, torch.get_rng_state())
        generator = DsrAnomalyGenerator(p_anomalous=1)
        torch.manual_seed(42)
        batch_masks = generator.augment_batch(torch.zeros(2, 3, 64, 64))
        following = torch.get_rng_state()
        torch.manual_seed(42)
        masks = []
        for _ in range(2):
            masks.append(torch.zeros(1, 64, 64) if torch.rand(1) > 1 else generator.generate_anomaly(64, 64))
        assert torch.equal(batch_masks, torch.stack(masks)) and torch.equal(following, torch.get_rng_state())
        torch.manual_seed(42)
        assert torch.equal(batch_masks, generator.augment_batch(torch.zeros(2, 3, 64, 64)))
        assert torch.count_nonzero(DsrAnomalyGenerator(p_anomalous=0).augment_batch(torch.zeros(2, 3, 64, 64))) == 0
    assert actual.shape == (1, 64, 64) and batch_masks.shape == (2, 1, 64, 64)
    assert ((actual == 0) | (actual == 1)).all() and ((batch_masks == 0) | (batch_masks == 1)).all()


def test_dsr_generator_strict_probability_and_threshold_boundaries(monkeypatch):
    """Preserve the equality branch at p=0 and exclude noise exactly equal to 0.5."""
    generator = DsrAnomalyGenerator(p_anomalous=0)
    monkeypatch.setattr(torch, "rand", lambda size: torch.zeros(size))
    monkeypatch.setattr(generator, "generate_anomaly", lambda height, width: torch.ones(1, height, width))
    assert torch.equal(generator.augment_batch(torch.zeros(2, 3, 64, 64)), torch.ones(2, 1, 64, 64))
    generator = DsrAnomalyGenerator()
    monkeypatch.setattr(generator, "rot", nn.Identity())
    monkeypatch.setattr(generator_module, "generate_perlin_noise",
                        lambda height, width, scale: torch.tensor([[0.5, 0.5001], [-0.1, 1.0]]))
    assert torch.equal(generator.generate_anomaly(2, 2), torch.tensor([[[0., 1.], [0., 1.]]]))


def test_dsr_losses_match_manual_reference():
    """Retain MSE means, image weight ten and alpha-one focal loss with squeezed integer targets."""
    from kornia.losses import FocalLoss
    from visionlab.anomaly_detection.models.dsr.loss import DsrSecondStageLoss, DsrThirdStageLoss

    hi, lo = torch.randn(2, 4, 4, 4), torch.randn(2, 4, 2, 2)
    qu_hi, qu_lo = torch.randn_like(hi), torch.randn_like(lo)
    images, reconstructed = torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8)
    prediction = torch.randn(2, 2, 8, 8)
    mask = torch.arange(2 * 8 * 8).remainder(2).reshape(2, 1, 8, 8).float()
    focal = FocalLoss(alpha=1, reduction="mean")(prediction, mask.squeeze(1).long())
    expected = (hi - qu_hi).square().mean() + (lo - qu_lo).square().mean()
    expected = expected + (images - reconstructed).square().mean() * 10 + focal
    second, third = DsrSecondStageLoss(), DsrThirdStageLoss()
    assert isinstance(second.l2_loss, nn.MSELoss)
    assert torch.allclose(second(hi, lo, qu_hi, qu_lo, images, reconstructed, prediction, mask), expected)
    assert torch.allclose(third(prediction, mask), focal)


def constructor_checks():
    """Reject file and network calls during construction and strict restoration before evaluation."""
    def forbidden(*args, **kwargs):
        raise AssertionError("DSR construction must not read files or access the network")

    with pytest.MonkeyPatch.context() as patch:
        for target in ("builtins.open", "io.open", "os.open", "torch.load", "socket.socket.connect",
                       "socket.socket.connect_ex", "socket.create_connection"):
            patch.setattr(target, forbidden)
        original, restored = DsrModel(), DsrModel()
        generator = DsrAnomalyGenerator()
        result = restored.load_state_dict(original.state_dict(), strict=True)
        assert result.missing_keys == result.unexpected_keys == []
    assert generator.p_anomalous == 0.5
    with torch.no_grad():
        prediction = restored.eval()(torch.rand(2, 3, 64, 64))
    assert prediction.pred_score.shape == (2,) and prediction.anomaly_map.shape == (2, 64, 64)
    assert torch.isfinite(prediction.pred_score).all() and torch.isfinite(prediction.anomaly_map).all()


def test_dsr_no_constructor_file_or_network_access():
    """Require asset-free construction and strict state restoration."""
    constructor_checks()


def test_dsr_state_dict_strict_round_trip(model):
    """Restore parameters and all EMA buffers; the original uses InstanceNorm without running statistics."""
    with torch.no_grad():
        model.discrete_latent_model._vq_vae_top._ema_cluster_size.fill_(3)
        model.discrete_latent_model._vq_vae_bot._ema_cluster_size.fill_(7)
    restored = DsrModel()
    result = restored.load_state_dict(model.state_dict(), strict=True)
    assert result.missing_keys == result.unexpected_keys == []
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in model.state_dict().items())
    images = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        expected, actual = model.eval()(images), restored.eval()(images)
    assert torch.equal(actual.pred_score, expected.pred_score)
    assert torch.equal(actual.anomaly_map, expected.anomaly_map)


def test_dsr_isolated_pure_dependencies():
    """Construct and evaluate without reference, Lightning, kornia, FrEIA or timm imports."""
    script = '''
import importlib.abc
import runpy
import sys
blocked = {"anomalib", "lightning", "pytorch_lightning", "kornia", "FrEIA", "timm"}
class Block(importlib.abc.MetaPathFinder):
    """Reject forbidden dependencies before importing the pure model or generator."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, Block())
runpy.run_path(sys.argv[1])["constructor_checks"]()
try:
    from visionlab.anomaly_detection.models.dsr.loss import DsrSecondStageLoss
except ModuleNotFoundError as error:
    assert error.name == "kornia"
else:
    raise AssertionError("Only the training loss must require kornia")
assert not blocked & {name.split(".")[0] for name in sys.modules}
'''
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.path.join(PROJECT_ROOT, "src")
    result = subprocess.run([sys.executable, "-c", script, os.path.abspath(__file__)],
                            env=environment, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_dsr_original_headers_and_dependency_classification():
    """Keep upstream licenses and classify losses like DRAEM while leaving inference kornia-free."""
    helpers = runpy.run_path(os.path.join(PROJECT_ROOT, "tests", "common", "test_dependency.py"))
    pure = {module for _, module in helpers["classified_modules"]()["pure"]}
    for name in ("__init__.py", "model.py", "anomaly_generator.py", "loss.py"):
        path = os.path.join(MODEL_ROOT, name)
        with open(path, encoding="utf-8") as stream:
            source = stream.read()
        assert "# Copyright (C) 2023-2024 Intel Corporation" in source
        assert "# SPDX-License-Identifier: Apache-2.0" in source
        if name == "model.py":
            assert "# Copyright (c) 2022 VitjanZ" in source
        imports = {module for module, _ in helpers["source_imports"](path)}
        assert not imports & {"anomalib", "lightning", "pytorch_lightning", "FrEIA", "timm"}
        assert ("kornia" in imports) == (name == "loss.py")
        suffix = "" if name == "__init__.py" else "." + name[:-3]
        assert "visionlab.anomaly_detection.models.dsr" + suffix in pure
