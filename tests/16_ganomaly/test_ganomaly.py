# tests/16_ganomaly/test_ganomaly.py: pytest pure GANomaly architecture, losses and dependency contracts
"""Use 64x64 inputs, n_features=8, latent_vec_size=16 and batch 2 instead of the large 256x256 configuration."""

import inspect
import math
import os
import runpy
import subprocess
import sys

import pytest
import torch
from torch import nn

from visionlab.anomaly_detection.common.batch import InferenceBatch
from visionlab.anomaly_detection.models.ganomaly import GanomalyModel
from visionlab.anomaly_detection.models.ganomaly.loss import DiscriminatorLoss, GeneratorLoss
from visionlab.anomaly_detection.models.ganomaly.model import Decoder, Discriminator, Encoder, Generator, pad_nextpow2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_ROOT = os.path.join(PROJECT_ROOT, "src", "visionlab", "anomaly_detection", "models", "ganomaly")


def make_model(extra_layers=0, add_final_conv_layer=True):
    """Build only the small GANomaly configuration without external assets."""
    return GanomalyModel(input_size=(64, 64), num_input_channels=3, n_features=8,
                         latent_vec_size=16, extra_layers=extra_layers,
                         add_final_conv_layer=add_final_conv_layer)


@pytest.fixture
def model():
    """Provide an independent small model for each test."""
    return make_model()


@pytest.mark.parametrize("cls,expected", (
    (GanomalyModel, {"input_size": inspect.Parameter.empty, "num_input_channels": inspect.Parameter.empty,
                     "n_features": inspect.Parameter.empty, "latent_vec_size": inspect.Parameter.empty,
                     "extra_layers": 0, "add_final_conv_layer": True}),
    (GeneratorLoss, {"wadv": 1, "wcon": 50, "wenc": 1}),
    (DiscriminatorLoss, {}),
))
def test_ganomaly_constructor_signatures(cls, expected):
    """Preserve constructor names, order, defaults and positional keyword arguments."""
    parameters = inspect.signature(cls).parameters
    assert list(parameters) == list(expected)
    assert {name: value.default for name, value in parameters.items()} == expected
    assert all(value.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD for value in parameters.values())


@pytest.mark.parametrize("height,width,size", ((64, 64, 64), (127, 128, 128), (100, 60, 128)))
def test_ganomaly_pad_nextpow2_values_and_shape(height, width, size):
    """Preserve values and the original asymmetric zero padding, including the documented example."""
    images = torch.arange(3 * height * width).reshape(1, 3, height, width).float() + 1
    padded = pad_nextpow2(images)
    top, left = math.ceil((size - height) / 2), math.ceil((size - width) / 2)
    expected = torch.zeros(1, 3, size, size)
    expected[:, :, top:top + height, left:left + width] = images
    assert padded.shape == (1, 3, size, size)
    assert torch.equal(padded, expected)


@pytest.mark.parametrize("extra_layers", (0, 2))
def test_ganomaly_architecture_names_shapes_and_activations(extra_layers):
    """Retain named encoder and decoder blocks and the discriminator's flattened temporary encoder."""
    model = make_model(extra_layers=extra_layers)
    generator, discriminator = model.generator, model.discriminator
    assert isinstance(generator, Generator) and isinstance(discriminator, Discriminator)
    assert list(model._modules) == ["generator", "discriminator"]
    assert list(generator._modules) == ["encoder1", "decoder", "encoder2"]
    assert list(discriminator._modules) == ["features", "classifier"]
    for encoder in (generator.encoder1, generator.encoder2):
        assert isinstance(encoder, Encoder)
        assert list(encoder.input_layers._modules) == ["initial-conv-3-8", "initial-relu-8"]
        initial = encoder.input_layers[0]
        assert initial.kernel_size == (4, 4) and initial.stride == (2, 2) and initial.padding == (4, 4)
        assert list(encoder.pyramid_features._modules) == [
            "pyramid-8-16-conv", "pyramid-16-batchnorm", "pyramid-16-relu",
            "pyramid-16-32-conv", "pyramid-32-batchnorm", "pyramid-32-relu",
            "pyramid-32-64-conv", "pyramid-64-batchnorm", "pyramid-64-relu",
        ]
        assert encoder.final_conv_layer.weight.shape == (16, 64, 4, 4)
        assert list(encoder.extra_layers._modules) == [
            f"extra-layers-{index}-8-{kind}" for index in range(extra_layers)
            for kind in ("conv", "batchnorm", "relu")
        ]
    decoder = generator.decoder
    assert isinstance(decoder, Decoder)
    assert list(decoder.latent_input._modules) == [
        "initial-16-64-convt", "initial-64-batchnorm", "initial-64-relu",
    ]
    assert list(decoder.inverse_pyramid._modules) == [
        "pyramid-64-32-convt", "pyramid-32-batchnorm", "pyramid-32-relu",
        "pyramid-32-16-convt", "pyramid-16-batchnorm", "pyramid-16-relu",
        "pyramid-16-8-convt", "pyramid-8-batchnorm", "pyramid-8-relu",
    ]
    assert list(decoder.extra_layers._modules) == [
        f"extra-layers-{index}-8-{kind}" for index in range(extra_layers)
        for kind in ("conv", "batchnorm", "relu")
    ]
    assert list(decoder.final_layers._modules) == ["final-8-3-convt", "final-3-tanh"]
    assert isinstance(decoder.final_layers[-1], nn.Tanh)
    assert all(module.inplace for module in generator.modules() if isinstance(module, (nn.ReLU, nn.LeakyReLU)))
    assert list(discriminator.classifier._modules) == ["0", "Sigmoid"]
    assert isinstance(discriminator.classifier[-1], nn.Sigmoid)
    expected_features = [*generator.encoder1.input_layers.children(),
                         *generator.encoder1.extra_layers.children(),
                         *generator.encoder1.pyramid_features.children()]
    assert [type(module) for module in discriminator.features] == [type(module) for module in expected_features]
    assert discriminator.classifier[0].weight.shape == (1, 64, 4, 4)
    images = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        fake, latent_i, latent_o = generator(images)
        pred, features = discriminator(images)
        decoded = decoder(latent_i)
    assert fake.shape == decoded.shape == images.shape
    assert latent_i.shape == latent_o.shape == (2, 16, 1, 1)
    assert fake.min() >= -1 and fake.max() <= 1
    assert pred.shape == (2,) and features.shape == (2, 64, 4, 4)
    assert ((pred > 0) & (pred < 1)).all()


def test_ganomaly_weights_init_preserves_default_initialization():
    """Keep constructor initialization unchanged while allowing direct convolution initialization."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = make_model()
        assert isinstance(inspect.getattr_static(GanomalyModel, "weights_init"), staticmethod)
        for network in (model.generator, model.discriminator):
            conv = next(module for module in network.modules() if isinstance(module, nn.Conv2d))
            bound = 1 / math.sqrt(3 * 4 * 4)
            assert conv.weight.abs().max() <= bound
            assert torch.isclose(conv.weight.std(), torch.tensor(bound / math.sqrt(3)), rtol=0.15)
            assert not torch.isclose(conv.weight.std(), torch.tensor(0.02), rtol=0.2)
            before = {name: value.clone() for name, value in network.state_dict().items()}
            GanomalyModel.weights_init(network)
            assert all(torch.equal(value, before[name]) for name, value in network.state_dict().items())
        conv = nn.Conv2d(64, 64, kernel_size=4, bias=False)
        GanomalyModel.weights_init(conv)
        assert abs(conv.weight.mean().item()) < 0.001
        assert torch.isclose(conv.weight.std(), torch.tensor(0.02), rtol=0.03)


def test_ganomaly_same_seed_and_component_creation_order():
    """Reproduce state and RNG consumption with generator then temporary-encoder discriminator creation."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        first = make_model()
        rng = torch.get_rng_state()
        torch.manual_seed(42)
        second = make_model()
        assert torch.equal(rng, torch.get_rng_state())
        assert first.state_dict().keys() == second.state_dict().keys()
        assert all(torch.equal(value, second.state_dict()[name]) for name, value in first.state_dict().items())
        torch.manual_seed(42)
        encoder1 = Encoder(input_size=(64, 64), latent_vec_size=16, num_input_channels=3, n_features=8)
        decoder = Decoder(input_size=(64, 64), latent_vec_size=16, num_input_channels=3, n_features=8)
        encoder2 = Encoder(input_size=(64, 64), latent_vec_size=16, num_input_channels=3, n_features=8)
        temporary = Encoder(input_size=(64, 64), latent_vec_size=1, num_input_channels=3, n_features=8)
        assert torch.equal(rng, torch.get_rng_state())
    for name, component in (("encoder1", encoder1), ("decoder", decoder), ("encoder2", encoder2)):
        actual = getattr(first.generator, name).state_dict()
        assert actual.keys() == component.state_dict().keys()
        assert all(torch.equal(value, actual[key]) for key, value in component.state_dict().items())
    layers = [*temporary.input_layers.children(), *temporary.extra_layers.children(),
              *temporary.pyramid_features.children()]
    expected = nn.Sequential(*layers).state_dict()
    assert expected.keys() == first.discriminator.features.state_dict().keys()
    assert all(torch.equal(value, first.discriminator.features.state_dict()[name]) for name, value in expected.items())
    assert torch.equal(temporary.final_conv_layer.weight, first.discriminator.classifier[0].weight)


def test_ganomaly_missing_final_conv_retains_original_attribute_error():
    """Document the original missing attribute when final convolution creation is disabled."""
    model = make_model(add_final_conv_layer=False)
    assert not hasattr(model.generator.encoder1, "final_conv_layer")
    assert not hasattr(model.generator.encoder2, "final_conv_layer")
    with pytest.raises(AttributeError, match="final_conv_layer"):
        model(torch.randn(2, 3, 64, 64))


@pytest.mark.parametrize("shape", ((64, 64), (63, 64), (50, 30)))
def test_ganomaly_training_forward_padding(model, shape):
    """Return padded images, reconstructions and both unsqueezed latent tensors during training."""
    images = torch.randn(2, 3, *shape)
    output = model(images)
    assert isinstance(output, tuple) and len(output) == 4
    padded, fake, latent_i, latent_o = output
    assert padded.shape == fake.shape == (2, 3, 64, 64)
    assert latent_i.shape == latent_o.shape == (2, 16, 1, 1)
    assert torch.equal(padded, pad_nextpow2(images))


def test_ganomaly_eval_scores_and_batch_independence(model):
    """Return only image scores with the original latent MSE formula and retain batch-one shape."""
    model.eval()
    images = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        _, latent_i, latent_o = model.generator(pad_nextpow2(images))
        expected = (latent_i - latent_o).square().mean(dim=1).view(-1)
        paired, single = model(images), model(images[:1])
    assert isinstance(paired, InferenceBatch) and isinstance(single, InferenceBatch)
    assert paired.anomaly_map is None and single.anomaly_map is None
    assert paired.pred_score.shape == (2,) and single.pred_score.shape == (1,)
    assert torch.allclose(paired.pred_score, expected, atol=1e-6, rtol=1e-5)
    assert torch.allclose(single.pred_score, paired.pred_score[:1], atol=1e-6, rtol=1e-5)
    assert torch.isfinite(paired.pred_score).all()


@pytest.mark.parametrize("weights", ((1, 50, 1), (2, 3, 4)))
def test_ganomaly_loss_formulas_and_determinism(weights):
    """Match manually reduced SmoothL1, L1, MSE and binary cross entropy losses."""
    latent_i, latent_o = torch.tensor([0.0, 3.0]), torch.tensor([0.5, 0.0])
    images, fake = torch.tensor([1.0, -2.0]), torch.tensor([-1.0, 1.0])
    pred_real, pred_fake = torch.tensor([0.8, 0.6]), torch.tensor([0.2, 0.3])
    wadv, wcon, wenc = weights
    expected_g = ((pred_real - pred_fake).square().mean() * wadv
                  + (images - fake).abs().mean() * wcon + torch.tensor((0.125 + 2.5) / 2) * wenc)
    expected_d = -(pred_real.log().mean() + (1 - pred_fake).log().mean()) * 0.5
    generator_loss = GeneratorLoss(wadv=wadv, wcon=wcon, wenc=wenc)
    discriminator_loss = DiscriminatorLoss()
    assert isinstance(generator_loss.loss_enc, nn.SmoothL1Loss)
    assert isinstance(generator_loss.loss_adv, nn.MSELoss) and isinstance(generator_loss.loss_con, nn.L1Loss)
    assert isinstance(discriminator_loss.loss_bce, nn.BCELoss)
    args = (latent_i, latent_o, images, fake, pred_real, pred_fake)
    actual_g, actual_d = generator_loss(*args), discriminator_loss(pred_real, pred_fake)
    assert torch.allclose(actual_g, expected_g) and torch.allclose(actual_d, expected_d)
    assert torch.equal(actual_g, generator_loss(*args))
    assert torch.equal(actual_d, discriminator_loss(pred_real, pred_fake))


def test_ganomaly_original_training_step_updates_both_networks(model):
    """Follow the original retained generator graph and detached fake discriminator update without Lightning."""
    g_opt = torch.optim.Adam(model.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(model.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    before_g = [parameter.detach().clone() for parameter in model.generator.parameters()]
    before_d = [parameter.detach().clone() for parameter in model.discriminator.parameters()]
    padded, fake, latent_i, latent_o = model(torch.randn(2, 3, 64, 64))
    pred_real, _ = model.discriminator(padded)
    pred_fake, _ = model.discriminator(fake)
    g_loss = GeneratorLoss()(latent_i, latent_o, padded, fake, pred_real, pred_fake)
    g_opt.zero_grad()
    g_loss.backward(retain_graph=True)
    g_opt.step()
    pred_fake, _ = model.discriminator(fake.detach())
    d_loss = DiscriminatorLoss()(pred_real, pred_fake)
    d_opt.zero_grad()
    d_loss.backward()
    d_opt.step()
    assert torch.isfinite(g_loss) and torch.isfinite(d_loss)
    assert any(not torch.equal(before, after) for before, after in zip(before_g, model.generator.parameters()))
    assert any(not torch.equal(before, after) for before, after in zip(before_d, model.discriminator.parameters()))


def constructor_checks():
    """Reject file and network access during construction and strict state loading, then run inference."""
    def forbidden(*args, **kwargs):
        raise AssertionError("Model construction must not read files or access the network")

    with pytest.MonkeyPatch.context() as patch:
        for target in ("builtins.open", "io.open", "os.open", "torch.load", "socket.socket.connect",
                       "socket.socket.connect_ex", "socket.create_connection"):
            patch.setattr(target, forbidden)
        original, restored = make_model(), make_model()
        result = restored.load_state_dict(original.state_dict(), strict=True)
        assert result.missing_keys == result.unexpected_keys == []
    with torch.no_grad():
        prediction = restored.eval()(torch.randn(2, 3, 64, 64))
    assert prediction.pred_score.shape == (2,) and prediction.anomaly_map is None
    assert torch.isfinite(prediction.pred_score).all()


def test_ganomaly_no_constructor_file_or_network_access():
    """Require no assets or network for construction and strict state loading."""
    constructor_checks()


def test_ganomaly_state_dict_strict_round_trip_includes_batchnorm(model):
    """Restore parameters and updated BatchNorm statistics before comparing evaluation outputs."""
    images = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        model(images)
    state = model.state_dict()
    assert any(name.endswith("running_mean") and torch.count_nonzero(value) > 0 for name, value in state.items())
    restored = make_model()
    result = restored.load_state_dict(state, strict=True)
    assert result.missing_keys == result.unexpected_keys == []
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in state.items())
    with torch.no_grad():
        expected, actual = model.eval()(images), restored.eval()(images)
    assert torch.equal(expected.pred_score, actual.pred_score)


def test_ganomaly_isolated_pure_dependencies():
    """Construct and evaluate in a fresh process without training frameworks or unrelated model dependencies."""
    script = '''
import importlib.abc
import runpy
import sys
blocked = {"anomalib", "lightning", "pytorch_lightning", "FrEIA", "timm", "torchvision", "kornia"}
class Block(importlib.abc.MetaPathFinder):
    """Reject forbidden packages before importing the model or tests."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, Block())
runpy.run_path(sys.argv[1])["constructor_checks"]()
assert not blocked & {name.split(".")[0] for name in sys.modules}
'''
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.path.join(PROJECT_ROOT, "src")
    result = subprocess.run([sys.executable, "-c", script, os.path.abspath(__file__)],
                            env=environment, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_ganomaly_pure_classification_and_original_headers():
    """Preserve source licenses and include every GANomaly module in the existing pure import group."""
    helpers = runpy.run_path(os.path.join(PROJECT_ROOT, "tests", "common", "test_dependency.py"))
    pure = {module for _, module in helpers["classified_modules"]()["pure"]}
    for name in ("__init__.py", "model.py", "loss.py"):
        path = os.path.join(MODEL_ROOT, name)
        with open(path, encoding="utf-8") as stream:
            source = stream.read()
        year = "2020-2022" if name == "model.py" else "2022-2025"
        assert f"# Copyright (C) {year} Intel Corporation" in source
        assert "# SPDX-License-Identifier: Apache-2.0" in source
        assert all(module not in ("anomalib", "lightning", "pytorch_lightning", "FrEIA", "timm", "torchvision", "kornia")
                   for module, _ in helpers["source_imports"](path))
        suffix = "" if name == "__init__.py" else "." + name[:-3]
        assert "visionlab.anomaly_detection.models.ganomaly" + suffix in pure
