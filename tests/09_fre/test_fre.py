# tests/09_fre/test_fre.py: pytest pure FRE, tied gradients, pooled reconstruction and offline construction

import inspect
import math
import os
import subprocess
import sys

import pytest
import torch
from torch.nn import functional as F

from visionlab.anomaly_detection.common.batch import InferenceBatch
from visionlab.anomaly_detection.models.components.feature_extractors import TimmFeatureExtractor
from visionlab.anomaly_detection.models.fre import FREModel
from visionlab.anomaly_detection.models.fre import model as fre_model
from visionlab.anomaly_detection.models.fre.model import TiedAE

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_fre(pooling_kernel_size=2):
    """Match a small nonsquare ResNet18 layer3 feature geometry to the tied input dimension."""
    input_dim = 256 * (4 // pooling_kernel_size) * (5 // pooling_kernel_size)
    return FREModel(backbone="resnet18", layer="layer3", input_dim=input_dim, latent_dim=7,
                    pre_trained=False, pooling_kernel_size=pooling_kernel_size)


def test_fre_tied_initialization_and_random_consumption():
    """Preserve parameter order, Xavier bounds, zero biases and the exact initialization RNG consumption."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        model = TiedAE(input_dim=17, latent_dim=5)
        following = torch.rand(4)
        torch.random.default_generator.manual_seed(42)
        repeated = TiedAE(input_dim=17, latent_dim=5)
        assert torch.equal(model.weight, repeated.weight)
        torch.random.default_generator.manual_seed(42)
        expected = torch.empty(5, 17)
        torch.nn.init.xavier_uniform_(expected)
        assert torch.equal(model.weight, expected) and torch.equal(following, torch.rand(4))
    parameters = dict(model.named_parameters())
    assert list(parameters) == ["weight", "decoder_bias", "encoder_bias"]
    assert model.weight.shape == (5, 17) and model.decoder_bias.shape == (17,) and model.encoder_bias.shape == (5,)
    assert (model.weight.abs() <= math.sqrt(6 / (17 + 5))).all()
    assert torch.count_nonzero(model.decoder_bias) == torch.count_nonzero(model.encoder_bias) == 0


def test_fre_tied_forward_and_shared_weight_gradient():
    """Accumulate encoder and transposed decoder gradients into the single shared weight."""
    model = TiedAE(input_dim=11, latent_dim=4)
    features, target = torch.randn(3, 11), torch.randn(3, 11)
    encoded = F.linear(features, model.weight, model.encoder_bias)
    expected = F.linear(encoded, model.weight.T, model.decoder_bias)
    actual = model(features)
    torch.testing.assert_close(actual, expected)
    encoder_weight = model.weight.detach().clone().requires_grad_()
    decoder_weight = model.weight.detach().T.clone().requires_grad_()
    independent = F.linear(F.linear(features, encoder_weight, model.encoder_bias.detach()),
                           decoder_weight, model.decoder_bias.detach())
    F.mse_loss(independent, target).backward()
    F.mse_loss(actual, target).backward()
    torch.testing.assert_close(model.weight.grad, encoder_weight.grad + decoder_weight.grad.T)
    assert len(list(model.parameters())) == 3
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert torch.count_nonzero(model.weight.grad) > 0


def test_fre_constructor_defaults_and_initialization_order(monkeypatch):
    """Keep six constructor arguments and build the tied autoencoder before an unpretrained extractor."""
    signature = inspect.signature(FREModel.__init__)
    assert list(signature.parameters) == ["self", "backbone", "layer", "input_dim", "latent_dim",
                                          "pre_trained", "pooling_kernel_size"]
    assert signature.parameters["backbone"].default is inspect.Parameter.empty
    assert signature.parameters["layer"].default is inspect.Parameter.empty
    assert {name: signature.parameters[name].default for name in
            ("input_dim", "latent_dim", "pre_trained", "pooling_kernel_size")} == {
                "input_dim": 65536, "latent_dim": 220, "pre_trained": True, "pooling_kernel_size": 4}
    calls = []

    def autoencoder(input_dim, latent_dim):
        calls.append("autoencoder")
        return TiedAE(input_dim=input_dim, latent_dim=latent_dim)

    def extractor(backbone, pre_trained, layers):
        calls.append(("extractor", pre_trained))
        return TimmFeatureExtractor(backbone=backbone, pre_trained=pre_trained, layers=layers)

    monkeypatch.setattr(fre_model, "TiedAE", autoencoder)
    monkeypatch.setattr(fre_model, "TimmFeatureExtractor", extractor)
    model = FREModel(backbone="resnet18", layer="layer3", input_dim=256, latent_dim=7)
    assert calls == ["autoencoder", ("extractor", False)]
    assert model.pooling_kernel_size == 4 and not model.feature_extractor.training


@pytest.mark.parametrize("pooling_kernel_size", (1, 2, 4))
def test_fre_features_pooling_detach_and_reconstruction_gradients(pooling_kernel_size, monkeypatch):
    """Return pooled shapes and detached inputs while retaining gradients only through TiedAE."""
    model = make_fre(pooling_kernel_size)
    image = torch.rand(2, 3, 64, 80, requires_grad=True)
    model.train()
    with torch.no_grad():
        model.feature_extractor.eval()
        raw = model.feature_extractor(image)["layer3"]
        expected = raw if pooling_kernel_size == 1 else F.avg_pool2d(raw, kernel_size=pooling_kernel_size)
    if pooling_kernel_size == 1:
        def forbidden(*args, **kwargs):
            raise AssertionError("pooling_kernel_size=1 must not pool")

        monkeypatch.setattr(fre_model.F, "avg_pool2d", forbidden)
    features_in, features_out, feature_shapes = model.get_features(image)
    assert feature_shapes == expected.shape == (2, 256, 4 // pooling_kernel_size, 5 // pooling_kernel_size)
    assert features_in.shape == features_out.shape == (2, model.fre_model.input_dim)
    assert not features_in.requires_grad and features_out.requires_grad
    torch.testing.assert_close(features_in, expected.flatten(1))
    assert not model.feature_extractor.training
    assert all(parameter.grad is None for parameter in model.feature_extractor.parameters())
    F.mse_loss(features_out, features_in).backward()
    assert image.grad is None and all(parameter.grad is None for parameter in model.feature_extractor.parameters())
    assert all(parameter.grad is not None for parameter in model.fre_model.parameters())


@pytest.mark.parametrize("training", (False, True))
def test_fre_forward_reference_score_before_interpolation(training):
    """Return InferenceBatch in both modes and sum scores on the pooled map before upsampling."""
    model = make_fre()
    model.train(training)
    image = torch.rand(2, 3, 64, 80)
    with torch.no_grad():
        features_in, features_out, feature_shapes = model.get_features(image)
        pooled_map = (features_in - features_out).square().reshape(feature_shapes).sum(dim=1)
        expected_score = pooled_map.sum(dim=(1, 2))
        expected_map = F.interpolate(pooled_map.unsqueeze(1), size=(64, 80), mode="bilinear", align_corners=False)
        output = model(image)
    assert isinstance(output, InferenceBatch)
    assert output.pred_score.shape == (2,) and output.anomaly_map.shape == (2, 1, 64, 80)
    assert not model.feature_extractor.training
    torch.testing.assert_close(output.pred_score, expected_score, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(output.anomaly_map, expected_map, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(output.pred_score, output.anomaly_map.sum(dim=(1, 2, 3)))


def test_fre_strict_state_dict_round_trip():
    """Restore the tied autoencoder and full backbone state with identical pure evaluation outputs."""
    model, restored = make_fre().eval(), make_fre().eval()
    state = model.state_dict()
    assert {"fre_model.weight", "fre_model.decoder_bias", "fre_model.encoder_bias"} <= state.keys()
    assert any(name.startswith("feature_extractor.") for name in state)
    result = restored.load_state_dict(state, strict=True)
    assert result.missing_keys == result.unexpected_keys == []
    image = torch.rand(2, 3, 64, 80)
    with torch.no_grad():
        expected, actual = model(image), restored(image)
    torch.testing.assert_close(actual.pred_score, expected.pred_score, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual.anomaly_map, expected.anomaly_map, rtol=1e-5, atol=1e-6)


def constructor_checks():
    """Construct with both pretrained arguments under forbidden file and network access and restore strictly."""
    calls = []

    def extractor(backbone, pre_trained, layers):
        calls.append(pre_trained)
        return TimmFeatureExtractor(backbone=backbone, pre_trained=pre_trained, layers=layers)

    def forbidden(*args, **kwargs):
        raise AssertionError("Forbidden FRE constructor file or network access")

    with pytest.MonkeyPatch.context() as patch:
        for target in ("builtins.open", "io.open", "os.open", "torch.load", "socket.socket.connect",
                       "socket.socket.connect_ex", "socket.create_connection"):
            patch.setattr(target, forbidden)
        patch.setattr(fre_model, "TimmFeatureExtractor", extractor)
        original = FREModel(backbone="resnet18", layer="layer3", input_dim=1024, latent_dim=7, pooling_kernel_size=2)
        restored = make_fre()
        assert calls == [False, False] and not original.feature_extractor.training
        result = restored.load_state_dict(original.state_dict(), strict=True)
        assert result.missing_keys == result.unexpected_keys == []
    with torch.no_grad():
        output = restored(torch.rand(1, 3, 64, 80))
    assert output.pred_score.shape == (1,) and output.anomaly_map.shape == (1, 1, 64, 80)


def test_fre_isolated_import_constructor_and_inference():
    """Exercise framework-independent import and offline construction in a fresh process."""
    script = '''
import importlib.abc
import runpy
import sys
import torch
import timm
class Block(importlib.abc.MetaPathFinder):
    """Reject reference and training frameworks before importing the pure FRE consumer."""
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


def test_fre_original_copyright_and_spdx_headers():
    """Retain the original 2024 copyright and SPDX markers in both copied FRE files."""
    root = os.path.join(PROJECT_ROOT, "src", "visionlab", "anomaly_detection", "models", "fre")
    for name in ("__init__.py", "model.py"):
        with open(os.path.join(root, name), encoding="utf-8") as stream:
            source = stream.read()
        assert "# Copyright (C) 2024 Intel Corporation" in source
        assert "# SPDX-License-Identifier: Apache-2.0" in source
