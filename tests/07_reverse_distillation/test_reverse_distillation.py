# tests/07_reverse_distillation/test_reverse_distillation.py: pytest pure decoder, bottleneck, loss and offline model contracts

import inspect
import os
import subprocess
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck

from visionlab.anomaly_detection.common.batch import InferenceBatch
from visionlab.anomaly_detection.models.components.feature_extractors import TimmFeatureExtractor
from visionlab.anomaly_detection.models.components.filters import GaussianBlur2d
from visionlab.anomaly_detection.models.components.tiling.tiler import Tiler
from visionlab.anomaly_detection.models.reverse_distillation import ReverseDistillationModel
from visionlab.anomaly_detection.models.reverse_distillation import model as reverse_model
from visionlab.anomaly_detection.models.reverse_distillation.anomaly_map import AnomalyMapGenerationMode, AnomalyMapGenerator
from visionlab.anomaly_detection.models.reverse_distillation.components import get_bottleneck_layer, get_decoder
from visionlab.anomaly_detection.models.reverse_distillation.components import bottleneck, de_resnet
from visionlab.anomaly_detection.models.reverse_distillation.loss import ReverseDistillationLoss

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKBONES = ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnext50_32x4d",
             "resnext101_32x8d", "wide_resnet50_2", "wide_resnet101_2")


@pytest.fixture(scope="module")
def shared_reverse_model():
    """Share a small randomly initialized encoder and trainable reconstruction path."""
    return ReverseDistillationModel(backbone="resnet18", input_size=(64, 64),
                                    layers=("layer1", "layer2", "layer3"),
                                    anomaly_map_mode=AnomalyMapGenerationMode.ADD, pre_trained=False)


@pytest.fixture
def model(shared_reverse_model):
    """Reset mode, tiling and accumulated gradients for independent pure-model checks."""
    shared_reverse_model.eval()
    shared_reverse_model.tiler = None
    shared_reverse_model.zero_grad(set_to_none=True)
    return shared_reverse_model


def test_reverse_distillation_loss_matches_flattened_reference_and_strict_layers():
    """Sum per-layer flattened cosine losses and reject unequal layer counts."""
    encoder = [torch.randn(2, 4, 8, 8), torch.randn(2, 8, 4, 4), torch.randn(2, 16, 2, 2)]
    decoder = [torch.randn_like(feature) for feature in encoder]
    expected = sum((1 - F.cosine_similarity(left.view(2, -1), right.view(2, -1))).mean()
                   for left, right in zip(encoder, decoder, strict=True))
    criterion = ReverseDistillationLoss()
    torch.testing.assert_close(criterion(encoder, decoder), expected)
    torch.testing.assert_close(criterion(encoder, encoder), torch.tensor(0.0), atol=1e-6, rtol=0)
    with pytest.raises(ValueError):
        criterion(encoder, decoder[:-1])


@pytest.mark.parametrize("mode", (AnomalyMapGenerationMode.ADD, AnomalyMapGenerationMode.MULTIPLY))
def test_reverse_distillation_anomaly_map_matches_cosine_reference(mode, monkeypatch):
    """Combine bilinearly resized cosine maps and create a fresh Gaussian blur on every forward."""
    encoder = [torch.randn(2, 4, 16, 12), torch.randn(2, 8, 8, 6)]
    decoder = [torch.randn_like(feature) for feature in encoder]
    generator = AnomalyMapGenerator(image_size=[48, 56], mode=mode)
    assert generator.image_size == (48, 56) and generator.sigma == 4 and generator.kernel_size == 33
    expected = torch.zeros(2, 1, 48, 56) if mode == AnomalyMapGenerationMode.ADD else torch.ones(2, 1, 48, 56)
    for left, right in zip(encoder, decoder, strict=True):
        distance = F.interpolate((1 - F.cosine_similarity(left, right)).unsqueeze(1),
                                 size=(48, 56), mode="bilinear", align_corners=True)
        expected = expected + distance if mode == AnomalyMapGenerationMode.ADD else expected * distance
    expected = GaussianBlur2d(kernel_size=(33, 33), sigma=(4, 4))(expected)
    created = []

    def blur(kernel_size, sigma):
        instance = GaussianBlur2d(kernel_size=kernel_size, sigma=sigma)
        created.append(instance)
        return instance

    monkeypatch.setattr(sys.modules[AnomalyMapGenerator.__module__], "GaussianBlur2d", blur)
    for _ in range(2):
        actual = generator(encoder, decoder)
        assert actual.shape == (2, 1, 48, 56)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert len(created) == 2 and created[0] is not created[1]


def test_reverse_distillation_anomaly_mode_defaults_and_validation():
    """Retain the generator's MULTIPLY default and the model's required mode argument."""
    assert AnomalyMapGenerator(image_size=(64, 64)).mode is AnomalyMapGenerationMode.MULTIPLY
    with pytest.raises(ValueError, match="Found mode invalid. Only multiply and add are supported."):
        AnomalyMapGenerator(image_size=(64, 64), mode="invalid")
    signature = inspect.signature(ReverseDistillationModel.__init__)
    assert list(signature.parameters) == ["self", "backbone", "input_size", "layers", "anomaly_map_mode", "pre_trained"]
    for name in ("backbone", "input_size", "layers", "anomaly_map_mode"):
        assert signature.parameters[name].default is inspect.Parameter.empty
    assert signature.parameters["pre_trained"].default is True


@pytest.mark.parametrize("backbone", BACKBONES)
def test_reverse_distillation_component_factory_dispatch(backbone, monkeypatch):
    """Check every original factory name without allocating all large ResNet variants."""
    calls = []
    decoder = nn.Identity()
    compression = nn.Identity()

    def make_decoder():
        return decoder

    def make_bottleneck(block, layers):
        calls.append((block, layers))
        return compression

    monkeypatch.setattr(de_resnet, "de_" + backbone, make_decoder)
    monkeypatch.setattr(bottleneck, "OCBE", make_bottleneck)
    assert get_decoder(backbone) is decoder and get_bottleneck_layer(backbone) is compression
    assert calls == [(BasicBlock, 2) if backbone in ("resnet18", "resnet34") else (Bottleneck, 3)]


def test_reverse_distillation_unsupported_decoder_and_bottleneck_fallback(monkeypatch):
    """Preserve decoder rejection and the bottleneck factory's original unrestricted fallback."""
    with pytest.raises(ValueError, match="^Decoder with architecture unsupported not supported$"):
        get_decoder("unsupported")
    calls = []

    def make_bottleneck(block, layers):
        calls.append((block, layers))
        return nn.Identity()

    monkeypatch.setattr(bottleneck, "OCBE", make_bottleneck)
    assert isinstance(get_bottleneck_layer("unsupported"), nn.Module) and calls == [(Bottleneck, 3)]


def test_reverse_distillation_wide_bottleneck_and_decoder_shapes():
    """Reconstruct all three wide-encoder channel scales using a small 64-pixel input geometry."""
    features = [torch.randn(1, 256, 16, 16), torch.randn(1, 512, 8, 8), torch.randn(1, 1024, 4, 4)]
    compression = get_bottleneck_layer("wide_resnet50_2").eval()
    decoder = get_decoder("wide_resnet50_2").eval()
    with torch.no_grad():
        embedding = compression(features)
        reconstructed = decoder(embedding)
    assert embedding.shape == (1, 2048, 2, 2)
    assert isinstance(reconstructed, list) and len(reconstructed) == 3
    assert [value.shape for value in reconstructed] == [value.shape for value in features]


def test_reverse_distillation_training_features_encoder_eval_and_gradients(model):
    """Keep encoder BatchNorm frozen across repeated training forwards and train only reconstruction."""
    images = torch.randn(2, 3, 64, 64)
    before = {name: value.clone() for name, value in model.encoder.named_buffers()}
    model.train()
    encoder, decoder = model(images)
    assert isinstance(encoder, list) and isinstance(decoder, list) and len(encoder) == len(decoder) == 3
    assert [value.shape for value in encoder] == [torch.Size((2, 64, 16, 16)), torch.Size((2, 128, 8, 8)),
                                                 torch.Size((2, 256, 4, 4))]
    assert [value.shape for value in encoder] == [value.shape for value in decoder]
    assert model.training and not model.encoder.training
    assert all(not module.training for module in model.encoder.modules())
    assert all(not feature.requires_grad for feature in encoder) and all(feature.requires_grad for feature in decoder)
    loss = ReverseDistillationLoss()(encoder, decoder)
    loss.backward()
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.bottleneck.parameters())
    assert any(parameter.grad is not None for parameter in model.decoder.parameters())
    model.train()
    with torch.no_grad():
        repeated = model(images)
    for first, second in zip((encoder, decoder), repeated, strict=True):
        assert [value.shape for value in first] == [value.shape for value in second]
    assert not model.encoder.training
    assert all(torch.equal(value, before[name]) for name, value in model.encoder.named_buffers())


def test_reverse_distillation_eval_and_strict_state_restore(model):
    """Restore all encoder and reconstruction weights and preserve original score reduction shape."""
    images = torch.randn(2, 3, 64, 64)
    assert model.tiler is None
    with torch.no_grad():
        expected = model(images)
    assert isinstance(expected, InferenceBatch) and expected.anomaly_map.shape == (2, 1, 64, 64)
    assert expected.pred_score.shape == (2, 1)
    torch.testing.assert_close(expected.pred_score, expected.anomaly_map.amax(dim=(-2, -1)))
    restored = ReverseDistillationModel(backbone="resnet18", input_size=(64, 64),
                                       layers=("layer1", "layer2", "layer3"),
                                       anomaly_map_mode=AnomalyMapGenerationMode.ADD, pre_trained=False).eval()
    result = restored.load_state_dict(model.state_dict(), strict=True)
    assert result.missing_keys == result.unexpected_keys == []
    with torch.no_grad():
        actual = restored(images)
    torch.testing.assert_close(actual.pred_score, expected.pred_score, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual.anomaly_map, expected.anomaly_map, rtol=1e-5, atol=1e-6)


def test_reverse_distillation_tiler_untile_feature_shapes(model):
    """Untile both encoder and decoder lists before constructing a full-size anomaly map."""
    model.tiler = Tiler(tile_size=32, stride=32)
    images = torch.randn(1, 3, 64, 64)
    model.train()
    with torch.no_grad():
        encoder, decoder = model(images)
    assert [value.shape for value in encoder] == [torch.Size((1, 64, 16, 16)), torch.Size((1, 128, 8, 8)),
                                                 torch.Size((1, 256, 4, 4))]
    assert [value.shape for value in encoder] == [value.shape for value in decoder]
    model.eval()
    with torch.no_grad():
        prediction = model(images)
    assert prediction.anomaly_map.shape == (1, 1, 64, 64) and prediction.pred_score.shape == (1, 1)


def constructor_checks():
    """Construct and strictly restore with file and network access forbidden after pure imports."""
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
        patch.setattr(reverse_model, "TimmFeatureExtractor", extractor)
        original = ReverseDistillationModel(backbone="resnet18", input_size=(64, 64),
                                           layers=("layer1", "layer2", "layer3"),
                                           anomaly_map_mode=AnomalyMapGenerationMode.ADD)
        restored = ReverseDistillationModel(backbone="resnet18", input_size=(64, 64),
                                           layers=("layer1", "layer2", "layer3"),
                                           anomaly_map_mode=AnomalyMapGenerationMode.ADD, pre_trained=False)
        assert calls == [False, False] and original.tiler is None
        state = original.state_dict()
        assert all(any(name.startswith(prefix) for name in state) for prefix in ("encoder.", "bottleneck.", "decoder."))
        result = restored.load_state_dict(state, strict=True)
        assert result.missing_keys == result.unexpected_keys == []
    restored.eval()
    with torch.no_grad():
        prediction = restored(torch.randn(1, 3, 64, 64))
    assert prediction.pred_score.shape == (1, 1) and prediction.anomaly_map.shape == (1, 1, 64, 64)


def test_reverse_distillation_isolated_constructor_imports_and_inference():
    """Keep pure imports, offline construction and inference independent of frameworks and OmegaConf."""
    script = '''
import importlib.abc
import runpy
import sys
import torch
import timm
class Block(importlib.abc.MetaPathFinder):
    """Reject model-side reference, training and configuration imports."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("anomalib", "lightning", "pytorch_lightning", "omegaconf"):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, Block())
sys.path.insert(0, sys.argv[2])
runpy.run_path(sys.argv[1])["constructor_checks"]()
assert not {"anomalib", "lightning", "pytorch_lightning", "omegaconf"} & set(sys.modules)
'''
    result = subprocess.run([sys.executable, "-c", script, os.path.abspath(__file__), os.path.join(PROJECT_ROOT, "src")],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_reverse_distillation_original_headers_and_removed_omegaconf():
    """Preserve original license markers on every copied file and remove the unused dependency."""
    root = os.path.join(PROJECT_ROOT, "src", "visionlab", "anomaly_detection", "models", "reverse_distillation")
    files = ("__init__.py", "model.py", "anomaly_map.py", "loss.py", "components/__init__.py",
             "components/bottleneck.py", "components/de_resnet.py")
    for relative in files:
        with open(os.path.join(root, relative), encoding="utf-8") as stream:
            source = stream.read()
        assert "# Copyright (C) 2022-2025 Intel Corporation" in source
        assert "# SPDX-License-Identifier: Apache-2.0" in source and "omegaconf" not in source
        if relative in ("anomaly_map.py", "components/bottleneck.py", "components/de_resnet.py"):
            assert "# Copyright (c) 2022 hq-deng" in source and "# SPDX-License-Identifier: MIT" in source
