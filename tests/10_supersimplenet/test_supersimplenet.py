# tests/10_supersimplenet/test_supersimplenet.py: pytest pure SuperSimpleNet, Perlin synthesis, losses and offline construction

from contextlib import contextmanager
import inspect
import os
import subprocess
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops.focal_loss import sigmoid_focal_loss

from visionlab.anomaly_detection.common.batch import InferenceBatch
from visionlab.anomaly_detection.common.perlin import generate_perlin_noise
from visionlab.anomaly_detection.models.components.feature_extractors import TimmFeatureExtractor
from visionlab.anomaly_detection.models.components.filters import GaussianBlur2d
from visionlab.anomaly_detection.models.supersimplenet import SupersimplenetModel
from visionlab.anomaly_detection.models.supersimplenet import model as ssn_model
from visionlab.anomaly_detection.models.supersimplenet import anomaly_generator as generator_module
from visionlab.anomaly_detection.models.supersimplenet.anomaly_generator import AnomalyGenerator
from visionlab.anomaly_detection.models.supersimplenet.loss import SSNLoss
from visionlab.anomaly_detection.models.supersimplenet.model import (
    AnomalyMapGenerator, FeatureAdapter, SegmentationDetectionModule, UpscalingFeatureExtractor, init_weights,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@contextmanager
def seeded_rng(seed):
    """Seed and restore CPU and available CUDA generators used by original Perlin synthesis."""
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed_all(seed)
        yield


@pytest.fixture(scope="module")
def shared_ssn():
    """Share one actual small backbone without changing the model constructor's dryrun."""
    return SupersimplenetModel(backbone="resnet18", layers=["layer2", "layer3"])


@pytest.fixture
def model(shared_ssn):
    """Reset mode and gradients without altering any original model buffers."""
    shared_ssn.eval()
    shared_ssn.zero_grad(set_to_none=True)
    return shared_ssn


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_ssn_perlin_shapes_range_and_seed(device):
    """Generate reproducible fixed-scale noise on CPU and on CUDA when available."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    with seeded_rng(42):
        first = generate_perlin_noise(height=45, width=70, scale=(4, 8), device=torch.device(device))
    with seeded_rng(42):
        repeated = generate_perlin_noise(height=45, width=70, scale=(4, 8), device=torch.device(device))
    with seeded_rng(43):
        changed = generate_perlin_noise(height=45, width=70, scale=(4, 8), device=torch.device(device))
    assert first.shape == (45, 70) and first.device.type == device
    assert torch.isfinite(first).all() and first.min() >= -1 and first.max() <= 1
    assert torch.equal(first, repeated) and not torch.equal(first, changed)


def test_ssn_perlin_signature_random_scale_and_default_device():
    """Keep required dimensions, optional scale and device and the original CUDA-preferring default."""
    signature = inspect.signature(generate_perlin_noise)
    assert list(signature.parameters) == ["height", "width", "scale", "device"]
    assert signature.parameters["height"].default is signature.parameters["width"].default is inspect.Parameter.empty
    assert signature.parameters["scale"].default is signature.parameters["device"].default is None
    with seeded_rng(17):
        first = generate_perlin_noise(height=64, width=64)
    with seeded_rng(17):
        repeated = generate_perlin_noise(height=64, width=64)
    assert first.shape == (64, 64) and first.device.type == ("cuda" if torch.cuda.is_available() else "cpu")
    assert torch.equal(first, repeated)


def test_ssn_anomaly_generator_signature_power_and_binary_masks():
    """Keep all three required arguments, power rounding and binary Perlin mask output."""
    signature = inspect.signature(AnomalyGenerator.__init__)
    assert list(signature.parameters) == ["self", "noise_mean", "noise_std", "threshold"]
    assert all(signature.parameters[name].default is inspect.Parameter.empty for name in ("noise_mean", "noise_std", "threshold"))
    generator = AnomalyGenerator(noise_mean=0, noise_std=0.015, threshold=0.2)
    assert [generator.next_power_2(value) for value in (1, 2, 3, 16, 17, 63)] == [1, 2, 4, 16, 32, 64]
    with seeded_rng(21):
        masks = generator.generate_perlin(batches=4, height=37, width=45)
    assert masks.shape == (4, 1, 37, 45) and ((masks == 0) | (masks == 1)).all()


def test_ssn_perlin_sample_zeroing_matches_cpu_rand(monkeypatch):
    """Zero masks only for the original per-sample torch.rand(1) greater-than-half decision."""
    def perlin(height, width):
        return torch.ones(height, width)

    monkeypatch.setattr(generator_module, "generate_perlin_noise", perlin)
    generator = AnomalyGenerator(noise_mean=0, noise_std=0.015, threshold=0.2)
    with seeded_rng(42):
        expected_zero = torch.rand(4) > 0.5
    with seeded_rng(42):
        actual = generator.generate_perlin(batches=4, height=21, width=27)
    expected = (~expected_zero).long().reshape(4, 1, 1, 1).expand(4, 1, 21, 27)
    assert torch.equal(actual, expected)


def test_ssn_anomaly_generator_seeded_double_batch_and_labels():
    """Duplicate features and set normal-image labels exactly where new noise masks are nonempty."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = AnomalyGenerator(noise_mean=0, noise_std=0.015, threshold=0.2)
    features = torch.ones(2, 3, 37, 45, device=device)
    masks, labels = torch.zeros(2, 1, 37, 45, device=device), torch.zeros(2, device=device)
    with seeded_rng(42):
        first = generator(features, masks, labels)
    with seeded_rng(42):
        repeated = generator(features, masks, labels)
    perturbed, mask, label = first
    assert perturbed.shape == (4, 3, 37, 45) and mask.shape == (4, 1, 37, 45) and label.shape == (4,)
    assert torch.equal(label, mask.flatten(1).any(dim=1).float())
    assert ((mask == 0) | (mask == 1)).all()
    assert torch.count_nonzero((perturbed - 1) * (1 - mask)) == 0
    for actual, expected in zip(first, repeated, strict=True):
        assert torch.equal(actual, expected)
    assert not masks.any() and not labels.any()


def test_ssn_anomaly_generator_nonoverlap_and_duplication_order(monkeypatch):
    """Apply known noise only to new anomalous pixels and preserve existing positive labels."""
    generator = AnomalyGenerator(noise_mean=0.25, noise_std=0, threshold=0.2)
    features = torch.arange(2 * 3 * 4 * 6).float().reshape(2, 3, 4, 6)
    masks, labels = torch.zeros(2, 1, 4, 6), torch.tensor([0., 1.])
    masks[1, :, :2] = 1
    perlin = torch.zeros(4, 1, 4, 6)
    perlin[0, :, 1:3, 1:4] = 1
    perlin[1] = 1
    monkeypatch.setattr(generator, "generate_perlin", lambda batches, height, width: perlin.clone())
    perturbed, actual_mask, actual_labels = generator(features, masks, labels)
    duplicated = torch.cat([features, features])
    old_mask = torch.cat([masks, masks])
    noise_mask = (1 - old_mask) * perlin
    torch.testing.assert_close(perturbed, duplicated + 0.25 * noise_mask)
    assert torch.equal(actual_mask, (old_mask + noise_mask > 0).float())
    assert torch.equal(actual_labels, torch.tensor([1., 1., 0., 1.]))


def test_ssn_loss_matches_focal_and_truncated_reference():
    """Sum two alpha-free gamma-four focal terms and the original truncated L1 means."""
    pred_map = torch.randn(2, 1, 3, 4)
    target_mask = torch.arange(pred_map.numel()).reshape(pred_map.shape).remainder(2).float()
    pred_score, target_label = torch.tensor([-0.4, 0.8]), torch.tensor([0., 1.])
    criterion = SSNLoss()
    normal, anomalous = pred_map[target_mask == 0], pred_map[target_mask > 0]
    truncation = (normal + 0.5).clamp_min(0).mean() + (-anomalous + 0.5).clamp_min(0).mean()
    expected = sigmoid_focal_loss(pred_map, target_mask, alpha=-1, gamma=4, reduction="mean") + truncation
    expected += sigmoid_focal_loss(pred_score, target_label, alpha=-1, gamma=4, reduction="mean")
    actual = criterion(pred_map, pred_score, target_mask, target_label)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(actual, criterion(pred_map, pred_score, target_mask, target_label))


def test_ssn_truncated_empty_loss_retains_cpu_integer_zero():
    """Keep the original empty-input torch.tensor(0) result rather than manufacturing a device-matched float."""
    zero = SSNLoss().trunc_l1_loss(torch.empty(0), torch.empty(0))
    assert torch.equal(zero, torch.tensor(0)) and zero.dtype == torch.int64 and zero.device.type == "cpu"
    criterion = SSNLoss()
    pred = torch.tensor([-0.7, 0.1])
    torch.testing.assert_close(criterion.trunc_l1_loss(pred, torch.zeros(2)), (pred + 0.5).clamp_min(0).mean())
    torch.testing.assert_close(criterion.trunc_l1_loss(pred, torch.ones(2)), (-pred + 0.5).clamp_min(0).mean())


def test_ssn_feature_extractor_upscale_pool_and_frozen_gradients():
    """Upscale both real ResNet feature levels to twice the first resolution and freeze the backbone."""
    extractor = UpscalingFeatureExtractor(backbone="resnet18", layers=["layer2", "layer3"])
    extractor.train()
    image = torch.rand(2, 3, 64, 80, requires_grad=True)
    actual = extractor(image)
    with torch.no_grad():
        features = extractor.feature_extractor(image)
        expected = extractor.pooler(torch.cat([F.interpolate(value, size=(16, 20), mode="bilinear")
                                              for value in features.values()], dim=1))
    assert actual.shape == (2, 384, 16, 20) and not actual.requires_grad
    torch.testing.assert_close(actual, expected)
    assert extractor.get_channels_dim() == 384
    assert extractor.pooler.kernel_size == 3 and extractor.pooler.stride == 1 and extractor.pooler.padding == 1
    assert not extractor.feature_extractor.training
    assert all(parameter.grad is None for parameter in extractor.parameters()) and image.grad is None


@pytest.mark.parametrize("stop_grad", (False, True))
def test_ssn_adapter_segmentation_and_classification_gradient_routes(stop_grad):
    """Return correct head shapes and detach every classification-to-segmentation route when requested."""
    adapter = FeatureAdapter(channel_dim=8)
    features = adapter(torch.randn(2, 8, 8, 10))
    assert features.shape == (2, 8, 8, 10)
    module = SegmentationDetectionModule(channel_dim=8, stop_grad=stop_grad)
    maps, scores = module(features)
    assert maps.shape == (2, 1, 8, 10) and scores.shape == (2,)
    seg, dec = module.get_params()
    assert isinstance(seg, list) and isinstance(dec, list)
    assert {id(value) for value in seg} == {id(value) for value in module.seg_head.parameters()}
    assert {id(value) for value in dec} == {
        id(value) for value in list(module.cls_conv.parameters()) + list(module.cls_fc.parameters())}
    assert {id(value) for value in seg + dec} == {id(value) for value in module.parameters()}
    scores.sum().backward()
    if stop_grad:
        assert all(value.grad is None for value in seg)
    else:
        assert any(value.grad is not None for value in seg)
    assert any(value.grad is not None for value in dec)
    assert adapter.projection.weight.grad is not None


def test_ssn_init_weights_order_and_unchanged_biases():
    """Apply Xavier normal weights and unit BatchNorm weights in traversal order without changing biases."""
    conv, norm, linear = nn.Conv2d(3, 5, 1), nn.BatchNorm2d(5), nn.Linear(5, 2)
    with torch.no_grad():
        conv.bias.fill_(2)
        norm.bias.fill_(3)
        linear.bias.fill_(4)
    with seeded_rng(19):
        nn.ModuleList([conv, norm, linear]).apply(init_weights)
    with seeded_rng(19):
        expected_conv, expected_linear = torch.empty_like(conv.weight), torch.empty_like(linear.weight)
        nn.init.xavier_normal_(expected_conv)
        nn.init.xavier_normal_(expected_linear)
    assert torch.equal(conv.weight, expected_conv) and torch.equal(linear.weight, expected_linear)
    assert torch.equal(norm.weight, torch.ones(5))
    assert (conv.bias == 2).all() and (norm.bias == 3).all() and (linear.bias == 4).all()


def test_ssn_constructor_signature_sequence_and_seed(monkeypatch):
    """Preserve four defaults, the random dryrun position and the complete initialization sequence."""
    signature = inspect.signature(SupersimplenetModel.__init__)
    expected = {"perlin_threshold": 0.2, "backbone": "wide_resnet50_2", "layers": ["layer2", "layer3"], "stop_grad": True}
    assert list(signature.parameters) == ["self"] + list(expected)
    assert {name: signature.parameters[name].default for name in expected} == expected
    events = []
    channels = UpscalingFeatureExtractor.get_channels_dim

    def extractor(backbone, layers, pre_trained):
        events.append(("extractor", pre_trained))
        return TimmFeatureExtractor(backbone=backbone, layers=layers, pre_trained=pre_trained)

    def dryrun(module):
        events.append("dryrun")
        observed = []
        handle = module.feature_extractor.register_forward_pre_hook(lambda child, inputs: observed.append(inputs[0].shape))
        try:
            result = channels(module)
        finally:
            handle.remove()
        assert observed == [torch.Size([1, 3, 256, 256])]
        return result

    def adapter(channel_dim):
        events.append("adaptor")
        return FeatureAdapter(channel_dim=channel_dim)

    def segdec(channel_dim, stop_grad):
        events.append("segdec")
        return SegmentationDetectionModule(channel_dim=channel_dim, stop_grad=stop_grad)

    def generator(noise_mean, noise_std, threshold):
        events.append("generator")
        return AnomalyGenerator(noise_mean=noise_mean, noise_std=noise_std, threshold=threshold)

    def map_generator(sigma):
        events.append("map")
        return AnomalyMapGenerator(sigma=sigma)

    monkeypatch.setattr(ssn_model, "TimmFeatureExtractor", extractor)
    monkeypatch.setattr(UpscalingFeatureExtractor, "get_channels_dim", dryrun)
    monkeypatch.setattr(ssn_model, "FeatureAdapter", adapter)
    monkeypatch.setattr(ssn_model, "SegmentationDetectionModule", segdec)
    monkeypatch.setattr(ssn_model, "AnomalyGenerator", generator)
    monkeypatch.setattr(ssn_model, "AnomalyMapGenerator", map_generator)
    with seeded_rng(42):
        first = SupersimplenetModel(backbone="resnet18", layers=["layer2", "layer3"])
        following = torch.rand(4)
    assert events == [("extractor", False), "dryrun", "adaptor", "segdec", "generator", "map"]
    events.clear()
    with seeded_rng(42):
        second = SupersimplenetModel(backbone="resnet18", layers=["layer2", "layer3"])
        assert torch.equal(following, torch.rand(4))
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name]), name


def test_ssn_training_double_batch_and_eval_scalar_score(model):
    """Keep real Perlin training outputs and the original batch-one classification squeeze."""
    # The original random Perlin scale reaches 32, so training uses a 32x32 feature grid.
    image = torch.rand(2, 3, 128, 128)
    masks, labels = torch.zeros(2, 128, 128, dtype=torch.int64), torch.zeros(2, dtype=torch.int64)
    model.train()
    with seeded_rng(23):
        outputs = model(image, masks=masks, labels=labels)
    assert isinstance(outputs, tuple) and len(outputs) == 4
    maps, scores, expanded_masks, expanded_labels = outputs
    assert maps.shape == expanded_masks.shape == (4, 1, 32, 32)
    assert scores.shape == expanded_labels.shape == (4,) and expanded_labels.dtype == torch.float32
    assert torch.isfinite(SSNLoss()(maps, scores, expanded_masks, expanded_labels))
    assert not masks.any() and not labels.any()
    assert not model.feature_extractor.feature_extractor.training
    model.eval()
    with torch.no_grad():
        prediction = model(image)
        single = model(image[:1])
    assert isinstance(prediction, InferenceBatch)
    assert prediction.anomaly_map.shape == (2, 1, 128, 128) and prediction.pred_score.shape == (2,)
    assert single.anomaly_map.shape == (1, 1, 128, 128) and single.pred_score.ndim == 0


def test_ssn_mask_downsampling_threshold_and_final_blur():
    """Keep the half-threshold boundary and bilinear map upsampling followed by sigma-four blur."""
    mask = torch.tensor([[[0, 1], [1, 0]], [[0, 1], [0, 0]]])
    actual = SupersimplenetModel.downsample_mask(mask, feat_h=1, feat_w=1)
    assert actual.shape == (2, 1, 1, 1) and torch.equal(actual, torch.tensor([[[[1.]]], [[[0.]]]]))
    generator = AnomalyMapGenerator(sigma=4)
    values = torch.rand(2, 1, 8, 10)
    expected = GaussianBlur2d(kernel_size=25, sigma=4)(F.interpolate(values, size=(64, 80), mode="bilinear"))
    torch.testing.assert_close(generator(values, final_size=(64, 80)), expected)


def test_ssn_strict_state_dict_round_trip(model):
    """Restore every feature, adapter and head tensor and reproduce pure inference."""
    restored = SupersimplenetModel(backbone="resnet18", layers=["layer2", "layer3"]).eval()
    state = model.state_dict()
    assert all(any(name.startswith(prefix) for name in state) for prefix in ("feature_extractor.", "adaptor.", "segdec."))
    result = restored.load_state_dict(state, strict=True)
    assert result.missing_keys == result.unexpected_keys == []
    image = torch.rand(2, 3, 64, 80)
    with torch.no_grad():
        expected, actual = model(image), restored(image)
    torch.testing.assert_close(actual.pred_score, expected.pred_score, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual.anomaly_map, expected.anomaly_map, rtol=1e-5, atol=1e-6)


def constructor_checks():
    """Build under forbidden file and network access and infer without reference or training frameworks."""
    calls = []

    def extractor(backbone, layers, pre_trained):
        calls.append(pre_trained)
        return TimmFeatureExtractor(backbone=backbone, layers=layers, pre_trained=pre_trained)

    def forbidden(*args, **kwargs):
        raise AssertionError("Forbidden SuperSimpleNet constructor file or network access")

    with pytest.MonkeyPatch.context() as patch:
        for target in ("builtins.open", "io.open", "os.open", "torch.load", "socket.socket.connect",
                       "socket.socket.connect_ex", "socket.create_connection"):
            patch.setattr(target, forbidden)
        patch.setattr(ssn_model, "TimmFeatureExtractor", extractor)
        original = SupersimplenetModel(backbone="resnet18", layers=["layer2", "layer3"])
        restored = SupersimplenetModel(backbone="resnet18", layers=["layer2", "layer3"])
        assert calls == [False, False]
        result = restored.load_state_dict(original.state_dict(), strict=True)
        assert result.missing_keys == result.unexpected_keys == []
    restored.eval()
    with torch.no_grad():
        prediction = restored(torch.rand(2, 3, 64, 64))
    assert prediction.pred_score.shape == (2,) and prediction.anomaly_map.shape == (2, 1, 64, 64)


def test_ssn_isolated_import_offline_constructor_and_inference():
    """Check framework-free pure imports, strict restoration and inference in a fresh process."""
    script = '''
import importlib.abc
import runpy
import sys
import torch
import timm
class Block(importlib.abc.MetaPathFinder):
    """Reject reference and training frameworks before loading pure SuperSimpleNet."""
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


def test_ssn_original_headers_and_perlin_function_only():
    """Preserve licenses and the noise function alongside the Phase 11 Perlin transform."""
    root = os.path.join(PROJECT_ROOT, "src", "visionlab", "anomaly_detection")
    for name in ("__init__.py", "model.py", "anomaly_generator.py", "loss.py"):
        with open(os.path.join(root, "models", "supersimplenet", name), encoding="utf-8") as stream:
            source = stream.read()
        assert "# Copyright (C) 2024 Intel Corporation" in source and "# SPDX-License-Identifier: Apache-2.0" in source
        if name == "model.py":
            assert "# SPDX-License-Identifier: MIT" in source
    with open(os.path.join(root, "common", "perlin.py"), encoding="utf-8") as stream:
        source = stream.read()
    assert "# Copyright (C) 2022-2025 Intel Corporation" in source and "# SPDX-License-Identifier: Apache-2.0" in source
    assert "def generate_perlin_noise(" in source
