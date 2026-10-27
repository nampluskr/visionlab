# tests/11_draem/test_draem.py: pytest pure DRAEM, SSPCAB, Perlin augmentation and dependency contracts

from contextlib import contextmanager
import inspect
import os
import subprocess
import sys

import pytest
import torch
from kornia.losses import FocalLoss, SSIMLoss
from torch import nn
from torchvision import io
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.transforms import v2

from visionlab.anomaly_detection.common.batch import InferenceBatch
from visionlab.anomaly_detection.common.multi_random_choice import MultiRandomChoice
from visionlab.anomaly_detection.common import perlin as perlin_module
from visionlab.anomaly_detection.common.perlin import PerlinAnomalyGenerator, generate_perlin_noise
from visionlab.anomaly_detection.models.components.layers import SSPCAB
from visionlab.anomaly_detection.models.components.layers.sspcab import AttentionModule
from visionlab.anomaly_detection.models.draem import DraemModel
from visionlab.anomaly_detection.models.draem.loss import DraemLoss
from visionlab.anomaly_detection.models.draem.model import DiscriminativeSubNetwork, ReconstructiveSubNetwork

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@contextmanager
def seeded_rng(seed):
    """Seed and restore CPU and available CUDA generators together."""
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed_all(seed)
        yield


@pytest.fixture
def source_images(tmp_path):
    """Create nested RGB sources with an ignored non-image file."""
    root = str(tmp_path)
    os.makedirs(os.path.join(root, "nested"))
    for name, size in (("first.png", 64), ("nested/second.png", 32), ("nested/third.png", 64)):
        image = torch.arange(3 * size * size).remainder(256).to(torch.uint8).reshape(3, size, size)
        io.write_png(image, os.path.join(root, name))
    io.write_jpeg(image, os.path.join(root, "nested", "fourth.jpg"))
    with open(os.path.join(root, "ignored.txt"), "w", encoding="utf-8") as stream:
        stream.write("ignored")
    return root


@pytest.fixture(scope="module")
def draem():
    """Share the full default-width pure model."""
    return DraemModel()


class RecordTransform:
    """Record selection order and transform one or several inputs."""

    def __init__(self, index, calls):
        self.index = index
        self.calls = calls

    def __call__(self, *inputs):
        self.calls.append(self.index)
        outputs = tuple(value * 10 + self.index for value in inputs)
        return outputs[0] if len(outputs) == 1 else outputs


def test_draem_multi_choice_validation_and_probabilities():
    """Reject invalid containers and lengths and normalize explicit or uniform weights."""
    with pytest.raises(TypeError):
        MultiRandomChoice(transforms=object())
    with pytest.raises(ValueError):
        MultiRandomChoice(transforms=[nn.Identity()], probabilities=[1, 2])
    transforms = [nn.Identity(), nn.Identity(), nn.Identity()]
    assert MultiRandomChoice(transforms=transforms, probabilities=[1, 2, 1]).probabilities == [0.25, 0.5, 0.25]
    assert MultiRandomChoice(transforms=transforms).probabilities == [1 / 3] * 3


@pytest.mark.parametrize("fixed", (False, True))
@pytest.mark.parametrize("multiple", (False, True))
def test_draem_multi_choice_seed_count_and_application_order(fixed, multiple):
    """Match randint and multinomial consumption and preserve multi-input application order."""
    calls = []
    transform = MultiRandomChoice(
        transforms=[RecordTransform(index, calls) for index in range(1, 7)],
        num_transforms=3, fixed_num_transforms=fixed,
    )
    inputs = (torch.tensor(0), torch.tensor(2)) if multiple else (torch.tensor(0),)
    selections = set()
    for seed in range(8):
        with seeded_rng(seed):
            count = 3 if fixed else int(torch.randint(3, (1,)) + 1)
            indices = torch.multinomial(torch.tensor(transform.probabilities), count).tolist()
        expected = list(inputs)
        for index in indices:
            expected = [value * 10 + index + 1 for value in expected]
        calls.clear()
        with seeded_rng(seed):
            actual = transform(*inputs)
        assert calls == [index + 1 for index in indices]
        assert len(calls) == count
        selections.add(tuple(calls))
        outputs = actual if multiple else (actual,)
        assert all(torch.equal(value, reference) for value, reference in zip(outputs, expected, strict=True))
        calls.clear()
        with seeded_rng(seed):
            repeated = transform(*inputs)
        repeated = repeated if multiple else (repeated,)
        assert all(torch.equal(value, reference) for value, reference in zip(outputs, repeated, strict=True))
    assert len(selections) > 1


def test_draem_perlin_signature_sources_and_augmenters(source_images):
    """Keep four defaults, extension-ordered recursive discovery and the ten original augmenters."""
    signature = inspect.signature(PerlinAnomalyGenerator)
    expected = {"anomaly_source_path": None, "probability": 0.5,
                "blend_factor": (0.2, 1.0), "rotation_range": (-90, 90)}
    assert list(signature.parameters) == list(expected)
    assert {name: value.default for name, value in signature.parameters.items()} == expected
    with seeded_rng(42):
        threshold = torch.empty(1).uniform_(32 / 255, 128 / 255).item()
        following = torch.rand(4)
    with seeded_rng(42):
        generator = PerlinAnomalyGenerator()
        assert torch.equal(torch.rand(4), following)
    assert generator.anomaly_source_paths == []
    assert generator.augmenters.transforms[4].threshold == threshold
    sourced = PerlinAnomalyGenerator(anomaly_source_path=source_images)
    expected_paths = []
    for extension in IMG_EXTENSIONS:
        for directory, _, names in os.walk(source_images):
            expected_paths.extend(os.path.join(directory, name) for name in names if name.endswith(extension))
    assert set(map(str, sourced.anomaly_source_paths)) == set(expected_paths)
    assert len(sourced.anomaly_source_paths) == 4
    extensions = [os.path.splitext(str(path))[1] for path in sourced.anomaly_source_paths]
    assert extensions == sorted(extensions, key=IMG_EXTENSIONS.index)
    assert [type(item) for item in generator.augmenters.transforms] == [
        v2.ColorJitter, v2.RandomPhotometricDistort, v2.RandomAdjustSharpness, v2.ColorJitter,
        v2.RandomSolarize, v2.RandomPosterize, v2.RandomInvert, v2.AutoAugment,
        v2.RandomEqualize, v2.RandomAffine,
    ]
    assert generator.augmenters.num_transforms == 3 and generator.augmenters.fixed_num_transforms
    assert generator.augmenters.probabilities == [0.1] * 10
    assert generator.perlin_rotation_transform.degrees == [-90.0, 90.0]
    assert generator.perlin_rotation_transform.interpolation == v2.InterpolationMode.BILINEAR


@pytest.mark.parametrize("with_source", (False, True))
def test_draem_perlin_perturbation_source_resize_and_mask(monkeypatch, source_images, with_source):
    """Check the strict half threshold, RGB source normalization and antialiased resize."""
    generator = PerlinAnomalyGenerator()
    noise = torch.zeros(64, 64)
    noise[:16] = 0.5
    noise[16:32] = 0.75
    observed = []

    def fixed_noise(height, width, device):
        """Return a known noise field on the explicitly requested device."""
        assert (height, width) == (64, 64) and device == torch.device("cpu")
        return noise

    def record_source(image):
        """Capture the channel-first augmentation input without consuming random numbers."""
        observed.append(image.clone())
        return image

    monkeypatch.setattr(perlin_module, "generate_perlin_noise", fixed_noise)
    generator.perlin_rotation_transform = nn.Identity()
    generator.augmenters = v2.Lambda(record_source)
    source = os.path.join(source_images, "nested", "second.png") if with_source else None
    perturbation, mask = generator.generate_perturbation(
        height=64, width=64, device=torch.device("cpu"), anomaly_source_path=source,
    )
    expected_mask = (noise > 0.5).float().unsqueeze(-1)
    assert torch.equal(mask, expected_mask)
    if with_source:
        expected_source = io.read_image(source, mode=io.ImageReadMode.RGB).float() / 255
        expected_source = v2.functional.resize(expected_source, [64, 64], antialias=True)
    else:
        expected_source = noise.unsqueeze(0).repeat(3, 1, 1) * 0.5 + 0.25
    torch.testing.assert_close(observed[0], expected_source)
    torch.testing.assert_close(perturbation, expected_source.permute(1, 2, 0) * expected_mask)


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("with_sources", (False, True))
def test_draem_perlin_seed_shapes_binary_masks_and_unmasked_pixels(device, with_sources, source_images):
    """Keep actual augmentation deterministic for both source modes on each available device."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    image = torch.linspace(0, 1, 2 * 3 * 64 * 64, device=device).reshape(2, 3, 64, 64)
    source = source_images if with_sources else None
    with seeded_rng(42):
        generator = PerlinAnomalyGenerator(anomaly_source_path=source, probability=1)
        actual, mask = generator(image)
    with seeded_rng(42):
        repeated_generator = PerlinAnomalyGenerator(anomaly_source_path=source, probability=1)
        repeated, repeated_mask = repeated_generator(image)
    assert actual.shape == image.shape and mask.shape == (2, 1, 64, 64)
    assert actual.device == mask.device == image.device
    assert ((mask == 0) | (mask == 1)).all()
    assert torch.equal(actual, repeated) and torch.equal(mask, repeated_mask)
    assert torch.equal(actual * (1 - mask), image * (1 - mask))
    with seeded_rng(19):
        single, single_mask = generator(image[0])
    assert single.shape == (3, 64, 64) and single_mask.shape == (1, 64, 64)
    with seeded_rng(19):
        sequential = [generator(sample) for sample in image]
    with seeded_rng(19):
        batched = generator(image)
    assert torch.equal(batched[0], torch.stack([item[0] for item in sequential]))
    assert torch.equal(batched[1], torch.stack([item[1] for item in sequential]))
    generator.probability = 0
    with seeded_rng(7):
        unchanged, empty_mask = generator(image)
    assert torch.equal(unchanged, image) and not empty_mask.any()


@pytest.mark.parametrize("blend", (0.3, (0.2, 0.8), None))
@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_draem_perlin_beta_formula_and_rng(monkeypatch, blend, device):
    """Check float, tuple and fallback beta with controlled nonempty perturbations."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    generator = PerlinAnomalyGenerator(probability=1, blend_factor=blend)
    image = torch.full((3, 64, 64), 0.25, device=device)
    mask = torch.zeros(64, 64, 1, device=device)
    mask[16:48, 16:48] = 1
    perturbation = torch.ones(64, 64, 3, device=device) * mask
    monkeypatch.setattr(generator, "generate_perturbation",
                        lambda height, width, device, anomaly_source_path: (perturbation, mask))
    with seeded_rng(29):
        torch.rand(1, device=device)
        beta = torch.rand(1, device=device) * 0.6 + 0.2 if isinstance(blend, tuple) else blend
        beta = 0.5 if beta is None else beta
        following = torch.rand(4, device=device)
    with seeded_rng(29):
        actual, actual_mask = generator(image)
        assert torch.equal(torch.rand(4, device=device), following)
    expected = image * (1 - actual_mask) + beta * perturbation.permute(2, 0, 1)
    expected += (1 - beta) * image * actual_mask
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert actual_mask.any() and (actual_mask == 0).any()


def test_draem_perlin_function_signature():
    """Retain the Phase 10 noise function contract."""
    signature = inspect.signature(generate_perlin_noise)
    assert list(signature.parameters) == ["height", "width", "scale", "device"]
    assert signature.parameters["height"].default is inspect.Parameter.empty
    assert signature.parameters["width"].default is inspect.Parameter.empty
    assert signature.parameters["scale"].default is signature.parameters["device"].default is None


@pytest.mark.parametrize("kernel_size, expected_count", ((1, 17736), (3, 148808)))
def test_draem_sspcab_shapes_parameter_counts_and_seed(kernel_size, expected_count):
    """Check four masked convolutions and the 64-to-8-to-64 attention parameter count."""
    with seeded_rng(42):
        block = SSPCAB(in_channels=64, kernel_size=kernel_size)
        following = torch.rand(4)
    with seeded_rng(42):
        repeated = SSPCAB(in_channels=64, kernel_size=kernel_size)
        assert torch.equal(torch.rand(4), following)
    # Attention: (64*8+8)+(8*64+64)=1096; convolutions: 4*(64*64*k*k+64).
    assert isinstance(block.attention_module, AttentionModule)
    assert sum(value.numel() for value in block.attention_module.parameters()) == 1096
    assert sum(value.numel() for value in block.parameters()) == expected_count
    assert block(torch.rand(2, 64, 64, 64)).shape == (2, 64, 64, 64)
    for name, value in block.state_dict().items():
        assert torch.equal(value, repeated.state_dict()[name]), name


@pytest.mark.parametrize("sspcab", (False, True))
def test_draem_constructor_defaults_seed_and_subnetwork_order(sspcab):
    """Match sequential full-width subnetwork initialization and following random draws."""
    signature = inspect.signature(DraemModel)
    assert list(signature.parameters) == ["sspcab"] and signature.parameters["sspcab"].default is False
    assert inspect.signature(ReconstructiveSubNetwork).parameters["base_width"].default == 128
    with seeded_rng(42):
        model = DraemModel(sspcab=sspcab)
        following = torch.rand(4)
    assert list(model._modules) == ["reconstructive_subnetwork", "discriminative_subnetwork"]
    assert isinstance(model.reconstructive_subnetwork.encoder.block5, SSPCAB) == sspcab
    with seeded_rng(42):
        reconstruction = ReconstructiveSubNetwork(sspcab=sspcab)
        for name, value in reconstruction.state_dict().items():
            assert torch.equal(value, model.reconstructive_subnetwork.state_dict()[name]), name
        del reconstruction
        discrimination = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        for name, value in discrimination.state_dict().items():
            assert torch.equal(value, model.discriminative_subnetwork.state_dict()[name]), name
        assert torch.equal(torch.rand(4), following)


def test_draem_training_and_eval_softmax_contract(draem):
    """Keep training tuples and three-dimensional evaluation maps from class-one softmax."""
    image = torch.rand(2, 3, 64, 64)
    draem.train()
    with torch.no_grad():
        training = draem(image)
    assert isinstance(training, tuple) and len(training) == 2
    assert training[0].shape == (2, 3, 64, 64) and training[1].shape == (2, 2, 64, 64)
    draem.eval()
    with torch.no_grad():
        reconstruction = draem.reconstructive_subnetwork(image)
        logits = draem.discriminative_subnetwork(torch.cat([image, reconstruction], axis=1))
        actual = draem(image)
    assert isinstance(actual, InferenceBatch)
    assert actual.anomaly_map.shape == (2, 64, 64) and actual.pred_score.shape == (2,)
    assert (actual.anomaly_map >= 0).all() and (actual.anomaly_map <= 1).all()
    torch.testing.assert_close(actual.anomaly_map, torch.softmax(logits, dim=1)[:, 1], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(actual.pred_score, actual.anomaly_map.amax(dim=(-2, -1)), atol=1e-6, rtol=1e-5)


def test_draem_loss_reference_positivity_and_perfect_reconstruction():
    """Sum the original MSE, twice SSIM and alpha-one focal loss without reordering arguments."""
    image, reconstruction = torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64)
    mask = torch.arange(2 * 64 * 64).remainder(2).reshape(2, 1, 64, 64)
    prediction = torch.randn(2, 2, 64, 64)
    criterion = DraemLoss()
    l2 = nn.MSELoss()(reconstruction, image)
    ssim = SSIMLoss(window_size=11)(reconstruction, image) * 2
    focal = FocalLoss(alpha=1, reduction="mean")(prediction, mask.squeeze(1).long())
    assert l2 > 0 and ssim > 0 and focal > 0
    actual = criterion(image, reconstruction, mask, prediction)
    torch.testing.assert_close(actual, l2 + ssim + focal)
    assert torch.equal(actual, criterion(image, reconstruction, mask, prediction))
    assert criterion.l2_loss(image, image).abs() < 1e-6
    assert criterion.ssim_loss(image, image).abs() < 1e-6


def test_draem_strict_state_dict_round_trip(draem):
    """Restore all tensors into a fresh full-width model and reproduce evaluation output."""
    restored = DraemModel().eval()
    result = restored.load_state_dict(draem.state_dict(), strict=True)
    assert result.missing_keys == result.unexpected_keys == []
    draem.eval()
    image = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        expected, actual = draem(image), restored(image)
    torch.testing.assert_close(actual.pred_score, expected.pred_score, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(actual.anomaly_map, expected.anomaly_map, atol=1e-6, rtol=1e-5)


def constructor_checks():
    """Construct and restore with file access and networking forbidden after guarded imports."""
    def forbidden(*args, **kwargs):
        raise AssertionError("Forbidden DRAEM constructor file or network access")

    with pytest.MonkeyPatch.context() as patch:
        for target in ("builtins.open", "io.open", "os.open", "torch.load", "socket.socket.connect",
                       "socket.socket.connect_ex", "socket.create_connection"):
            patch.setattr(target, forbidden)
        original = DraemModel()
        restored = DraemModel()
        result = restored.load_state_dict(original.state_dict(), strict=True)
        assert result.missing_keys == result.unexpected_keys == []
    restored.eval()
    with torch.no_grad():
        prediction = restored(torch.rand(2, 3, 64, 64))
    assert prediction.pred_score.shape == (2,) and prediction.anomaly_map.shape == (2, 64, 64)


def test_draem_isolated_import_offline_constructor_and_inference():
    """Block reference and Lightning imports before importing any pure DRAEM modules."""
    script = '''
import importlib.abc
import runpy
import sys
class Block(importlib.abc.MetaPathFinder):
    """Reject reference and training frameworks in a fresh interpreter."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("anomalib", "lightning", "pytorch_lightning"):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, Block())
runpy.run_path(sys.argv[1])["constructor_checks"]()
assert not {"anomalib", "lightning", "pytorch_lightning"} & set(sys.modules)
'''
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.path.join(PROJECT_ROOT, "src")
    result = subprocess.run([sys.executable, "-c", script, os.path.abspath(__file__)],
                            env=environment, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_draem_original_spdx_headers():
    """Preserve upstream licenses in every copied pure module including the expanded Perlin file."""
    root = os.path.join(PROJECT_ROOT, "src", "visionlab", "anomaly_detection")
    for relative in ("common/perlin.py", "common/multi_random_choice.py", "models/components/layers/__init__.py",
                     "models/components/layers/sspcab.py", "models/draem/__init__.py",
                     "models/draem/model.py", "models/draem/loss.py"):
        with open(os.path.join(root, relative), encoding="utf-8") as stream:
            source = stream.read()
        assert "# SPDX-License-Identifier: Apache-2.0" in source, relative
        assert "# Copyright (C)" in source, relative
        if relative == "models/draem/model.py":
            assert "# SPDX-License-Identifier: MIT" in source and "# Copyright (c) 2021 VitjanZ" in source
        if relative == "common/perlin.py":
            assert "class PerlinAnomalyGenerator" in source
