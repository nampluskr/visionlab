# tests/18_winclip/test_winclip.py: Pure WinCLIP scoring, dynamic buffers and isolated dependency tests

import inspect
import os
import subprocess
import sys

import pytest
import torch
from torch.nn import functional as F

from visionlab.anomaly_detection.models.winclip import model as winclip_module
from visionlab.anomaly_detection.models.winclip.clip import CLIP
from visionlab.anomaly_detection.models.winclip.prompting import create_prompt_ensemble
from visionlab.anomaly_detection.models.winclip.tokenizer import tokenize
from visionlab.anomaly_detection.models.winclip.utils import (
    class_scores, cosine_similarity, harmonic_aggregation, make_masks, visual_association_score,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_small_clip():
    """Build a seeded width-64, two-layer CLIP with embed_dim 32, context 77 and a 3x3 grid."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        return CLIP(embed_dim=32,
                    vision_cfg=dict(image_size=48, patch_size=16, width=64, layers=2, head_width=64),
                    text_cfg=dict(context_length=77, vocab_size=49408, width=64, heads=4, layers=2))


@pytest.fixture
def small_model(monkeypatch):
    """Replace only the CLIP construction point and keep the full original WinCLIP scoring path."""
    monkeypatch.setattr(winclip_module, "CLIP", make_small_clip)
    return winclip_module.WinClipModel().eval()


def reference_masks(scale):
    """Enumerate windows in row-major order with columns containing flattened patch indices."""
    return torch.tensor([[row * 3 + column + dy * 3 + dx
                          for dy in range(scale) for dx in range(scale)]
                         for row in range(4 - scale) for column in range(4 - scale)]).T


def reference_embeddings(clip, images):
    """Finish full-image and masked forwards from explicit pre-dropout tokens without hooks."""
    visual = clip.visual
    if images.shape[-2:] != visual.image_size:
        images = F.interpolate(images, size=visual.image_size, mode="bicubic", align_corners=False, antialias=True)
    patches = visual.conv1(images).flatten(2).permute(0, 2, 1)
    cls = visual.class_embedding.view(1, 1, -1).expand(images.shape[0], 1, -1)
    feature_map = torch.cat((cls, patches), dim=1) + visual.positional_embedding
    full = visual.ln_pre(feature_map)
    full = visual.transformer(full.permute(1, 0, 2)).permute(1, 0, 2)
    full = visual.ln_post(full)
    windows = []
    for scale in (2, 3):
        masks = reference_masks(scale)
        indices = torch.cat((torch.zeros(1, masks.shape[1], dtype=torch.long), masks + 1)).T
        selected = torch.cat([feature_map[:, index] for index in indices])
        selected = visual.ln_pre(selected)
        selected = visual.transformer(selected.permute(1, 0, 2)).permute(1, 0, 2)
        selected = visual.ln_post(selected)
        pooled = selected[:, 0] @ visual.proj
        windows.append(pooled.reshape(len(indices), len(images), -1).permute(1, 0, 2))
    return full[:, 0] @ visual.proj, windows, full[:, 1:]


def reference_class_scores(embeddings, text, temperature):
    """Compute the anomalous softmax probability from normalized dot products."""
    return (F.normalize(embeddings, dim=-1) @ F.normalize(text, dim=-1).T / temperature).softmax(-1)[..., 1]


def reference_harmonic(scores, masks):
    """Calculate the harmonic mean over windows covering each of nine patches."""
    values = []
    for patch in range(9):
        selected = scores[:, (masks == patch).any(dim=0)]
        values.append(selected.shape[1] / selected.reciprocal().sum(dim=1))
    return torch.stack(values, dim=1).reshape(-1, 3, 3).nan_to_num(posinf=0.0)


def reference_association(embeddings, references):
    """Compute minimum half cosine distance to every reference patch."""
    references = references.reshape(-1, embeddings.shape[-1])
    similarity = F.normalize(embeddings, dim=-1) @ F.normalize(references, dim=-1).T
    return (1 - similarity).min(dim=-1).values / 2


def reference_forward(model, images):
    """Follow the original zero/few-shot score equations using independent window calculations."""
    image, windows, patches = reference_embeddings(model.clip, images)
    score = reference_class_scores(image, model.text_embeddings, model.temperature)
    scales = [score[:, None, None].repeat(1, 3, 3)]
    for scale, window in zip((2, 3), windows):
        scales.append(reference_harmonic(reference_class_scores(window, model.text_embeddings, model.temperature),
                                          reference_masks(scale)))
    anomaly = len(scales) / torch.stack(scales).reciprocal().sum(dim=0)
    if model.k_shot:
        few = [reference_association(patches, model.patch_embeddings).reshape(-1, 3, 3)]
        for scale, window, reference in zip((2, 3), windows, model.visual_embeddings):
            few.append(reference_harmonic(reference_association(window, reference), reference_masks(scale)))
        few = torch.stack(few).mean(dim=0)
        anomaly = (anomaly + few) / 2
        score = (score + few.amax(dim=(-2, -1))) / 2
    anomaly = F.interpolate(anomaly.unsqueeze(1), size=images.shape[-2:], mode="bilinear").squeeze(1)
    return score, anomaly


def test_winclip_constructor_offline_and_unsupported_transform(small_model, monkeypatch):
    """Keep four original defaults and build width-64 CLIP without any explicit file or network access."""
    expected = dict(class_name=None, reference_images=None, scales=(2, 3), apply_transform=False)
    parameters = inspect.signature(winclip_module.WinClipModel).parameters
    assert list(parameters) == list(expected)
    assert {name: value.default for name, value in parameters.items()} == expected
    assert all(value.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD for value in parameters.values())

    def reject(*args, **kwargs):
        raise AssertionError("Unexpected pure model file or network access")

    with monkeypatch.context() as patch:
        for target in ("builtins.open", "io.open", "os.open", "torch.load", "socket.create_connection",
                       "socket.socket.connect", "socket.socket.connect_ex", "torch.hub.load_state_dict_from_url"):
            patch.setattr(target, reject)
        model = winclip_module.WinClipModel()
    assert model.clip.visual.output_tokens and model.grid_size == (3, 3)
    assert not hasattr(winclip_module, "PRETRAINED")
    source = inspect.getsource(winclip_module)
    assert "torch.load" not in source and "open(" not in source
    assert not hasattr(model, "pretrained") and not hasattr(model, "_transform") and not hasattr(model, "transform")
    with pytest.raises(NotImplementedError, match="apply_transform"):
        winclip_module.WinClipModel(apply_transform=True)
    for attribute in ("text_embeddings", "visual_embeddings", "patch_embeddings"):
        with pytest.raises(RuntimeError):
            getattr(model, attribute)


def test_winclip_utils_hand_calculations():
    """Check both 3x3 masks, harmonic aggregation, cosine similarities and anomalous probabilities."""
    assert torch.equal(make_masks((3, 3), 2), torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5],
                                                          [3, 4, 6, 7], [4, 5, 7, 8]], dtype=torch.int32))
    for scale in (2, 3):
        assert torch.equal(make_masks((3, 3), scale), reference_masks(scale).int())
    with pytest.raises(ValueError):
        make_masks((2, 2), 3)
    scores = torch.tensor([[1.0, 0.75, 0.5, 0.25]])
    expected = torch.tensor([[[1.0, 6 / 7, 0.75], [2 / 3, 0.48, 0.375], [0.5, 1 / 3, 0.25]]])
    torch.testing.assert_close(harmonic_aggregation(scores, (3, 3), make_masks((3, 3), 2)), expected)
    torch.testing.assert_close(harmonic_aggregation(torch.zeros(1, 4), (3, 3), make_masks((3, 3), 2)),
                               torch.zeros(1, 3, 3))
    unit = torch.eye(2)
    torch.testing.assert_close(cosine_similarity(unit, unit), unit)
    torch.testing.assert_close(cosine_similarity(unit.unsqueeze(0), unit), unit.unsqueeze(0))
    probabilities = torch.tensor([1 / (1 + torch.exp(torch.tensor(1.0))),
                                  1 / (1 + torch.exp(torch.tensor(-1.0)))])
    torch.testing.assert_close(class_scores(unit, unit, target_class=1), probabilities)
    torch.testing.assert_close(visual_association_score(unit.unsqueeze(0), unit.unsqueeze(0)), torch.zeros(1, 2))


def test_winclip_prompts_and_original_text_mean(small_model):
    """Average the unnormalized original text features over all 147 normal and 84 anomalous prompts."""
    normal, anomalous = create_prompt_ensemble("bottle")
    assert len(normal) == 147 and len(anomalous) == 84
    assert normal[0] == "a cropped photo of the bottle."
    assert normal[-1] == "a photo of a bottle without damage for anomaly detection."
    assert anomalous[0] == "a cropped photo of the damaged bottle."
    assert anomalous[-1] == "a photo of a bottle with damage for anomaly detection."
    small_model.setup("bottle")
    with torch.no_grad():
        expected = torch.cat([small_model.clip.encode_text(tokenize(prompts)).mean(dim=0, keepdim=True)
                              for prompts in (normal, anomalous)])
    assert small_model.text_embeddings.shape == (2, 32)
    torch.testing.assert_close(small_model.text_embeddings, expected, atol=1e-6, rtol=1e-5)
    created = winclip_module.WinClipModel(class_name="bottle")
    torch.testing.assert_close(created.text_embeddings, expected, atol=1e-6, rtol=1e-5)
    small_model.setup("candle")
    assert small_model.class_name == "candle" and not torch.equal(small_model.text_embeddings, expected)


@pytest.mark.parametrize("size,batch_size", ((48, 2), (64, 1)))
def test_winclip_independent_window_forward_and_resize(small_model, monkeypatch, size, batch_size):
    """Compare original window equations using width-64, two-layer CLIP on native and resized CPU inputs."""
    small_model.setup("bottle")
    images = torch.rand(batch_size, 3, size, size, generator=torch.Generator().manual_seed(11))
    with torch.no_grad():
        expected_score, expected_map = reference_forward(small_model, images)
    calls = []
    interpolate = F.interpolate

    def record_interpolate(input, *args, **kwargs):
        calls.append(kwargs.get("mode"))
        return interpolate(input, *args, **kwargs)

    monkeypatch.setattr(F, "interpolate", record_interpolate)
    actual = small_model(images)
    assert calls == (["bicubic", "bilinear"] if size != 48 else ["bilinear"])
    assert actual.pred_score.shape == (batch_size,) and actual.anomaly_map.shape == (batch_size, size, size)
    torch.testing.assert_close(actual.pred_score, expected_score, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(actual.anomaly_map, expected_map, atol=1e-6, rtol=1e-5)
    assert not actual.pred_score.requires_grad and not actual.anomaly_map.requires_grad


def test_winclip_hook_cleanup_success_and_failure(small_model, monkeypatch):
    """Release the feature hook after repeated calls and after an image encoder exception."""
    images = torch.rand(2, 3, 48, 48)
    with torch.no_grad():
        first = small_model.encode_image(images)
        second = small_model.encode_image(images)
    assert not small_model.clip.visual.patch_dropout._forward_hooks
    torch.testing.assert_close(first[0], second[0], atol=0, rtol=0)
    for expected, actual in zip(first[1], second[1]):
        torch.testing.assert_close(expected, actual, atol=0, rtol=0)
    torch.testing.assert_close(first[2], second[2], atol=0, rtol=0)

    def fail(image):
        raise RuntimeError("encoder failure")

    monkeypatch.setattr(small_model.clip, "encode_image", fail)
    with pytest.raises(RuntimeError, match="encoder failure"):
        small_model.encode_image(images)
    assert not small_model.clip.visual.patch_dropout._forward_hooks


def test_winclip_dynamic_buffers_and_strict_restore(small_model):
    """Strictly restore all small CLIP weights and dynamically shaped text buffers without tokenization."""
    small_model.setup("bottle")
    state = small_model.state_dict()
    assert "_text_embeddings" in state and "clip.token_embedding.weight" in state
    assert not any(name.startswith("_masks_") for name in state)
    restored = winclip_module.WinClipModel().eval()
    assert restored._text_embeddings.numel() == 0
    restored.load_state_dict(state, strict=True)
    assert restored.text_embeddings.shape == (2, 32)
    images = torch.rand(2, 3, 48, 48)
    expected, actual = small_model(images), restored(images)
    torch.testing.assert_close(expected.pred_score, actual.pred_score, atol=0, rtol=0)
    torch.testing.assert_close(expected.anomaly_map, actual.anomaly_map, atol=0, rtol=0)


def test_winclip_small_few_shot_reference_formula(small_model):
    """Check small synthetic reference buffers and the original equal zero/few-shot mixture only."""
    references = torch.rand(2, 3, 48, 48, generator=torch.Generator().manual_seed(17))
    small_model.setup("bottle", references)
    assert small_model.k_shot == 2
    assert [value.shape for value in small_model.visual_embeddings] == [(2, 4, 32), (2, 1, 32)]
    assert small_model.patch_embeddings.shape == (2, 9, 64)
    state = small_model.state_dict()
    assert state["__visual_embeddings_0"].shape == (2, 4, 32)
    assert state["_patch_embeddings"].shape == (2, 9, 64)
    images = torch.rand(1, 3, 48, 48)
    with torch.no_grad():
        score, anomaly = reference_forward(small_model, images)
    actual = small_model(images)
    torch.testing.assert_close(actual.pred_score, score, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(actual.anomaly_map, anomaly, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("block_regex", (False, True))
def test_winclip_pure_isolated_setup_and_forward(block_regex):
    """Block all optional model frameworks in a fresh process and require regex only for text setup."""
    script = '''
import importlib.abc
import runpy
import sys
import torch
forbidden = {"anomalib", "lightning", "pytorch_lightning", "open_clip", "ftfy", "timm", "torchvision", "kornia"}
if sys.argv[3] == "True":
    forbidden.add("regex")
assert not forbidden & {name.split(".")[0] for name in sys.modules}
class Block(importlib.abc.MetaPathFinder):
    """Reject optional dependencies before loading pure WinCLIP."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in forbidden:
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, Block())
sys.path.insert(0, sys.argv[2])
helpers = runpy.run_path(sys.argv[1])
module = helpers["winclip_module"]
module.CLIP = helpers["make_small_clip"]
model = module.WinClipModel().eval()
if "regex" in forbidden:
    try:
        model.setup("bottle")
    except ModuleNotFoundError as error:
        assert error.name == "regex"
    else:
        raise AssertionError("Text setup must require regex")
else:
    model.setup("bottle")
    prediction = model(torch.zeros(1, 3, 64, 64))
    assert prediction.pred_score.shape == (1,)
    assert prediction.anomaly_map.shape == (1, 64, 64)
assert not forbidden & {name.split(".")[0] for name in sys.modules}
'''
    result = subprocess.run([sys.executable, "-c", script, os.path.abspath(__file__),
                             os.path.join(PROJECT_ROOT, "src"), str(block_regex)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
