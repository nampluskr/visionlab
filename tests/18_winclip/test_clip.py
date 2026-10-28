# tests/18_winclip/test_clip.py: Independent CLIP parity, BPE and isolated dependency contracts

import ast
import hashlib
import importlib.util
import math
import os
import runpy
import subprocess
import sys

import pytest
import torch
from torch.nn import functional as F

from visionlab.anomaly_detection.models.winclip.clip import CLIP
from visionlab.anomaly_detection.models.winclip.tokenizer import SimpleTokenizer, tokenize

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_ROOT = os.path.join(PROJECT_ROOT, "src", "visionlab", "anomaly_detection", "models", "winclip")
WEIGHTS = r"D:\backbones\vit_base_patch16_plus_clip_240.laion400m_e31\open_clip_pytorch_model.bin"
SMALL_VISION = dict(image_size=48, patch_size=16, width=64, layers=2, head_width=64)
SMALL_TEXT = dict(context_length=77, vocab_size=49408, width=64, heads=4, layers=2)
VISUAL_COUNTS = {
    "conv1": 688128, "class_embedding": 896, "positional_embedding": 202496,
    "ln_pre": 1792, "transformer": 115745280, "ln_post": 1792, "proj": 573440,
}
TEXT_COUNTS = {
    "token_embedding": 31621120, "positional_embedding": 49280, "transformer": 59082240,
    "ln_final": 1280, "text_projection": 409600, "logit_scale": 1,
}


@pytest.fixture(scope="module")
def small_pair():
    """Reuse CPU CLIPs with width 64, two layers, embed_dim 32, 48x48 images and context 77."""
    from open_clip.model import CLIP as ReferenceCLIP

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        reference = ReferenceCLIP(embed_dim=32, vision_cfg=SMALL_VISION, text_cfg=SMALL_TEXT).eval()
        actual = CLIP(embed_dim=32, vision_cfg=SMALL_VISION, text_cfg=SMALL_TEXT).eval()
        actual.load_state_dict(reference.state_dict(), strict=True)
    return actual, reference


def test_clip_default_schema_counts_and_local_strict_load():
    """Compare default ViT-B-16-plus-240 schemas on meta and read the 795 MiB checkpoint once."""
    from open_clip.model import CLIP as ReferenceCLIP

    with torch.device("meta"):
        actual = CLIP()
        reference = ReferenceCLIP(
            embed_dim=640,
            vision_cfg=dict(image_size=240, patch_size=16, width=896, layers=12),
            text_cfg=dict(context_length=77, vocab_size=49408, width=640, heads=10, layers=12),
        )
    state = actual.state_dict()
    expected = reference.state_dict()
    assert len(state) == 302 and set(state) == set(expected)
    assert {key: value.shape for key, value in state.items()} == {
        key: value.shape for key, value in expected.items()
    }
    visual_counts, text_counts = {}, {}
    for name, value in actual.named_parameters():
        if name.startswith("visual."):
            group = name.split(".")[1]
            visual_counts[group] = visual_counts.get(group, 0) + value.numel()
        else:
            group = name.split(".")[0]
            text_counts[group] = text_counts.get(group, 0) + value.numel()
    assert visual_counts == VISUAL_COUNTS and text_counts == TEXT_COUNTS
    assert sum(visual_counts.values()) == 117213824
    assert sum(text_counts.values()) == 91163521
    assert sum(value.numel() for value in actual.parameters()) == 208377345
    assert "attn_mask" not in state and actual.attn_mask.shape == (77, 77)
    assert actual.visual.grid_size == (15, 15)
    assert actual.visual.transformer.resblocks[0].attn.num_heads == 14
    assert isinstance(actual.visual.patch_dropout, torch.nn.Identity)
    assert all(isinstance(block.mlp.gelu, torch.nn.GELU)
               for tower in (actual.visual.transformer, actual.transformer) for block in tower.resblocks)
    del reference, expected
    if not os.path.isfile(WEIGHTS):
        pytest.skip("Local CLIP checkpoint is unavailable; schemas and counts were checked")
    checkpoint = torch.load(WEIGHTS, weights_only=True, map_location="cpu")
    assert set(checkpoint) == set(state)
    assert {key: value.shape for key, value in checkpoint.items()} == {
        key: value.shape for key, value in state.items()
    }
    assert all(value.dtype == torch.float32 for value in checkpoint.values())
    # Assign into the meta model to avoid allocating a second 795 MiB parameter copy.
    loaded = actual.load_state_dict(checkpoint, strict=True, assign=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    assert all(value.device.type == "cpu" for value in actual.parameters())


def window_embeddings(visual, feature_map, masks):
    """Follow WinClipModel._get_window_embeddings without importing anomalib."""
    batch_size = feature_map.shape[0]
    n_masks = masks.shape[1]
    class_index = torch.zeros(1, n_masks, dtype=int).to(feature_map.device)
    masks = torch.cat((class_index, masks + 1)).T
    masked = torch.cat([torch.index_select(feature_map, 1, mask) for mask in masks])
    masked = visual.patch_dropout(masked)
    masked = visual.ln_pre(masked)
    masked = visual.transformer(masked.permute(1, 0, 2)).permute(1, 0, 2)
    masked = visual.ln_post(masked)
    pooled, _ = visual._global_pool(masked)
    pooled = pooled @ visual.proj
    return pooled.reshape(n_masks, batch_size, -1).permute(1, 0, 2)


def test_clip_cpu_encoders_hook_and_window_parity(small_pair):
    """Match width-64 two-layer encoders and scale-2/3 windows with CPU batch 2 at FR-5 tolerances."""
    actual, reference = small_pair
    actual.visual.output_tokens = reference.visual.output_tokens = True
    captured = []

    def capture(module, inputs, output):
        captured.append(inputs[0].detach().clone())

    handles = [model.visual.patch_dropout.register_forward_hook(capture) for model in small_pair]
    try:
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(7)
            images = torch.randn(2, 3, 48, 48)
            image, patches = actual.encode_image(images)
            ref_image, ref_patches = reference.encode_image(images)
    finally:
        for handle in handles:
            handle.remove()
    assert image.shape == (2, 32) and patches.shape == (2, 9, 64)
    assert len(captured) == 2 and captured[0].shape == (2, 10, 64)
    torch.testing.assert_close(image, ref_image, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(patches, ref_patches, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(captured[0], captured[1], atol=1e-6, rtol=1e-5)
    with torch.no_grad():
        raw = actual.visual.conv1(images).flatten(2).permute(0, 2, 1)
        torch.testing.assert_close(captured[0][:, 1:], raw + actual.visual.positional_embedding[1:],
                                   atol=1e-6, rtol=1e-5)
        for scale in (2, 3):
            masks = F.unfold(torch.arange(9).reshape(1, 3, 3).float(), kernel_size=scale).int()
            windows = window_embeddings(actual.visual, captured[0], masks)
            ref_windows = window_embeddings(reference.visual, captured[1], masks)
            assert windows.shape == (2, (4 - scale) ** 2, 32)
            torch.testing.assert_close(windows, ref_windows, atol=1e-6, rtol=1e-5)
        tokens = tokenize(["a photo of a bottle.", "a photo of a damaged candle."])
        torch.testing.assert_close(actual.encode_text(tokens), reference.encode_text(tokens), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(actual.encode_text(tokens, normalize=True),
                                   reference.encode_text(tokens, normalize=True), atol=1e-6, rtol=1e-5)
        normalized, normalized_patches = actual.encode_image(images, normalize=True)
        torch.testing.assert_close(normalized, F.normalize(ref_image, dim=-1), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(normalized_patches, ref_patches, atol=1e-6, rtol=1e-5)
        actual.visual.output_tokens = reference.visual.output_tokens = False
        torch.testing.assert_close(actual.encode_image(images), reference.encode_image(images), atol=1e-6, rtol=1e-5)


def test_clip_seed_and_reference_initialization():
    """Match repeated width-64 two-layer initialization, reference RNG consumption and logit scale."""
    from open_clip.model import CLIP as ReferenceCLIP

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        first = CLIP(embed_dim=32, vision_cfg=SMALL_VISION, text_cfg=SMALL_TEXT)
        following = torch.get_rng_state()
        torch.manual_seed(19)
        second = CLIP(embed_dim=32, vision_cfg=SMALL_VISION, text_cfg=SMALL_TEXT)
        assert torch.equal(following, torch.get_rng_state())
        assert all(torch.equal(value, second.state_dict()[key]) for key, value in first.state_dict().items())
        del second
        torch.manual_seed(19)
        reference = ReferenceCLIP(embed_dim=32, vision_cfg=SMALL_VISION, text_cfg=SMALL_TEXT)
        assert torch.equal(following, torch.get_rng_state())
        assert all(torch.equal(value, reference.state_dict()[key]) for key, value in first.state_dict().items())
    assert first.logit_scale.item() == pytest.approx(math.log(1 / 0.07))


def prompt_ensemble(class_name):
    """Reproduce all 231 prompts from anomalib 2.0.0 prompting.py without importing it."""
    # Copyright (C) 2024 Intel Corporation; SPDX-License-Identifier: Apache-2.0
    states = ["{}", "flawless {}", "perfect {}", "unblemished {}", "{} without flaw",
              "{} without defect", "{} without damage", "damaged {}", "{} with flaw",
              "{} with defect", "{} with damage"]
    templates = [
        "a cropped photo of the {}.", "a close-up photo of a {}.", "a close-up photo of the {}.",
        "a bright photo of a {}.", "a bright photo of the {}.", "a dark photo of the {}.",
        "a dark photo of a {}.", "a jpeg corrupted photo of the {}.", "a jpeg corrupted photo of the {}.",
        "a blurry photo of the {}.", "a blurry photo of a {}.", "a photo of a {}.", "a photo of the {}.",
        "a photo of a small {}.", "a photo of the small {}.", "a photo of a large {}.",
        "a photo of the large {}.", "a photo of the {} for visual inspection.",
        "a photo of a {} for visual inspection.", "a photo of the {} for anomaly detection.",
        "a photo of a {} for anomaly detection.",
    ]
    return [template.format(state.format(class_name)) for state in states for template in templates]


def test_clip_tokenizer_exact_reference_tokens():
    """Compare default context 77 on all bottle/candle prompts and ASCII boundary inputs."""
    from open_clip.tokenizer import SimpleTokenizer as ReferenceTokenizer
    from open_clip.tokenizer import tokenize as reference_tokenize

    texts = prompt_ensemble("bottle") + prompt_ensemble("candle")
    assert len(texts) == 462
    texts += ["", " \t\n ", " a  PHOTO\t of\n a bottle. ", "bottle " * 200,
              "a &amp;amp; b &#39;s", "it's 123.45!", "<start_of_text> bottle <end_of_text>"]
    actual = tokenize(texts)
    assert actual.dtype == torch.long and actual.shape == (len(texts), 77)
    assert torch.equal(actual, reference_tokenize(texts))
    tokenizer, reference = SimpleTokenizer(), ReferenceTokenizer()
    assert tokenizer.vocab_size == 49408
    assert (tokenizer.sot_token_id, tokenizer.eot_token_id) == (49406, 49407)
    for length in (1, 2, 7, 77):
        assert torch.equal(tokenizer(texts, context_length=length), reference(texts, context_length=length))
    assert torch.equal(tokenize(""), torch.tensor([[49406, 49407] + [0] * 75]))
    assert actual[-4, -1] == tokenizer.eot_token_id
    assert tokenize([]).shape == (0, 77)
    for text in texts:
        assert tokenizer.encode(text) == reference.encode(text)
        assert tokenizer.decode(tokenizer.encode(text)) == reference.decode(reference.encode(text))


def test_clip_bpe_copy_bytes():
    """Keep the bundled OpenAI CLIP vocabulary byte-identical to installed open_clip."""
    spec = importlib.util.find_spec("open_clip")
    original = os.path.join(os.path.dirname(spec.origin), "bpe_simple_vocab_16e6.txt.gz")
    copied = os.path.join(MODEL_ROOT, "bpe_simple_vocab_16e6.txt.gz")
    with open(original, "rb") as stream:
        expected = stream.read()
    with open(copied, "rb") as stream:
        actual = stream.read()
    assert len(actual) == len(expected) == 1356917
    assert actual == expected
    assert hashlib.sha256(actual).digest() == hashlib.sha256(expected).digest()


@pytest.mark.parametrize("block_regex", (False, True))
def test_clip_isolated_dependencies_and_constructor_io(block_regex):
    """Import and build width-64 two-layer CLIP without I/O; require regex only at tokenizer creation."""
    script = '''
import builtins
import importlib.abc
import io
import os
import socket
import sys
import torch
forbidden = {"anomalib", "lightning", "pytorch_lightning", "open_clip", "ftfy", "timm", "torchvision"}
if sys.argv[2] == "True":
    forbidden.add("regex")
assert not forbidden & {name.split(".")[0] for name in sys.modules}
class Block(importlib.abc.MetaPathFinder):
    """Reject forbidden imports in a fresh interpreter."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in forbidden:
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, Block())
sys.path.insert(0, sys.argv[1])
def reject(*args, **kwargs):
    """Reject explicit file and network access while importing and constructing encoders."""
    raise AssertionError("Unexpected CLIP file or network access")
targets = [(builtins, "open"), (io, "open"), (os, "open"), (torch, "load"),
           (socket.socket, "connect"), (socket.socket, "connect_ex"), (socket, "create_connection"),
           (torch.hub, "load_state_dict_from_url")]
originals = [(owner, name, getattr(owner, name)) for owner, name in targets]
for owner, name in targets:
    setattr(owner, name, reject)
try:
    from visionlab.anomaly_detection.models.winclip.clip import CLIP
    from visionlab.anomaly_detection.models.winclip.tokenizer import SimpleTokenizer, tokenize
    model = CLIP(embed_dim=32,
                 vision_cfg=dict(image_size=48, patch_size=16, width=64, layers=2, head_width=64),
                 text_cfg=dict(context_length=77, vocab_size=49408, width=64, heads=4, layers=2)).eval()
    model.load_state_dict(model.state_dict(), strict=True)
finally:
    for owner, name, original in originals:
        setattr(owner, name, original)
socket.socket.connect = socket.socket.connect_ex = socket.create_connection = reject
if "regex" in forbidden:
    try:
        SimpleTokenizer()
    except ModuleNotFoundError as error:
        assert str(error) == "regex"
    else:
        raise AssertionError("Tokenizer construction must require regex")
    tokens = torch.zeros(2, 77, dtype=torch.long)
    tokens[:, :2] = torch.tensor([49406, 49407])
else:
    tokens = tokenize(["a photo of a bottle.", "a photo of a candle."])
with torch.no_grad():
    assert model.encode_text(tokens).shape == (2, 32)
    assert model.encode_image(torch.zeros(2, 3, 48, 48)).shape == (2, 32)
assert not forbidden & {name.split(".")[0] for name in sys.modules}
'''
    result = subprocess.run([sys.executable, "-c", script, os.path.join(PROJECT_ROOT, "src"), str(block_regex)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_clip_no_file_loading_methods_or_forbidden_imports():
    """Reject file loaders and non-torch dependencies in the independent encoder source."""
    with open(os.path.join(MODEL_ROOT, "clip.py"), encoding="utf-8") as stream:
        tree = ast.parse(stream.read())
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
            assert name not in {"open", "load", "load_state_dict_from_url", "urlopen"}
    assert imports == {"math", "collections", "torch"}


@pytest.mark.parametrize("tower,option", (("vision_cfg", "patch_dropout"), ("text_cfg", "embed_cls")))
def test_clip_rejects_unused_configuration(tower, option):
    """Reject unsupported encoder branches before allocating default model parameters."""
    with pytest.raises(ValueError, match="Unsupported CLIP configuration"):
        if tower == "vision_cfg":
            CLIP(vision_cfg={option: True})
        else:
            CLIP(text_cfg={option: True})


def test_clip_regex_allowlist_scope(tmp_path, monkeypatch):
    """Reject top-level, nested, aliased, duplicate and mismatched tokenizer regex imports."""
    helpers = runpy.run_path(os.path.join(PROJECT_ROOT, "tests", "common", "test_dependency.py"))
    check = helpers["allowed_regex_import_lines"]
    monkeypatch.setitem(check.__globals__, "SOURCE_ROOT", str(tmp_path))
    relative, cls, method, module = helpers["REGEX_IMPORTS"][0]
    path = os.path.join(tmp_path, relative)
    os.makedirs(os.path.dirname(path))
    valid = f"class {cls}:\n    def {method}(self):\n        import {module}\n"
    invalid = [f"import {module}\n", valid.replace(cls, "OtherClass"),
               valid.replace(method, "other_method"), valid.replace("import regex", "import regex as re"),
               valid.replace("import regex", "from regex import compile"),
               valid.replace("import regex", "import regex, os"),
               valid + "        import regex\n",
               valid.replace("import regex", "def nested():\n            import regex")]
    for source in [valid] + invalid:
        with open(path, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(source)
        assert bool(check(path)) is (source == valid)
