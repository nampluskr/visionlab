# src/visionlab/anomaly_detection/models/winclip/clip.py: Independent fp32 WinCLIP encoders
# Copyright (c) 2012-2021 Gabriel Ilharco, Mitchell Wortsman, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, John Miller, Hongseok Namkoong, Hannaneh Hajishirzi, Ali Farhadi, Ludwig Schmidt
# derived from open_clip (MIT license)

import math
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm(nn.LayerNorm):
    """Apply the open_clip fp32 layer normalization path."""

    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps).to(x.dtype)


class ResidualAttentionBlock(nn.Module):
    """Apply self attention and a GELU MLP with residual connections."""

    def __init__(self, d_model, n_head, mlp_ratio=4.0):
        super().__init__()
        self.ln_1 = LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ls_1 = nn.Identity()
        self.ln_2 = LayerNorm(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(mlp_width, d_model)),
        ]))
        self.ls_2 = nn.Identity()

    def attention(self, q_x, attn_mask=None):
        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(q_x, q_x, q_x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, q_x, attn_mask=None):
        x = q_x + self.ls_1(self.attention(self.ln_1(q_x), attn_mask=attn_mask))
        return x + self.ls_2(self.mlp(self.ln_2(x)))


class Transformer(nn.Module):
    """Transform sequence-first tokens using the WinCLIP residual blocks."""

    def __init__(self, width, layers, heads, mlp_ratio=4.0):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads, mlp_ratio) for _ in range(layers)
        ])

    def forward(self, x, attn_mask=None):
        for block in self.resblocks:
            x = block(x, attn_mask=attn_mask)
        return x


class VisionTransformer(nn.Module):
    """Encode images and expose pre-transformer tokens for WinCLIP windows."""

    def __init__(self, image_size, patch_size, width, layers, heads, mlp_ratio=4.0,
                 output_dim=640, output_tokens=False):
        super().__init__()
        self.output_tokens = output_tokens
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size)
        self.grid_size = tuple(size // patch for size, patch in zip(self.image_size, self.patch_size))
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width)
        )
        self.patch_dropout = nn.Identity()
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads, mlp_ratio)
        self.attn_pool = None
        self.pool_type = "tok"
        self.final_ln_after_pool = False
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def _global_pool(self, x):
        return x[:, 0], x[:, 1:]

    def forward(self, x):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        class_token = self.class_embedding.view(1, 1, -1).expand(x.shape[0], -1, -1).to(x.dtype)
        x = torch.cat([class_token, x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.patch_dropout(x)
        x = self.ln_pre(x)
        x = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        pooled, tokens = self._global_pool(self.ln_post(x))
        pooled = pooled @ self.proj
        return (pooled, tokens) if self.output_tokens else pooled


class CLIP(nn.Module):
    """Provide the ViT-B-16-plus-240 encoders without loading external weights."""

    def __init__(self, embed_dim=640, vision_cfg=None, text_cfg=None):
        super().__init__()
        vision = dict(image_size=240, patch_size=16, width=896, layers=12,
                      head_width=64, mlp_ratio=4.0, output_tokens=False)
        text = dict(context_length=77, vocab_size=49408, width=640, heads=10,
                    layers=12, mlp_ratio=4.0)
        for config, supplied in ((vision, vision_cfg), (text, text_cfg)):
            if supplied is not None:
                unknown = set(supplied) - set(config)
                if unknown:
                    raise ValueError(f"Unsupported CLIP configuration: {sorted(unknown)}")
                config.update(supplied)
        self.visual = VisionTransformer(
            image_size=vision["image_size"], patch_size=vision["patch_size"],
            width=vision["width"], layers=vision["layers"],
            heads=vision["width"] // vision["head_width"], mlp_ratio=vision["mlp_ratio"],
            output_dim=embed_dim, output_tokens=vision["output_tokens"],
        )
        self.context_length = text["context_length"]
        self.vocab_size = text["vocab_size"]
        width = text["width"]
        self.token_embedding = nn.Embedding(self.vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, width))
        self.transformer = Transformer(width, text["layers"], text["heads"], text["mlp_ratio"])
        self.ln_final = LayerNorm(width)
        self.text_pool_type = "argmax"
        mask = torch.empty(self.context_length, self.context_length).fill_(float("-inf")).triu_(1)
        self.register_buffer("attn_mask", mask, persistent=False)
        self.text_projection = nn.Parameter(torch.empty(width, embed_dim))
        self.init_parameters()
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        width = self.transformer.width
        proj_std = (width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=width ** -0.5)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=(2 * width) ** -0.5)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        nn.init.normal_(self.text_projection, std=width ** -0.5)

    def encode_image(self, image, normalize=False):
        features = self.visual(image)
        if self.visual.output_tokens:
            pooled, tokens = features
            return (F.normalize(pooled, dim=-1) if normalize else pooled), tokens
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize=False):
        x = self.token_embedding(text) + self.positional_embedding
        x = self.transformer(x.permute(1, 0, 2), attn_mask=self.attn_mask).permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x
