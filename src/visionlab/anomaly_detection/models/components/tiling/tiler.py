# src/visionlab/anomaly_detection/models/components/tiling/tiler.py: image tiling and scaled feature reconstruction

from collections.abc import Sequence
from enum import Enum
from itertools import product
from math import ceil

import torch
from torch.nn import functional
from torchvision.transforms import RandomCrop


class ImageUpscaleMode(str, Enum):
    """Select zero padding or nearest interpolation for image resizing."""

    PADDING = "padding"
    INTERPOLATION = "interpolation"


class StrideSizeError(Exception):
    """Report a stride that leaves gaps between tiles."""


class Tiler:
    """Split images into tiles and reconstruct images or scaled feature maps."""

    def __init__(self, tile_size, stride=None, remove_border_count=0, mode=ImageUpscaleMode.PADDING):
        self.tile_size_h, self.tile_size_w = self.validate_size_type(tile_size)
        self.stride_h, self.stride_w = self.validate_size_type(tile_size if stride is None else stride)
        self.remove_border_count = remove_border_count
        self.random_tile_count = 4
        self.overlapping = (self.stride_h, self.stride_w) != (self.tile_size_h, self.tile_size_w)
        self.mode = mode
        if self.stride_h > self.tile_size_h or self.stride_w > self.tile_size_w:
            raise StrideSizeError("Stride size larger than tile size produces unreliable results. Ensure stride size <= tile size.")
        if mode not in (ImageUpscaleMode.PADDING, ImageUpscaleMode.INTERPOLATION):
            raise ValueError(f"Unknown mode {mode}. Available modes: padding and interpolation")

    @staticmethod
    def validate_size_type(parameter):
        if isinstance(parameter, int):
            return parameter, parameter
        if isinstance(parameter, Sequence):
            return parameter[0], parameter[1]
        raise TypeError(f"Invalid type {type(parameter)} for tile/stride size. Must be int or Sequence.")

    def tile(self, image, use_random_tiling=False):
        if image.dim() == 3:
            image = image.unsqueeze(0)
        self.batch_size, self.num_channels, self.input_h, self.input_w = image.shape
        if self.input_h < self.tile_size_h or self.input_w < self.tile_size_w:
            raise ValueError(f"Tile size {self.tile_size_h, self.tile_size_w} exceeds image size {self.input_h, self.input_w}")
        self.resized_h = ceil((self.input_h - self.tile_size_h) / self.stride_h) * self.stride_h + self.tile_size_h
        self.resized_w = ceil((self.input_w - self.tile_size_w) / self.stride_w) * self.stride_w + self.tile_size_w
        if self.mode == ImageUpscaleMode.PADDING:
            image = functional.pad(image, [0, self.resized_w - self.input_w, 0, self.resized_h - self.input_h])
        else:
            image = functional.interpolate(image, size=(self.resized_h, self.resized_w))
        if use_random_tiling:
            return torch.vstack([RandomCrop(self.tile_size_h)(image) for _ in range(self.random_tile_count)])
        self.num_patches_h = (self.resized_h - self.tile_size_h) // self.stride_h + 1
        self.num_patches_w = (self.resized_w - self.tile_size_w) // self.stride_w + 1
        patches = image.unfold(2, self.tile_size_h, self.stride_h).unfold(3, self.tile_size_w, self.stride_w)
        # The reference allocates tiles with the default floating dtype.
        return patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(
            -1, self.num_channels, self.tile_size_h, self.tile_size_w,
        ).to(dtype=torch.get_default_dtype(), copy=True)

    def untile(self, tiles):
        _, channels, height, width = tiles.shape
        scale_h = height / self.tile_size_h
        scale_w = width / self.tile_size_w
        border = self.remove_border_count
        inner_h, inner_w = height - 2 * border, width - 2 * border
        output_h, output_w = int(self.resized_h * scale_h), int(self.resized_w * scale_w)
        patches = tiles.contiguous().view(
            self.batch_size, self.num_patches_h, self.num_patches_w, channels, height, width,
        ).permute(1, 2, 0, 3, 4, 5).reshape(-1, self.batch_size, channels, height, width)
        patches = patches[:, :, :, border:border + inner_h, border:border + inner_w]
        output = torch.zeros((self.batch_size, channels, output_h, output_w), device=tiles.device)
        coverage = torch.zeros_like(output)
        coordinates = product(
            range(border, output_h - inner_h + 1, int(self.stride_h * scale_h)),
            range(border, output_w - inner_w + 1, int(self.stride_w * scale_w)),
        )
        for patch, (top, left) in zip(patches, coordinates, strict=True):
            output[:, :, top:top + inner_h, left:left + inner_w] += patch
            coverage[:, :, top:top + inner_h, left:left + inner_w] += 1
        output = output / coverage
        output[output != output] = 0
        if self.mode == ImageUpscaleMode.PADDING:
            return output[:, :, :self.input_h, :self.input_w]
        return functional.interpolate(output, size=(self.input_h, self.input_w))
