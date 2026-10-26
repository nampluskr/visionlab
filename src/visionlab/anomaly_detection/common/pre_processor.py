# src/visionlab/anomaly_detection/common/pre_processor.py: validated normalization and optional aligned center crops

import torch
from torchvision.transforms.v2.functional import center_crop

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PreProcessor:
    """Validate RGB float batches and optionally crop and normalize without modifying inputs."""

    def __init__(self, mean=None, std=None, center_crop=None):
        if (mean is None) != (std is None):
            raise ValueError("mean and std must both be provided or both be None")
        self.mean = None
        self.std = None
        if center_crop is not None:
            if not isinstance(center_crop, (tuple, list)) or len(center_crop) != 2 or \
                    any(type(size) is not int or size <= 0 for size in center_crop):
                raise ValueError("center_crop must contain two positive integer sizes")
            center_crop = tuple(center_crop)
        self.center_crop = center_crop
        if mean is not None:
            mean = torch.as_tensor(mean, dtype=torch.float64)
            std = torch.as_tensor(std, dtype=torch.float64)
            if mean.shape != (3,) or std.shape != (3,):
                raise ValueError("mean and std must contain three channel values")
            if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
                raise ValueError("mean and std must be finite and std must be positive")
            self.mean = mean.reshape(1, 3, 1, 1).clone()
            self.std = std.reshape(1, 3, 1, 1).clone()

    def crop(self, tensor):
        if self.center_crop is None:
            return tensor
        if not isinstance(tensor, torch.Tensor) or tensor.ndim < 2:
            raise ValueError("crop input must be a tensor with spatial dimensions")
        if any(size > extent for size, extent in zip(self.center_crop, tensor.shape[-2:], strict=True)):
            raise ValueError("center_crop cannot exceed input spatial dimensions")
        return center_crop(tensor, output_size=list(self.center_crop))

    def __call__(self, image):
        if not isinstance(image, torch.Tensor) or not image.is_floating_point():
            raise ValueError("image must be a floating point tensor")
        if image.ndim != 4 or image.shape[1] != 3 or any(size == 0 for size in image.shape):
            raise ValueError("image must have nonempty shape [B, 3, H, W]")
        if not torch.isfinite(image).all():
            raise ValueError("image must contain finite values")
        image = self.crop(image)
        if self.mean is None:
            return image
        return (image - self.mean.to(image)) / self.std.to(image)
