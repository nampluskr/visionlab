# src/visionlab/anomaly_detection/common/pre_processor.py: validated batch normalization without resizing

import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PreProcessor:
    """Validate RGB float batches and optionally normalize without modifying inputs."""

    def __init__(self, mean=None, std=None):
        if (mean is None) != (std is None):
            raise ValueError("mean and std must both be provided or both be None")
        self.mean = None
        self.std = None
        if mean is not None:
            mean = torch.as_tensor(mean, dtype=torch.float64)
            std = torch.as_tensor(std, dtype=torch.float64)
            if mean.shape != (3,) or std.shape != (3,):
                raise ValueError("mean and std must contain three channel values")
            if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
                raise ValueError("mean and std must be finite and std must be positive")
            self.mean = mean.reshape(1, 3, 1, 1).clone()
            self.std = std.reshape(1, 3, 1, 1).clone()

    def __call__(self, image):
        if not isinstance(image, torch.Tensor) or not image.is_floating_point():
            raise ValueError("image must be a floating point tensor")
        if image.ndim != 4 or image.shape[1] != 3 or any(size == 0 for size in image.shape):
            raise ValueError("image must have nonempty shape [B, 3, H, W]")
        if not torch.isfinite(image).all():
            raise ValueError("image must contain finite values")
        if self.mean is None:
            return image
        return (image - self.mean.to(image)) / self.std.to(image)
