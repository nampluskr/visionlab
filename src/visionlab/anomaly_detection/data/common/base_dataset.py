# src/visionlab/anomaly_detection/data/common/base_dataset.py: shared local image dataset loading and sample operations

import copy
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional


def make_samples(root_category, split, extensions):
    """Read and pair sorted images and masks in the MVTec directory layout."""
    if not os.path.isdir(root_category):
        raise FileNotFoundError(root_category)
    if split not in (None, "train", "test"):
        raise ValueError(f"Unsupported dataset split: {split}")
    images = []
    masks = []
    for folder, _, filenames in os.walk(root_category):
        for filename in filenames:
            if os.path.splitext(filename)[1] not in extensions:
                continue
            path = os.path.join(folder, filename)
            parts = os.path.relpath(path, root_category).split(os.sep)
            if len(parts) != 3 or parts[0] not in ("train", "test", "ground_truth"):
                continue
            if parts[0] == "ground_truth":
                masks.append(path)
            else:
                images.append({
                    "image_path": path,
                    "mask_path": "",
                    "split": parts[0],
                    "label": parts[1],
                    "label_index": int(parts[1] != "good"),
                })
    images.sort(key=lambda sample: sample["image_path"])
    masks.sort()
    abnormal = [sample for sample in images if sample["split"] == "test" and sample["label_index"] == 1]
    if len(abnormal) != len(masks):
        raise ValueError("Anomalous test images and ground truth masks have different counts")
    for sample, mask_path in zip(abnormal, masks):
        image_stem = os.path.splitext(os.path.basename(sample["image_path"]))[0]
        mask_stem = os.path.splitext(os.path.basename(mask_path))[0]
        if image_stem not in mask_stem:
            raise ValueError(f"Image and mask names do not match: {sample['image_path']}, {mask_path}")
        sample["mask_path"] = mask_path
    for sample in images:
        if sample["label_index"] and not sample["mask_path"]:
            raise ValueError(f"Missing mask for anomalous image: {sample['image_path']}")
        sample["sample_id"] = os.path.relpath(sample["image_path"], os.path.dirname(root_category))
    samples = [sample for sample in images if split is None or sample["split"] == split]
    if not samples:
        raise RuntimeError(f"Found 0 images for split {split} in {root_category}")
    return samples


class BaseDataset(Dataset):
    """Read local anomaly images as resized tensors and target dictionaries."""

    def __init__(self, root, category, augmentations=None, split=None):
        if augmentations is not None:
            raise ValueError("Augmentations are not supported in Phase 0")
        self.root = os.path.abspath(os.fspath(root))
        self.root_category = os.path.join(self.root, category)
        self.category = category
        self.split = split
        self.samples = make_samples(self.root_category, split=split, extensions=self.extensions)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(sample["image_path"]) as source:
            image = functional.to_dtype(functional.to_image(source.convert("RGB")), torch.float32, scale=True)
        if sample["label_index"]:
            with Image.open(sample["mask_path"]) as source:
                mask = functional.to_image(source.convert("L")).to(torch.float32).div(255).to(torch.uint8)
            if mask.shape[-2:] != image.shape[-2:]:
                raise ValueError(f"Image and mask dimensions differ: {sample['image_path']}")
        else:
            mask = torch.zeros((1, *image.shape[-2:]), dtype=torch.uint8)
        image = functional.resize(image, size=[256, 256], interpolation=InterpolationMode.BILINEAR, antialias=True)
        mask = functional.resize(mask, size=[256, 256], interpolation=InterpolationMode.NEAREST).squeeze(0)
        target = {
            "label": sample["label_index"],
            "mask": mask.to(torch.int64),
            "sample_id": sample["sample_id"],
            "image_path": sample["image_path"],
            "mask_path": sample["mask_path"],
        }
        return image, target

    def subsample(self, indices):
        indices = [int(index) for index in indices]
        if len(set(indices)) != len(indices):
            raise ValueError("No duplicates allowed in indices")
        subset = copy.copy(self)
        subset.samples = sorted(
            [self.samples[index].copy() for index in indices], key=lambda sample: sample["image_path"],
        )
        return subset

    def __add__(self, other):
        if type(other) is not type(self):
            raise TypeError("Cannot concatenate datasets of different types")
        combined = copy.copy(self)
        combined.samples = sorted(
            [sample.copy() for sample in self.samples + other.samples], key=lambda sample: sample["image_path"],
        )
        return combined
