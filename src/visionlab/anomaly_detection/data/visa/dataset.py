# src/visionlab/anomaly_detection/data/visa/dataset.py: local VisA images from a prepared visa_pytorch directory

import os

from visionlab.anomaly_detection.data.common.base_dataset import BaseDataset


class VisADataset(BaseDataset):
    """Read prepared VisA data without creating or changing the dataset copy."""

    extensions = (".png", ".jpg", ".JPG")

    def __init__(self, root, category, augmentations=None, split=None):
        super().__init__(
            root=os.path.join(root, "visa_pytorch"), category=category, augmentations=augmentations, split=split,
        )
