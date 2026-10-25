# src/visionlab/anomaly_detection/data/mvtec/dataset.py: local MVTec AD dataset defaults

from visionlab.anomaly_detection.data.common.base_dataset import BaseDataset


class MVTecDataset(BaseDataset):
    """Read local MVTec AD images as resized tensors and target dictionaries."""

    extensions = (".png", ".PNG")

    def __init__(self, root="./datasets/MVTecAD", category="bottle", augmentations=None, split=None):
        super().__init__(root=root, category=category, augmentations=augmentations, split=split)
