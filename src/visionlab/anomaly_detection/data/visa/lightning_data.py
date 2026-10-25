# src/visionlab/anomaly_detection/data/visa/lightning_data.py: native Lightning data module for prepared local VisA

import os

from visionlab.anomaly_detection.data.common.split import TestSplitMode, ValSplitMode
from visionlab.anomaly_detection.data.mvtec.lightning_data import MVTecLitData
from visionlab.anomaly_detection.data.visa.dataset import VisADataset


class VisALitData(MVTecLitData):
    """Connect prepared VisA data to Lightning without copying or downloading files."""

    dataset_class = VisADataset

    def __init__(
        self,
        root="./datasets/visa",
        category="capsules",
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=8,
        train_augmentations=None,
        val_augmentations=None,
        test_augmentations=None,
        augmentations=None,
        test_split_mode=TestSplitMode.FROM_DIR,
        test_split_ratio=0.2,
        val_split_mode=ValSplitMode.SAME_AS_TEST,
        val_split_ratio=0.5,
        seed=None,
    ):
        super().__init__(
            root=root,
            category=category,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            train_augmentations=train_augmentations,
            val_augmentations=val_augmentations,
            test_augmentations=test_augmentations,
            augmentations=augmentations,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )
        self.split_root = os.path.join(self.root, "visa_pytorch")

    def prepare_data(self):
        path = os.path.join(self.split_root, self.category)
        if not os.path.isdir(path):
            raise FileNotFoundError(path)
