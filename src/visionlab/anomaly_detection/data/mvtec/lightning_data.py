# src/visionlab/anomaly_detection/data/mvtec/lightning_data.py: native Lightning data module for local MVTec AD

import os

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from visionlab.anomaly_detection.data.common.split import TestSplitMode, ValSplitMode, split_datasets
from visionlab.anomaly_detection.data.mvtec.dataset import MVTecDataset


class MVTecLitData(LightningDataModule):
    """Connect the pure MVTec dataset and supported splits to native Lightning."""

    dataset_class = MVTecDataset

    def __init__(
        self,
        root="./datasets/MVTecAD",
        category="bottle",
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
        super().__init__()
        if any(value is not None for value in (
            train_augmentations, val_augmentations, test_augmentations, augmentations,
        )):
            raise ValueError("Augmentations are not supported in Phase 0")
        self.root = os.path.abspath(os.fspath(root))
        self.category = category
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.test_split_mode = TestSplitMode(test_split_mode)
        self.test_split_ratio = test_split_ratio
        self.val_split_mode = ValSplitMode(val_split_mode)
        self.val_split_ratio = val_split_ratio
        self.seed = seed
        self._is_setup = False

    @property
    def name(self):
        return self.__class__.__name__

    def prepare_data(self):
        for folder in (os.path.join("train", "good"), "test", "ground_truth"):
            path = os.path.join(self.root, self.category, folder)
            if not os.path.isdir(path):
                raise FileNotFoundError(path)

    def setup(self, stage=None):
        if self._is_setup:
            return
        train_data = self.dataset_class(root=self.root, category=self.category, split="train")
        test_data = self.dataset_class(root=self.root, category=self.category, split="test")
        self.train_data, self.val_data, self.test_data = split_datasets(
            train_data=train_data,
            test_data=test_data,
            test_split_mode=self.test_split_mode,
            test_split_ratio=self.test_split_ratio,
            val_split_mode=self.val_split_mode,
            val_split_ratio=self.val_split_ratio,
            seed=self.seed,
        )
        self._is_setup = True

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data, batch_size=self.train_batch_size, num_workers=self.num_workers, shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_data, batch_size=self.eval_batch_size, num_workers=self.num_workers, shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data, batch_size=self.eval_batch_size, num_workers=self.num_workers, shuffle=False,
        )

    def predict_dataloader(self):
        return self.test_dataloader()
