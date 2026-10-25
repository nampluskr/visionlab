# tests/00_dataset/test_data.py: standalone tests for pure datasets and reproducible splits

import importlib.abc
import os
import sys
import tempfile
import unittest
from dataclasses import asdict, is_dataclass


class BlockFrameworkImports(importlib.abc.MetaPathFinder):
    """Reject imports of the reference and training frameworks during these tests."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("anomalib", "lightning", "pytorch_lightning"):
            raise ModuleNotFoundError(f"Forbidden framework import: {fullname}")
        return None


sys.meta_path.insert(0, BlockFrameworkImports())
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import torch
from PIL import Image
from torch.utils.data import DataLoader

from visionlab.anomaly_detection.common.batch import Batch, InferenceBatch
from visionlab.anomaly_detection.data.common.base_dataset import BaseDataset
from visionlab.anomaly_detection.data.common.split import (
    TestSplitMode,
    ValSplitMode,
    random_split,
    split_datasets,
)
from visionlab.anomaly_detection.data.mvtec.dataset import MVTecDataset
from visionlab.anomaly_detection.data.visa.dataset import VisADataset


def make_fixture(root, category, extension):
    """Write a small prepared dataset with normal images and binary anomaly masks."""
    for split, label, count in (("train", "good", 8), ("test", "good", 10), ("test", "bad", 10)):
        folder = os.path.join(root, category, split, label)
        os.makedirs(folder, exist_ok=True)
        for index in reversed(range(count)):
            filename = f"{index:03d}{extension}"
            with Image.new("RGB", (19, 13), color=(64, 128, 255)) as image:
                image.save(os.path.join(folder, filename))
            if label == "bad":
                mask_folder = os.path.join(root, category, "ground_truth", label)
                os.makedirs(mask_folder, exist_ok=True)
                with Image.new("L", (19, 13), color=0) as mask:
                    mask.paste(255, (0, 6, 19, 13))
                    mask.save(os.path.join(mask_folder, f"{index:03d}_mask.png"))


def image_paths(dataset):
    """Return the image paths in dataset order."""
    return [sample["image_path"] for sample in dataset.samples]


class DatasetTests(unittest.TestCase):
    """Check FR-1, FR-3 and local-only dependency behavior using temporary files."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = self.temporary.name
        self.mvtec_root = os.path.join(self.root, "mvtec")
        self.visa_root = os.path.join(self.root, "visa")
        make_fixture(self.mvtec_root, category="bottle", extension=".png")
        make_fixture(os.path.join(self.visa_root, "visa_pytorch"), category="candle", extension=".JPG")
        self.cases = (
            (MVTecDataset, self.mvtec_root, "bottle"),
            (VisADataset, self.visa_root, "candle"),
        )

    def test_dataset_inheritance(self):
        self.assertTrue(issubclass(MVTecDataset, BaseDataset))
        self.assertTrue(issubclass(VisADataset, BaseDataset))
        self.assertFalse(issubclass(MVTecDataset, VisADataset))
        self.assertFalse(issubclass(VisADataset, MVTecDataset))

    def test_item_contract(self):
        for dataset_class, root, category in self.cases:
            for split in ("train", "test"):
                with self.subTest(dataset=dataset_class.__name__, split=split):
                    dataset = dataset_class(root=root, category=category, split=split)
                    self.assertEqual(image_paths(dataset), sorted(image_paths(dataset)))
                    for index in (0, len(dataset) - 1):
                        image, target = dataset[index]
                        self.assertEqual(tuple(image.shape), (3, 256, 256))
                        self.assertEqual(image.dtype, torch.float32)
                        self.assertGreaterEqual(image.min().item(), 0)
                        self.assertLessEqual(image.max().item(), 1)
                        self.assertEqual(tuple(target["mask"].shape), (256, 256))
                        self.assertEqual(target["mask"].dtype, torch.int64)
                        self.assertIn(target["label"], (0, 1))
                        self.assertTrue(set(torch.unique(target["mask"]).tolist()).issubset({0, 1}))
                        self.assertTrue(os.path.isfile(target["image_path"]))
                        self.assertEqual(target["image_path"], dataset.samples[index]["image_path"])
                        self.assertEqual(target["sample_id"], os.path.relpath(target["image_path"], dataset.root))
                        if target["label"]:
                            self.assertTrue(os.path.isfile(target["mask_path"]))
                            self.assertEqual(target["mask"][0, 0].item(), 0)
                            self.assertEqual(target["mask"][-1, -1].item(), 1)
                        else:
                            self.assertEqual(target["mask_path"], "")
                            self.assertEqual(target["mask"].sum().item(), 0)
                    self.assertEqual(len({sample["sample_id"] for sample in dataset.samples}), len(dataset))
                    image_batch, target_batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=0)))
                    self.assertEqual(tuple(image_batch.shape), (2, 3, 256, 256))
                    self.assertEqual(tuple(target_batch["mask"].shape), (2, 256, 256))

    def test_rgb_conversion_without_normalization(self):
        image_path = os.path.join(self.mvtec_root, "bottle", "train", "good", "000.png")
        with Image.new("L", (19, 13), color=128) as image:
            image.save(image_path)
        dataset = MVTecDataset(root=self.mvtec_root, category="bottle", split="train")
        image, _ = dataset[0]
        torch.testing.assert_close(image, torch.full((3, 256, 256), 128 / 255))

    def test_from_test_matches_seed_42_membership(self):
        for dataset_class, root, category in self.cases:
            with self.subTest(dataset=dataset_class.__name__):
                runs = []
                for _ in range(2):
                    train = dataset_class(root=root, category=category, split="train")
                    test = dataset_class(root=root, category=category, split="test")
                    runs.append(split_datasets(train, test, val_split_mode="from_test", val_split_ratio=0.4, seed=42))
                for first, second in zip(runs[0], runs[1]):
                    self.assertEqual(image_paths(first), image_paths(second))
                    self.assertEqual(image_paths(first), sorted(image_paths(first)))
                train, valid, test = runs[0]
                self.assertEqual(len(train), 8)
                self.assertEqual(len(valid), 8)
                self.assertEqual(len(test), 12)
                self.assertFalse(set(image_paths(valid)) & set(image_paths(test)))
                self.assertFalse(set(image_paths(train)) & set(image_paths(test)))
                original = dataset_class(root=root, category=category, split="test")
                self.assertEqual(set(image_paths(valid)) | set(image_paths(test)), set(image_paths(original)))
                for label in (0, 1):
                    valid_names = [os.path.splitext(os.path.basename(sample["image_path"]))[0]
                                   for sample in valid.samples if sample["label_index"] == label]
                    test_names = [os.path.splitext(os.path.basename(sample["image_path"]))[0]
                                  for sample in test.samples if sample["label_index"] == label]
                    self.assertEqual(valid_names, ["000", "003", "007", "009"])
                    self.assertEqual(test_names, ["001", "002", "004", "005", "006", "008"])

    def test_same_as_test_defaults_and_independent_metadata(self):
        train = MVTecDataset(root=self.mvtec_root, split="train")
        test = MVTecDataset(root=self.mvtec_root, split="test")
        _, valid, result = split_datasets(train, test)
        self.assertEqual(image_paths(valid), image_paths(result))
        self.assertIsNot(valid, result)
        valid.samples[0]["sample_id"] = "changed"
        self.assertNotEqual(valid.samples[0]["sample_id"], result.samples[0]["sample_id"])

    def test_no_normal_test_samples_uses_training_split(self):
        train = MVTecDataset(root=self.mvtec_root, split="train")
        test = MVTecDataset(root=self.mvtec_root, split="test")
        test = test.subsample([index for index, sample in enumerate(test.samples) if sample["label_index"]])
        remaining, valid, test = split_datasets(train, test, seed=42)
        self.assertEqual(len(remaining), 7)
        self.assertEqual(len(test), 11)
        self.assertEqual(image_paths(valid), image_paths(test))
        self.assertFalse(set(image_paths(remaining)) & set(image_paths(test)))
        self.assertEqual(len(train), 8)

    def test_seed_zero_uses_global_rng(self):
        test = MVTecDataset(root=self.mvtec_root, split="test")
        state = torch.random.get_rng_state()
        self.addCleanup(torch.random.set_rng_state, state)
        torch.manual_seed(123)
        expected = random_split(test, split_ratio=0.4, label_aware=True, seed=None)
        expected_state = torch.random.get_rng_state()
        torch.manual_seed(123)
        actual = random_split(test, split_ratio=0.4, label_aware=True, seed=0)
        for left, right in zip(expected, actual):
            self.assertEqual(image_paths(left), image_paths(right))
        self.assertTrue(torch.equal(expected_state, torch.random.get_rng_state()))

    def test_invalid_modes_ratios_and_augmentations(self):
        train = MVTecDataset(root=self.mvtec_root, split="train")
        test = MVTecDataset(root=self.mvtec_root, split="test")
        for mode in ("none", "synthetic", "unknown"):
            with self.subTest(test_mode=mode), self.assertRaises(ValueError):
                split_datasets(train, test, test_split_mode=mode)
        for mode in ("none", "from_train", "synthetic", "unknown"):
            with self.subTest(val_mode=mode), self.assertRaises(ValueError):
                split_datasets(train, test, val_split_mode=mode)
        for ratio in (-0.1, 0.0, 1.0, 1.1, float("nan")):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                random_split(test, split_ratio=ratio)
        with self.assertRaises(ValueError):
            random_split(test.subsample([0]), split_ratio=0.4)
        for dataset_class, root, category in self.cases:
            with self.subTest(dataset=dataset_class.__name__), self.assertRaises(ValueError):
                dataset_class(root=root, category=category, augmentations=object())
            with self.assertRaises(ValueError):
                dataset_class(root=root, category=category, split="val")

    def test_missing_local_assets_and_bad_masks(self):
        for dataset_class, _, category in self.cases:
            with self.subTest(dataset=dataset_class.__name__), self.assertRaises(FileNotFoundError):
                dataset_class(root=os.path.join(self.root, "missing"), category=category)
        with self.assertRaises(FileNotFoundError):
            VisADataset(root=os.path.join(self.visa_root, "visa_pytorch"), category="candle")
        mask_path = os.path.join(self.mvtec_root, "bottle", "ground_truth", "bad", "000_mask.png")
        renamed = os.path.join(os.path.dirname(mask_path), "zzz_mask.png")
        os.rename(mask_path, renamed)
        with self.assertRaises(ValueError):
            MVTecDataset(root=self.mvtec_root, split="test")
        os.remove(renamed)
        with self.assertRaises(ValueError):
            MVTecDataset(root=self.mvtec_root, split="test")
        empty_root = os.path.join(self.root, "empty")
        os.makedirs(os.path.join(empty_root, "bottle"))
        with self.assertRaises(RuntimeError):
            MVTecDataset(root=empty_root)

    def test_dataclasses_and_enum_names(self):
        batch = Batch(image=torch.zeros(1, 3, 2, 2), sample_id=["sample"])
        prediction = InferenceBatch(pred_score=torch.ones(1), anomaly_map=torch.zeros(1, 1, 2, 2))
        self.assertTrue(is_dataclass(batch))
        self.assertTrue(is_dataclass(prediction))
        self.assertEqual(asdict(batch)["sample_id"], ["sample"])
        self.assertIsNone(batch.gt_mask)
        self.assertIsNone(prediction.pred_label)
        self.assertEqual(TestSplitMode.FROM_DIR.value, "from_dir")
        self.assertEqual(ValSplitMode.FROM_TEST.value, "from_test")
        self.assertFalse({"anomalib", "lightning", "pytorch_lightning"} & set(sys.modules))


if __name__ == "__main__":
    unittest.main(verbosity=2)
