# tests/00_dataset/test_data.py: pytest contracts for prepared datasets and reproducible splits

import os
from dataclasses import asdict, is_dataclass

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from visionlab.anomaly_detection.common.batch import Batch, InferenceBatch
from visionlab.anomaly_detection.data.common.base_dataset import BaseDataset
from visionlab.anomaly_detection.data.common.split import TestSplitMode as SplitMode, ValSplitMode, random_split, split_datasets
from visionlab.anomaly_detection.data.mvtec.dataset import MVTecDataset
from visionlab.anomaly_detection.data.visa.dataset import VisADataset


def make_fixture(root, category, extension):
    """Write prepared images and paired anomaly masks in deliberately reversed creation order."""
    for split, label, count in (("train", "good", 8), ("test", "good", 10), ("test", "bad", 10)):
        folder = os.path.join(root, category, split, label)
        os.makedirs(folder, exist_ok=True)
        for index in reversed(range(count)):
            with Image.new("RGB", (19, 13), color=(64, 128, 255)) as image:
                image.save(os.path.join(folder, f"{index:03d}{extension}"))
            if label == "bad":
                mask_folder = os.path.join(root, category, "ground_truth", label)
                os.makedirs(mask_folder, exist_ok=True)
                with Image.new("L", (19, 13), color=0) as mask:
                    mask.paste(255, (0, 6, 19, 13))
                    mask.save(os.path.join(mask_folder, f"{index:03d}_mask.png"))


def image_paths(dataset):
    """Return image paths in dataset order."""
    return [sample["image_path"] for sample in dataset.samples]


@pytest.fixture
def prepared_data(tmp_path):
    """Prepare independent MVTec and VisA roots for each test."""
    mvtec, visa = os.path.join(tmp_path, "mvtec"), os.path.join(tmp_path, "visa")
    make_fixture(mvtec, category="bottle", extension=".png")
    make_fixture(os.path.join(visa, "visa_pytorch"), category="candle", extension=".JPG")
    return ((MVTecDataset, mvtec, "bottle"), (VisADataset, visa, "candle"))


def test_dataset_inheritance():
    """Keep both concrete datasets independent children of the common base."""
    assert issubclass(MVTecDataset, BaseDataset) and issubclass(VisADataset, BaseDataset)
    assert not issubclass(MVTecDataset, VisADataset) and not issubclass(VisADataset, MVTecDataset)


def test_item_contract(prepared_data):
    """Check FR-1 fields, sorting, binary masks and default collation for both datasets."""
    for dataset_class, root, category in prepared_data:
        for split in ("train", "test"):
            dataset = dataset_class(root=root, category=category, split=split)
            assert image_paths(dataset) == sorted(image_paths(dataset))
            for index in (0, len(dataset) - 1):
                image, target = dataset[index]
                assert image.shape == (3, 256, 256) and image.dtype == torch.float32
                assert 0 <= image.min() <= image.max() <= 1
                assert target["mask"].shape == (256, 256) and target["mask"].dtype == torch.int64
                assert target["label"] in (0, 1)
                assert set(torch.unique(target["mask"]).tolist()) <= {0, 1}
                assert os.path.isfile(target["image_path"])
                assert target["image_path"] == dataset.samples[index]["image_path"]
                assert target["sample_id"] == os.path.relpath(target["image_path"], dataset.root)
                if target["label"]:
                    assert os.path.isfile(target["mask_path"])
                    assert target["mask"][0, 0] == 0 and target["mask"][-1, -1] == 1
                else:
                    assert target["mask_path"] == "" and target["mask"].sum() == 0
            assert len({sample["sample_id"] for sample in dataset.samples}) == len(dataset)
            images, targets = next(iter(DataLoader(dataset, batch_size=2, num_workers=0)))
            assert images.shape == (2, 3, 256, 256) and targets["mask"].shape == (2, 256, 256)


def test_rgb_conversion_without_normalization(prepared_data):
    """Convert grayscale inputs to RGB without changing their numeric intensity."""
    root = prepared_data[0][1]
    with Image.new("L", (19, 13), color=128) as image:
        image.save(os.path.join(root, "bottle", "train", "good", "000.png"))
    image, _ = MVTecDataset(root=root, split="train")[0]
    torch.testing.assert_close(image, torch.full((3, 256, 256), 128 / 255))


def test_from_test_matches_seed_42_membership(prepared_data):
    """Preserve reference memberships, order, reproducibility and split disjointness."""
    for dataset_class, root, category in prepared_data:
        runs = [split_datasets(dataset_class(root=root, category=category, split="train"),
                               dataset_class(root=root, category=category, split="test"),
                               val_split_mode="from_test", val_split_ratio=0.4, seed=42) for _ in range(2)]
        for first, second in zip(*runs):
            assert image_paths(first) == image_paths(second) == sorted(image_paths(first))
        train, valid, test = runs[0]
        assert (len(train), len(valid), len(test)) == (8, 8, 12)
        assert not set(image_paths(valid)) & set(image_paths(test))
        assert not set(image_paths(train)) & set(image_paths(test))
        original = dataset_class(root=root, category=category, split="test")
        assert set(image_paths(valid)) | set(image_paths(test)) == set(image_paths(original))
        for label in (0, 1):
            names = [[os.path.splitext(os.path.basename(sample["image_path"]))[0]
                      for sample in subset.samples if sample["label_index"] == label] for subset in (valid, test)]
            assert names == [["000", "003", "007", "009"], ["001", "002", "004", "005", "006", "008"]]


def test_same_as_test_defaults_and_independent_metadata(prepared_data):
    """Share membership without sharing mutable sample metadata."""
    root = prepared_data[0][1]
    _, valid, result = split_datasets(MVTecDataset(root=root, split="train"), MVTecDataset(root=root, split="test"))
    assert image_paths(valid) == image_paths(result) and valid is not result
    valid.samples[0]["sample_id"] = "changed"
    assert valid.samples[0]["sample_id"] != result.samples[0]["sample_id"]


def test_no_normal_test_samples_uses_training_split(prepared_data):
    """Borrow normal training samples only when the test folder has none."""
    root = prepared_data[0][1]
    train, test = MVTecDataset(root=root, split="train"), MVTecDataset(root=root, split="test")
    test = test.subsample([index for index, sample in enumerate(test.samples) if sample["label_index"]])
    remaining, valid, test = split_datasets(train, test, seed=42)
    assert (len(remaining), len(test), len(train)) == (7, 11, 8)
    assert image_paths(valid) == image_paths(test)
    assert not set(image_paths(remaining)) & set(image_paths(test))


def test_seed_zero_uses_global_rng(prepared_data):
    """Match the reference seed-zero global generator behavior."""
    test = MVTecDataset(root=prepared_data[0][1], split="test")
    with torch.random.fork_rng():
        torch.manual_seed(123)
        expected = random_split(test, split_ratio=0.4, label_aware=True, seed=None)
        expected_state = torch.random.get_rng_state()
        torch.manual_seed(123)
        actual = random_split(test, split_ratio=0.4, label_aware=True, seed=0)
        assert [image_paths(item) for item in expected] == [image_paths(item) for item in actual]
        assert torch.equal(expected_state, torch.random.get_rng_state())


def test_invalid_modes_ratios_and_augmentations(prepared_data):
    """Reject unsupported modes, degenerate ratios, augmentation and invalid subsets."""
    root = prepared_data[0][1]
    train, test = MVTecDataset(root=root, split="train"), MVTecDataset(root=root, split="test")
    for mode in ("none", "synthetic", "unknown"):
        with pytest.raises(ValueError):
            split_datasets(train, test, test_split_mode=mode)
    for mode in ("none", "from_train", "synthetic", "unknown"):
        with pytest.raises(ValueError):
            split_datasets(train, test, val_split_mode=mode)
    for ratio in (-0.1, 0.0, 1.0, 1.1, float("nan")):
        with pytest.raises(ValueError):
            random_split(test, split_ratio=ratio)
    with pytest.raises(ValueError):
        random_split(test.subsample([0]), split_ratio=0.4)
    for dataset_class, root, category in prepared_data:
        with pytest.raises(ValueError):
            dataset_class(root=root, category=category, augmentations=object())
        with pytest.raises(ValueError):
            dataset_class(root=root, category=category, split="val")


def test_missing_local_assets_and_bad_masks(prepared_data, tmp_path):
    """Reject missing roots, mismatched masks and empty local categories."""
    for dataset_class, _, category in prepared_data:
        with pytest.raises(FileNotFoundError):
            dataset_class(root=os.path.join(tmp_path, "missing"), category=category)
    with pytest.raises(FileNotFoundError):
        VisADataset(root=os.path.join(prepared_data[1][1], "visa_pytorch"), category="candle")
    root = prepared_data[0][1]
    mask = os.path.join(root, "bottle", "ground_truth", "bad", "000_mask.png")
    renamed = os.path.join(os.path.dirname(mask), "zzz_mask.png")
    os.rename(mask, renamed)
    with pytest.raises(ValueError):
        MVTecDataset(root=root, split="test")
    os.remove(renamed)
    with pytest.raises(ValueError):
        MVTecDataset(root=root, split="test")
    empty = os.path.join(tmp_path, "empty")
    os.makedirs(os.path.join(empty, "bottle"))
    with pytest.raises(RuntimeError):
        MVTecDataset(root=empty)


def test_dataclasses_and_enum_names():
    """Keep the shared batch defaults and public split enum names."""
    batch = Batch(image=torch.zeros(1, 3, 2, 2), sample_id=["sample"])
    prediction = InferenceBatch(pred_score=torch.ones(1), anomaly_map=torch.zeros(1, 1, 2, 2))
    assert is_dataclass(batch) and is_dataclass(prediction)
    assert asdict(batch)["sample_id"] == ["sample"]
    assert batch.gt_mask is None and prediction.pred_label is None
    assert SplitMode.FROM_DIR.value == "from_dir" and ValSplitMode.FROM_TEST.value == "from_test"
