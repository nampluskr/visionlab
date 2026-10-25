# src/visionlab/anomaly_detection/data/common/split.py: reproducible anomaly dataset splitting

import math
import warnings
from enum import Enum

import torch


class TestSplitMode(str, Enum):
    """Name the test split modes used by anomalib 2.0.0."""

    NONE = "none"
    FROM_DIR = "from_dir"
    SYNTHETIC = "synthetic"


class ValSplitMode(str, Enum):
    """Name the validation split modes used by anomalib 2.0.0."""

    NONE = "none"
    SAME_AS_TEST = "same_as_test"
    FROM_TRAIN = "from_train"
    FROM_TEST = "from_test"
    SYNTHETIC = "synthetic"


def random_split(dataset, split_ratio, label_aware=False, seed=None):
    """Split sorted samples using anomalib 2.0.0 rounding and generator semantics."""
    ratios = [1 - split_ratio, split_ratio] if isinstance(split_ratio, float) else list(split_ratio)
    if not (math.isclose(sum(ratios), 1) and sum(ratios) <= 1):
        raise ValueError("Split ratios must sum to 1")
    if not all(0 < ratio < 1 for ratio in ratios):
        raise ValueError("Split ratios must be between 0 and 1")

    groups = [dataset]
    if label_aware:
        labels = sorted({sample["label_index"] for sample in dataset.samples})
        groups = [dataset.subsample([
            index for index, sample in enumerate(dataset.samples) if sample["label_index"] == label
        ]) for label in labels]

    samples_per_subset = [[] for _ in ratios]
    for group in groups:
        lengths = [math.floor(len(group) * ratio) for ratio in ratios]
        if len(group) and not sum(lengths):
            raise ValueError("Too few samples for the requested split ratios")
        for index in range(len(group) - sum(lengths)):
            lengths[index % sum(lengths)] += 1
        if 0 in lengths:
            warnings.warn("Zero subset length encountered during splitting", UserWarning)
        # A fresh generator per label, and seed=0 using global RNG, match the reference.
        generator = torch.Generator().manual_seed(seed) if seed else None
        indices = torch.randperm(len(group), generator=generator)
        for subset_index, selected in enumerate(torch.split(indices, lengths)):
            samples_per_subset[subset_index].extend(group.samples[index] for index in selected.tolist())

    index_by_path = {sample["image_path"]: index for index, sample in enumerate(dataset.samples)}
    return [
        dataset.subsample([index_by_path[sample["image_path"]] for sample in samples])
        for samples in samples_per_subset
    ]


def split_by_label(dataset):
    """Return normal and anomalous datasets in image path order."""
    return tuple(dataset.subsample([
        index for index, sample in enumerate(dataset.samples) if sample["label_index"] == label
    ]) for label in (0, 1))


def split_datasets(
    train_data,
    test_data,
    test_split_mode=TestSplitMode.FROM_DIR,
    test_split_ratio=0.2,
    val_split_mode=ValSplitMode.SAME_AS_TEST,
    val_split_ratio=0.5,
    seed=None,
):
    """Create train, validation and test datasets for the supported Phase 0 modes."""
    test_mode = TestSplitMode(test_split_mode)
    val_mode = ValSplitMode(val_split_mode)
    if test_mode != TestSplitMode.FROM_DIR:
        raise ValueError(f"Unsupported test split mode: {test_mode.value}")
    if val_mode not in (ValSplitMode.SAME_AS_TEST, ValSplitMode.FROM_TEST):
        raise ValueError(f"Unsupported validation split mode: {val_mode.value}")

    normal_data, abnormal_data = split_by_label(test_data)
    if not len(normal_data):
        train_data, normal_data = random_split(train_data, split_ratio=test_split_ratio, seed=seed)
    test_data = abnormal_data + normal_data
    if val_mode == ValSplitMode.FROM_TEST:
        test_data, val_data = random_split(test_data, split_ratio=val_split_ratio, label_aware=True, seed=seed)
    else:
        val_data = test_data.subsample(range(len(test_data)))
    return train_data, val_data, test_data
