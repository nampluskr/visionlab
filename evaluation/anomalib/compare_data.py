# evaluation/anomalib/compare_data.py: compare native datasets with the installed anomalib reference

import argparse
import hashlib
import inspect
import json
import math
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import torch
import yaml
from torchvision.transforms.v2 import Resize

from visionlab.anomaly_detection.data.mvtec.lightning_data import MVTecLitData
from visionlab.anomaly_detection.data.visa.lightning_data import VisALitData

from anomalib.data import MVTecAD, Visa
from anomalib.pre_processing import PreProcessor
from local_data import LocalMVTecAD, LocalVisa

RTOL = 1e-5
ATOL = 1e-6


def normalize_path(path):
    """Normalize separators while preserving an empty normal-image mask path."""
    return os.path.normpath(os.fspath(path)) if path else ""


def record_check(checks, name, passed, details=None):
    """Record a comparison and print each failed item without stopping later checks."""
    checks.append({"name": name, "passed": bool(passed), "details": details})
    print(f"{'PASS' if passed else 'FAIL'}: {name}")
    if not passed and details is not None:
        print(json.dumps(details, ensure_ascii=True))


def signature_fields(cls):
    """Describe constructor names, order, kinds and exact default representations."""
    return [{
        "name": parameter.name,
        "kind": parameter.kind.name,
        "has_default": parameter.default is not inspect.Parameter.empty,
        "default": repr(parameter.default) if parameter.default is not inspect.Parameter.empty else None,
    } for parameter in inspect.signature(cls.__init__).parameters.values()]


def sample_records(rows):
    """Describe samples in their existing order with raw and normalized paths."""
    return [{
        "image_path": normalize_path(sample["image_path"]),
        "mask_path": normalize_path(sample["mask_path"]),
        "raw_image_path": os.fspath(sample["image_path"]),
        "raw_mask_path": os.fspath(sample["mask_path"]) if sample["mask_path"] else "",
        "label": int(sample["label_index"]),
    } for sample in rows]


def count_files(folder, extensions):
    """Count image files recursively without reading or modifying their contents."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(folder)
    return sum(os.path.splitext(name)[1] in extensions
               for _, _, names in os.walk(folder) for name in names)


def count_layout(root_category, extensions):
    """Recount the prepared train, normal test, anomalous test and mask files."""
    test_folder = os.path.join(root_category, "test")
    if not os.path.isdir(test_folder):
        raise FileNotFoundError(test_folder)
    abnormal = {
        name: count_files(os.path.join(test_folder, name), extensions)
        for name in sorted(os.listdir(test_folder))
        if name != "good" and os.path.isdir(os.path.join(test_folder, name))
    }
    return {
        "train_good": count_files(os.path.join(root_category, "train", "good"), extensions),
        "test_good": count_files(os.path.join(test_folder, "good"), extensions),
        "test_bad": sum(abnormal.values()),
        "mask": count_files(os.path.join(root_category, "ground_truth"), extensions),
        "test_bad_by_label": abnormal,
    }


def tensor_summary(tensor):
    """Describe the shape, dtype and numeric range without changing a tensor."""
    if tensor is None:
        return None
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "range": [tensor.min().item(), tensor.max().item()] if tensor.numel() else None,
    }


def compare_tensor(checks, name, reference, actual, exact=False):
    """Compare tensor metadata separately from image tolerance or exact mask values."""
    expected = tensor_summary(reference)
    observed = tensor_summary(actual)
    details = {"anomalib": expected, "visionlab": observed}
    if reference is None or actual is None:
        record_check(checks, f"{name}.presence", reference is not None and actual is not None, details)
        return details
    if exact:
        details["known_difference"] = {
            "anomalib_dtype": expected["dtype"],
            "visionlab_dtype": observed["dtype"],
            "reason": "Mask dtype is informational; visionlab uses int64 (FR-1)",
        }
        print(f"KNOWN DIFFERENCE: {name}.dtype {json.dumps(details['known_difference'])}")
    else:
        record_check(checks, f"{name}.dtype", reference.dtype == actual.dtype,
                     {"anomalib": expected["dtype"], "visionlab": observed["dtype"]})
    same_shape = reference.shape == actual.shape
    record_check(checks, f"{name}.shape", same_shape,
                 {"anomalib": expected["shape"], "visionlab": observed["shape"]})
    if not exact:
        same_range = expected["range"] is not None and observed["range"] is not None and all(
            math.isclose(left, right, rel_tol=RTOL, abs_tol=ATOL)
            for left, right in zip(expected["range"], observed["range"])
        )
        record_check(checks, f"{name}.range", same_range,
                     {"anomalib": expected["range"], "visionlab": observed["range"]})
    equal_values = False
    if same_shape:
        equal_values = torch.equal(reference.to(torch.int64), actual.to(torch.int64)) if exact else torch.allclose(
            reference.to(torch.float64), actual.to(torch.float64), rtol=RTOL, atol=ATOL,
        )
    record_check(checks, f"{name}.{'equal' if exact else 'allclose'}", equal_values)
    return details


def compare_subset(name, reference, native, sample_count, pre_processor, checks):
    """Compare ordered metadata and the first samples before model normalization."""
    expected = sample_records(reference.samples.to_dict("records"))
    actual = sample_records(native.samples)
    result = {"anomalib": expected, "visionlab": actual, "sample_comparisons": []}
    record_check(checks, f"{name}.count", len(expected) == len(actual),
                 {"anomalib": len(expected), "visionlab": len(actual)})
    for field in ("image_path", "label", "mask_path"):
        left = [sample[field] for sample in expected]
        right = [sample[field] for sample in actual]
        mismatches = [index for index in range(max(len(left), len(right)))
                      if index >= len(left) or index >= len(right) or left[index] != right[index]]
        record_check(checks, f"{name}.{field}.ordered", not mismatches, {
            "mismatch_indices": mismatches,
            "examples": [{"index": index,
                          "anomalib": left[index] if index < len(left) else None,
                          "visionlab": right[index] if index < len(right) else None}
                         for index in mismatches[:5]],
        })
    original_augmentations = reference.augmentations
    for index in range(min(sample_count, len(expected), len(actual))):
        item = {"index": index, "anomalib_image_path": expected[index]["image_path"],
                "visionlab_image_path": actual[index]["image_path"]}
        try:
            raw_reference_item = reference[index]
            item["raw_anomalib_image"] = tensor_summary(raw_reference_item.image)
            item["raw_anomalib_mask"] = tensor_summary(raw_reference_item.gt_mask)
            reference.augmentations = pre_processor.transform
            reference_item = reference[index]
            image, target = native[index]
            item["image"] = compare_tensor(checks, f"{name}[{index}].image", reference_item.image, image)
            item["mask"] = compare_tensor(checks, f"{name}[{index}].mask", reference_item.gt_mask, target["mask"], exact=True)
            record_check(checks, f"{name}[{index}].label", int(reference_item.gt_label) == target["label"])
        except Exception as error:
            item["error"] = f"{type(error).__name__}: {error}"
            record_check(checks, f"{name}[{index}].load", False, item["error"])
        finally:
            reference.augmentations = original_augmentations
        result["sample_comparisons"].append(item)
    return result


def create_modules(data, category):
    """Read configured local roots and create independent reference and native modules."""
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", f"{data}.yaml")
    with open(config_file, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)["data"]["init_args"]
    reference_root = os.path.abspath(config["root"])
    if data == "mvtec":
        native_root = reference_root
        reference = LocalMVTecAD(
            root=reference_root, category=category, train_batch_size=8, eval_batch_size=8,
            num_workers=0, test_split_mode="from_dir", val_split_mode="from_test", val_split_ratio=0.4, seed=42,
        )
        native = MVTecLitData(
            root=native_root, category=category, train_batch_size=8, eval_batch_size=8,
            num_workers=0, test_split_mode="from_dir", val_split_mode="from_test", val_split_ratio=0.4, seed=42,
        )
        reference_split_root = reference_root
        native_split_root = native.root
    else:
        reference_split_root = os.path.abspath(config["split_root"])
        native_root = os.path.dirname(reference_split_root)
        reference = LocalVisa(
            root=reference_root, split_root=reference_split_root, category=category,
            train_batch_size=8, eval_batch_size=8, num_workers=0, test_split_mode="from_dir",
            val_split_mode="from_test", val_split_ratio=0.4, seed=42,
        )
        native = VisALitData(
            root=native_root, category=category, train_batch_size=8, eval_batch_size=8,
            num_workers=0, test_split_mode="from_dir", val_split_mode="from_test", val_split_ratio=0.4, seed=42,
        )
        native_split_root = native.split_root
    roots = {
        "config_file": config_file,
        "anomalib_root": reference_root,
        "anomalib_split_root": reference_split_root,
        "visionlab_root": native_root,
        "visionlab_split_root": native_split_root,
        "anomalib_category_root": os.path.join(reference_split_root, category),
        "visionlab_category_root": os.path.join(native_split_root, category),
    }
    return reference, native, roots


def main():
    """Compare both constructor contracts and one dataset, then save a hashed manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=("mvtec", "visa"), required=True)
    parser.add_argument("--category")
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    category = args.category or ("bottle" if args.data == "mvtec" else "candle")
    if category in (".", "..") or os.path.basename(category) != category:
        parser.error("--category must be a single directory name")
    checks = []
    manifest = {
        "data": args.data,
        "category": category,
        "settings": {"test_split_mode": "from_dir", "test_split_ratio": 0.2,
                     "val_split_mode": "from_test", "val_split_ratio": 0.4,
                     "seed": 42, "train_batch_size": 8, "eval_batch_size": 8, "num_workers": 0},
        "preprocessing": {"scope": "dataset input before model normalization", "size": [256, 256],
                          "image_interpolation": "bilinear", "antialias": True, "mask_interpolation": "nearest",
                          "normalization": None, "anomalib_transform": "Dataset.augmentations = PreProcessor.transform(Resize)",
                          "visionlab_transform": "Dataset resize", "rtol": RTOL, "atol": ATOL},
        "sampling": {"method": "ordered dataset prefix", "requested_per_subset": args.samples},
        "signatures": {},
        "splits": {},
        "checks": checks,
    }
    for name, reference_class, native_class in (("mvtec", MVTecAD, MVTecLitData), ("visa", Visa, VisALitData)):
        expected = signature_fields(reference_class)
        actual = signature_fields(native_class)
        manifest["signatures"][name] = {"anomalib": expected, "visionlab": actual,
                                        "anomalib_raw": str(inspect.signature(reference_class.__init__)),
                                        "visionlab_raw": str(inspect.signature(native_class.__init__))}
        record_check(checks, f"{name}.constructor", expected == actual,
                     {"anomalib": expected, "visionlab": actual})
    try:
        reference, native, roots = create_modules(args.data, category)
        manifest["roots"] = roots
        record_check(checks, "roots.same_prepared_category",
                     normalize_path(roots["anomalib_category_root"]) == normalize_path(roots["visionlab_category_root"]), roots)
        # LocalVisa.prepare_data would copy files if this directory were missing.
        for path in (roots["anomalib_category_root"], roots["visionlab_category_root"]):
            if not os.path.isdir(path):
                raise FileNotFoundError(path)
        manifest["counts"] = {
            "anomalib": count_layout(roots["anomalib_category_root"], native.dataset_class.extensions),
            "visionlab": count_layout(roots["visionlab_category_root"], native.dataset_class.extensions),
        }
        record_check(checks, "source_file_counts", manifest["counts"]["anomalib"] == manifest["counts"]["visionlab"],
                     manifest["counts"])
        reference.prepare_data()
        native.prepare_data()
        reference.setup()
        native.setup()
        pre_processor = PreProcessor(transform=Resize(size=(256, 256), antialias=True))
        for name in ("train", "val", "test"):
            reference_loader = getattr(reference, f"{name}_dataloader")()
            native_loader = getattr(native, f"{name}_dataloader")()
            reference_settings = {"batch_size": reference_loader.batch_size, "num_workers": reference_loader.num_workers,
                                  "sampler": type(reference_loader.sampler).__name__}
            native_settings = {"batch_size": native_loader.batch_size, "num_workers": native_loader.num_workers,
                               "sampler": type(native_loader.sampler).__name__}
            record_check(checks, f"{name}.dataloader", reference_settings == native_settings,
                         {"anomalib": reference_settings, "visionlab": native_settings})
            manifest["splits"][name] = compare_subset(
                name, getattr(reference, f"{name}_data"), getattr(native, f"{name}_data"),
                args.samples, pre_processor, checks,
            )
    except Exception as error:
        manifest["error"] = f"{type(error).__name__}: {error}"
        record_check(checks, "dataset_setup_or_comparison", False, manifest["error"])
    manifest["passed"] = all(check["passed"] for check in checks)
    output_dir = os.path.join(PROJECT_ROOT, "outputs", "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{args.data}_{category}_manifest.json")
    with open(output_path, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")
    digest = hashlib.sha256()
    with open(output_path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    print(f"Manifest: {output_path}")
    print(f"Data verification ID (SHA-256): {digest.hexdigest()}")
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
