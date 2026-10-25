# tests/common/test_dependency.py: pytest static and isolated runtime dependency boundaries

import ast
import importlib
import importlib.abc
import os
import runpy
import subprocess
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE_ROOT = os.path.join(PROJECT_ROOT, "src", "visionlab")
LIGHTNING_FILES = {
    "anomaly_detection/data/mvtec/lightning_data.py", "anomaly_detection/data/visa/lightning_data.py",
    "anomaly_detection/models/stfpm/lightning_model.py", "anomaly_detection/common/lightning_visualizer.py",
    "anomaly_detection/trainers/lightning_cli.py",
}


class BlockImports(importlib.abc.MetaPathFinder):
    """Reject forbidden frameworks before loading the isolated module group."""

    def __init__(self, forbidden):
        self.forbidden = forbidden

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.forbidden:
            raise ModuleNotFoundError(f"Forbidden import: {fullname}")
        return None


def python_files(root):
    """List all Python sources while excluding interpreter cache directories."""
    return sorted(os.path.join(directory, name) for directory, folders, names in os.walk(root)
                  if "__pycache__" not in os.path.relpath(directory, root).split(os.sep)
                  for name in names if name.endswith(".py"))


def source_imports(path):
    """Extract actual import statements while ignoring comments and string literals."""
    with open(path, encoding="utf-8") as stream:
        tree = ast.parse(stream.read(), filename=path)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((alias.name.split(".")[0], node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.append((node.module.split(".")[0], node.lineno))
    return imports


def classified_modules():
    """Classify every source module into its declared pure or native boundary."""
    groups = {"pure": [], "lightning": []}
    for path in python_files(SOURCE_ROOT):
        relative = os.path.relpath(path, SOURCE_ROOT).replace(os.sep, "/")
        name = os.path.basename(path)
        if relative in LIGHTNING_FILES:
            group = "lightning"
        elif relative.startswith(("anomaly_detection/data/common/", "anomaly_detection/models/components/",
                                  "anomaly_detection/common/", "inference/", "utils/")) or \
                (relative.startswith("anomaly_detection/data/") and name == "dataset.py") or \
                (relative.startswith("anomaly_detection/models/stfpm/") and
                 name in ("model.py", "loss.py", "anomaly_map.py", "__init__.py")) or name == "__init__.py":
            group = "pure"
        else:
            raise AssertionError(f"Unclassified source file: {relative}")
        parts = relative[:-3].split("/")
        if parts[-1] == "__init__":
            parts.pop()
        groups[group].append((path, ".".join(["visionlab"] + parts)))
    return groups


def import_group(group):
    """Import modules or calculate metrics only after installing fresh-process import guards."""
    forbidden = {"anomalib"} if group == "lightning" else {"anomalib", "lightning", "pytorch_lightning"}
    sys.meta_path.insert(0, BlockImports(forbidden))
    from visionlab.utils.io import block_network

    block_network()
    if group == "metrics":
        import torch
        from visionlab.anomaly_detection.common.metric import AUROC, F1
        from visionlab.anomaly_detection.common.post_processor import PostProcessor

        scores, labels = torch.arange(4, dtype=torch.float32), torch.tensor([0, 0, 1, 1])
        for metric, prediction in ((AUROC(), scores), (F1(), scores > 1)):
            metric.update(prediction, labels)
            assert metric.compute() == 1.0 and metric.reason is None
        processor = PostProcessor()
        processor.update({"pred_score": scores, "gt_label": labels,
                          "anomaly_map": scores.reshape(4, 1, 1, 1), "gt_mask": labels.reshape(4, 1, 1)})
        processor.finalize()
        assert processor.image_threshold == processor.pixel_threshold == 2
        assert all(value.device.type == "cpu" for value in processor.state_dict().values())
    else:
        failures = []
        for path, module in classified_modules()[group]:
            try:
                importlib.import_module(module)
                print(f"PASS {module}")
            except Exception as error:
                failures.append(f"{path}: {type(error).__name__}: {error}")
        assert not failures, "\n".join(failures)
        if group == "pure":
            runpy.run_path(os.path.join(PROJECT_ROOT, "evaluation", "visionlab", "predict.py"),
                           run_name="pure_prediction_import_check")
    assert not any(name.split(".")[0] in forbidden for name in sys.modules)


def test_no_reference_imports_in_source():
    """Keep reference and sklearn imports absent and native imports within the four allowed file names."""
    for path in python_files(SOURCE_ROOT):
        relative = os.path.relpath(path, SOURCE_ROOT).replace(os.sep, "/")
        for name, line in source_imports(path):
            assert name not in ("anomalib", "sklearn", "scikit-learn"), f"{path}:{line}: {name}"
            if name in ("lightning", "pytorch_lightning"):
                assert relative in LIGHTNING_FILES, f"{path}:{line}: {name}"


def test_pytest_only_and_unique_basenames():
    """Enforce NFR-6 across all test sources and prevent pytest module name collisions."""
    paths = python_files(os.path.join(PROJECT_ROOT, "tests"))
    names = [os.path.basename(path) for path in paths]
    assert len(names) == len(set(names))
    for path in paths:
        assert all(name != "unittest" for name, _ in source_imports(path)), path


@pytest.fixture(params=("pure", "lightning", "metrics"))
def import_boundary(request):
    """Give each dependency boundary its own collected pytest case."""
    return request.param


def test_isolated_import_groups(import_boundary):
    """Check NFR-1, NFR-2 and FR-24 with a fresh interpreter for every boundary."""
    group = import_boundary
    script = '''
import runpy
import sys
sys.path.insert(0, sys.argv[2])
runpy.run_path(sys.argv[1])["import_group"](sys.argv[3])
'''
    result = subprocess.run([sys.executable, "-c", script, os.path.abspath(__file__),
                             os.path.join(PROJECT_ROOT, "src"), group], capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"{group}\n{result.stdout}\n{result.stderr}"


def test_latest_export_selection(tmp_path, monkeypatch):
    """Choose numeric versions and reject a latest run with no export instead of falling back."""
    namespace = runpy.run_path(os.path.join(PROJECT_ROOT, "evaluation", "visionlab", "predict.py"))
    select_export = namespace["latest_export"]
    monkeypatch.setitem(select_export.__globals__, "PROJECT_ROOT", str(tmp_path))
    run_root = os.path.join(tmp_path, "outputs", "visionlab", "stfpm", "mvtec_bottle")
    for version in (2, 10):
        directory = os.path.join(run_root, f"v{version}", "export")
        os.makedirs(directory)
        with open(os.path.join(directory, "model.pt"), "wb"):
            pass
    assert select_export("stfpm", "mvtec", "bottle") == os.path.join(run_root, "v10", "export", "model.pt")
    os.makedirs(os.path.join(run_root, "v11"))
    with pytest.raises(FileNotFoundError):
        select_export("stfpm", "mvtec", "bottle")
