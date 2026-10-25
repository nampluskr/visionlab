# tests/common/test_cli.py: pytest native CLI layering, provenance and checkpoint execution contracts

import json
import os
import runpy
import socket

import pytest

from visionlab.anomaly_detection.trainers.lightning_cli import CONFIG_DIR, LitCLI, expand_args
from visionlab.utils import io as io_utils

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def cli_helpers():
    """Reuse the native asset and image builders without changing import paths."""
    return runpy.run_path(os.path.join(PROJECT_ROOT, "tests", "01_stfpm", "test_lightning.py"))


@pytest.fixture(scope="module")
def cli_assets(tmp_path_factory, cli_helpers):
    """Share one random teacher file and restore its TOML mapping after the module."""
    root = str(tmp_path_factory.mktemp("cli_assets"))
    io_utils.block_network()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(io_utils, "PATHS_FILE", cli_helpers["make_assets"](root))
        yield root


@pytest.fixture
def cli_args(tmp_path, cli_assets, cli_helpers):
    """Prepare independent offline CLI runs with a single CPU training step."""
    data_root = os.path.join(tmp_path, "data")
    cli_helpers["make_fixture"](data_root)

    def arguments(command):
        return [command, "--data", "mvtec", "--model", "stfpm", "--data.init_args.root", data_root,
                "--data.init_args.train_batch_size", "2", "--trainer.default_root_dir", os.path.join(tmp_path, "runs"),
                "--trainer.accelerator", "cpu", "--trainer.devices", "1", "--trainer.max_epochs", "1",
                "--trainer.limit_train_batches", "1", "--trainer.logger", "false",
                "--trainer.enable_progress_bar", "false", "--trainer.enable_model_summary", "false"]

    return arguments


@pytest.fixture(scope="module")
def fitted_cli(tmp_path_factory, cli_assets, cli_helpers):
    """Fit once for checkpoint, export and run provenance comparisons."""
    root = str(tmp_path_factory.mktemp("fitted_cli"))
    data_root = os.path.join(root, "data")
    cli_helpers["make_fixture"](data_root)
    args = ["--data", "mvtec", "--model", "stfpm", "--data.init_args.root", data_root,
            "--data.init_args.train_batch_size", "2", "--trainer.default_root_dir", os.path.join(root, "runs"),
            "--trainer.accelerator", "cpu", "--trainer.devices", "1", "--trainer.max_epochs", "1",
            "--trainer.limit_train_batches", "1", "--trainer.logger", "false",
            "--trainer.enable_progress_bar", "false", "--trainer.enable_model_summary", "false"]
    return LitCLI(args=["fit"] + args), args


def test_name_expansion_and_four_layer_merging(cli_args):
    """Merge all four configuration layers and keep explicit overrides and trainer arguments."""
    arguments = expand_args(["fit", "--data", "mvtec", "--model=stfpm", "--data.init_args.category", "cable"])
    configs = [arguments[index + 1] for index, argument in enumerate(arguments) if argument == "--config"]
    assert configs == [os.path.join(CONFIG_DIR, "default.yaml"), os.path.join(CONFIG_DIR, "trainer.yaml"),
                       os.path.join(CONFIG_DIR, "data", "mvtec.yaml"), os.path.join(CONFIG_DIR, "models", "stfpm.yaml")]
    cli = LitCLI(args=cli_args("fit")[1:] + ["--data.init_args.category", "cable", "--trainer.gradient_clip_val", "5",
                                            "--trainer.num_sanity_val_steps", "2"], run=False)
    assert cli.config.seed_everything == 42 and cli.datamodule.category == "cable"
    assert cli.datamodule.val_split_ratio == 0.4 and cli.datamodule.seed == 42
    assert cli.model.backbone == "resnet18" and cli.model.visualizer is None
    assert cli.trainer.gradient_clip_val == cli.trainer.num_sanity_val_steps == 0
    assert cli.trainer.default_root_dir.endswith(os.path.join("stfpm", "mvtec_cable", "v0"))


def test_unknown_argument_is_rejected(cli_args):
    """Preserve Lightning CLI errors for unrecognized arguments."""
    with pytest.raises(SystemExit) as error:
        LitCLI(args=cli_args("fit") + ["--unrecognized_option", "1"])
    assert error.value.code != 0


def test_fit_test_predict_checkpoint_and_run_json(fitted_cli, cli_assets):
    """Recover fit, test and predict checkpoints with exact provenance and split manifests."""
    fitted, args = fitted_cli
    checkpoint = os.path.join(fitted.trainer.default_root_dir, "weights", "lightning", "model.ckpt")
    assert os.path.isfile(checkpoint)
    assert os.path.isfile(os.path.join(fitted.trainer.default_root_dir, "export", "model.pt"))
    with open(os.path.join(fitted.trainer.default_root_dir, "run.json"), encoding="utf-8") as stream:
        record = json.load(stream)
    assert record["seed"] == 42
    assert record["config"]["trainer"]["gradient_clip_val"] == record["config"]["trainer"]["num_sanity_val_steps"] == 0
    assert record["config"]["trainer"]["default_root_dir"] == fitted.trainer.default_root_dir
    assert record["teacher_weights"]["sha256"] == io_utils.file_sha256(os.path.join(cli_assets, "resnet18.pth"))
    assert record["checkpoint"]["sha256"] == io_utils.file_sha256(checkpoint)
    for stage in ("train", "val", "test"):
        assert record["splits"][stage] == [sample["image_path"] for sample in getattr(fitted.datamodule, f"{stage}_data").samples]
    tested = LitCLI(args=["test"] + args + ["--ckpt_path", checkpoint])
    assert "image_AUROC" in tested.model.last_metrics["test"]
    assert tested.trainer.default_root_dir.endswith("v1")
    with open(os.path.join(tested.trainer.default_root_dir, "run.json"), encoding="utf-8") as stream:
        test_record = json.load(stream)
    assert test_record["checkpoint"]["sha256"] == record["checkpoint"]["sha256"]
    predicted = LitCLI(args=["predict"] + args + ["--ckpt_path", checkpoint])
    assert predicted.trainer.default_root_dir.endswith("v2")
    assert os.path.isfile(os.path.join(predicted.trainer.default_root_dir, "run.json"))


def test_versioned_directories_and_network_block(tmp_path, cli_assets):
    """Create sequential version directories without latest links and reject outgoing sockets."""
    root = os.path.join(tmp_path, "versions")
    assert os.path.basename(io_utils.create_versioned_dir(root)) == "v0"
    assert os.path.basename(io_utils.create_versioned_dir(root)) == "v1"
    assert not os.path.lexists(os.path.join(root, "latest"))
    with pytest.raises(RuntimeError, match="network access is blocked"):
        socket.create_connection(("example.invalid", 443), timeout=1)


def test_train_entry_runs_fit_then_test(cli_args):
    """Keep the train entry command as fit followed by test on the trained model."""
    entry = runpy.run_path(os.path.join(PROJECT_ROOT, "evaluation", "visionlab", "train.py"))
    cli = entry["main"](args=["train"] + cli_args("fit")[1:])
    assert "test" in cli.model.last_metrics
    assert os.path.isfile(os.path.join(cli.trainer.default_root_dir, "run.json"))
