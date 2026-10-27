# tests/common/test_cli.py: pytest native CLI layering, provenance and checkpoint execution contracts

import json
import os
import runpy
import socket

import pytest

from visionlab.anomaly_detection.trainers.lightning_cli import CONFIG_DIR, LitCLI, expand_args
from visionlab.utils import io as io_utils

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.mark.parametrize("data, category", (("mvtec", "bottle"), ("visa", "candle")))
@pytest.mark.parametrize("max_epochs", (None, 5))
def test_fastflow_config_merging_and_output_name(tmp_path, cli_assets, data, category, max_epochs):
    """Merge FastFlow after each dataset and preserve shared epochs and explicit budget overrides."""
    import yaml

    with open(os.path.join(CONFIG_DIR, "models", "fastflow.yaml"), encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    assert config["model"]["class_path"] == "visionlab.anomaly_detection.models.fastflow.lightning_model.FastflowLitModel"
    assert config["model"]["init_args"] == {
        "backbone": "resnet18", "pre_trained": True, "flow_steps": 8, "conv3x3_only": False,
        "hidden_ratio": 1.0, "visualizer": False,
    }
    assert "max_epochs" not in config["trainer"]
    arguments = ["--data", data, "--model", "fastflow", "--model.init_args.flow_steps", "2",
                 "--trainer.num_sanity_val_steps", "2", "--trainer.gradient_clip_val", "5",
                 "--trainer.default_root_dir", os.path.join(tmp_path, "runs"), "--trainer.accelerator", "cpu",
                 "--trainer.devices", "1", "--trainer.logger", "false"]
    if max_epochs is not None:
        arguments += ["--trainer.max_epochs", str(max_epochs)]
    cli = LitCLI(args=arguments, run=False)
    assert cli.trainer.max_epochs == (10 if max_epochs is None else max_epochs)
    assert cli.trainer.num_sanity_val_steps == cli.trainer.gradient_clip_val == 0
    assert cli.model.backbone == "resnet18" and cli.model.pre_trained
    assert cli.model.flow_steps == 2 and not cli.model.conv3x3_only and cli.model.hidden_ratio == 1.0
    assert cli.model.visualizer is None and cli.model.pre_processor.center_crop is None
    assert cli.datamodule.train_batch_size == cli.datamodule.eval_batch_size == 8
    assert cli.trainer.default_root_dir.endswith(os.path.join("fastflow", f"{data}_{category}", "v0"))


@pytest.mark.parametrize("max_epochs", (None, 5))
def test_draem_config_merging_beta_and_output_name(tmp_path, max_epochs):
    """Merge DRAEM after MVTec with tuple beta, shared epochs and explicit budget overrides."""
    import yaml

    with open(os.path.join(CONFIG_DIR, "models", "draem.yaml"), encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    assert config["model"]["init_args"]["beta"] == [0.1, 1.0]
    assert config["model"]["init_args"]["anomaly_source_path"] == "D:/datasets/dtd/images"
    assert "max_epochs" not in config["trainer"]
    arguments = ["--data", "mvtec", "--model", "draem",
                 "--trainer.num_sanity_val_steps", "2", "--trainer.gradient_clip_val", "5",
                 "--trainer.default_root_dir", os.path.join(tmp_path, "runs"), "--trainer.accelerator", "cpu",
                 "--trainer.devices", "1", "--trainer.logger", "false"]
    if max_epochs is not None:
        arguments += ["--trainer.max_epochs", str(max_epochs)]
    cli = LitCLI(args=arguments, run=False)
    assert cli.trainer.max_epochs == (10 if max_epochs is None else max_epochs)
    assert cli.trainer.num_sanity_val_steps == cli.trainer.gradient_clip_val == 0
    assert cli.model.augmenter.blend_factor == (0.1, 1.0)
    assert type(cli.model.augmenter.blend_factor) is tuple
    assert cli.model.teacher_weights_path is None and cli.model.pre_processor.mean is None
    assert cli.model.visualizer is None and not cli.model.sspcab
    assert cli.datamodule.train_batch_size == cli.datamodule.eval_batch_size == 8
    assert cli.trainer.default_root_dir.endswith(os.path.join("draem", "mvtec_bottle", "v0"))


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


def test_efficient_ad_config_merging_and_output_name(tmp_path, monkeypatch):
    """Apply EfficientAD data overrides while retaining shared epochs and its package name."""
    helpers = runpy.run_path(os.path.join(PROJECT_ROOT, "tests", "02_efficient_ad", "test_efficient_ad.py"))
    paths_file, imagenet_dir, _ = helpers["make_efficient_assets"](str(tmp_path), teacher_out_channels=384)
    monkeypatch.setattr(io_utils, "PATHS_FILE", paths_file)
    cli = LitCLI(args=["--data", "mvtec", "--model", "efficient_ad", "--model.init_args.imagenet_dir", imagenet_dir,
                       "--trainer.default_root_dir", os.path.join(tmp_path, "runs"), "--trainer.accelerator", "cpu",
                       "--trainer.devices", "1", "--trainer.logger", "false", "--trainer.num_sanity_val_steps", "2"],
                 run=False)
    assert cli.datamodule.train_batch_size == 1 and cli.datamodule.eval_batch_size == 16
    assert cli.trainer.max_epochs == 10 and cli.trainer.num_sanity_val_steps == 0
    assert cli.model.model_size.value == "small" and cli.model.visualizer is None
    assert cli.model.pre_processor.mean is None
    assert cli.trainer.default_root_dir.endswith(os.path.join("efficient_ad", "mvtec_bottle", "v0"))


def test_padim_config_merging_and_output_name(tmp_path, cli_assets):
    """Load PaDiM after the shared data config and retain its fixed fitting epoch and package name."""
    cli = LitCLI(args=["--data", "mvtec", "--model", "padim", "--trainer.max_epochs", "5",
                       "--trainer.num_sanity_val_steps", "2", "--trainer.val_check_interval", "0.5",
                       "--trainer.default_root_dir", os.path.join(tmp_path, "runs"), "--trainer.accelerator", "cpu",
                       "--trainer.devices", "1", "--trainer.logger", "false"], run=False)
    assert cli.trainer.max_epochs == 1 and cli.trainer.num_sanity_val_steps == 0
    assert cli.trainer.val_check_interval == 1.0
    assert cli.datamodule.train_batch_size == cli.datamodule.eval_batch_size == 8
    assert cli.model.model.n_features == 100 and cli.model.visualizer is None
    assert cli.model.pre_trained is True
    assert cli.trainer.default_root_dir.endswith(os.path.join("padim", "mvtec_bottle", "v0"))


def test_dfm_config_merging_and_output_name(tmp_path, monkeypatch):
    """Load the default ResNet50 DFM config and enforce its fixed fitting trainer arguments."""
    import timm
    import torch

    weights = os.path.join(tmp_path, "resnet50.pth")
    torch.save(timm.create_model("resnet50", pretrained=False).state_dict(), weights)
    mapping = os.path.join(tmp_path, "paths.toml")
    with open(mapping, "w", encoding="utf-8", newline="\n") as stream:
        stream.write('[backbones]\nresnet50 = "' + weights.replace("\\", "/") + '"\n')
    monkeypatch.setattr(io_utils, "PATHS_FILE", mapping)
    cli = LitCLI(args=["--data", "mvtec", "--model", "dfm", "--trainer.max_epochs", "5",
                       "--trainer.num_sanity_val_steps", "2", "--trainer.gradient_clip_val", "5",
                       "--trainer.default_root_dir", os.path.join(tmp_path, "runs"), "--trainer.accelerator", "cpu",
                       "--trainer.devices", "1", "--trainer.logger", "false"], run=False)
    assert cli.trainer.max_epochs == 1 and cli.trainer.num_sanity_val_steps == cli.trainer.gradient_clip_val == 0
    assert cli.datamodule.train_batch_size == cli.datamodule.eval_batch_size == 8
    assert cli.model.backbone == "resnet50" and cli.model.layer == "layer3"
    assert cli.model.pca_level == 0.97 and cli.model.score_type == "fre" and cli.model.visualizer is None
    assert cli.model.teacher_weights_path == weights
    assert cli.trainer.default_root_dir.endswith(os.path.join("dfm", "mvtec_bottle", "v0"))


def test_dfkde_config_merging_and_output_name(tmp_path, cli_assets):
    """Load image-only DFKDE after data settings and enforce its fixed fitting trainer arguments."""
    cli = LitCLI(args=["--data", "mvtec", "--model", "dfkde", "--trainer.max_epochs", "5",
                       "--trainer.num_sanity_val_steps", "2", "--trainer.gradient_clip_val", "5",
                       "--trainer.default_root_dir", os.path.join(tmp_path, "runs"), "--trainer.accelerator", "cpu",
                       "--trainer.devices", "1", "--trainer.logger", "false"], run=False)
    assert cli.trainer.max_epochs == 1 and cli.trainer.num_sanity_val_steps == cli.trainer.gradient_clip_val == 0
    assert cli.datamodule.train_batch_size == cli.datamodule.eval_batch_size == 8
    assert cli.model.layers == ["layer4"] and cli.model.n_pca_components == 16
    assert cli.model.max_training_points == 40000 and cli.model.feature_scaling_method.value == "scale"
    assert cli.model.visualizer is None
    assert [entry[0] for entry in cli.model.evaluator.test_metrics] == ["image_AUROC", "image_F1Score"]
    assert cli.trainer.default_root_dir.endswith(os.path.join("dfkde", "mvtec_bottle", "v0"))


def test_patchcore_config_merging_and_output_name(tmp_path, cli_assets):
    """Merge PatchCore defaults and fixed fitting arguments with a small locally mapped backbone."""
    import yaml

    with open(os.path.join(CONFIG_DIR, "models", "patchcore.yaml"), encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    assert config["model"]["init_args"]["backbone"] == "wide_resnet50_2"
    cli = LitCLI(args=["--data", "mvtec", "--model", "patchcore", "--model.init_args.backbone", "resnet18",
                       "--trainer.max_epochs", "5", "--trainer.num_sanity_val_steps", "2",
                       "--trainer.gradient_clip_val", "5", "--trainer.default_root_dir", os.path.join(tmp_path, "runs"),
                       "--trainer.accelerator", "cpu", "--trainer.devices", "1", "--trainer.logger", "false"], run=False)
    assert cli.trainer.max_epochs == 1 and cli.trainer.num_sanity_val_steps == cli.trainer.gradient_clip_val == 0
    assert cli.datamodule.train_batch_size == cli.datamodule.eval_batch_size == 8
    assert cli.model.layers == ["layer2", "layer3"] and cli.model.num_neighbors == 9
    assert cli.model.coreset_sampling_ratio == 0.1 and cli.model.visualizer is None
    assert cli.model.pre_processor.center_crop == (224, 224)
    assert cli.trainer.default_root_dir.endswith(os.path.join("patchcore", "mvtec_bottle", "v0"))


@pytest.mark.parametrize("max_epochs", (None, 5))
def test_reverse_distillation_config_merging_and_output_name(tmp_path, cli_assets, max_epochs):
    """Keep shared epochs overridable while enforcing model hooks and its package output folder."""
    import yaml

    with open(os.path.join(CONFIG_DIR, "models", "reverse_distillation.yaml"), encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    assert config["model"]["init_args"]["backbone"] == "wide_resnet50_2"
    assert "max_epochs" not in config["trainer"]
    arguments = ["--data", "mvtec", "--model", "reverse_distillation", "--model.init_args.backbone", "resnet18",
                 "--trainer.num_sanity_val_steps", "2", "--trainer.gradient_clip_val", "5",
                 "--trainer.default_root_dir", os.path.join(tmp_path, "runs"), "--trainer.accelerator", "cpu",
                 "--trainer.devices", "1", "--trainer.logger", "false"]
    if max_epochs is not None:
        arguments += ["--trainer.max_epochs", str(max_epochs)]
    cli = LitCLI(args=arguments, run=False)
    assert cli.trainer.max_epochs == (10 if max_epochs is None else max_epochs)
    assert cli.trainer.num_sanity_val_steps == cli.trainer.gradient_clip_val == 0
    assert cli.datamodule.train_batch_size == cli.datamodule.eval_batch_size == 8
    assert cli.model.layers == ("layer1", "layer2", "layer3") and cli.model.visualizer is None
    assert cli.model.anomaly_map_mode.value == "add"
    assert cli.trainer.default_root_dir.endswith(os.path.join("reverse_distillation", "mvtec_bottle", "v0"))


@pytest.mark.parametrize("max_epochs", (None, 5))
def test_cfa_config_merging_and_output_name(tmp_path, monkeypatch, max_epochs):
    """Load CFA defaults and shared epochs while preserving command-line budget overrides."""
    import torch
    import torchvision
    import yaml

    weights = os.path.join(tmp_path, "resnet18.pth")
    torch.save(torchvision.models.resnet18(weights=None).state_dict(), weights)
    mapping = os.path.join(tmp_path, "paths.toml")
    with open(mapping, "w", encoding="utf-8", newline="\n") as stream:
        stream.write('[backbones]\nresnet18 = "' + weights.replace("\\", "/") + '"\n')
    monkeypatch.setattr(io_utils, "PATHS_FILE", mapping)
    with open(os.path.join(CONFIG_DIR, "models", "cfa.yaml"), encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    assert config["model"]["init_args"]["backbone"] == "wide_resnet50_2" and "max_epochs" not in config["trainer"]
    arguments = ["--data", "mvtec", "--model", "cfa", "--model.init_args.backbone", "resnet18",
                 "--trainer.num_sanity_val_steps", "2", "--trainer.gradient_clip_val", "5",
                 "--trainer.default_root_dir", os.path.join(tmp_path, "runs"), "--trainer.accelerator", "cpu",
                 "--trainer.devices", "1", "--trainer.logger", "false"]
    if max_epochs is not None:
        arguments += ["--trainer.max_epochs", str(max_epochs)]
    cli = LitCLI(args=arguments, run=False)
    assert cli.trainer.max_epochs == (10 if max_epochs is None else max_epochs)
    assert cli.trainer.num_sanity_val_steps == cli.trainer.gradient_clip_val == 0
    assert cli.datamodule.train_batch_size == cli.datamodule.eval_batch_size == 8
    assert cli.model.model_args["gamma_c"] == cli.model.model_args["gamma_d"] == 1
    assert cli.model.model_args["num_nearest_neighbors"] == cli.model.model_args["num_hard_negative_features"] == 3
    assert cli.model.model_args["radius"] == 1e-5 and cli.model.visualizer is None
    assert cli.trainer.default_root_dir.endswith(os.path.join("cfa", "mvtec_bottle", "v0"))


def test_fre_config_merging_and_output_name(tmp_path, cli_assets):
    """Load FRE defaults and enforce the user-selected fixed ten-epoch budget over CLI overrides."""
    import yaml

    with open(os.path.join(CONFIG_DIR, "models", "fre.yaml"), encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    assert config["model"]["init_args"]["backbone"] == "resnet50"
    assert config["model"]["init_args"]["input_dim"] == 65536
    assert config["model"]["init_args"]["latent_dim"] == 220
    cli = LitCLI(args=["--data", "mvtec", "--model", "fre", "--model.init_args.backbone", "resnet18",
                       "--model.init_args.input_dim", "16384", "--model.init_args.latent_dim", "7",
                       "--trainer.max_epochs", "5", "--trainer.num_sanity_val_steps", "2",
                       "--trainer.gradient_clip_val", "5", "--trainer.default_root_dir", os.path.join(tmp_path, "runs"),
                       "--trainer.accelerator", "cpu", "--trainer.devices", "1", "--trainer.logger", "false"], run=False)
    assert cli.trainer.max_epochs == 10 and cli.trainer.num_sanity_val_steps == cli.trainer.gradient_clip_val == 0
    assert cli.datamodule.train_batch_size == cli.datamodule.eval_batch_size == 8
    assert cli.model.layer == "layer3" and cli.model.pooling_kernel_size == 2 and cli.model.visualizer is None
    assert cli.trainer.default_root_dir.endswith(os.path.join("fre", "mvtec_bottle", "v0"))


@pytest.mark.parametrize("max_epochs", (None, 5))
def test_supersimplenet_config_merging_and_output_name(tmp_path, cli_assets, max_epochs):
    """Keep shared epochs overridable while enforcing SuperSimpleNet hooks and its model output folder."""
    import yaml

    with open(os.path.join(CONFIG_DIR, "models", "supersimplenet.yaml"), encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    assert config["model"]["init_args"]["backbone"] == "wide_resnet50_2" and "max_epochs" not in config["trainer"]
    arguments = ["--data", "mvtec", "--model", "supersimplenet", "--model.init_args.backbone", "resnet18",
                 "--trainer.num_sanity_val_steps", "2", "--trainer.gradient_clip_val", "5",
                 "--trainer.default_root_dir", os.path.join(tmp_path, "runs"), "--trainer.accelerator", "cpu",
                 "--trainer.devices", "1", "--trainer.logger", "false"]
    if max_epochs is not None:
        arguments += ["--trainer.max_epochs", str(max_epochs)]
    cli = LitCLI(args=arguments, run=False)
    assert cli.trainer.max_epochs == (10 if max_epochs is None else max_epochs)
    assert cli.trainer.num_sanity_val_steps == cli.trainer.gradient_clip_val == 0
    assert cli.datamodule.train_batch_size == cli.datamodule.eval_batch_size == 8
    assert cli.model.layers == ["layer2", "layer3"] and cli.model.perlin_threshold == 0.2
    assert not cli.model.supervised and cli.model.model.segdec.stop_grad and cli.model.visualizer is None
    assert cli.trainer.default_root_dir.endswith(os.path.join("supersimplenet", "mvtec_bottle", "v0"))


@pytest.mark.parametrize("data,category", (("mvtec", "bottle"), ("visa", "candle")))
@pytest.mark.parametrize("max_epochs", (None, 5))
def test_cflow_config_merging_and_output_name(tmp_path, cli_assets, data, category, max_epochs):
    """Apply CFLOW's one-epoch YAML budget while preserving explicit CLI overrides and model folders."""
    import yaml

    with open(os.path.join(CONFIG_DIR, "models", "cflow.yaml"), encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    assert config["model"]["class_path"].endswith("cflow.lightning_model.CflowLitModel")
    assert config["model"]["init_args"] == {
        "backbone": "wide_resnet50_2", "layers": ["layer2", "layer3", "layer4"], "pre_trained": True,
        "fiber_batch_size": 64, "decoder": "freia-cflow", "condition_vector": 128,
        "coupling_blocks": 8, "clamp_alpha": 1.9, "permute_soft": False, "lr": 0.0001, "visualizer": False,
    }
    assert config["trainer"] == {"max_epochs": 1, "num_sanity_val_steps": 0, "gradient_clip_val": 0}
    arguments = ["--data", data, "--model", "cflow", "--model.init_args.backbone", "resnet18",
                 "--model.init_args.layers", '["layer1", "layer2", "layer3"]',
                 "--model.init_args.condition_vector", "16", "--model.init_args.coupling_blocks", "2",
                 "--model.init_args.fiber_batch_size", "16",
                 "--trainer.num_sanity_val_steps", "2", "--trainer.gradient_clip_val", "5",
                 "--trainer.default_root_dir", os.path.join(tmp_path, "runs"), "--trainer.accelerator", "cpu",
                 "--trainer.devices", "1", "--trainer.logger", "false"]
    if max_epochs is not None:
        arguments += ["--trainer.max_epochs", str(max_epochs)]
    cli = LitCLI(args=arguments, run=False)
    assert cli.trainer.max_epochs == (1 if max_epochs is None else max_epochs)
    assert cli.trainer.num_sanity_val_steps == cli.trainer.gradient_clip_val == 0
    assert cli.datamodule.train_batch_size == cli.datamodule.eval_batch_size == 8
    assert cli.model.automatic_optimization is False and cli.model.visualizer is None
    assert cli.model.teacher_weights_path == os.path.join(cli_assets, "resnet18.pth")
    assert cli.trainer.default_root_dir.endswith(os.path.join("cflow", f"{data}_{category}", "v0"))
