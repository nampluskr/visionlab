# tests/common/test_cli.py: pytest native CLI layering, provenance and checkpoint execution contracts

import json
import os
import runpy
import socket

import pytest

from visionlab.anomaly_detection.trainers.lightning_cli import CONFIG_DIR, LitCLI, expand_args
from visionlab.utils import io as io_utils

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def winclip_cli_assets(tmp_path, monkeypatch):
    """Provide only a width-64 two-layer CLIP and a temporary teacher map for new WinCLIP CLI tests."""
    import torch
    from visionlab.anomaly_detection.models.winclip import model as winclip_module

    helpers = runpy.run_path(os.path.join(PROJECT_ROOT, "tests", "18_winclip", "test_winclip.py"))
    factory = helpers["make_small_clip"]
    weights = os.path.join(tmp_path, "clip.bin")
    torch.save(factory().state_dict(), weights)
    mapping = os.path.join(tmp_path, "paths.toml")
    with open(mapping, "w", encoding="utf-8", newline="\n") as stream:
        stream.write('[teachers]\nwinclip_clip = "' + weights.replace("\\", "/") + '"\n')
    monkeypatch.setattr(io_utils, "PATHS_FILE", mapping)
    monkeypatch.setattr(winclip_module, "CLIP", factory)
    return weights


@pytest.mark.parametrize("data,category", (("mvtec", "bottle"), ("visa", "candle")))
def test_winclip_config_merging_and_output_name(tmp_path, winclip_cli_assets, data, category):
    """Merge data before zero-shot model config without changing shared batch sizes or run naming."""
    import yaml

    with open(os.path.join(CONFIG_DIR, "models", "winclip.yaml"), encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    assert config["model"]["class_path"] == "visionlab.anomaly_detection.models.winclip.lightning_model.WinClipLitModel"
    assert config["model"]["init_args"] == {
        "class_name": None, "k_shot": 0, "scales": [2, 3], "few_shot_source": None, "visualizer": False,
    }
    assert config["trainer"] == {} and "data" not in config
    expanded = expand_args(["--data", data, "--model", "winclip"], run=False)
    configs = [expanded[index + 1] for index, value in enumerate(expanded) if value == "--config"]
    assert configs == [os.path.join(CONFIG_DIR, "default.yaml"), os.path.join(CONFIG_DIR, "trainer.yaml"),
                       os.path.join(CONFIG_DIR, "data", f"{data}.yaml"),
                       os.path.join(CONFIG_DIR, "models", "winclip.yaml")]
    cli = LitCLI(args=["--data", data, "--model", "winclip", "--trainer.default_root_dir", os.path.join(tmp_path, "runs"),
                       "--trainer.accelerator", "cpu", "--trainer.devices", "1", "--trainer.logger", "false"], run=False)
    assert cli.datamodule.train_batch_size == cli.datamodule.eval_batch_size == 8
    assert cli.model.teacher_weights_path == winclip_cli_assets
    assert cli.model.visualizer is None and not cli.model.is_setup
    assert cli.model.learning_type.value == "zero_shot" and cli.model.trainer_arguments == {}
    assert cli.trainer.default_root_dir.endswith(os.path.join("winclip", f"{data}_{category}", "v0"))


def test_winclip_train_entry_validates_then_tests(tmp_path, monkeypatch, winclip_cli_assets):
    """Run the real train entry on tiny CPU data and verify validate, export and test without fit or checkpoints."""
    from PIL import Image

    root = os.path.join(tmp_path, "data")
    for group in ("train/good", "test/good", "test/defect", "ground_truth/defect"):
        os.makedirs(os.path.join(root, "bottle", group))
    for index in range(2):
        for group, value in (("train/good", 40), ("test/good", 80), ("test/defect", 180)):
            Image.new("RGB", (24, 24), color=(value + index, 100, 120)).save(
                os.path.join(root, "bottle", group, f"{index}.png"),
            )
        Image.new("L", (24, 24), color=255).save(
            os.path.join(root, "bottle", "ground_truth", "defect", f"{index}_mask.png"),
        )
    events = []
    run_subcommand = LitCLI._run_subcommand

    def observe_dispatch(cli, subcommand):
        validate, test = cli.trainer.validate, cli.trainer.test

        def capture_validate(*args, **kwargs):
            events.append("validate")
            return validate(*args, **kwargs)

        def capture_test(*args, **kwargs):
            events.append("test")
            assert os.path.isfile(os.path.join(cli.trainer.default_root_dir, "export", "model.pt"))
            return test(*args, **kwargs)

        def forbid_fit(*args, **kwargs):
            raise AssertionError("Zero-shot train must never call fit")

        # Install instance spies after CLI parsing so Trainer method signatures remain intact.
        monkeypatch.setattr(cli.trainer, "validate", capture_validate)
        monkeypatch.setattr(cli.trainer, "test", capture_test)
        monkeypatch.setattr(cli.trainer, "fit", forbid_fit)
        return run_subcommand(cli, subcommand)

    monkeypatch.setattr(LitCLI, "_run_subcommand", observe_dispatch)
    entry = runpy.run_path(os.path.join(PROJECT_ROOT, "evaluation", "visionlab", "train.py"))
    cli = entry["main"](["train", "--data", "mvtec", "--model", "winclip", "--data.init_args.root", root,
                         "--data.init_args.val_split_mode", "same_as_test", "--trainer.accelerator", "cpu",
                         "--trainer.devices", "1", "--trainer.logger", "false", "--trainer.enable_progress_bar", "false",
                         "--trainer.enable_model_summary", "false", "--trainer.default_root_dir", os.path.join(tmp_path, "runs")])
    assert events == ["validate", "test"] and cli.trainer.optimizers == []
    assert set(cli.model.last_metrics) == {"val", "test"}
    assert not any(name.endswith(".ckpt") for _, _, files in os.walk(cli.trainer.default_root_dir) for name in files)
    with open(os.path.join(cli.trainer.default_root_dir, "run.json"), encoding="utf-8") as stream:
        record = json.load(stream)
    assert record["checkpoint"] is None and record["teacher_weights"]["path"] == winclip_cli_assets


def test_winclip_cli_preserves_one_class_and_direct_subcommand_dispatch(monkeypatch):
    """Keep the parent's fit path for ONE_CLASS and all direct subcommands while switching only zero/few-shot train."""
    from types import SimpleNamespace
    from lightning.pytorch.cli import LightningCLI
    from visionlab.anomaly_detection.common.learning_type import LearningType

    calls = []

    def parent_run(self, command):
        calls.append(("parent", command))

    monkeypatch.setattr(LightningCLI, "_run_subcommand", parent_run)
    for learning_type in LearningType:
        for train_and_test in (False, True):
            for command in ("fit", "validate", "test", "predict"):
                calls.clear()
                fake = SimpleNamespace(
                    train_and_test=train_and_test, model=SimpleNamespace(learning_type=learning_type), datamodule=object(),
                    trainer=SimpleNamespace(validate=lambda **kwargs: calls.append(("validate", kwargs))),
                    config={command: {"ckpt_path": "resume.ckpt"}}, record_run=lambda: calls.append(("record",)),
                )
                # A real uninitialized CLI instance preserves super() binding without constructing old models.
                cli = object.__new__(LitCLI)
                cli.__dict__.update(fake.__dict__)
                cli._run_subcommand(command)
                if train_and_test and command == "fit" and learning_type in (LearningType.ZERO_SHOT, LearningType.FEW_SHOT):
                    assert calls[0] == ("validate", {"model": fake.model, "datamodule": fake.datamodule,
                                                      "ckpt_path": "resume.ckpt"})
                else:
                    assert calls[0] == ("parent", command)
                assert calls[-1] == ("record",)


@pytest.fixture(scope="module")
def dsr_cli_weights(tmp_path_factory):
    """Create a synthetic full-size VQ state for DSR CLI construction without external weights."""
    import torch
    from visionlab.anomaly_detection.models.dsr.model import DsrModel

    path = os.path.join(str(tmp_path_factory.mktemp("dsr_cli")), "vq.pckl")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        torch.save(DsrModel().discrete_latent_model.state_dict(), path)
    return path


@pytest.mark.parametrize("data,category", (("mvtec", "bottle"), ("visa", "candle")))
@pytest.mark.parametrize("max_epochs", (None, 5))
def test_dsr_config_merging_and_output_name(tmp_path, monkeypatch, dsr_cli_weights, data, category, max_epochs):
    """Keep shared batches and epochs after the DSR config while preserving explicit epoch overrides."""
    import yaml

    mapping = os.path.join(tmp_path, "paths.toml")
    with open(mapping, "w", encoding="utf-8", newline="\n") as stream:
        stream.write('[teachers]\ndsr_vq = "' + dsr_cli_weights.replace("\\", "/") + '"\n')
    monkeypatch.setattr(io_utils, "PATHS_FILE", mapping)
    with open(os.path.join(CONFIG_DIR, "models", "dsr.yaml"), encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    assert config["model"]["class_path"] == "visionlab.anomaly_detection.models.dsr.lightning_model.DsrLitModel"
    assert config["model"]["init_args"] == {"latent_anomaly_strength": 0.2, "upsampling_train_ratio": 0.7,
                                             "visualizer": False}
    assert config["trainer"] == {"num_sanity_val_steps": 0} and "data" not in config
    expanded = expand_args(["--data", data, "--model", "dsr"], run=False)
    configs = [expanded[index + 1] for index, value in enumerate(expanded) if value == "--config"]
    assert configs == [os.path.join(CONFIG_DIR, "default.yaml"), os.path.join(CONFIG_DIR, "trainer.yaml"),
                       os.path.join(CONFIG_DIR, "data", f"{data}.yaml"), os.path.join(CONFIG_DIR, "models", "dsr.yaml")]
    arguments = ["--data", data, "--model", "dsr", "--trainer.num_sanity_val_steps", "2",
                 "--trainer.default_root_dir", os.path.join(tmp_path, "runs"), "--trainer.accelerator", "cpu",
                 "--trainer.devices", "1", "--trainer.logger", "false"]
    if max_epochs is not None:
        arguments += ["--trainer.max_epochs", str(max_epochs)]
    cli = LitCLI(args=arguments, run=False)
    assert cli.trainer.max_epochs == (10 if max_epochs is None else max_epochs)
    assert cli.trainer.num_sanity_val_steps == cli.config.trainer.num_sanity_val_steps == 0
    assert cli.datamodule.train_batch_size == cli.datamodule.eval_batch_size == 8
    assert cli.model.automatic_optimization is False and cli.model.visualizer is None
    assert cli.model.teacher_weights_path == dsr_cli_weights
    assert cli.model.pre_processor.mean is cli.model.pre_processor.std is None
    assert not hasattr(cli.model, "second_phase")
    assert cli.trainer.default_root_dir.endswith(os.path.join("dsr", f"{data}_{category}", "v0"))


@pytest.mark.parametrize("data,category", (("mvtec", "bottle"), ("visa", "candle")))
@pytest.mark.parametrize("max_epochs", (None, 5))
def test_ganomaly_config_merging_and_output_name(tmp_path, monkeypatch, data, category, max_epochs):
    """Merge GANomaly's data batches and manual trainer settings while retaining shared or explicit epoch budgets."""
    import yaml

    monkeypatch.setattr(io_utils, "PATHS_FILE", os.path.join(tmp_path, "absent.toml"))
    with open(os.path.join(CONFIG_DIR, "models", "ganomaly.yaml"), encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    assert config["model"]["class_path"] == "visionlab.anomaly_detection.models.ganomaly.lightning_model.GanomalyLitModel"
    assert config["model"]["init_args"] == {
        "batch_size": 32, "n_features": 64, "latent_vec_size": 100, "extra_layers": 0,
        "add_final_conv_layer": True, "wadv": 1, "wcon": 50, "wenc": 1, "lr": 0.0002,
        "beta1": 0.5, "beta2": 0.999, "visualizer": False,
    }
    assert config["data"]["init_args"] == {"train_batch_size": 32, "eval_batch_size": 32}
    assert config["trainer"] == {"gradient_clip_val": 0, "num_sanity_val_steps": 0}
    expanded = expand_args(["--data", data, "--model", "ganomaly"], run=False)
    configs = [expanded[index + 1] for index, value in enumerate(expanded) if value == "--config"]
    assert configs == [os.path.join(CONFIG_DIR, "default.yaml"), os.path.join(CONFIG_DIR, "trainer.yaml"),
                       os.path.join(CONFIG_DIR, "data", f"{data}.yaml"), os.path.join(CONFIG_DIR, "models", "ganomaly.yaml")]
    arguments = ["--data", data, "--model", "ganomaly", "--model.init_args.n_features", "4",
                 "--model.init_args.latent_vec_size", "16", "--trainer.num_sanity_val_steps", "2",
                 "--trainer.gradient_clip_val", "5", "--trainer.default_root_dir", os.path.join(tmp_path, "runs"),
                 "--trainer.accelerator", "cpu", "--trainer.devices", "1", "--trainer.logger", "false"]
    if max_epochs is not None:
        arguments += ["--trainer.max_epochs", str(max_epochs)]
    cli = LitCLI(args=arguments, run=False)
    assert cli.trainer.max_epochs == (10 if max_epochs is None else max_epochs)
    assert cli.trainer.gradient_clip_val == cli.config.trainer.gradient_clip_val == 0
    assert cli.trainer.num_sanity_val_steps == cli.config.trainer.num_sanity_val_steps == 0
    assert cli.datamodule.train_batch_size == cli.datamodule.eval_batch_size == 32
    assert cli.model.real_label.shape == cli.model.fake_label.shape == (32,)
    assert cli.model.model.generator.encoder1.final_conv_layer.in_channels == \
        cli.model.model.generator.decoder.latent_input[0].out_channels == cli.model.n_features * 2**5 == 128
    assert cli.model.n_features == 4 and cli.model.latent_vec_size == 16
    assert cli.model.automatic_optimization is False and cli.model.visualizer is None
    assert cli.model.teacher_weights_path is None
    assert cli.trainer.default_root_dir.endswith(os.path.join("ganomaly", f"{data}_{category}", "v0"))


@pytest.fixture(scope="module")
def csflow_cli_weights(tmp_path_factory):
    """Create one full torchvision teacher file for the small CSFlow CLI cases."""
    import torch
    from torchvision.models import efficientnet_b5

    path = os.path.join(str(tmp_path_factory.mktemp("csflow_cli")), "efficientnet_b5.pth")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        torch.save(efficientnet_b5(weights=None).state_dict(), path)
    return path


@pytest.mark.parametrize("data,category", (("mvtec", "bottle"), ("visa", "candle")))
@pytest.mark.parametrize("max_epochs", (None, 5))
def test_csflow_config_merging_and_output_name(tmp_path, monkeypatch, csflow_cli_weights, data, category, max_epochs):
    """Merge shared epochs and fixed CSFlow clipping after each data config while retaining explicit budgets."""
    import yaml

    mapping = os.path.join(tmp_path, "paths.toml")
    with open(mapping, "w", encoding="utf-8", newline="\n") as stream:
        stream.write('[teachers]\ncsflow_efficientnet_b5 = "' + csflow_cli_weights.replace("\\", "/") + '"\n')
    monkeypatch.setattr(io_utils, "PATHS_FILE", mapping)
    with open(os.path.join(CONFIG_DIR, "models", "csflow.yaml"), encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    assert config["model"]["class_path"] == "visionlab.anomaly_detection.models.csflow.lightning_model.CsflowLitModel"
    assert config["model"]["init_args"] == {"cross_conv_hidden_channels": 1024, "n_coupling_blocks": 4,
                                             "clamp": 3, "num_channels": 3, "visualizer": False}
    assert config["trainer"] == {"gradient_clip_val": 1, "num_sanity_val_steps": 0}
    expanded = expand_args(["--data", data, "--model", "csflow"], run=False)
    configs = [expanded[index + 1] for index, value in enumerate(expanded) if value == "--config"]
    assert configs == [os.path.join(CONFIG_DIR, "default.yaml"), os.path.join(CONFIG_DIR, "trainer.yaml"),
                       os.path.join(CONFIG_DIR, "data", f"{data}.yaml"), os.path.join(CONFIG_DIR, "models", "csflow.yaml")]
    arguments = ["--data", data, "--model", "csflow", "--model.init_args.cross_conv_hidden_channels", "16",
                 "--model.init_args.n_coupling_blocks", "2", "--trainer.num_sanity_val_steps", "2",
                 "--trainer.gradient_clip_val", "5", "--trainer.default_root_dir", os.path.join(tmp_path, "runs"),
                 "--trainer.accelerator", "cpu", "--trainer.devices", "1", "--trainer.logger", "false"]
    if max_epochs is not None:
        arguments += ["--trainer.max_epochs", str(max_epochs)]
    cli = LitCLI(args=arguments, run=False)
    assert cli.trainer.max_epochs == (10 if max_epochs is None else max_epochs)
    assert cli.trainer.gradient_clip_val == cli.config.trainer.gradient_clip_val == 1
    assert cli.trainer.num_sanity_val_steps == cli.config.trainer.num_sanity_val_steps == 0
    assert cli.model.model.input_dims == (3, 256, 256) and cli.model.cross_conv_hidden_channels == 16
    assert cli.model.n_coupling_blocks == 2 and cli.model.clamp == cli.model.num_channels == 3
    assert cli.model.automatic_optimization is True and cli.model.visualizer is None
    assert cli.model.teacher_weights_path == csflow_cli_weights
    assert cli.datamodule.train_batch_size == cli.datamodule.eval_batch_size == 8
    assert cli.trainer.default_root_dir.endswith(os.path.join("csflow", f"{data}_{category}", "v0"))


@pytest.mark.parametrize("data,category", (("mvtec", "bottle"), ("visa", "candle")))
@pytest.mark.parametrize("max_epochs", (None, 5))
def test_uflow_config_merging_and_output_name(tmp_path, cli_assets, data, category, max_epochs):
    """Merge UFlow after each dataset with shared epochs, fixed input size and its own output folder."""
    import yaml

    with open(os.path.join(CONFIG_DIR, "models", "uflow.yaml"), encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    assert config["model"]["class_path"] == "visionlab.anomaly_detection.models.uflow.lightning_model.UflowLitModel"
    assert config["model"]["init_args"] == {
        "backbone": "resnet18", "flow_steps": 4, "affine_clamp": 2.0,
        "affine_subnet_channels_ratio": 1.0, "permute_soft": False, "visualizer": False,
    }
    assert config["trainer"] == {"num_sanity_val_steps": 0}
    expanded = expand_args(["--data", data, "--model", "uflow"], run=False)
    configs = [expanded[index + 1] for index, value in enumerate(expanded) if value == "--config"]
    assert configs == [os.path.join(CONFIG_DIR, "default.yaml"), os.path.join(CONFIG_DIR, "trainer.yaml"),
                       os.path.join(CONFIG_DIR, "data", f"{data}.yaml"), os.path.join(CONFIG_DIR, "models", "uflow.yaml")]
    arguments = ["--data", data, "--model", "uflow", "--model.init_args.flow_steps", "1",
                 "--trainer.num_sanity_val_steps", "2",
                 "--trainer.default_root_dir", os.path.join(tmp_path, "runs"), "--trainer.accelerator", "cpu",
                 "--trainer.devices", "1", "--trainer.logger", "false"]
    if max_epochs is not None:
        arguments += ["--trainer.max_epochs", str(max_epochs)]
    cli = LitCLI(args=arguments, run=False)
    assert cli.trainer.max_epochs == (10 if max_epochs is None else max_epochs)
    assert cli.trainer.num_sanity_val_steps == 0
    assert cli.model.model.input_size == (256, 256) and cli.model.flow_steps == 1
    assert cli.model.backbone == "resnet18" and cli.model.affine_clamp == 2.0
    assert cli.model.affine_subnet_channels_ratio == 1.0 and cli.model.permute_soft is False
    assert cli.model.automatic_optimization is True and cli.model.visualizer is None
    assert cli.model.teacher_weights_path == os.path.join(cli_assets, "resnet18.pth")
    assert cli.datamodule.train_batch_size == cli.datamodule.eval_batch_size == 8
    assert cli.trainer.default_root_dir.endswith(os.path.join("uflow", f"{data}_{category}", "v0"))


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
