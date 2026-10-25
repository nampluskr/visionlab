# src/visionlab/anomaly_detection/trainers/lightning_cli.py: layered native Lightning CLI and per-run provenance

import json
import logging
import os
import sys

import torch
from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI

from visionlab.utils.io import create_versioned_dir, write_run_info

CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))
logger = logging.getLogger(__name__)


def expand_args(args, run=True):
    """Prepend shared configs and expand named model or data configs in argument order."""
    args = list(sys.argv[1:] if args is None else args)
    command = []
    if args and args[0] in LightningCLI.subcommands():
        command = [args.pop(0)]
    elif run:
        return args
    result = command + ["--config", os.path.join(CONFIG_DIR, "default.yaml"),
                        "--config", os.path.join(CONFIG_DIR, "trainer.yaml")]
    index = 0
    while index < len(args):
        token = args[index]
        key, separator, inline = token.partition("=")
        if key in ("--data", "--model") and (separator or index + 1 < len(args)):
            value = inline if separator else args[index + 1]
            folder = "data" if key == "--data" else "models"
            candidate = os.path.join(CONFIG_DIR, folder, f"{value}.yaml")
            if os.path.basename(value) == value and not os.path.splitext(value)[1] and os.path.isfile(candidate):
                result.extend(["--config", candidate])
                index += 1 if separator else 2
                continue
        result.append(token)
        index += 1
    return result


class LitCLI(LightningCLI):
    """Run native model and data subclasses with final configuration and split records."""

    def __init__(self, args=None, run=True):
        super().__init__(
            model_class=LightningModule,
            datamodule_class=LightningDataModule,
            subclass_mode_model=True,
            subclass_mode_data=True,
            args=expand_args(args, run=run),
            run=run,
            auto_configure_optimizers=False,
            save_config_kwargs={"overwrite": True},
        )

    def _parse_ckpt_path(self):
        command = self.config.get("subcommand")
        if not command:
            return
        path = self.config[command].get("ckpt_path")
        if not path or not os.path.isfile(path):
            return
        state = torch.load(path, map_location="cpu", weights_only=True)
        parameters = dict(state.get("hyper_parameters", {}))
        if not parameters:
            return
        class_path = parameters.pop("_class_path", self.config[command]["model"]["class_path"])
        parameters.pop("_instantiator", None)
        # The installed CLI merges flat hyperparameters into a subclass specification.
        restored = {command: {"model": {"class_path": class_path, "init_args": parameters}}}
        self.config = self.parser.parse_object(restored, self.config)

    def before_instantiate_classes(self):
        config = self.config[self.subcommand] if self.subcommand else self.config
        model_name = config["model"]["class_path"].rsplit(".", 1)[-1].removesuffix("LitModel").lower()
        data_name = config["data"]["class_path"].rsplit(".", 1)[-1].removesuffix("LitData").lower()
        category = config["data"]["init_args"]["category"]
        if os.path.basename(category) != category or category in (".", ".."):
            raise ValueError("category must be a single directory name")
        root = os.path.join(config["trainer"]["default_root_dir"], model_name, f"{data_name}_{category}")
        config["trainer"]["default_root_dir"] = create_versioned_dir(root)

    def instantiate_trainer(self):
        config = self._get(self.config_init, "trainer")
        parsed = self.config[self.subcommand] if self.subcommand else self.config
        for key, value in self.model.trainer_arguments.items():
            if config.get(key) != value:
                logger.info("Overriding %s from %s with %s for %s", key, config.get(key), value, type(self.model).__name__)
            config[key] = value
            parsed["trainer"][key] = value
        if config.get("enable_checkpointing") is False:
            config["callbacks"] = [callback for callback in (config.get("callbacks") or [])
                                   if not isinstance(callback, ModelCheckpoint)]
            parsed["trainer"]["callbacks"] = [callback for callback in (parsed["trainer"].get("callbacks") or [])
                                            if not callback["class_path"].endswith(".ModelCheckpoint")]
        else:
            directory = os.path.join(config["default_root_dir"], "weights", "lightning")
            for callback in config.get("callbacks") or []:
                if isinstance(callback, ModelCheckpoint) and callback.dirpath is None:
                    callback.dirpath = directory
            for callback in parsed["trainer"].get("callbacks") or []:
                if callback["class_path"].endswith(".ModelCheckpoint") and callback["init_args"].get("dirpath") is None:
                    callback["init_args"]["dirpath"] = directory
        return super().instantiate_trainer()

    def _run_subcommand(self, subcommand):
        super()._run_subcommand(subcommand)
        self.record_run()

    def record_run(self):
        if not self.trainer.is_global_zero:
            return
        config = self.config[self.subcommand] if self.subcommand else self.config
        resolved = json.loads(self._parser(self.subcommand).dump(config, format="json", skip_none=False))
        checkpoint = self.trainer.checkpoint_callback
        checkpoint_path = None if checkpoint is None else checkpoint.best_model_path or checkpoint.last_model_path or None
        if not checkpoint_path:
            checkpoint_path = self.trainer.ckpt_path
        write_run_info(
            run_dir=self.trainer.default_root_dir,
            config=resolved,
            datamodule=self.datamodule,
            teacher_path=self.model.teacher_weights_path,
            seed=config["seed_everything"],
            checkpoint_path=checkpoint_path,
        )
