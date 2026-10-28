# evaluation/visionlab/train.py: offline native Lightning entry point with learning-type-aware train command

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from visionlab.utils.io import block_network

block_network()

import logging

from visionlab.anomaly_detection.trainers.lightning_cli import LitCLI


def main(args=None):
    """Run a native subcommand or train and test the current model in one process."""
    args = list(sys.argv[1:] if args is None else args)
    train_and_test = bool(args and args[0] == "train")
    if train_and_test:
        args[0] = "fit"
    logging.basicConfig(level=logging.INFO)
    cli = LitCLI(args=args, train_and_test=train_and_test)
    if train_and_test:
        cli.trainer.test(model=cli.model, datamodule=cli.datamodule, ckpt_path=None)
        cli.record_run()
    return cli


if __name__ == "__main__":
    main()
