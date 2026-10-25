# tests/common/test_history.py: pytest history tag selection and backward-compatible command defaults

import os
import runpy
import sys
from types import SimpleNamespace

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_history_selects_requested_tag(tmp_path, monkeypatch):
    """Read each selected scalar tag while retaining the default loss history."""
    namespace = runpy.run_path(os.path.join(PROJECT_ROOT, "evaluation", "anomalib", "compare_history.py"))
    read_history = namespace["read_history"]
    directory = os.path.join(tmp_path, "lightning_logs", "version_0")
    os.makedirs(directory)
    with open(os.path.join(directory, "events.out.tfevents.fixture"), "wb"):
        pass

    class Events:
        """Expose two distinct epoch tags in the event accumulator contract."""

        def Reload(self):
            pass

        def Tags(self):
            return {"scalars": ["epoch", "train_loss_epoch", "train_st_epoch"], "tensors": []}

        def Scalars(self, tag):
            offset = {"epoch": 0, "train_loss_epoch": 10, "train_st_epoch": 20}[tag]
            return [SimpleNamespace(step=index, value=offset + index, wall_time=float(index)) for index in range(2)]

    monkeypatch.setitem(read_history.__globals__, "EventAccumulator", lambda path, size_guidance: Events())
    assert read_history(str(tmp_path)) == {0: 10, 1: 11}
    assert read_history(str(tmp_path), tag="train_st_epoch") == {0: 20, 1: 21}
    assert read_history(str(tmp_path), tag="absent") == {}


@pytest.mark.parametrize("tags", ([], ["train_st_epoch", "train_ae_epoch", "train_stae_epoch", "train_loss_epoch"]))
def test_history_command_default_and_multiple_tags(tags, monkeypatch, capsys):
    """Print a comparison table for every requested tag and preserve existing invocations."""
    namespace = runpy.run_path(os.path.join(PROJECT_ROOT, "evaluation", "anomalib", "compare_history.py"))
    calls = []

    def history(run, tag):
        calls.append(tag)
        return {0: 1.0}

    monkeypatch.setitem(namespace["main"].__globals__, "read_history", history)
    arguments = ["compare_history.py", "--reference", "reference", "--native", "native"]
    monkeypatch.setattr(sys, "argv", arguments + (["--tags"] + tags if tags else []))
    namespace["main"]()
    expected = tags or ["train_loss_epoch"]
    assert calls == [tag for tag in expected for _ in range(2)]
    output = capsys.readouterr().out
    assert output.count("| Epoch |") == len(expected)
    assert all(f"Tag: {tag}" in output for tag in expected)
