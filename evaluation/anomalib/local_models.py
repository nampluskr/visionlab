# evaluation/anomalib/local_models.py: anomalib model subclasses for local assets and phase-specific overrides

import os

import torch
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor

from anomalib import models
from anomalib.models.image.cfa import torch_model as cfa_torch_model
from anomalib.models.image.csflow import torch_model as csflow_torch_model
from anomalib.models.image.winclip import torch_model as winclip_torch_model


# The file arguments default to None so that the classes parse without them;
# the model yaml sets the local paths.
class LocalEfficientAd(models.EfficientAd):
    """EfficientAd that loads the teacher and Imagenette from local paths."""

    def __init__(self, teacher_file=None, **kwargs):
        super().__init__(**kwargs)
        self.teacher_file = teacher_file

    def prepare_pretrained_model(self):
        # Runs first in on_train_start, before anomalib could download Imagenette.
        if not os.path.isdir(self.imagenet_dir):
            raise FileNotFoundError(self.imagenet_dir)
        if not os.path.isfile(self.teacher_file):
            raise FileNotFoundError(self.teacher_file)
        state_dict = torch.load(self.teacher_file, map_location=self.device, weights_only=True)
        self.model.teacher.load_state_dict(state_dict)


class LocalCfa(models.Cfa):
    """Cfa whose torchvision backbone is built from a local weights file."""

    def __init__(self, weights_file=None, **kwargs):
        # cfa.torch_model downloads torchvision weights by backbone name, so its
        # get_feature_extractor is replaced before the model is built.
        def get_feature_extractor(backbone, return_nodes):
            model = getattr(torchvision.models, backbone)(weights=None)
            model.load_state_dict(torch.load(weights_file, map_location="cpu", weights_only=True))
            return create_feature_extractor(model=model, return_nodes=return_nodes).eval()

        cfa_torch_model.get_feature_extractor = get_feature_extractor
        super().__init__(**kwargs)


class LocalCsflow(models.Csflow):
    """Csflow whose EfficientNet-B5 is built from a local weights file."""

    def __init__(self, weights_file=None, **kwargs):
        # csflow.torch_model calls efficientnet_b5(weights=IMAGENET1K_V1), which downloads.
        def efficientnet_b5(weights=None):
            model = torchvision.models.efficientnet_b5(weights=None)
            model.load_state_dict(torch.load(weights_file, map_location="cpu", weights_only=True))
            return model

        csflow_torch_model.efficientnet_b5 = efficientnet_b5
        super().__init__(**kwargs)
        # the extractor constructor dry-runs in train mode on uninitialized memory and corrupts BatchNorm running statistics; keep the train-mode dry-run so the random number order matches the original, then restore the local weights (Phase 15, user decision 2026-10-28).
        self.model.feature_extractor.feature_extractor.backbone.load_state_dict(
            torch.load(weights_file, map_location="cpu", weights_only=True), strict=True,
        )


class LocalDsr(models.Dsr):
    """Dsr that loads the pretrained VQ-VAE from a local file."""

    def __init__(self, weights_file=None, **kwargs):
        super().__init__(**kwargs)
        self.weights_file = weights_file

    def prepare_pretrained_model(self):
        # The original reads ./pre_trained/ and downloads when the file is missing.
        if not os.path.isfile(self.weights_file):
            raise FileNotFoundError(self.weights_file)
        return self.weights_file


class LocalWinClip(models.WinClip):
    """WinClip whose CLIP checkpoint is a local file."""

    def __init__(self, weights_file=None, **kwargs):
        # winclip.torch_model reads the module constant PRETRAINED; open_clip accepts a file path.
        winclip_torch_model.PRETRAINED = weights_file
        super().__init__(**kwargs)


class LocalFre(models.Fre):
    """Fre using the shared ten-epoch budget by user decision."""

    @property
    def trainer_arguments(self):
        arguments = super().trainer_arguments.copy()
        # The original pins max_epochs=220; this phase uses the shared 10-epoch budget by user decision.
        arguments["max_epochs"] = 10
        return arguments
