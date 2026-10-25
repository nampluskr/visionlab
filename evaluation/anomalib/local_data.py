# evaluation/anomalib/local_data.py: anomalib datamodule subclasses that read local dataset folders

import os
from pathlib import Path

from anomalib.data import MVTecAD, Visa


class LocalMVTecAD(MVTecAD):
    """MVTecAD that checks local folders instead of downloading."""

    def prepare_data(self):
        for folder in ("train/good", "test", "ground_truth"):
            path = os.path.join(self.root, self.category, folder)
            if not os.path.isdir(path):
                raise FileNotFoundError(path)


class LocalVisa(Visa):
    """Visa that writes its split outside the original dataset folder (D-17)."""

    def __init__(self, split_root=None, **kwargs):
        super().__init__(**kwargs)
        # anomalib's Visa joins split_root with "/", so it must stay a Path.
        self.split_root = Path(split_root)

    def prepare_data(self):
        if os.path.isdir(os.path.join(self.split_root, self.category)):
            return
        category_dir = os.path.join(self.root, self.category)
        if not os.path.isdir(category_dir):
            raise FileNotFoundError(category_dir)
        # Copy images and binarize masks with anomalib's own 1cls split.
        self.apply_cls1_split()
