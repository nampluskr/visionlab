# src/visionlab/anomaly_detection/models/efficient_ad/lightning_model.py: native Lightning EfficientAD training and evaluation

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import math
import os

import torch
import tqdm
from lightning.pytorch import LightningModule
from lightning.pytorch.trainer.states import TrainerFn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms.v2 import CenterCrop, Compose, RandomGrayscale, Resize, ToTensor

from visionlab.anomaly_detection.common.evaluator import Evaluator
from visionlab.anomaly_detection.common.learning_type import LearningType
from visionlab.anomaly_detection.common.lightning_visualizer import LitVisualizer
from visionlab.anomaly_detection.common.post_processor import PostProcessor
from visionlab.anomaly_detection.common.pre_processor import PreProcessor
from visionlab.anomaly_detection.common.visualizer import ImageVisualizer
from visionlab.anomaly_detection.models.efficient_ad.model import EfficientAdModel, EfficientAdModelSize, reduce_tensor_elems
from visionlab.utils.io import load_pretrained_teacher, save_export


class EfficientAdLitModel(LightningModule):
    """Connect the original EfficientAD algorithm to native Lightning and pure processing."""

    def __init__(self, imagenet_dir="./datasets/imagenette", teacher_out_channels=384,
                 model_size=EfficientAdModelSize.S, lr=0.0001, weight_decay=0.00001,
                 padding=False, pad_maps=True, pre_processor=True, post_processor=True,
                 evaluator=True, visualizer=True):
        super().__init__()
        self.pre_processor = self._resolve(pre_processor, PreProcessor, PreProcessor)
        self.post_processor = self._resolve(post_processor, PostProcessor, PostProcessor)
        self.evaluator = self._resolve(evaluator, Evaluator, Evaluator.default)
        self.visualizer = self._resolve(visualizer, ImageVisualizer, ImageVisualizer)
        self.imagenet_dir = os.fspath(imagenet_dir)
        if not isinstance(model_size, EfficientAdModelSize):
            model_size = EfficientAdModelSize(model_size)
        self.model_size = model_size
        self.model = EfficientAdModel(teacher_out_channels=teacher_out_channels, model_size=model_size,
                                      padding=padding, pad_maps=pad_maps)
        # D-19: Inject the local teacher here so resume never reloads it in on_train_start.
        self.teacher_weights_path = load_pretrained_teacher(self.model.teacher, f"efficient_ad_{model_size.value}")
        self.batch_size = 1
        self.lr = lr
        self.weight_decay = weight_decay
        self.model_args = {"teacher_out_channels": teacher_out_channels, "model_size": model_size.value,
                           "padding": padding, "pad_maps": pad_maps}
        self.save_hyperparameters({"imagenet_dir": self.imagenet_dir, **self.model_args,
                                  "lr": lr, "weight_decay": weight_decay,
                                  "_class_path": f"{type(self).__module__}.{type(self).__name__}"})
        self.metric_reasons = {}
        self.last_metrics = {}

    @staticmethod
    def _resolve(value, component_type, factory):
        if value is True:
            return factory()
        if value is False:
            return None
        if isinstance(value, component_type):
            return value
        raise TypeError(f"Expected bool or {component_type.__name__}, received {type(value).__name__}")

    def forward(self, image):
        if self.pre_processor is not None:
            image = self.pre_processor(image)
        return self.model(image)

    def prepare_imagenette_data(self, image_size):
        self.data_transforms_imagenet = Compose([
            Resize((image_size[0] * 2, image_size[1] * 2)),
            RandomGrayscale(p=0.3),
            CenterCrop((image_size[0], image_size[1])),
            ToTensor(),
        ])
        if not os.path.isdir(self.imagenet_dir):
            raise FileNotFoundError(self.imagenet_dir)
        imagenet_dataset = ImageFolder(self.imagenet_dir, transform=self.data_transforms_imagenet)
        self.imagenet_loader = DataLoader(imagenet_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        self.imagenet_iterator = iter(self.imagenet_loader)

    @torch.no_grad()
    def teacher_channel_mean_std(self, dataloader):
        arrays_defined = False
        n = None
        channel_sum = None
        channel_sum_sqr = None
        for image, _ in tqdm.tqdm(dataloader, desc="Calculate teacher channel mean & std", position=0, leave=True):
            y = self.model.teacher(image.to(self.device))
            if not arrays_defined:
                _, num_channels, _, _ = y.shape
                n = torch.zeros((num_channels,), dtype=torch.int64, device=y.device)
                channel_sum = torch.zeros((num_channels,), dtype=torch.float32, device=y.device)
                channel_sum_sqr = torch.zeros((num_channels,), dtype=torch.float32, device=y.device)
                arrays_defined = True
            n += y[:, 0].numel()
            channel_sum += torch.sum(y, dim=[0, 2, 3])
            channel_sum_sqr += torch.sum(y**2, dim=[0, 2, 3])
        if n is None:
            raise ValueError("The value of 'n' cannot be None.")
        channel_mean = channel_sum / n
        channel_std = (torch.sqrt((channel_sum_sqr / n) - (channel_mean**2))).float()[None, :, None, None]
        channel_mean = channel_mean.float()[None, :, None, None]
        return {"mean": channel_mean, "std": channel_std}

    @torch.no_grad()
    def map_norm_quantiles(self, dataloader):
        maps_st = []
        maps_ae = []
        for image, target in tqdm.tqdm(dataloader, desc="Calculate Validation Dataset Quantiles", position=0, leave=True):
            for img, label in zip(image, target["label"], strict=True):
                if label == 0:
                    map_st, map_ae = self.model.get_maps(img.to(self.device), normalize=False)
                    maps_st.append(map_st)
                    maps_ae.append(map_ae)
        qa_st, qb_st = self._get_quantiles_of_maps(maps_st)
        qa_ae, qb_ae = self._get_quantiles_of_maps(maps_ae)
        return {"qa_st": qa_st, "qa_ae": qa_ae, "qb_st": qb_st, "qb_ae": qb_ae}

    def _get_quantiles_of_maps(self, maps):
        maps_flat = reduce_tensor_elems(torch.cat(maps))
        qa = torch.quantile(maps_flat, q=0.9).to(self.device)
        qb = torch.quantile(maps_flat, q=0.995).to(self.device)
        return qa, qb

    def on_train_start(self):
        if self.trainer.datamodule.train_batch_size != 1:
            raise ValueError("train_batch_size for EfficientAd should be 1.")
        if self.pre_processor is not None and self.pre_processor.mean is not None:
            raise ValueError("Transforms for EfficientAd should not contain Normalize.")
        image, _ = next(iter(self.trainer.train_dataloader))
        self.prepare_imagenette_data(image.shape[-2:])
        if not self.model.is_set(self.model.mean_std):
            self.model.mean_std.update(self.teacher_channel_mean_std(self.trainer.datamodule.train_dataloader()))

    def training_step(self, batch, batch_idx):
        image, _ = batch
        if self.pre_processor is not None:
            image = self.pre_processor(image)
        try:
            batch_imagenet = next(self.imagenet_iterator)[0].to(self.device)
        except StopIteration:
            self.imagenet_iterator = iter(self.imagenet_loader)
            batch_imagenet = next(self.imagenet_iterator)[0].to(self.device)
        loss_st, loss_ae, loss_stae = self.model(batch=image, batch_imagenet=batch_imagenet)
        loss = loss_st + loss_ae + loss_stae
        self.log("train_st", loss_st.item(), on_epoch=True, prog_bar=True, logger=True, batch_size=1)
        self.log("train_ae", loss_ae.item(), on_epoch=True, prog_bar=True, logger=True, batch_size=1)
        self.log("train_stae", loss_stae.item(), on_epoch=True, prog_bar=True, logger=True, batch_size=1)
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True, batch_size=1)
        return {"loss": loss}

    def _outputs(self, batch):
        image, target = batch
        prediction = self(image)
        return {"image": image, "gt_label": target.get("label"), "gt_mask": target.get("mask"),
                "pred_score": prediction.pred_score, "anomaly_map": prediction.anomaly_map,
                "image_path": target.get("image_path"), "sample_id": target.get("sample_id")}

    def _post_process(self, outputs):
        if self.post_processor is not None:
            processed = self.post_processor(outputs["pred_score"], outputs["anomaly_map"])
            outputs = dict(outputs)
            outputs["pred_label"] = processed["pred_label"]
            outputs["pred_mask"] = processed["pred_mask"]
            outputs["normalized_pred_score"] = processed["pred_score"]
            outputs["normalized_anomaly_map"] = processed["anomaly_map"]
        return outputs

    def _fit_validation(self):
        return (self._trainer is not None and self.trainer.state.fn == TrainerFn.FITTING
                and not self.trainer.sanity_checking)

    def on_validation_start(self):
        # D-8: Standalone evaluation reads the calibration saved during fit.
        if self._fit_validation():
            self.model.quantiles.update(self.map_norm_quantiles(self.trainer.datamodule.val_dataloader()))

    def validation_step(self, batch, batch_idx):
        outputs = self._outputs(batch)
        if self.post_processor is not None and self._fit_validation():
            self.post_processor.update(outputs)
        if self.evaluator is not None:
            self.evaluator.update("val", outputs)
        return outputs

    def test_step(self, batch, batch_idx):
        outputs = self._post_process(self._outputs(batch))
        if self.evaluator is not None:
            self.evaluator.update("test", outputs)
        return outputs

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self._post_process(self._outputs(batch))

    def on_validation_epoch_start(self):
        if self.evaluator is not None:
            self.evaluator.reset("val")

    def on_test_epoch_start(self):
        if self.evaluator is not None:
            self.evaluator.reset("test")

    def _log_metrics(self, stage, prefix):
        if self.evaluator is None:
            return
        values, reasons = self.evaluator.compute(stage)
        self.last_metrics[stage] = dict(values)
        for name, value in values.items():
            key = prefix + name
            self.log(key, value, on_epoch=True, logger=True)
            if math.isnan(value):
                self.metric_reasons[key] = reasons[name]
            else:
                self.metric_reasons.pop(key, None)
        self.evaluator.reset(stage)

    def on_validation_epoch_end(self):
        if self.post_processor is not None and self._fit_validation():
            self.post_processor.finalize()
        self._log_metrics("val", "val_")

    def on_test_epoch_end(self):
        self._log_metrics("test", "")

    def on_save_checkpoint(self, checkpoint):
        checkpoint["post_processor"] = None if self.post_processor is None else self.post_processor.state_dict()

    def on_load_checkpoint(self, checkpoint):
        state = checkpoint.get("post_processor")
        if self.post_processor is not None and state is not None:
            self.post_processor.load_state_dict(state)

    def on_fit_end(self):
        if self.trainer.is_global_zero:
            save_export(path=os.path.join(self.trainer.default_root_dir, "export", "model.pt"),
                        model=self.model, model_name="efficient_ad", model_args=self.model_args,
                        pre_processor=self.pre_processor, post_processor=self.post_processor)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.model.student.parameters()) + list(self.model.ae.parameters()),
                                     lr=self.lr, weight_decay=self.weight_decay)
        if self.trainer.max_epochs < 0 and self.trainer.max_steps < 0:
            raise ValueError("A finite number of steps or epochs must be defined")
        if self.trainer.max_epochs < 0:
            num_steps = self.trainer.max_steps
        elif self.trainer.max_steps < 0:
            num_steps = self.trainer.max_epochs * len(self.trainer.datamodule.train_dataloader())
        else:
            num_steps = min(self.trainer.max_steps,
                            self.trainer.max_epochs * len(self.trainer.datamodule.train_dataloader()))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.95 * num_steps), gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def configure_callbacks(self):
        return [] if self.visualizer is None else [LitVisualizer(self.visualizer)]

    @property
    def trainer_arguments(self):
        return {"num_sanity_val_steps": 0}

    @property
    def learning_type(self):
        return LearningType.ONE_CLASS
