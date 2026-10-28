# src/visionlab/anomaly_detection/models/dsr/lightning_model.py: native Lightning DSR manual optimization and evaluation

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import math
import os

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.trainer.states import TrainerFn

from visionlab.anomaly_detection.common.evaluator import Evaluator
from visionlab.anomaly_detection.common.learning_type import LearningType
from visionlab.anomaly_detection.common.lightning_visualizer import LitVisualizer
from visionlab.anomaly_detection.common.perlin import PerlinAnomalyGenerator
from visionlab.anomaly_detection.common.post_processor import PostProcessor
from visionlab.anomaly_detection.common.pre_processor import PreProcessor
from visionlab.anomaly_detection.common.visualizer import ImageVisualizer
from visionlab.anomaly_detection.models.dsr.anomaly_generator import DsrAnomalyGenerator
from visionlab.anomaly_detection.models.dsr.loss import DsrSecondStageLoss, DsrThirdStageLoss
from visionlab.anomaly_detection.models.dsr.model import DsrModel
from visionlab.utils.io import load_pretrained_teacher, save_export

logger = logging.getLogger(__name__)


class DsrLitModel(LightningModule):
    """Train DSR with separate reconstruction and upsampling optimizers using native Lightning hooks."""

    def __init__(self, latent_anomaly_strength=0.2, upsampling_train_ratio=0.7,
                 pre_processor=True, post_processor=True, evaluator=True, visualizer=True):
        super().__init__()
        self.automatic_optimization = False
        self.upsampling_train_ratio = upsampling_train_ratio
        self.quantized_anomaly_generator = DsrAnomalyGenerator()
        self.perlin_generator = PerlinAnomalyGenerator()
        self.model = DsrModel(latent_anomaly_strength)
        self.second_stage_loss = DsrSecondStageLoss()
        self.third_stage_loss = DsrThirdStageLoss()
        self.second_phase: int
        # D-19 injects VQ weights at construction instead of on_train_start; loading consumes no RNG.
        self.teacher_weights_path = load_pretrained_teacher(self.model.discrete_latent_model, "dsr_vq")
        self.pre_processor = self._resolve(pre_processor, PreProcessor, PreProcessor)
        self.post_processor = self._resolve(post_processor, PostProcessor, PostProcessor)
        self.evaluator = self._resolve(evaluator, Evaluator, Evaluator.default)
        self.visualizer = self._resolve(visualizer, ImageVisualizer, ImageVisualizer)
        self.save_hyperparameters({"latent_anomaly_strength": latent_anomaly_strength,
                                  "upsampling_train_ratio": upsampling_train_ratio,
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

    def configure_optimizers(self):
        num_steps = max(
            self.trainer.max_steps // len(self.trainer.datamodule.train_dataloader()),
            self.trainer.max_epochs,
        )
        self.second_phase = int(num_steps * self.upsampling_train_ratio)
        anneal = int(0.8 * self.second_phase)
        optimizer_d = torch.optim.Adam(
            params=list(self.model.image_reconstruction_network.parameters())
            + list(self.model.subspace_restriction_module_hi.parameters())
            + list(self.model.subspace_restriction_module_lo.parameters())
            + list(self.model.anomaly_detection_module.parameters()),
            lr=0.0002,
        )
        scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, anneal, gamma=0.1)
        optimizer_u = torch.optim.Adam(params=self.model.upsampling_module.parameters(), lr=0.0002)
        return ({"optimizer": optimizer_d, "lr_scheduler": scheduler_d}, {"optimizer": optimizer_u})

    def on_train_epoch_start(self):
        if self.current_epoch == self.second_phase:
            logger.info("Now training upsampling module.")

    def training_step(self, batch, batch_idx):
        ph1_opt, ph2_opt = self.optimizers()
        input_image, _ = batch
        if self.pre_processor is not None:
            input_image = self.pre_processor(input_image)
        if self.current_epoch < self.second_phase:
            anomaly_mask = self.quantized_anomaly_generator.augment_batch(input_image)
            model_outputs = self.model(input_image, anomaly_mask)
            loss = self.second_stage_loss(
                model_outputs["recon_feat_hi"], model_outputs["recon_feat_lo"],
                model_outputs["embedding_bot"], model_outputs["embedding_top"], input_image,
                model_outputs["obj_spec_image"], model_outputs["anomaly_map"], model_outputs["true_anomaly_map"],
            )
            ph1_opt.zero_grad()
            self.manual_backward(loss)
            ph1_opt.step()
        else:
            input_image, anomaly_maps = self.perlin_generator(input_image)
            model_outputs = self.model(input_image)
            loss = self.third_stage_loss(model_outputs["anomaly_map"], anomaly_maps)
            ph2_opt.zero_grad()
            self.manual_backward(loss)
            ph2_opt.step()
        # Match the reference epoch aggregation, where every step counts once.
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
            save_export(
                path=os.path.join(self.trainer.default_root_dir, "export", "model.pt"),
                model=self.model, model_name="dsr",
                model_args={"latent_anomaly_strength": self.model.latent_anomaly_strength},
                pre_processor=self.pre_processor, post_processor=self.post_processor,
            )

    @property
    def trainer_arguments(self):
        return {"num_sanity_val_steps": 0}

    @property
    def learning_type(self):
        return LearningType.ONE_CLASS

    def configure_callbacks(self):
        return [] if self.visualizer is None else [LitVisualizer(self.visualizer)]
