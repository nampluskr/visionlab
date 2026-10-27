# src/visionlab/anomaly_detection/models/draem/lightning_model.py: native Lightning DRAEM synthesis training and evaluation

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import math
import os

from lightning.pytorch import LightningModule
from lightning.pytorch.trainer.states import TrainerFn
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR

from visionlab.anomaly_detection.common.evaluator import Evaluator
from visionlab.anomaly_detection.common.learning_type import LearningType
from visionlab.anomaly_detection.common.lightning_visualizer import LitVisualizer
from visionlab.anomaly_detection.common.perlin import PerlinAnomalyGenerator
from visionlab.anomaly_detection.common.post_processor import PostProcessor
from visionlab.anomaly_detection.common.pre_processor import PreProcessor
from visionlab.anomaly_detection.common.visualizer import ImageVisualizer
from visionlab.anomaly_detection.models.draem.model import DraemModel
from visionlab.anomaly_detection.models.draem.loss import DraemLoss
from visionlab.utils.io import save_export


class DraemLitModel(LightningModule):
    """Train synthetic image anomalies and evaluate pure DRAEM with native Lightning hooks."""

    def __init__(self, enable_sspcab=False, sspcab_lambda=0.1, anomaly_source_path=None, beta=(0.1, 1.0),
                 pre_processor=True, post_processor=True, evaluator=True, visualizer=True):
        super().__init__()
        self.pre_processor = self._resolve(pre_processor, PreProcessor, PreProcessor)
        self.post_processor = self._resolve(post_processor, PostProcessor, PostProcessor)
        self.evaluator = self._resolve(evaluator, Evaluator, Evaluator.default)
        self.visualizer = self._resolve(visualizer, ImageVisualizer, ImageVisualizer)
        # Untyped YAML lists must use the original tuple branch for random blending.
        beta = tuple(beta) if isinstance(beta, list) else beta
        self.augmenter = PerlinAnomalyGenerator(anomaly_source_path=anomaly_source_path, blend_factor=beta)
        self.model = DraemModel(sspcab=enable_sspcab)
        self.loss = DraemLoss()
        self.sspcab = enable_sspcab
        if self.sspcab:
            self.sspcab_activations = {}
            self.setup_sspcab()
            self.sspcab_loss = nn.MSELoss()
            self.sspcab_lambda = sspcab_lambda
        self.teacher_weights_path = None
        self.save_hyperparameters({"enable_sspcab": enable_sspcab, "sspcab_lambda": sspcab_lambda,
                                  "anomaly_source_path": anomaly_source_path, "beta": beta,
                                  "_class_path": f"{type(self).__module__}.{type(self).__name__}"})
        self.metric_reasons = {}
        self.last_metrics = {}

    def setup_sspcab(self):
        def get_activation(name):
            """Create a hook retaining the original activation tensor."""
            def hook(module, inputs, output):
                """Retain the activation without detaching its gradient."""
                self.sspcab_activations[name] = output
            return hook

        self.model.reconstructive_subnetwork.encoder.mp4.register_forward_hook(get_activation("input"))
        self.model.reconstructive_subnetwork.encoder.block5.register_forward_hook(get_activation("output"))

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

    def training_step(self, batch, batch_idx):
        input_image, _ = batch
        if self.pre_processor is not None:
            input_image = self.pre_processor(input_image)
        augmented_image, anomaly_mask = self.augmenter(input_image)
        reconstruction, prediction = self.model(augmented_image)
        loss = self.loss(input_image, reconstruction, anomaly_mask, prediction)
        if self.sspcab:
            loss += self.sspcab_lambda * self.sspcab_loss(
                self.sspcab_activations["input"], self.sspcab_activations["output"],
            )
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
                model=self.model,
                model_name="draem",
                model_args={"sspcab": self.sspcab},
                pre_processor=self.pre_processor,
                post_processor=self.post_processor,
            )

    @property
    def trainer_arguments(self):
        return {"gradient_clip_val": 0, "num_sanity_val_steps": 0}

    @property
    def learning_type(self):
        return LearningType.ONE_CLASS

    def configure_optimizers(self):
        optimizer = optim.Adam(params=self.model.parameters(), lr=0.0001)
        scheduler = MultiStepLR(optimizer, milestones=[400, 600], gamma=0.1)
        return [optimizer], [scheduler]

    def configure_callbacks(self):
        return [] if self.visualizer is None else [LitVisualizer(self.visualizer)]
