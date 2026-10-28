# src/visionlab/anomaly_detection/models/ganomaly/lightning_model.py: native Lightning GANomaly manual training and image evaluation

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import math
import os

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.trainer.states import TrainerFn
from torch import optim

from visionlab.anomaly_detection.common.evaluator import Evaluator
from visionlab.anomaly_detection.common.learning_type import LearningType
from visionlab.anomaly_detection.common.lightning_visualizer import LitVisualizer
from visionlab.anomaly_detection.common.metric import AUROC, F1
from visionlab.anomaly_detection.common.post_processor import PostProcessor
from visionlab.anomaly_detection.common.pre_processor import IMAGENET_MEAN, IMAGENET_STD, PreProcessor
from visionlab.anomaly_detection.common.visualizer import ImageVisualizer
from visionlab.anomaly_detection.models.ganomaly.loss import DiscriminatorLoss, GeneratorLoss
from visionlab.anomaly_detection.models.ganomaly.model import GanomalyModel
from visionlab.utils.io import save_export

# input size is the anomalib default pre-processor size (256); the original reads it from the pre-processor
GANOMALY_INPUT_SIZE = (256, 256)


class GanomalyLitModel(LightningModule):
    """Train GANomaly with two manual Adam optimizers and evaluate image-only anomaly scores."""

    def __init__(self, batch_size=32, n_features=64, latent_vec_size=100, extra_layers=0,
                 add_final_conv_layer=True, wadv=1, wcon=50, wenc=1, lr=0.0002, beta1=0.5, beta2=0.999,
                 pre_processor=True, post_processor=True, evaluator=True, visualizer=True):
        super().__init__()
        self.pre_processor = self._resolve(
            pre_processor, PreProcessor, lambda: PreProcessor(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        )
        self.post_processor = self._resolve(post_processor, PostProcessor, PostProcessor)
        self.evaluator = self._resolve(evaluator, Evaluator, self.configure_evaluator)
        self.visualizer = self._resolve(visualizer, ImageVisualizer, ImageVisualizer)
        self.n_features = n_features
        self.latent_vec_size = latent_vec_size
        self.extra_layers = extra_layers
        self.add_final_conv_layer = add_final_conv_layer
        self.real_label = torch.ones(size=(batch_size,), dtype=torch.float32)
        self.fake_label = torch.zeros(size=(batch_size,), dtype=torch.float32)
        self.min_scores = torch.tensor(float("inf"), dtype=torch.float32)
        self.max_scores = torch.tensor(float("-inf"), dtype=torch.float32)
        self.model = GanomalyModel(
            input_size=GANOMALY_INPUT_SIZE, num_input_channels=3, n_features=self.n_features,
            latent_vec_size=self.latent_vec_size, extra_layers=self.extra_layers,
            add_final_conv_layer=self.add_final_conv_layer,
        )
        self.generator_loss = GeneratorLoss(wadv, wcon, wenc)
        self.discriminator_loss = DiscriminatorLoss()
        self.automatic_optimization = False
        self.learning_rate = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.teacher_weights_path = None
        self.save_hyperparameters({"batch_size": batch_size, "n_features": n_features,
                                  "latent_vec_size": latent_vec_size, "extra_layers": extra_layers,
                                  "add_final_conv_layer": add_final_conv_layer, "wadv": wadv, "wcon": wcon,
                                  "wenc": wenc, "lr": lr, "beta1": beta1, "beta2": beta2,
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

    @staticmethod
    def configure_evaluator():
        # D-8 adds validation raw AUROC to the original image-only test metrics.
        return Evaluator(val_metrics=[("image_AUROC", AUROC(), "pred_score", "gt_label")],
                         test_metrics=[("image_AUROC", AUROC(), "pred_score", "gt_label"),
                                       ("image_F1Score", F1(), "pred_label", "gt_label")])

    def forward(self, image):
        if self.pre_processor is not None:
            image = self.pre_processor(image)
        return self.model(image)

    def training_step(self, batch, batch_idx):
        d_opt, g_opt = self.optimizers()
        image, _ = batch
        if self.pre_processor is not None:
            image = self.pre_processor(image)
        padded, fake, latent_i, latent_o = self.model(image)
        pred_real, _ = self.model.discriminator(padded)
        pred_fake, _ = self.model.discriminator(fake)
        g_loss = self.generator_loss(latent_i, latent_o, padded, fake, pred_real, pred_fake)
        g_opt.zero_grad()
        self.manual_backward(g_loss, retain_graph=True)
        g_opt.step()
        pred_fake, _ = self.model.discriminator(fake.detach())
        d_loss = self.discriminator_loss(pred_real, pred_fake)
        d_opt.zero_grad()
        self.manual_backward(d_loss)
        d_opt.step()
        # Match the reference epoch aggregation, where every step counts once.
        self.log_dict({"generator_loss": g_loss.item(), "discriminator_loss": d_loss.item()},
                      on_epoch=True, prog_bar=True, logger=True, batch_size=1)
        return {"generator_loss": g_loss, "discriminator_loss": d_loss}

    def configure_optimizers(self):
        optimizer_d = optim.Adam(self.model.discriminator.parameters(), lr=self.learning_rate,
                                 betas=(self.beta1, self.beta2))
        optimizer_g = optim.Adam(self.model.generator.parameters(), lr=self.learning_rate,
                                 betas=(self.beta1, self.beta2))
        return [optimizer_d, optimizer_g]

    def _reset_min_max(self):
        self.min_scores = torch.tensor(float("inf"), dtype=torch.float32)
        self.max_scores = torch.tensor(float("-inf"), dtype=torch.float32)

    def _normalize(self, scores):
        return (scores - self.min_scores.to(scores.device)) / (
            self.max_scores.to(scores.device) - self.min_scores.to(scores.device)
        )

    def on_validation_start(self):
        self._reset_min_max()

    def on_test_start(self):
        self._reset_min_max()

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
        self.max_scores = max(self.max_scores, torch.max(outputs["pred_score"]))
        self.min_scores = min(self.min_scores, torch.min(outputs["pred_score"]))
        if self.post_processor is not None and self._fit_validation():
            self.post_processor.update(outputs)
        if self.evaluator is not None:
            self.evaluator.update("val", outputs)
        return outputs

    def test_step(self, batch, batch_idx):
        outputs = self._post_process(self._outputs(batch))
        self.max_scores = max(self.max_scores, torch.max(outputs["pred_score"]))
        self.min_scores = min(self.min_scores, torch.min(outputs["pred_score"]))
        if self.evaluator is not None:
            self.evaluator.update("test", outputs)
        return outputs

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        # Normalize after evaluator and post-processor updates, matching the reference callback-before-module order.
        outputs["pred_score"] = self._normalize(outputs["pred_score"])

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        outputs["pred_score"] = self._normalize(outputs["pred_score"])

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
                model=self.model, model_name="ganomaly",
                model_args={"input_size": list(GANOMALY_INPUT_SIZE), "num_input_channels": 3,
                            "n_features": self.n_features, "latent_vec_size": self.latent_vec_size,
                            "extra_layers": self.extra_layers, "add_final_conv_layer": self.add_final_conv_layer},
                pre_processor=self.pre_processor, post_processor=self.post_processor,
            )

    @property
    def trainer_arguments(self):
        return {"gradient_clip_val": 0, "num_sanity_val_steps": 0}

    @property
    def learning_type(self):
        return LearningType.ONE_CLASS

    def configure_callbacks(self):
        return [] if self.visualizer is None else [LitVisualizer(self.visualizer)]
