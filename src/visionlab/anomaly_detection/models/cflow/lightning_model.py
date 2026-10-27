# src/visionlab/anomaly_detection/models/cflow/lightning_model.py: native Lightning CFLOW manual training and evaluation

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import math
import os

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.trainer.states import TrainerFn
from torch import optim
from torch.nn import functional as F

from visionlab.anomaly_detection.common.evaluator import Evaluator
from visionlab.anomaly_detection.common.learning_type import LearningType
from visionlab.anomaly_detection.common.lightning_visualizer import LitVisualizer
from visionlab.anomaly_detection.common.post_processor import PostProcessor
from visionlab.anomaly_detection.common.pre_processor import IMAGENET_MEAN, IMAGENET_STD, PreProcessor
from visionlab.anomaly_detection.common.visualizer import ImageVisualizer
from visionlab.anomaly_detection.models.cflow.model import CflowModel
from visionlab.anomaly_detection.models.cflow.utils import get_logp, positional_encoding_2d
from visionlab.utils.io import load_teacher_weights, save_export


class CflowLitModel(LightningModule):
    """Train conditional flows with manual fiber steps and evaluate pure CFLOW."""

    def __init__(self, backbone="wide_resnet50_2", layers=("layer2", "layer3", "layer4"), pre_trained=True,
                 fiber_batch_size=64, decoder="freia-cflow", condition_vector=128, coupling_blocks=8,
                 clamp_alpha=1.9, permute_soft=False, lr=0.0001,
                 pre_processor=True, post_processor=True, evaluator=True, visualizer=True):
        super().__init__()
        self.pre_processor = self._resolve(
            pre_processor, PreProcessor, lambda: PreProcessor(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        )
        self.post_processor = self._resolve(post_processor, PostProcessor, PostProcessor)
        self.evaluator = self._resolve(evaluator, Evaluator, Evaluator.default)
        self.visualizer = self._resolve(visualizer, ImageVisualizer, ImageVisualizer)
        self.model = CflowModel(
            backbone=backbone, pre_trained=pre_trained, layers=layers, fiber_batch_size=fiber_batch_size,
            decoder=decoder, condition_vector=condition_vector, coupling_blocks=coupling_blocks,
            clamp_alpha=clamp_alpha, permute_soft=permute_soft,
        )
        self.automatic_optimization = False
        self.learning_rate = lr
        # D-19: The encoder exposes the timm backbone through its feature_extractor.
        self.teacher_weights_path = load_teacher_weights(self.model.encoder, backbone) if pre_trained else None
        self.model_args = {"backbone": backbone, "layers": list(layers), "pre_trained": False,
                           "fiber_batch_size": fiber_batch_size, "decoder": decoder,
                           "condition_vector": condition_vector, "coupling_blocks": coupling_blocks,
                           "clamp_alpha": clamp_alpha, "permute_soft": permute_soft}
        self.save_hyperparameters(dict(self.model_args, pre_trained=pre_trained, lr=lr,
                                       _class_path=f"{type(self).__module__}.{type(self).__name__}"))
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

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        images, _ = batch
        if self.pre_processor is not None:
            images = self.pre_processor(images)
        activation = self.model.encoder(images)
        avg_loss = torch.zeros([1], dtype=torch.float64).to(images.device)

        height = []
        width = []
        for layer_idx, layer in enumerate(self.model.pool_layers):
            encoder_activations = activation[layer].detach()
            batch_size, dim_feature_vector, im_height, im_width = encoder_activations.size()
            image_size = im_height * im_width
            embedding_length = batch_size * image_size
            height.append(im_height)
            width.append(im_width)
            # Replace einops repeat with the same batch tiling as the pure model.
            pos_encoding = (
                positional_encoding_2d(self.model.condition_vector, im_height, im_width)
                .unsqueeze(0).repeat(batch_size, 1, 1, 1)
            ).to(images.device)
            # Replace einops rearrange with the same BHWxP ordering.
            c_r = pos_encoding.permute(0, 2, 3, 1).reshape(-1, self.model.condition_vector)
            # Replace einops rearrange with the same BHWxC ordering.
            e_r = encoder_activations.permute(0, 2, 3, 1).reshape(-1, dim_feature_vector)
            perm = torch.randperm(embedding_length)
            decoder = self.model.decoders[layer_idx].to(images.device)
            fiber_batches = embedding_length // self.model.fiber_batch_size
            if fiber_batches <= 0:
                msg = "Make sure we have enough fibers, otherwise decrease N or batch-size!"
                raise ValueError(msg)

            for batch_num in range(fiber_batches):
                opt.zero_grad()
                if batch_num < (fiber_batches - 1):
                    idx = torch.arange(batch_num * self.model.fiber_batch_size,
                                       (batch_num + 1) * self.model.fiber_batch_size)
                else:
                    idx = torch.arange(batch_num * self.model.fiber_batch_size, embedding_length)
                c_p = c_r[perm[idx]]
                e_p = e_r[perm[idx]]
                p_u, log_jac_det = decoder(e_p, [c_p])
                decoder_log_prob = get_logp(dim_feature_vector, p_u, log_jac_det)
                log_prob = decoder_log_prob / dim_feature_vector
                loss = -F.logsigmoid(log_prob)
                self.manual_backward(loss.mean())
                opt.step()
                avg_loss += loss.sum()

        # Match the reference epoch aggregation, where every step counts once.
        self.log("train_loss", avg_loss.item(), on_epoch=True, prog_bar=True, logger=True, batch_size=1)
        return {"loss": avg_loss}

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
                model=self.model, model_name="cflow", model_args=dict(self.model_args),
                pre_processor=self.pre_processor, post_processor=self.post_processor,
            )

    @property
    def trainer_arguments(self):
        return {"gradient_clip_val": 0, "num_sanity_val_steps": 0}

    @property
    def learning_type(self):
        return LearningType.ONE_CLASS

    def configure_optimizers(self):
        decoders_parameters = []
        for decoder_idx in range(len(self.model.pool_layers)):
            decoders_parameters.extend(list(self.model.decoders[decoder_idx].parameters()))
        return optim.Adam(params=decoders_parameters, lr=self.learning_rate)

    def configure_callbacks(self):
        return [] if self.visualizer is None else [LitVisualizer(self.visualizer)]
