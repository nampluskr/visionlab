# src/visionlab/anomaly_detection/common/batch.py: pure dataclasses for image batches and inference outputs

from dataclasses import field, make_dataclass


# Runtime fields keep dataclass support without annotation syntax.
Batch = make_dataclass(
    "Batch",
    [(name, object, field(default=None)) for name in (
        "image", "gt_label", "gt_mask", "image_path", "mask_path", "sample_id",
    )],
    namespace={"__module__": __name__, "__doc__": "Store image tensors and sample metadata."},
)

InferenceBatch = make_dataclass(
    "InferenceBatch",
    [(name, object, field(default=None)) for name in (
        "pred_score", "anomaly_map", "pred_label", "pred_mask",
    )],
    namespace={"__module__": __name__, "__doc__": "Store model predictions without framework dependencies."},
)
