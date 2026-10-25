# src/visionlab/anomaly_detection/common/learning_type.py: anomaly model learning types

from enum import Enum


class LearningType(str, Enum):
    """Name the supported anomaly learning paradigms."""

    ONE_CLASS = "one_class"
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
