"""Base classes for all anomaly components.

This module provides the foundational classes used across anomalib's model
components. These include:

- ``AnomalibModule``: Base class for all anomaly detection modules
- ``BufferListMixin``: Mixin for managing lists of model buffers
- ``DynamicBufferMixin``: Mixin for handling dynamic model buffers
- ``MemoryBankMixin``: Mixin for models requiring feature memory banks

Example:
    >>> from anomalib.models.components.base import AnomalibModule
    >>> class MyAnomalyModel(AnomalibModule):
    ...     def __init__(self):
    ...         super().__init__()
    ...     def forward(self, x):
    ...         return x
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .dynamic_buffer import DynamicBufferMixin

__all__ = ["DynamicBufferMixin"]
