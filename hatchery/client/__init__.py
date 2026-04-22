# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Hatchery Python client SDK.

Lightweight HTTP client for the Hatchery training API. No torch or
transformers dependency — just httpx and pydantic.

    pip install hatchery-client

Usage::

    from hatchery.client import HatcheryClient

    client = HatcheryClient(base_url="http://127.0.0.1:8420", token="dev")
    tc = client.create_lora_training_client("Qwen/Qwen2-0.5B-Instruct", rank=32)

    tc.forward_backward(data).result()
    tc.optim_step(learning_rate=1e-4).result()
"""

from hatchery.client._client import (
    HatcheryClient,
    HatcheryClientError,
    RequestFailedError,
    SamplingClient,
    TrainingClient,
)

__all__ = [
    "HatcheryClient",
    "HatcheryClientError",
    "RequestFailedError",
    "SamplingClient",
    "TrainingClient",
]
