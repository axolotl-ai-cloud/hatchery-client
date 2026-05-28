# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Unit tests for the Tinker-parity additions.

Three new client surfaces are exercised here in isolation (no
hatchery-core dependency) via ``respx`` httpx mocking:

* :meth:`HatcheryClient.create_sampling_client`
* :meth:`TrainingClient.forward`
* :meth:`SamplingClient.compute_logprobs`

The end-to-end round-trip against a real local-dev gateway lives in
hatchery-core's integration tests; here we assert the wire-level
request shape and response decoding only.
"""

from __future__ import annotations

import threading

import pytest
import respx
from httpx import Response

from hatchery.client import HatcheryClient, SamplingClient, TrainingClient


@pytest.fixture
def base_url() -> str:
    return "http://test.hatchery"


@pytest.fixture
async def client(base_url):
    c = HatcheryClient(base_url=base_url, token="test-token")
    try:
        yield c
    finally:
        await c.aclose()


def _make_sampling_client(
    client: HatcheryClient, *, sampling_session_id: str
) -> SamplingClient:
    """Construct a SamplingClient bypassing the network round-trip.

    Mirrors what ``HatcheryClient.create_sampling_client`` does after
    the server confirms the session — handy for testing
    :meth:`SamplingClient.compute_logprobs` in isolation.
    """
    sc = SamplingClient.__new__(SamplingClient)
    sc._client = client
    sc.model_id = "smp_test"
    sc.sampling_session_id = sampling_session_id
    sc._seq_lock = threading.Lock()
    sc._seq_id = 0
    return sc


def _make_training_client(client: HatcheryClient) -> TrainingClient:
    tc = TrainingClient.__new__(TrainingClient)
    tc._client = client
    tc.model_id = "mdl_test"
    tc.base_model = "base/test"
    tc.lora_rank = 8
    tc._seq_lock = threading.Lock()
    tc._seq_id = 0
    return tc


# ── SamplingClient.compute_logprobs ─────────────────────────────────────


@respx.mock
async def test_compute_logprobs_posts_degenerate_sample(base_url, client):
    """``compute_logprobs`` issues a max_tokens=1 + prompt_logprobs=True
    sample call and returns the response's ``prompt_logprobs`` list.

    Matches the official Tinker SDK's own implementation strategy
    (``sampling_client.py:399-406`` in ``tinker==0.22.2``).
    """
    asample = respx.post(f"{base_url}/api/v1/asample").mock(
        return_value=Response(
            200, json={"future_id": "fut-cl", "request_id": "fut-cl"}
        ),
    )
    respx.post(f"{base_url}/api/v1/retrieve_future").mock(
        return_value=Response(
            200,
            json={"type": "sample", "prompt_logprobs": [None, -0.1, -0.2, -0.3]},
        ),
    )

    sc = _make_sampling_client(client, sampling_session_id="samp-smp_test-0-abcdef")

    result = await sc.compute_logprobs_async([10, 11, 12, 13])

    assert result == [None, -0.1, -0.2, -0.3]
    assert asample.called
    import json

    body = json.loads(asample.calls.last.request.read())
    # The strictly-required body fields the upstream-parity contract depends on:
    assert body["sampling_session_id"] == "samp-smp_test-0-abcdef"
    assert body["num_samples"] == 1
    assert body["prompt_logprobs"] is True
    assert body["sampling_params"]["max_tokens"] == 1
    assert body["prompt"]["chunks"][0]["tokens"] == [10, 11, 12, 13]


@respx.mock
async def test_compute_logprobs_returns_empty_list_when_server_omits_field(
    base_url, client
):
    """If the gateway returns a response without ``prompt_logprobs``
    (e.g., an older worker that ignores ``include_prompt_logprobs``),
    we return ``[]`` rather than ``None`` so callers can call ``len()``
    safely.
    """
    respx.post(f"{base_url}/api/v1/asample").mock(
        return_value=Response(200, json={"future_id": "f", "request_id": "f"}),
    )
    respx.post(f"{base_url}/api/v1/retrieve_future").mock(
        return_value=Response(200, json={"type": "sample"}),  # no prompt_logprobs
    )
    sc = _make_sampling_client(client, sampling_session_id="samp-smp_test-0-deadbeef")
    assert await sc.compute_logprobs_async([1, 2, 3]) == []


# ── TrainingClient.forward ──────────────────────────────────────────────


@respx.mock
async def test_forward_posts_to_forward_route_not_forward_only(base_url, client):
    """``TrainingClient.forward`` must hit ``/api/v1/forward`` (the
    per-position logprobs path), NOT ``/api/v1/forward_only`` (scalar
    loss only). Wrapping the wrong route silently swaps semantics.
    """
    fwd = respx.post(f"{base_url}/api/v1/forward").mock(
        return_value=Response(
            200, json={"future_id": "fut-fw", "request_id": "fut-fw"}
        ),
    )
    forward_only = respx.post(
        f"{base_url}/api/v1/forward_only"
    )  # not mocked → asserts unused
    # Mock the actual wire shape Hatchery emits — tinker's
    # ForwardBackwardOutput (loss_fn_outputs with TensorData logprobs),
    # not the worker's internal per_datum_logprobs key.
    respx.post(f"{base_url}/api/v1/retrieve_future").mock(
        return_value=Response(
            200,
            json={
                "loss_fn_output_type": "cross_entropy",
                "loss_fn_outputs": [
                    {
                        "logprobs": {
                            "data": [0.0, -0.5, -0.3],
                            "dtype": "float32",
                            "shape": [3],
                        }
                    }
                ],
                "metrics": {},
            },
        ),
    )

    tc = _make_training_client(client)
    datum = {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
        "loss_fn_inputs": {},
    }
    fut = await tc.forward_async([datum], loss_fn="cross_entropy")
    result = await fut.result_async()

    assert fwd.called
    assert not forward_only.called
    # Tinker-shaped ForwardBackwardOutput: extract per-datum logprobs
    # from loss_fn_outputs[i].logprobs.data.
    assert result["loss_fn_output_type"] == "cross_entropy"
    assert result["loss_fn_outputs"][0]["logprobs"]["data"] == [0.0, -0.5, -0.3]

    import json

    body = json.loads(fwd.calls.last.request.read())
    assert body["forward_input"]["loss_fn"] == "cross_entropy"
    assert body["model_id"] == "mdl_test"


# ── HatcheryClient.create_sampling_client ───────────────────────────────


def test_create_sampling_client_requires_at_least_one_arg():
    """Mirrors ``tinker.ServiceClient.create_sampling_client``'s
    ``ValueError("Either model_path or base_model must be provided")``.
    """
    client = HatcheryClient(base_url="http://test", token="t")
    try:
        with pytest.raises(ValueError, match="model_path|base_model"):
            client.create_sampling_client()
    finally:
        client.close()


@respx.mock
async def test_create_sampling_client_base_model(base_url, client):
    """Base-model path: posts to /create_sampling_session, returns a
    SamplingClient wired to the minted sampling_session_id.
    """
    create_route = respx.post(f"{base_url}/api/v1/create_sampling_session").mock(
        return_value=Response(
            200,
            json={
                "future_id": "fut-cs",
                "request_id": "fut-cs",
                "model_id": "smp_xyz",
            },
        ),
    )
    respx.post(f"{base_url}/api/v1/retrieve_future").mock(
        return_value=Response(
            200,
            json={
                "type": "create_sampling_session",
                "sampling_session_id": "samp-smp_xyz-0-cafef00d",
                "model_id": "smp_xyz",
                "base_model": "Qwen/Qwen2-0.5B",
                "expires_at": 1234567890.0,
            },
        ),
    )

    sc = await client.create_sampling_client_async(base_model="Qwen/Qwen2-0.5B")

    assert isinstance(sc, SamplingClient)
    assert sc.model_id == "smp_xyz"
    assert sc.sampling_session_id == "samp-smp_xyz-0-cafef00d"

    import json

    body = json.loads(create_route.calls.last.request.read())
    assert body == {"base_model": "Qwen/Qwen2-0.5B"}


@respx.mock
async def test_create_sampling_client_model_path_wins(base_url, client):
    """When both are passed, both go on the wire — the server resolves
    precedence (model_path wins) per the Tinker spec.
    """
    create_route = respx.post(f"{base_url}/api/v1/create_sampling_session").mock(
        return_value=Response(
            200,
            json={"future_id": "f", "request_id": "f", "model_id": "smp_a"},
        ),
    )
    respx.post(f"{base_url}/api/v1/retrieve_future").mock(
        return_value=Response(
            200,
            json={
                "type": "create_sampling_session",
                "sampling_session_id": "samp-smp_a-0-1234",
                "model_id": "smp_a",
                "base_model": "Qwen/Qwen2-0.5B",
                "expires_at": 0,
            },
        ),
    )

    await client.create_sampling_client_async(
        base_model="Qwen/Qwen2-0.5B",
        model_path="tinker://mdl_parent/checkpoints/step-5",
        ttl_seconds=300,
    )

    import json

    body = json.loads(create_route.calls.last.request.read())
    assert body["base_model"] == "Qwen/Qwen2-0.5B"
    assert body["model_path"] == "tinker://mdl_parent/checkpoints/step-5"
    assert body["ttl_seconds"] == 300
