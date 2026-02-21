from __future__ import annotations

import io

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from inference_atlas.api_server import create_app


def _client() -> TestClient:
    app = create_app()
    return TestClient(app)


def test_healthz() -> None:
    client = _client()
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_catalog_browse_endpoint() -> None:
    client = _client()
    response = client.get("/api/v1/catalog", params={"workload_type": "llm"})
    assert response.status_code == 200
    body = response.json()
    assert body["total"] >= 1
    assert all(row["workload_type"] == "llm" for row in body["rows"])


def test_rank_catalog_endpoint_includes_relaxation_fields() -> None:
    client = _client()
    payload = {
        "workload_type": "vision",
        "allowed_providers": [],
        "unit_name": "audio_min",
        "monthly_usage": 10.0,
        "monthly_budget_max_usd": 0.0,
        "top_k": 5,
        "confidence_weighted": True,
        "comparator_mode": "listed",
        "throughput_aware": False,
        "peak_to_avg": 2.5,
        "util_target": 0.75,
        "strict_capacity_check": False,
    }
    response = client.post("/api/v1/rank/catalog", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "relaxation_applied" in body
    assert "relaxation_steps" in body
    assert "exclusion_breakdown" in body


def test_plan_llm_endpoint() -> None:
    client = _client()
    payload = {
        "tokens_per_day": 5_000_000,
        "model_bucket": "7b",
        "provider_ids": [],
        "peak_to_avg": 2.5,
        "util_target": 0.75,
        "beta": 0.08,
        "alpha": 1.0,
        "autoscale_inefficiency": 1.15,
        "monthly_budget_max_usd": 0.0,
        "output_token_ratio": 0.3,
        "top_k": 3,
    }
    response = client.post("/api/v1/plan/llm", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["plans"]
    assert body["plans"][0]["rank"] == 1


def test_ai_assist_endpoint() -> None:
    client = _client()
    payload = {
        "message": "cheapest speech to text provider",
        "context": {"workload_type": "speech_to_text", "providers": [], "recent_results": None},
    }
    response = client.post("/api/v1/ai/assist", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "reply" in body
    assert isinstance(body["reply"], str)


def test_invoice_analyze_endpoint() -> None:
    client = _client()
    content = (
        "provider,workload_type,usage_qty,usage_unit,amount_usd\n"
        "openai,speech_to_text,120,audio_min,12.0\n"
    ).encode("utf-8")
    response = client.post(
        "/api/v1/invoice/analyze",
        files={"file": ("invoice.csv", io.BytesIO(content), "text/csv")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["grand_total"] > 0
    assert body["line_items"]
