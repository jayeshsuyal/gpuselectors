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


def test_quality_catalog_endpoint() -> None:
    client = _client()
    response = client.get("/api/v1/quality/catalog", params={"workload_type": "llm"})
    assert response.status_code == 200
    body = response.json()
    assert body["total"] >= 1
    assert "mapped_count" in body
    assert "unmapped_count" in body
    assert all(row["workload_type"] == "llm" for row in body["rows"])
    assert "quality_mapped" in body["rows"][0]


def test_quality_catalog_endpoint_mapped_only() -> None:
    client = _client()
    response = client.get(
        "/api/v1/quality/catalog",
        params={"workload_type": "llm", "mapped_only": "true"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total"] >= 1
    assert all(row["quality_mapped"] for row in body["rows"])


def test_quality_insights_endpoint() -> None:
    client = _client()
    response = client.get("/api/v1/quality/insights", params={"workload_type": "llm"})
    assert response.status_code == 200
    body = response.json()
    assert "points" in body
    assert "frontier_count" in body
    assert body["total_points"] >= 1
    assert body["frontier_count"] >= 1
    assert "is_pareto_frontier" in body["points"][0]


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


def test_cost_audit_endpoint() -> None:
    client = _client()
    payload = {
        "modality": "llm",
        "model_name": "Llama 3.1 70B",
        "model_precision": "fp16",
        "fine_tuned": False,
        "pricing_model": "token_api",
        "monthly_input_tokens": 500000000,
        "monthly_output_tokens": 100000000,
        "traffic_pattern": "steady",
        "workload_execution": "latency_sensitive",
        "caching_enabled": "no",
        "providers": ["openai"],
        "autoscaling": "yes",
        "quantization_applied": "no",
        "monthly_ai_spend_usd": 8000,
    }
    response = client.post("/api/v1/audit/cost", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "efficiency_score" in body
    assert "recommendations" in body
    assert "score_breakdown" in body
    assert body["score_breakdown"]["post_cap_score"] == body["efficiency_score"]
    assert "per_modality_audits" in body
    assert "hardware_recommendation" in body
    assert "pricing_model_verdict" in body
    assert "pricing_source" in body
    assert "pricing_source_provider" in body
    assert "pricing_source_gpu" in body


def test_plan_scaling_endpoint() -> None:
    client = _client()
    payload = {
        "mode": "catalog",
        "catalog_ranking": {
            "offers": [
                {
                    "rank": 1,
                    "provider": "openai",
                    "sku_name": "whisper-1",
                    "billing_mode": "per_unit",
                    "unit_price_usd": 0.006,
                    "normalized_price": 0.36,
                    "unit_name": "audio_min",
                    "confidence": "official",
                    "monthly_estimate_usd": 10.0,
                    "required_replicas": 1,
                    "capacity_check": "ok",
                    "previous_unit_price_usd": None,
                    "price_change_abs_usd": None,
                    "price_change_pct": None,
                }
            ],
            "provider_diagnostics": [],
            "excluded_count": 0,
            "warnings": [],
            "relaxation_applied": False,
            "relaxation_steps": [],
            "exclusion_breakdown": {},
        },
    }
    response = client.post("/api/v1/plan/scaling", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "catalog"
    assert body["deployment_mode"] == "serverless"
    assert body["estimated_gpu_count"] >= 0


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


def test_copilot_endpoint_accepts_frontend_shape() -> None:
    client = _client()
    payload = {"message": "Need llm under $300", "history": [], "workload_type": "llm"}
    response = client.post("/api/v1/ai/copilot", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "reply" in body
    assert "is_ready" in body


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


def test_rank_catalog_endpoint_rejects_invalid_payload() -> None:
    client = _client()
    payload = {
        "workload_type": "llm",
        "allowed_providers": [],
        "unit_name": None,
        "monthly_usage": 10.0,
        "monthly_budget_max_usd": 0.0,
        "top_k": 5,
        "confidence_weighted": True,
        "comparator_mode": "invalid_mode",
        "throughput_aware": False,
        "peak_to_avg": 2.5,
        "util_target": 0.75,
        "strict_capacity_check": False,
    }
    response = client.post("/api/v1/rank/catalog", json=payload)
    assert response.status_code == 422


def test_invoice_analyze_endpoint_requires_file() -> None:
    client = _client()
    response = client.post("/api/v1/invoice/analyze", files={})
    assert response.status_code == 422


def test_generate_report_endpoint() -> None:
    client = _client()
    payload = {
        "mode": "catalog",
        "title": "API Report",
        "catalog_ranking": {
            "offers": [
                {
                    "rank": 1,
                    "provider": "openai",
                    "sku_name": "whisper-1",
                    "billing_mode": "per_unit",
                    "unit_price_usd": 0.006,
                    "normalized_price": 0.36,
                    "unit_name": "audio_min",
                    "confidence": "official",
                    "monthly_estimate_usd": 10.0,
                    "required_replicas": None,
                    "capacity_check": "unknown",
                    "previous_unit_price_usd": None,
                    "price_change_abs_usd": None,
                    "price_change_pct": None,
                }
            ],
            "provider_diagnostics": [],
            "excluded_count": 0,
            "warnings": [],
            "relaxation_applied": False,
            "relaxation_steps": [],
            "exclusion_breakdown": {},
        },
    }
    response = client.post("/api/v1/report/generate", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "catalog"
    assert body["report_id"].startswith("rep_")
    assert "markdown" in body
    assert "charts" in body
    assert "chart_data" in body
    assert "metadata" in body
    assert "csv_exports" in body


def test_generate_report_endpoint_supports_pdf_format() -> None:
    client = _client()
    payload = {
        "mode": "catalog",
        "title": "API PDF Report",
        "output_format": "pdf",
        "catalog_ranking": {
            "offers": [
                {
                    "rank": 1,
                    "provider": "openai",
                    "sku_name": "whisper-1",
                    "billing_mode": "per_unit",
                    "unit_price_usd": 0.006,
                    "normalized_price": 0.36,
                    "unit_name": "audio_min",
                    "confidence": "official",
                    "monthly_estimate_usd": 10.0,
                    "required_replicas": None,
                    "capacity_check": "unknown",
                    "previous_unit_price_usd": None,
                    "price_change_abs_usd": None,
                    "price_change_pct": None,
                }
            ],
            "provider_diagnostics": [],
            "excluded_count": 0,
            "warnings": [],
            "relaxation_applied": False,
            "relaxation_steps": [],
            "exclusion_breakdown": {},
        },
    }
    response = client.post("/api/v1/report/generate", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["output_format"] == "pdf"
    assert body["pdf_base64"]


def _catalog_report_payload(output_format: str) -> dict:
    return {
        "mode": "catalog",
        "title": f"API {output_format.upper()} Report",
        "output_format": output_format,
        "catalog_ranking": {
            "offers": [
                {
                    "rank": 1,
                    "provider": "openai",
                    "sku_name": "whisper-1",
                    "billing_mode": "per_unit",
                    "unit_price_usd": 0.006,
                    "normalized_price": 0.36,
                    "unit_name": "audio_min",
                    "confidence": "official",
                    "monthly_estimate_usd": 10.0,
                    "required_replicas": None,
                    "capacity_check": "unknown",
                    "previous_unit_price_usd": None,
                    "price_change_abs_usd": None,
                    "price_change_pct": None,
                }
            ],
            "provider_diagnostics": [],
            "excluded_count": 0,
            "warnings": [],
            "relaxation_applied": False,
            "relaxation_steps": [],
            "exclusion_breakdown": {},
        },
    }


def test_report_download_endpoint_markdown() -> None:
    client = _client()
    response = client.post("/api/v1/report/download", json=_catalog_report_payload("markdown"))
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert "attachment; filename=" in response.headers["content-disposition"]
    assert response.text.startswith("# ")


def test_report_download_endpoint_html() -> None:
    client = _client()
    response = client.post("/api/v1/report/download", json=_catalog_report_payload("html"))
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.text.lower().startswith("<!doctype html>")


def test_report_download_endpoint_pdf() -> None:
    client = _client()
    response = client.post("/api/v1/report/download", json=_catalog_report_payload("pdf"))
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/pdf")
    assert response.content.startswith(b"%PDF")
