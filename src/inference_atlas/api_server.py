"""Optional FastAPI server for InferenceAtlas endpoints."""

from __future__ import annotations

import os

from inference_atlas.api_models import (
    AIAssistRequest,
    AIAssistResponse,
    CatalogBrowseResponse,
    CatalogRankingRequest,
    CatalogRankingResponse,
    CopilotTurnRequest,
    CopilotTurnResponse,
    InvoiceAnalysisResponse,
    LLMPlanningRequest,
    LLMPlanningResponse,
    ReportGenerateRequest,
    ReportGenerateResponse,
)
from inference_atlas.api_service import (
    run_ai_assist,
    run_browse_catalog,
    run_copilot_turn,
    run_generate_report,
    run_invoice_analyze,
    run_plan_llm,
    run_rank_catalog,
)


def create_app():
    """Create FastAPI app lazily so base package has no hard FastAPI dependency."""
    try:
        from fastapi import FastAPI, File, HTTPException, Response, UploadFile
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "FastAPI is not installed. Install with: "
            "pip install 'fastapi>=0.110,<1.0' 'uvicorn>=0.30,<1.0'"
        ) from exc

    app = FastAPI(title="InferenceAtlas API", version="0.1.1")

    origins_raw = os.getenv("INFERENCE_ATLAS_CORS_ORIGINS", "*")
    allow_origins = [origin.strip() for origin in origins_raw.split(",") if origin.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/v1/ai/copilot", response_model=CopilotTurnResponse)
    def copilot_turn(payload: CopilotTurnRequest) -> CopilotTurnResponse:
        try:
            return run_copilot_turn(payload)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/v1/plan/llm", response_model=LLMPlanningResponse)
    def plan_llm(payload: LLMPlanningRequest) -> LLMPlanningResponse:
        try:
            return run_plan_llm(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/v1/rank/catalog", response_model=CatalogRankingResponse)
    def rank_catalog(payload: CatalogRankingRequest) -> CatalogRankingResponse:
        try:
            return run_rank_catalog(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/api/v1/catalog", response_model=CatalogBrowseResponse)
    def browse_catalog(
        workload_type: str | None = None,
        provider: str | None = None,
        unit_name: str | None = None,
    ) -> CatalogBrowseResponse:
        try:
            return run_browse_catalog(
                workload_type=workload_type,
                provider=provider,
                unit_name=unit_name,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/v1/invoice/analyze", response_model=InvoiceAnalysisResponse)
    async def analyze_invoice(file: UploadFile = File(...)) -> InvoiceAnalysisResponse:
        try:
            content = await file.read()
            return run_invoice_analyze(content)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/v1/ai/assist", response_model=AIAssistResponse)
    def ai_assist(payload: AIAssistRequest) -> AIAssistResponse:
        try:
            return run_ai_assist(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/v1/report/generate", response_model=ReportGenerateResponse)
    def generate_report(payload: ReportGenerateRequest) -> ReportGenerateResponse:
        try:
            return run_generate_report(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/v1/report/download")
    def download_report(payload: ReportGenerateRequest) -> Response:
        try:
            report = run_generate_report(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        safe_id = report.report_id
        if report.output_format == "pdf":
            if not report.pdf_base64:
                raise HTTPException(status_code=500, detail="PDF output was not generated.")
            import base64

            body = base64.b64decode(report.pdf_base64.encode("ascii"))
            media_type = "application/pdf"
            filename = f"{safe_id}.pdf"
        elif report.output_format == "html":
            if report.html is None:
                raise HTTPException(status_code=500, detail="HTML output was not generated.")
            body = report.html.encode("utf-8")
            media_type = "text/html; charset=utf-8"
            filename = f"{safe_id}.html"
        else:
            body = report.markdown.encode("utf-8")
            media_type = "text/markdown; charset=utf-8"
            filename = f"{safe_id}.md"

        headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
        return Response(content=body, media_type=media_type, headers=headers)

    return app
