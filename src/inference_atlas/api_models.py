"""Pydantic API contracts for backend endpoints."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RiskBreakdown(BaseModel):
    model_config = ConfigDict(extra="ignore")

    risk_overload: float
    risk_complexity: float
    total_risk: float


class ProviderDiagnostic(BaseModel):
    model_config = ConfigDict(extra="ignore")

    provider: str
    status: Literal["included", "excluded", "not_selected"]
    reason: str


class RankedPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    rank: int
    provider_id: str
    provider_name: str
    offering_id: str
    billing_mode: str
    confidence: str
    monthly_cost_usd: float
    score: float
    utilization_at_peak: Optional[float]
    risk: RiskBreakdown
    assumptions: dict[str, float]
    why: str


class LLMPlanningRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    tokens_per_day: float = Field(gt=0)
    model_bucket: str = Field(min_length=1)
    provider_ids: list[str] = Field(default_factory=list)
    peak_to_avg: float = Field(gt=0, default=2.5)
    util_target: float = Field(gt=0, lt=1, default=0.75)
    beta: float = Field(ge=0, default=0.08)
    alpha: float = Field(ge=0, default=1.0)
    autoscale_inefficiency: float = Field(ge=1, default=1.15)
    monthly_budget_max_usd: float = Field(ge=0, default=0.0)
    output_token_ratio: float = Field(ge=0, le=1, default=0.30)
    top_k: int = Field(ge=1, le=50, default=10)


class LLMPlanningResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    plans: list[RankedPlan]
    provider_diagnostics: list[ProviderDiagnostic]
    excluded_count: int
    warnings: list[str] = Field(default_factory=list)


class CatalogRankingRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    workload_type: str = Field(min_length=1)
    allowed_providers: list[str] = Field(default_factory=list)
    unit_name: Optional[str] = None
    monthly_usage: float = Field(ge=0, default=0.0)
    monthly_budget_max_usd: float = Field(ge=0, default=0.0)
    top_k: int = Field(ge=1, le=50, default=10)
    confidence_weighted: bool = True
    comparator_mode: Literal["normalized", "listed", "raw"] = "normalized"
    throughput_aware: bool = False
    peak_to_avg: float = Field(gt=0, default=2.5)
    util_target: float = Field(gt=0, lt=1, default=0.75)
    strict_capacity_check: bool = False


class RankedCatalogOffer(BaseModel):
    model_config = ConfigDict(extra="ignore")

    rank: int
    provider: str
    sku_name: str
    billing_mode: str
    unit_price_usd: float
    normalized_price: Optional[float]
    unit_name: str
    confidence: str
    monthly_estimate_usd: Optional[float]
    required_replicas: Optional[int]
    capacity_check: Literal["ok", "insufficient", "unknown"]
    previous_unit_price_usd: Optional[float] = None
    price_change_abs_usd: Optional[float] = None
    price_change_pct: Optional[float] = None


class CatalogRankingResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    offers: list[RankedCatalogOffer]
    provider_diagnostics: list[ProviderDiagnostic]
    excluded_count: int
    warnings: list[str] = Field(default_factory=list)
    relaxation_applied: bool = False
    relaxation_steps: list[dict[str, Any]] = Field(default_factory=list)
    exclusion_breakdown: dict[str, int] = Field(default_factory=dict)


class CatalogBrowseResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    rows: list[dict[str, Any]]
    total: int


class InvoiceLineItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    provider: str
    workload_type: str
    line_item: str
    quantity: float
    unit: str
    unit_price: float
    total: float


class InvoiceAnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    line_items: list[InvoiceLineItem]
    totals_by_provider: dict[str, float]
    grand_total: float
    detected_workloads: list[str]
    warnings: list[str] = Field(default_factory=list)


class AIAssistContext(BaseModel):
    model_config = ConfigDict(extra="ignore")

    workload_type: Optional[str] = None
    providers: list[str] = Field(default_factory=list)
    recent_results: Optional[list[dict[str, Any]]] = None


class AIAssistRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    message: str = Field(min_length=1, max_length=4000)
    context: AIAssistContext = Field(default_factory=AIAssistContext)


class AIAssistResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    reply: str
    suggested_action: Optional[str] = None


class CopilotMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    content: str


class CopilotTurnRequest(BaseModel):
    """Request payload for one IA copilot turn.

    Supports both:
    - frontend payload: {message, history, workload_type}
    - internal payload: {user_text, state}
    """

    model_config = ConfigDict(extra="ignore")

    message: Optional[str] = Field(default=None, min_length=1, max_length=2000)
    history: list[CopilotMessage] = Field(default_factory=list)
    workload_type: Optional[str] = None

    user_text: Optional[str] = Field(default=None, min_length=1, max_length=2000)
    state: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_input_shape(self) -> "CopilotTurnRequest":
        if self.user_text:
            return self
        if self.message:
            return self
        raise ValueError("Either 'user_text' or 'message' must be provided.")


class CopilotTurnResponse(BaseModel):
    """Structured response payload for IA copilot UI."""

    model_config = ConfigDict(extra="ignore")

    reply: str
    extracted_spec: dict[str, Any]
    missing_fields: list[str]
    follow_up_questions: list[str]
    apply_payload: Optional[dict[str, Any]]
    is_ready: bool


class ReportSection(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str
    bullets: list[str] = Field(default_factory=list)


class ReportGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mode: Literal["llm", "catalog"]
    title: str = Field(default="InferenceAtlas Optimization Report", min_length=1, max_length=200)
    output_format: Literal["markdown", "html", "pdf"] = "markdown"
    include_charts: bool = True
    include_csv_exports: bool = True
    include_narrative: bool = False
    llm_planning: Optional[LLMPlanningResponse] = None
    catalog_ranking: Optional[CatalogRankingResponse] = None

    @model_validator(mode="after")
    def validate_mode_payload(self) -> "ReportGenerateRequest":
        if self.mode == "llm" and self.llm_planning is None:
            raise ValueError("llm_planning is required when mode='llm'")
        if self.mode == "catalog" and self.catalog_ranking is None:
            raise ValueError("catalog_ranking is required when mode='catalog'")
        return self


class ReportGenerateResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    report_id: str
    generated_at_utc: str
    title: str
    mode: Literal["llm", "catalog"]
    sections: list[ReportSection] = Field(default_factory=list)
    chart_data: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    output_format: Literal["markdown", "html", "pdf"] = "markdown"
    narrative: Optional[str] = None
    csv_exports: dict[str, str] = Field(default_factory=dict)
    markdown: str
    html: Optional[str] = None
    pdf_base64: Optional[str] = None
