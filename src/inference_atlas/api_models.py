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


class QualityCatalogRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    provider: str
    workload_type: str
    model_key: str
    sku_name: str
    billing_mode: str
    unit_price_usd: float
    unit_name: str
    quality_mapped: bool
    quality_model_id: Optional[str] = None
    quality_score_0_100: Optional[float] = None
    quality_score_adjusted_0_100: Optional[float] = None
    quality_confidence: Optional[str] = None
    quality_confidence_weight: Optional[float] = None
    quality_matched_by: Optional[str] = None


class QualityCatalogResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    rows: list[QualityCatalogRow]
    total: int
    mapped_count: int
    unmapped_count: int


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


class ReportChartSeries(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    label: str
    unit: Optional[str] = None
    points: list[dict[str, Any]] = Field(default_factory=list)


class ReportChart(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    title: str
    type: Literal["bar", "line", "stacked_bar", "step_line"]
    x_label: str
    y_label: str
    series: list[ReportChartSeries] = Field(default_factory=list)
    legend: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


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


class ScalingPlanRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mode: Literal["llm", "catalog"]
    llm_planning: Optional[LLMPlanningResponse] = None
    catalog_ranking: Optional[CatalogRankingResponse] = None

    @model_validator(mode="after")
    def validate_mode_payload(self) -> "ScalingPlanRequest":
        if self.mode == "llm" and self.llm_planning is None:
            raise ValueError("llm_planning is required when mode='llm'")
        if self.mode == "catalog" and self.catalog_ranking is None:
            raise ValueError("catalog_ranking is required when mode='catalog'")
        return self


class ScalingPlanResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mode: Literal["llm", "catalog"]
    deployment_mode: Literal["serverless", "dedicated", "autoscale", "unknown"]
    estimated_gpu_count: int = Field(ge=0, default=0)
    suggested_gpu_type: Optional[str] = None
    projected_utilization: Optional[float] = None
    utilization_target: Optional[float] = None
    risk_band: Literal["low", "medium", "high", "unknown"] = "unknown"
    capacity_check: Literal["ok", "insufficient", "unknown"] = "unknown"
    rationale: str
    assumptions: list[str] = Field(default_factory=list)


class CostAuditRecommendation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    recommendation_type: Literal[
        "pricing_model_switch",
        "procurement",
        "quantization",
        "autoscaling",
        "caching",
        "hardware_match",
        "other",
    ] = "other"
    title: str
    rationale: str
    estimated_savings_pct: float = Field(ge=0, le=100)
    priority: Literal["high", "medium", "low"] = "medium"


class CostAuditHardwareRecommendation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    tier: Literal["serverless", "single_gpu", "multi_gpu", "hybrid", "unknown"] = "unknown"
    gpu_family: Optional[str] = None
    deployment_shape: str = "unknown"
    reasoning: str


class CostAuditPricingVerdict(BaseModel):
    model_config = ConfigDict(extra="ignore")

    current_model: Literal["token_api", "dedicated_gpu", "mixed"]
    verdict: Literal["appropriate", "consider_switch", "suboptimal", "unknown"] = "unknown"
    reason: str


class CostAuditSavingsEstimate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    low_usd: float = Field(ge=0)
    high_usd: float = Field(ge=0)
    basis: str


class CostAuditScoreBreakdown(BaseModel):
    model_config = ConfigDict(extra="ignore")

    base_score: int = Field(ge=0, le=100, default=100)
    penalty_points: int = Field(ge=0, default=0)
    bonus_points: int = Field(ge=0, default=0)
    pre_cap_score: int = Field(ge=0, le=100)
    post_cap_score: int = Field(ge=0, le=100)
    major_flags: int = Field(ge=0, default=0)
    caps_applied: list[str] = Field(default_factory=list)
    combined_savings_pct: float = Field(ge=0, le=100, default=0.0)


class CostAuditDataGap(BaseModel):
    model_config = ConfigDict(extra="ignore")

    field: str
    impact: Literal["high", "medium", "low"]
    why_it_matters: str


class CostAuditLegAudit(BaseModel):
    model_config = ConfigDict(extra="ignore")

    modality: str
    model_name: str
    estimated_spend_usd: float = Field(ge=0, default=0)
    efficiency_score: int = Field(ge=0, le=100)
    top_recommendation: Optional[str] = None
    estimated_savings_high_usd: float = Field(ge=0, default=0)


class CostAuditRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    modality: Literal[
        "llm",
        "asr",
        "tts",
        "embeddings",
        "image_gen",
        "video_gen",
        "mixed",
    ]
    model_name: str = Field(min_length=1)
    model_precision: Literal["fp16", "bf16", "fp8", "int8", "int4", "unknown"] = "unknown"
    fine_tuned: bool = False

    pricing_model: Literal["token_api", "dedicated_gpu", "mixed"]
    monthly_input_tokens: Optional[float] = Field(default=None, ge=0)
    monthly_output_tokens: Optional[float] = Field(default=None, ge=0)
    gpu_type: Optional[str] = None
    gpu_count: Optional[int] = Field(default=None, ge=0)
    gpu_procurement_type: Literal["on_demand", "reserved", "spot", "mixed", "unknown"] = "unknown"
    traffic_pattern: Literal["steady", "bursty", "business_hours", "batch_offline", "unknown"] = "unknown"
    peak_concurrency: Optional[int] = Field(default=None, ge=0)

    avg_input_tokens: Optional[float] = Field(default=None, ge=0)
    avg_output_tokens: Optional[float] = Field(default=None, ge=0)
    workload_execution: Literal[
        "latency_sensitive",
        "throughput_optimized",
        "mixed",
        "unknown",
    ] = "unknown"
    caching_enabled: Literal["yes", "no", "unknown"] = "unknown"

    providers: list[str] = Field(default_factory=list)
    autoscaling: Literal["yes", "no", "unknown"] = "unknown"
    quantization_applied: Literal["yes", "no", "unknown"] = "unknown"
    multi_model_pipeline: bool = False
    pipeline_models: list[str] = Field(default_factory=list)

    use_case: Optional[str] = None
    business_context: Literal["consumer", "b2b", "internal_tooling", "unknown"] = "unknown"
    latency_sla: Optional[str] = None
    monthly_ai_spend_usd: Optional[float] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_payload(self) -> "CostAuditRequest":
        if self.pricing_model == "token_api":
            if self.monthly_input_tokens is None and self.monthly_output_tokens is None:
                raise ValueError(
                    "token_api audits require monthly_input_tokens and/or monthly_output_tokens"
                )
        if self.pricing_model == "dedicated_gpu":
            if self.gpu_count is None or self.gpu_count <= 0:
                raise ValueError("dedicated_gpu audits require gpu_count > 0")
        if not self.multi_model_pipeline and self.pipeline_models:
            raise ValueError("pipeline_models should be empty when multi_model_pipeline is false")
        return self


class CostAuditResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    efficiency_score: int = Field(ge=0, le=100)
    recommendations: list[CostAuditRecommendation] = Field(default_factory=list)
    hardware_recommendation: CostAuditHardwareRecommendation
    pricing_model_verdict: CostAuditPricingVerdict
    pricing_source: Literal["provider_csv", "heuristic_prior", "unknown"] = "unknown"
    pricing_source_provider: Optional[str] = None
    pricing_source_gpu: Optional[str] = None
    red_flags: list[str] = Field(default_factory=list)
    estimated_monthly_savings: CostAuditSavingsEstimate
    score_breakdown: CostAuditScoreBreakdown
    per_modality_audits: list[CostAuditLegAudit] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)
    data_gaps_detailed: list[CostAuditDataGap] = Field(default_factory=list)


class ReportGenerateResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    report_id: str
    generated_at_utc: str
    title: str
    mode: Literal["llm", "catalog"]
    sections: list[ReportSection] = Field(default_factory=list)
    charts: list[ReportChart] = Field(default_factory=list)
    chart_data: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    output_format: Literal["markdown", "html", "pdf"] = "markdown"
    narrative: Optional[str] = None
    csv_exports: dict[str, str] = Field(default_factory=dict)
    markdown: str
    html: Optional[str] = None
    pdf_base64: Optional[str] = None
