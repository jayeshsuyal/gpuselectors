"""InferenceAtlas: LLM Deployment Cost Optimizer.

A planning model for estimating GPU requirements and monthly costs for LLM deployments.
Supports multi-GPU scaling, traffic pattern modeling, and cross-platform cost comparison.
"""

from __future__ import annotations

__version__ = "0.1.1"

# Public API exports
from inference_atlas.cost_model import (
    CostBreakdown,
    calculate_gpu_monthly_cost,
    calculate_per_token_monthly_cost,
)
from inference_atlas.catalog_ranking import (
    RankedCatalogOffer,
    build_provider_diagnostics,
    confidence_multiplier,
    get_catalog_tuning_preset,
    normalize_unit_price_for_workload,
    rank_catalog_offers,
)
from inference_atlas.ai_copilot import (
    build_apply_payload,
    extract_spec_updates,
    get_follow_up_questions,
    get_missing_fields,
    get_suggested_presets,
    next_copilot_turn,
)
from inference_atlas.api_models import CopilotTurnRequest, CopilotTurnResponse
from inference_atlas.api_service import run_copilot_turn
from inference_atlas.data_loader import (
    get_catalog_v2_metadata,
    get_catalog_v2_rows,
    get_huggingface_catalog_metadata,
    get_huggingface_models,
    get_mvp_catalog,
    get_model_display_name,
    get_models,
    get_platforms,
    get_pricing_by_workload,
    get_pricing_records,
    validate_mvp_catalogs,
    validate_pricing_datasets,
    validate_huggingface_catalog,
    refresh_huggingface_catalog_cache,
    refresh_catalog_v2_cache,
)
from inference_atlas.llm import (
    GPT52Adapter,
    LLMAdapter,
    LLMRouter,
    Opus46Adapter,
    ParseWorkloadResult,
    RouterConfig,
    WorkloadSpec,
    parse_workload_text,
    validate_workload_payload,
)
from inference_atlas.invoice_analyzer import analyze_invoice_csv, canonical_workload_from_invoice
from inference_atlas.mvp_planner import (
    CapacityEstimate,
    NormalizedWorkload,
    PlannerConfig,
    ProviderCompatibility,
    RankedPlan,
    RiskBreakdown,
    capacity,
    compute_monthly_cost,
    enumerate_configs,
    enumerate_configs_for_providers,
    get_provider_compatibility,
    get_tuning_preset,
    normalize_workload,
    rank_configs,
    risk_score,
)
from inference_atlas.recommender import Recommendation, compute_penalty, get_recommendations
from inference_atlas.scaling import (
    TrafficProfile,
    UtilizationEstimate,
    calculate_utilization,
    get_traffic_profile,
    latency_risk_band,
)

__all__ = [
    # Version
    "__version__",
    # Recommendation engine
    "get_recommendations",
    "Recommendation",
    "compute_penalty",
    # Cost modeling
    "calculate_gpu_monthly_cost",
    "calculate_per_token_monthly_cost",
    "CostBreakdown",
    # Scaling & utilization
    "calculate_utilization",
    "get_traffic_profile",
    "latency_risk_band",
    "UtilizationEstimate",
    "TrafficProfile",
    # Data loading
    "get_platforms",
    "get_models",
    "get_model_display_name",
    "get_pricing_records",
    "get_catalog_v2_rows",
    "get_catalog_v2_metadata",
    "get_pricing_by_workload",
    "validate_pricing_datasets",
    "validate_mvp_catalogs",
    "get_mvp_catalog",
    "get_huggingface_models",
    "get_huggingface_catalog_metadata",
    "validate_huggingface_catalog",
    "refresh_huggingface_catalog_cache",
    "refresh_catalog_v2_cache",
    # MVP planner
    "normalize_workload",
    "capacity",
    "enumerate_configs",
    "enumerate_configs_for_providers",
    "get_provider_compatibility",
    "compute_monthly_cost",
    "risk_score",
    "rank_configs",
    "get_tuning_preset",
    "NormalizedWorkload",
    "CapacityEstimate",
    "PlannerConfig",
    "ProviderCompatibility",
    "RiskBreakdown",
    "RankedPlan",
    # LLM adapter layer
    "LLMAdapter",
    "GPT52Adapter",
    "Opus46Adapter",
    "LLMRouter",
    "RouterConfig",
    "ParseWorkloadResult",
    "parse_workload_text",
    "WorkloadSpec",
    "validate_workload_payload",
    # Invoice analyzer
    "analyze_invoice_csv",
    "canonical_workload_from_invoice",
    # Catalog ranking helpers
    "normalize_unit_price_for_workload",
    "confidence_multiplier",
    "rank_catalog_offers",
    "get_catalog_tuning_preset",
    "build_provider_diagnostics",
    "RankedCatalogOffer",
    # AI copilot helpers
    "extract_spec_updates",
    "get_missing_fields",
    "get_follow_up_questions",
    "get_suggested_presets",
    "build_apply_payload",
    "next_copilot_turn",
    # API contracts/services
    "CopilotTurnRequest",
    "CopilotTurnResponse",
    "run_copilot_turn",
]
