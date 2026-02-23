"""Data contracts and validation for InferenceAtlas pipeline.

This module defines Pydantic models for validating inputs and outputs across
the planning pipeline. All validation happens at contract boundaries to ensure
type safety and data integrity.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# =============================================================================
# Enums (Single Source of Truth)
# =============================================================================


class ConfidenceLevel(str, Enum):
    """Pricing confidence levels with clear semantics.

    Confidence indicates the reliability and freshness of pricing data:
    - HIGH/OFFICIAL: Verified from official provider documentation
    - MEDIUM/ESTIMATED: Community-verified or derived from similar offerings
    - LOW/VENDOR_LIST: Listed but unconfirmed or potentially stale
    """

    HIGH = "high"
    OFFICIAL = "official"  # Alias for HIGH
    MEDIUM = "medium"
    ESTIMATED = "estimated"  # Alias for MEDIUM
    LOW = "low"
    VENDOR_LIST = "vendor_list"  # Alias for LOW

    @property
    def score(self) -> int:
        """Numeric ordering for confidence comparison (higher is better)."""
        return {
            "high": 3,
            "official": 3,
            "medium": 2,
            "estimated": 2,
            "low": 1,
            "vendor_list": 1,
        }[self.value]

    @property
    def price_penalty_multiplier(self) -> float:
        """Multiplier used when confidence-weighted ranking is enabled."""
        return {
            "high": 1.00,
            "official": 1.00,
            "medium": 1.10,
            "estimated": 1.10,
            "low": 1.25,
            "vendor_list": 1.25,
        }[self.value]


class ModelBucket(str, Enum):
    """Model size buckets for capacity planning.

    Buckets group models by parameter count for throughput estimation:
    - 7B: ~7 billion parameters (e.g., Llama 3.3 7B, Mistral 7B)
    - 13B: ~13 billion parameters
    - 34B: ~34 billion parameters
    - 70B: ~70 billion parameters (e.g., Llama 3.3 70B)
    - 405B: ~405 billion parameters (e.g., Llama 3.1 405B)
    """

    BUCKET_7B = "7b"
    BUCKET_13B = "13b"
    BUCKET_34B = "34b"
    BUCKET_70B = "70b"
    BUCKET_405B = "405b"


class TrafficPattern(str, Enum):
    """Traffic patterns for workload modeling.

    Patterns determine peak-to-average ratio defaults:
    - STEADY: Uniform load throughout the day (peak_to_avg ~1.5)
    - BUSINESS_HOURS: Office-hour heavy (peak_to_avg ~2.5)
    - BURSTY: Spiky, unpredictable load (peak_to_avg ~3.5)
    """

    STEADY = "steady"
    BUSINESS_HOURS = "business_hours"
    BURSTY = "bursty"


class BillingMode(str, Enum):
    """Billing modes for provider offerings.

    - PER_TOKEN: Pay per million tokens processed
    - DEDICATED_HOURLY: Dedicated GPU instances, billed hourly
    - AUTOSCALE_HOURLY: Auto-scaling GPU instances, billed hourly
    """

    PER_TOKEN = "per_token"
    DEDICATED_HOURLY = "dedicated_hourly"
    AUTOSCALE_HOURLY = "autoscale_hourly"


# =============================================================================
# Input Contracts
# =============================================================================


class WorkloadInput(BaseModel):
    """Validated LLM workload planning input.

    This contract enforces validation at the service boundary, ensuring
    all parameters are within acceptable bounds before planning begins.

    Example:
        >>> workload = WorkloadInput(
        ...     tokens_per_day=5_000_000,
        ...     model_bucket=ModelBucket.BUCKET_70B,
        ...     provider_ids={"anthropic", "openai"},
        ... )
        >>> workload.tokens_per_day
        5000000.0
    """

    model_config = ConfigDict(protected_namespaces=())

    # Required fields
    tokens_per_day: float = Field(
        gt=0,
        description="Daily token volume (input + output tokens combined)",
    )
    model_bucket: ModelBucket = Field(
        description="Model size bucket for capacity planning",
    )
    provider_ids: set[str] = Field(
        min_length=1,
        description="Set of provider IDs to include in ranking",
    )

    # Tuning parameters with validated bounds
    peak_to_avg: float = Field(
        default=2.5,
        gt=0,
        le=10,
        description="Peak-to-average traffic ratio (higher = spikier load)",
    )
    util_target: float = Field(
        default=0.75,
        gt=0,
        lt=1,
        description="Target utilization at peak (lower = more headroom)",
    )
    beta: float = Field(
        default=0.08,
        ge=0,
        le=1,
        description="Multi-GPU scaling efficiency penalty (0 = perfect scaling)",
    )
    alpha: float = Field(
        default=1.0,
        ge=0,
        le=5,
        description="Risk penalty multiplier in score calculation (0 = ignore risk)",
    )
    autoscale_inefficiency: float = Field(
        default=1.15,
        ge=1.0,
        le=2.0,
        description="Autoscale overhead factor (e.g., 1.15 = 15% overhead)",
    )
    monthly_budget_max_usd: float = Field(
        default=0.0,
        ge=0,
        description="Optional monthly budget cap (0 = no budget filter)",
    )
    output_token_ratio: float = Field(
        default=0.30,
        ge=0,
        le=1,
        description="Fraction of tokens that are output (for I/O pricing blending)",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of top-ranked plans to return",
    )

    @field_validator("provider_ids")
    @classmethod
    def validate_provider_ids(cls, v: set[str]) -> set[str]:
        """Ensure all provider IDs are non-empty strings."""
        if not all(isinstance(p, str) and p.strip() for p in v):
            raise ValueError("All provider_ids must be non-empty strings")
        return {p.strip() for p in v}


class CatalogRankingInput(BaseModel):
    """Input contract for non-LLM catalog ranking (beta).

    This simpler contract is used for price-based ranking of non-LLM
    workloads (e.g., speech-to-text, image generation) where throughput
    modeling is not yet implemented.
    """

    workload_type: str = Field(
        description="Workload type (e.g., 'speech_to_text', 'image_generation')",
    )
    provider_ids: set[str] = Field(
        min_length=1,
        description="Set of provider IDs to include in ranking",
    )
    unit_name: Optional[str] = Field(
        default=None,
        description="Optional unit filter (e.g., 'audio_hour', 'image')",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top offers to return",
    )


# =============================================================================
# Output Contracts
# =============================================================================


class RiskBreakdownContract(BaseModel):
    """Risk breakdown for a ranked plan.

    Risk scores are in [0, 1] where:
    - risk_overload: Capacity margin risk (low margin = high risk)
    - risk_complexity: Operational complexity risk (more GPUs = higher risk)
    - total_risk: Weighted combination (70% overload + 30% complexity)
    """

    risk_overload: float = Field(ge=0, le=1, description="Capacity overload risk")
    risk_complexity: float = Field(ge=0, le=1, description="Operational complexity risk")
    total_risk: float = Field(ge=0, le=1, description="Combined risk score")


class RankedPlanContract(BaseModel):
    """A single ranked deployment plan with cost and risk analysis.

    This contract represents one viable deployment configuration with
    all relevant cost, capacity, and risk metrics computed.
    """

    rank: int = Field(ge=1, description="Rank position (1 = best)")
    provider_id: str = Field(description="Provider identifier")
    provider_name: str = Field(description="Human-readable provider name")
    offering_id: str = Field(description="Specific offering/SKU identifier")
    billing_mode: BillingMode = Field(description="Billing model for this offering")
    confidence: ConfidenceLevel = Field(description="Pricing confidence level")

    # Cost and scoring
    monthly_cost_usd: float = Field(ge=0, description="Estimated monthly cost in USD")
    score: float = Field(ge=0, description="Ranking score (lower is better)")

    # Capacity and utilization
    utilization_at_peak: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Utilization ratio at peak load (if known)",
    )
    gpu_type: Optional[str] = Field(
        default=None,
        description="GPU type for dedicated/autoscale offerings",
    )
    gpu_count: int = Field(
        default=0,
        ge=0,
        description="Number of GPUs (0 for per_token offerings)",
    )

    # Risk and metadata
    risk: RiskBreakdownContract = Field(description="Risk breakdown")
    assumptions: dict[str, float] = Field(
        description="Planning assumptions (peak_to_avg, util_target, etc.)",
    )
    why: str = Field(description="Explanation of why this plan was ranked here")

    @model_validator(mode="after")
    def validate_gpu_fields(self) -> RankedPlanContract:
        """Enforce consistent gpu_type/gpu_count based on billing mode."""
        if self.billing_mode == BillingMode.PER_TOKEN:
            if self.gpu_count != 0 or self.gpu_type is not None:
                raise ValueError(
                    "per_token plans must set gpu_count=0 and gpu_type=None"
                )
            return self
        if self.gpu_count < 1:
            raise ValueError("hourly plans must set gpu_count >= 1")
        if not self.gpu_type:
            raise ValueError("hourly plans must set gpu_type")
        return self


class PlanningResult(BaseModel):
    """Complete planning result with ranked plans and metadata.

    This top-level contract represents the output of a planning request,
    including all ranked plans, warnings, and summary information.
    """

    plans: list[RankedPlanContract] = Field(
        description="Ranked deployment plans (sorted by score)",
    )
    workload_summary: dict[str, float] = Field(
        default_factory=dict,
        description="Summary of workload parameters",
    )
    excluded_providers: list[str] = Field(
        default_factory=list,
        description="Providers that were filtered out or had no viable configs",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Validation warnings or informational messages",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (service version, timestamps, etc.)",
    )


# =============================================================================
# Catalog Ranking Output (Beta)
# =============================================================================


class CatalogOfferContract(BaseModel):
    """A single catalog offer from price-based ranking (non-LLM)."""

    rank: int = Field(ge=1)
    provider: str
    offering: str
    billing: str
    unit_price_usd: float = Field(ge=0)
    unit_name: str
    confidence: str
    monthly_estimate_usd: Optional[float] = Field(default=None, ge=0)


class CatalogRankingResult(BaseModel):
    """Result from catalog-based ranking for non-LLM workloads (beta)."""

    offers: list[CatalogOfferContract]
    workload_type: str
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
