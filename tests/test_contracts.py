"""Tests for Pydantic validation contracts in inference_atlas.contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inference_atlas.contracts import (
    BillingMode,
    CatalogOfferContract,
    CatalogRankingInput,
    CatalogRankingResult,
    ConfidenceLevel,
    ModelBucket,
    PlanningResult,
    RankedPlanContract,
    RiskBreakdownContract,
    TrafficPattern,
    WorkloadInput,
)


# =============================================================================
# Enum Tests
# =============================================================================


def test_confidence_level_enum_values() -> None:
    """Confidence level enum has all expected values."""
    assert ConfidenceLevel.HIGH.value == "high"
    assert ConfidenceLevel.OFFICIAL.value == "official"
    assert ConfidenceLevel.MEDIUM.value == "medium"
    assert ConfidenceLevel.ESTIMATED.value == "estimated"
    assert ConfidenceLevel.LOW.value == "low"
    assert ConfidenceLevel.VENDOR_LIST.value == "vendor_list"


def test_confidence_level_numeric_ordering() -> None:
    """Confidence levels have correct numeric scores for comparison."""
    assert ConfidenceLevel.HIGH.score == 3
    assert ConfidenceLevel.OFFICIAL.score == 3  # Alias
    assert ConfidenceLevel.MEDIUM.score == 2
    assert ConfidenceLevel.ESTIMATED.score == 2  # Alias
    assert ConfidenceLevel.LOW.score == 1
    assert ConfidenceLevel.VENDOR_LIST.score == 1  # Alias


def test_model_bucket_enum_values() -> None:
    """Model bucket enum has all expected size buckets."""
    assert ModelBucket.BUCKET_7B.value == "7b"
    assert ModelBucket.BUCKET_13B.value == "13b"
    assert ModelBucket.BUCKET_34B.value == "34b"
    assert ModelBucket.BUCKET_70B.value == "70b"
    assert ModelBucket.BUCKET_405B.value == "405b"


def test_traffic_pattern_enum_values() -> None:
    """Traffic pattern enum has all expected patterns."""
    assert TrafficPattern.STEADY.value == "steady"
    assert TrafficPattern.BUSINESS_HOURS.value == "business_hours"
    assert TrafficPattern.BURSTY.value == "bursty"


def test_billing_mode_enum_values() -> None:
    """Billing mode enum has all expected billing types."""
    assert BillingMode.PER_TOKEN.value == "per_token"
    assert BillingMode.DEDICATED_HOURLY.value == "dedicated_hourly"
    assert BillingMode.AUTOSCALE_HOURLY.value == "autoscale_hourly"


# =============================================================================
# WorkloadInput Tests
# =============================================================================


def test_workload_input_valid() -> None:
    """Valid workload input passes validation with defaults."""
    workload = WorkloadInput(
        tokens_per_day=5_000_000,
        model_bucket=ModelBucket.BUCKET_70B,
        provider_ids={"aws", "gcp"},
    )
    assert workload.tokens_per_day == 5_000_000
    assert workload.model_bucket == ModelBucket.BUCKET_70B
    assert workload.provider_ids == {"aws", "gcp"}
    # Check defaults
    assert workload.peak_to_avg == 2.5
    assert workload.util_target == 0.75
    assert workload.beta == 0.08
    assert workload.alpha == 1.0
    assert workload.autoscale_inefficiency == 1.15
    assert workload.output_token_ratio == 0.30
    assert workload.top_k == 3


def test_workload_input_negative_tokens_rejected() -> None:
    """Negative tokens_per_day raises ValidationError."""
    with pytest.raises(ValidationError, match="greater than 0"):
        WorkloadInput(
            tokens_per_day=-1000,
            model_bucket=ModelBucket.BUCKET_70B,
            provider_ids={"aws"},
        )


def test_workload_input_zero_tokens_rejected() -> None:
    """Zero tokens_per_day raises ValidationError."""
    with pytest.raises(ValidationError, match="greater than 0"):
        WorkloadInput(
            tokens_per_day=0,
            model_bucket=ModelBucket.BUCKET_7B,
            provider_ids={"azure"},
        )


def test_workload_input_empty_providers_rejected() -> None:
    """Empty provider set raises ValidationError."""
    with pytest.raises(ValidationError, match="at least 1 item"):
        WorkloadInput(
            tokens_per_day=1000,
            model_bucket=ModelBucket.BUCKET_7B,
            provider_ids=set(),
        )


def test_workload_input_whitespace_providers_rejected() -> None:
    """Provider IDs with only whitespace are rejected."""
    with pytest.raises(ValidationError, match="non-empty strings"):
        WorkloadInput(
            tokens_per_day=1000,
            model_bucket=ModelBucket.BUCKET_7B,
            provider_ids={"  ", ""},
        )


def test_workload_input_providers_trimmed() -> None:
    """Provider IDs with leading/trailing whitespace are trimmed."""
    workload = WorkloadInput(
        tokens_per_day=1000,
        model_bucket=ModelBucket.BUCKET_7B,
        provider_ids={" aws ", "gcp  ", "  azure"},
    )
    assert workload.provider_ids == {"aws", "gcp", "azure"}


def test_workload_input_peak_to_avg_bounds() -> None:
    """peak_to_avg must be > 0 and <= 10."""
    # Valid boundary
    workload = WorkloadInput(
        tokens_per_day=1000,
        model_bucket=ModelBucket.BUCKET_7B,
        provider_ids={"aws"},
        peak_to_avg=10.0,
    )
    assert workload.peak_to_avg == 10.0

    # Zero rejected
    with pytest.raises(ValidationError, match="greater than 0"):
        WorkloadInput(
            tokens_per_day=1000,
            model_bucket=ModelBucket.BUCKET_7B,
            provider_ids={"aws"},
            peak_to_avg=0,
        )

    # Negative rejected
    with pytest.raises(ValidationError, match="greater than 0"):
        WorkloadInput(
            tokens_per_day=1000,
            model_bucket=ModelBucket.BUCKET_7B,
            provider_ids={"aws"},
            peak_to_avg=-1.5,
        )

    # Too high rejected
    with pytest.raises(ValidationError, match="less than or equal to 10"):
        WorkloadInput(
            tokens_per_day=1000,
            model_bucket=ModelBucket.BUCKET_7B,
            provider_ids={"aws"},
            peak_to_avg=11.0,
        )


def test_workload_input_util_target_bounds() -> None:
    """util_target must be 0 < x < 1."""
    # Valid range
    workload = WorkloadInput(
        tokens_per_day=1000,
        model_bucket=ModelBucket.BUCKET_7B,
        provider_ids={"aws"},
        util_target=0.50,
    )
    assert workload.util_target == 0.50

    # Zero rejected
    with pytest.raises(ValidationError, match="greater than 0"):
        WorkloadInput(
            tokens_per_day=1000,
            model_bucket=ModelBucket.BUCKET_7B,
            provider_ids={"aws"},
            util_target=0.0,
        )

    # Exactly 1.0 rejected
    with pytest.raises(ValidationError, match="less than 1"):
        WorkloadInput(
            tokens_per_day=1000,
            model_bucket=ModelBucket.BUCKET_7B,
            provider_ids={"aws"},
            util_target=1.0,
        )


def test_workload_input_alpha_bounds() -> None:
    """alpha must be 0 <= x <= 5."""
    # Valid boundaries
    workload_min = WorkloadInput(
        tokens_per_day=1000,
        model_bucket=ModelBucket.BUCKET_7B,
        provider_ids={"aws"},
        alpha=0.0,
    )
    assert workload_min.alpha == 0.0

    workload_max = WorkloadInput(
        tokens_per_day=1000,
        model_bucket=ModelBucket.BUCKET_7B,
        provider_ids={"aws"},
        alpha=5.0,
    )
    assert workload_max.alpha == 5.0

    # Negative rejected
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        WorkloadInput(
            tokens_per_day=1000,
            model_bucket=ModelBucket.BUCKET_7B,
            provider_ids={"aws"},
            alpha=-1.0,
        )

    # Too high rejected
    with pytest.raises(ValidationError, match="less than or equal to 5"):
        WorkloadInput(
            tokens_per_day=1000,
            model_bucket=ModelBucket.BUCKET_7B,
            provider_ids={"aws"},
            alpha=6.0,
        )


def test_workload_input_output_token_ratio_bounds() -> None:
    """output_token_ratio must be 0 <= x <= 1."""
    # Valid boundaries
    workload_min = WorkloadInput(
        tokens_per_day=1000,
        model_bucket=ModelBucket.BUCKET_7B,
        provider_ids={"aws"},
        output_token_ratio=0.0,
    )
    assert workload_min.output_token_ratio == 0.0

    workload_max = WorkloadInput(
        tokens_per_day=1000,
        model_bucket=ModelBucket.BUCKET_7B,
        provider_ids={"aws"},
        output_token_ratio=1.0,
    )
    assert workload_max.output_token_ratio == 1.0

    # Negative rejected
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        WorkloadInput(
            tokens_per_day=1000,
            model_bucket=ModelBucket.BUCKET_7B,
            provider_ids={"aws"},
            output_token_ratio=-0.1,
        )

    # Too high rejected
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        WorkloadInput(
            tokens_per_day=1000,
            model_bucket=ModelBucket.BUCKET_7B,
            provider_ids={"aws"},
            output_token_ratio=1.1,
        )


def test_ranked_plan_contract_rejects_per_token_gpu_fields() -> None:
    with pytest.raises(ValidationError, match="per_token plans must set gpu_count=0"):
        RankedPlanContract(
            rank=1,
            provider_id="p",
            provider_name="Provider",
            offering_id="sku",
            billing_mode=BillingMode.PER_TOKEN,
            confidence=ConfidenceLevel.HIGH,
            monthly_cost_usd=10.0,
            score=10.0,
            utilization_at_peak=None,
            gpu_type="a100_80gb",
            gpu_count=1,
            risk=RiskBreakdownContract(risk_overload=0.1, risk_complexity=0.1, total_risk=0.1),
            assumptions={"peak_to_avg": 2.5},
            why="test",
        )


def test_ranked_plan_contract_rejects_hourly_without_gpu_type() -> None:
    with pytest.raises(ValidationError, match="hourly plans must set gpu_type"):
        RankedPlanContract(
            rank=1,
            provider_id="p",
            provider_name="Provider",
            offering_id="sku",
            billing_mode=BillingMode.DEDICATED_HOURLY,
            confidence=ConfidenceLevel.HIGH,
            monthly_cost_usd=10.0,
            score=10.0,
            utilization_at_peak=0.6,
            gpu_type=None,
            gpu_count=1,
            risk=RiskBreakdownContract(risk_overload=0.1, risk_complexity=0.1, total_risk=0.1),
            assumptions={"peak_to_avg": 2.5},
            why="test",
        )


# =============================================================================
# Output Contract Tests
# =============================================================================


def test_risk_breakdown_contract_valid() -> None:
    """Valid risk breakdown passes validation."""
    risk = RiskBreakdownContract(
        risk_overload=0.2,
        risk_complexity=0.3,
        total_risk=0.27,
    )
    assert risk.risk_overload == 0.2
    assert risk.risk_complexity == 0.3
    assert risk.total_risk == 0.27


def test_risk_breakdown_contract_bounds() -> None:
    """Risk values must be in [0, 1]."""
    # Valid boundaries
    risk_min = RiskBreakdownContract(
        risk_overload=0.0,
        risk_complexity=0.0,
        total_risk=0.0,
    )
    assert risk_min.total_risk == 0.0

    risk_max = RiskBreakdownContract(
        risk_overload=1.0,
        risk_complexity=1.0,
        total_risk=1.0,
    )
    assert risk_max.total_risk == 1.0

    # Negative rejected
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        RiskBreakdownContract(
            risk_overload=-0.1,
            risk_complexity=0.3,
            total_risk=0.2,
        )

    # Too high rejected
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        RiskBreakdownContract(
            risk_overload=0.5,
            risk_complexity=1.5,
            total_risk=0.8,
        )


def test_ranked_plan_contract_minimal() -> None:
    """Minimal valid ranked plan with required fields only."""
    risk = RiskBreakdownContract(
        risk_overload=0.2,
        risk_complexity=0.1,
        total_risk=0.17,
    )

    plan = RankedPlanContract(
        rank=1,
        provider_id="aws",
        provider_name="AWS",
        offering_id="sagemaker_per_token",
        billing_mode=BillingMode.PER_TOKEN,
        confidence=ConfidenceLevel.HIGH,
        monthly_cost_usd=1500.0,
        score=1600.0,
        risk=risk,
        assumptions={"peak_to_avg": 2.5, "util_target": 0.75},
        why="Lowest cost with high confidence",
    )

    assert plan.rank == 1
    assert plan.monthly_cost_usd == 1500.0
    assert plan.utilization_at_peak is None  # Optional field


def test_planning_result_empty_plans() -> None:
    """PlanningResult can have empty plans list with warnings."""
    result = PlanningResult(
        plans=[],
        excluded_providers=["aws", "gcp"],
        warnings=["No feasible configurations found for workload"],
    )

    assert len(result.plans) == 0
    assert len(result.warnings) == 1
    assert len(result.excluded_providers) == 2


def test_planning_result_with_plans() -> None:
    """PlanningResult with multiple ranked plans."""
    risk = RiskBreakdownContract(
        risk_overload=0.2,
        risk_complexity=0.1,
        total_risk=0.17,
    )

    plan1 = RankedPlanContract(
        rank=1,
        provider_id="aws",
        provider_name="AWS",
        offering_id="sagemaker",
        billing_mode=BillingMode.PER_TOKEN,
        confidence=ConfidenceLevel.HIGH,
        monthly_cost_usd=1500.0,
        score=1600.0,
        risk=risk,
        assumptions={},
        why="Best option",
    )

    plan2 = RankedPlanContract(
        rank=2,
        provider_id="gcp",
        provider_name="GCP",
        offering_id="vertex_ai",
        billing_mode=BillingMode.PER_TOKEN,
        confidence=ConfidenceLevel.MEDIUM,
        monthly_cost_usd=1800.0,
        score=1900.0,
        risk=risk,
        assumptions={},
        why="Second best",
    )

    result = PlanningResult(
        plans=[plan1, plan2],
        workload_summary={"tokens_per_day": 5_000_000},
    )

    assert len(result.plans) == 2
    assert result.plans[0].rank == 1
    assert result.plans[1].rank == 2


# =============================================================================
# Catalog Ranking Tests
# =============================================================================


def test_catalog_ranking_input_valid() -> None:
    """Valid catalog ranking input passes validation."""
    input_data = CatalogRankingInput(
        workload_type="speech_to_text",
        provider_ids={"deepgram", "assemblyai"},
        unit_name="audio_hour",
        top_k=5,
    )

    assert input_data.workload_type == "speech_to_text"
    assert input_data.unit_name == "audio_hour"
    assert input_data.top_k == 5


def test_catalog_offer_contract_with_estimate() -> None:
    """Catalog offer with monthly estimate."""
    offer = CatalogOfferContract(
        rank=1,
        provider="deepgram",
        offering="Nova-2",
        billing="per_hour",
        unit_price_usd=0.0043,
        unit_name="audio_hour",
        confidence="high",
        monthly_estimate_usd=310.0,
    )

    assert offer.monthly_estimate_usd == 310.0


def test_catalog_ranking_result() -> None:
    """Complete catalog ranking result."""
    offer1 = CatalogOfferContract(
        rank=1,
        provider="deepgram",
        offering="Nova-2",
        billing="per_hour",
        unit_price_usd=0.0043,
        unit_name="audio_hour",
        confidence="high",
    )

    result = CatalogRankingResult(
        offers=[offer1],
        workload_type="speech_to_text",
        warnings=["Unit price ranking only - throughput not modeled"],
    )

    assert len(result.offers) == 1
    assert result.workload_type == "speech_to_text"
    assert len(result.warnings) == 1
