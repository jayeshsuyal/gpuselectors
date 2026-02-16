"""MVP planning engine built on schema-validated JSON catalogs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from inference_atlas.data_loader import get_mvp_catalog


DEFAULT_UTIL_TARGET = 0.75
DEFAULT_PEAK_TO_AVG = 2.5
DEFAULT_SCALING_BETA = 0.08
DEFAULT_AUTOSCALE_INEFFICIENCY = 1.15
DEFAULT_ALPHA = 1.0
DEFAULT_OUTPUT_TOKEN_RATIO = 0.30
CONFIDENCE_ORDER = {
    "high": 3,
    "official": 3,
    "medium": 2,
    "estimated": 2,
    "low": 1,
    "vendor_list": 1,
}


@dataclass(frozen=True)
class NormalizedWorkload:
    tokens_per_day: float
    peak_to_avg: float
    util_target: float
    tokens_per_month: float
    avg_tok_s: float
    peak_tok_s: float
    required_capacity_tok_s: float


@dataclass(frozen=True)
class CapacityEstimate:
    tok_s_total: float
    tok_s_per_gpu: float
    efficiency: float
    p95_latency_est_ms: float | None
    mem_ok: bool


@dataclass(frozen=True)
class PlannerConfig:
    provider_id: str
    provider_name: str
    offering_id: str
    billing_mode: str
    gpu_type: str | None
    gpu_count: int
    price_per_gpu_hour_usd: float | None
    price_per_1m_tokens_usd: float | None
    tps_cap: float | None
    region: str
    confidence: str
    notes: str


@dataclass(frozen=True)
class RiskBreakdown:
    risk_overload: float
    risk_complexity: float
    total_risk: float


@dataclass(frozen=True)
class RankedPlan:
    rank: int
    provider_id: str
    provider_name: str
    offering_id: str
    billing_mode: str
    gpu_type: str | None
    gpu_count: int
    confidence: str
    monthly_cost_usd: float
    score: float
    utilization_at_peak: float | None
    headroom_pct: float | None
    required_capacity_tok_s: float
    provided_capacity_tok_s: float | None
    risk: RiskBreakdown
    why: str
    assumptions: dict[str, float]


@dataclass(frozen=True)
class ProviderCompatibility:
    provider_id: str
    provider_name: str
    compatible: bool
    reason: str


def normalize_workload(
    tokens_per_day: float,
    peak_to_avg: float = DEFAULT_PEAK_TO_AVG,
    util_target: float = DEFAULT_UTIL_TARGET,
) -> NormalizedWorkload:
    """Convert raw workload inputs into canonical planning metrics."""
    tokens = float(tokens_per_day)
    p2a = float(peak_to_avg)
    util = float(util_target)

    if tokens <= 0:
        raise ValueError("tokens_per_day must be > 0")
    if p2a <= 0:
        raise ValueError("peak_to_avg must be > 0")
    if not 0 < util < 1:
        raise ValueError("util_target must be between 0 and 1")

    avg_tok_s = tokens / 86_400
    peak_tok_s = avg_tok_s * p2a
    required_capacity = peak_tok_s / util
    return NormalizedWorkload(
        tokens_per_day=tokens,
        peak_to_avg=p2a,
        util_target=util,
        tokens_per_month=tokens * 30,
        avg_tok_s=avg_tok_s,
        peak_tok_s=peak_tok_s,
        required_capacity_tok_s=required_capacity,
    )


def _bucket_token(model_bucket: str) -> str:
    token = model_bucket.strip().lower()
    if token.startswith("bucket_"):
        token = token.removeprefix("bucket_")
    return token


def _lower_confidence(a: str, b: str) -> str:
    a_score = CONFIDENCE_ORDER.get(a, 0)
    b_score = CONFIDENCE_ORDER.get(b, 0)
    return a if a_score <= b_score else b


def capacity(
    model_bucket: str,
    gpu_type: str,
    gpus: int,
    beta: float = DEFAULT_SCALING_BETA,
    capacity_entries: list[dict[str, Any]] | None = None,
) -> CapacityEstimate:
    """Estimate total throughput for model bucket + GPU count."""
    if gpus < 1:
        raise ValueError("gpus must be >= 1")
    if beta < 0:
        raise ValueError("beta must be >= 0")

    entries = capacity_entries
    if entries is None:
        entries = get_mvp_catalog("capacity_table")["entries"]

    bucket = _bucket_token(model_bucket)
    match = next(
        (
            entry
            for entry in entries
            if entry["model_size_bucket"] == bucket and entry["gpu_type"] == gpu_type
        ),
        None,
    )
    if match is None:
        raise ValueError(
            f"No capacity entry for model_bucket='{bucket}' gpu_type='{gpu_type}'"
        )

    base = float(match["tok_s_1gpu"])
    efficiency = 1.0 / (1.0 + beta * (gpus - 1))
    tok_s_total = base * gpus * efficiency
    tok_s_per_gpu = tok_s_total / gpus
    return CapacityEstimate(
        tok_s_total=tok_s_total,
        tok_s_per_gpu=tok_s_per_gpu,
        efficiency=efficiency,
        p95_latency_est_ms=None,
        mem_ok=True,
    )


def enumerate_configs(
    model_bucket: str,
    output_token_ratio: float = DEFAULT_OUTPUT_TOKEN_RATIO,
) -> list[PlannerConfig]:
    """Enumerate provider/offering configurations from providers catalog."""
    if not 0 <= output_token_ratio <= 1:
        raise ValueError("output_token_ratio must be between 0 and 1")

    providers = get_mvp_catalog("providers")["providers"]
    models = get_mvp_catalog("models")["models"]
    model_to_bucket = {row["model_id"]: row["size_bucket"] for row in models}
    target_bucket = _bucket_token(model_bucket)
    target_bucket_ref = f"bucket_{target_bucket}"

    configs: list[PlannerConfig] = []
    for provider in providers:
        paired_per_token: dict[tuple[str, str], dict[str, Any]] = {}
        single_per_token: list[dict[str, Any]] = []

        for offering in provider["offerings"]:
            model_refs: list[str] = offering.get("model_refs", [])
            supports_model = any(
                ref == target_bucket_ref or model_to_bucket.get(ref) == target_bucket
                for ref in model_refs
            )
            if not supports_model:
                continue

            mode = offering["billing_mode"]
            if mode == "per_token":
                offering_id = offering["offering_id"]
                model_ref = model_refs[0] if model_refs else offering_id
                if offering_id.endswith("_input"):
                    base_id = offering_id[: -len("_input")]
                    bucket = paired_per_token.setdefault((model_ref, base_id), {})
                    bucket["input"] = offering
                elif offering_id.endswith("_output"):
                    base_id = offering_id[: -len("_output")]
                    bucket = paired_per_token.setdefault((model_ref, base_id), {})
                    bucket["output"] = offering
                else:
                    single_per_token.append(offering)
                continue

            for gpu_count in offering["allowed_gpu_counts"]:
                configs.append(
                    PlannerConfig(
                        provider_id=provider["provider_id"],
                        provider_name=provider["provider_name"],
                        offering_id=offering["offering_id"],
                        billing_mode=mode,
                        gpu_type=offering["gpu_type"],
                        gpu_count=int(gpu_count),
                        price_per_gpu_hour_usd=float(offering["price_per_gpu_hour_usd"]),
                        price_per_1m_tokens_usd=None,
                        tps_cap=None,
                        region=offering["region"],
                        confidence=offering["confidence"],
                        notes=offering.get("notes", ""),
                    )
                )

        for (model_ref, base_id), pair in paired_per_token.items():
            if "input" in pair and "output" in pair:
                input_price = float(pair["input"]["price_per_1m_tokens_usd"])
                output_price = float(pair["output"]["price_per_1m_tokens_usd"])
                blended = ((1.0 - output_token_ratio) * input_price) + (
                    output_token_ratio * output_price
                )
                input_tps_cap = (
                    float(pair["input"]["tps_cap"])
                    if pair["input"].get("tps_cap") is not None
                    else None
                )
                output_tps_cap = (
                    float(pair["output"]["tps_cap"])
                    if pair["output"].get("tps_cap") is not None
                    else None
                )
                tps_caps = [cap for cap in (input_tps_cap, output_tps_cap) if cap is not None]
                blended_tps_cap = min(tps_caps) if tps_caps else None
                configs.append(
                    PlannerConfig(
                        provider_id=provider["provider_id"],
                        provider_name=provider["provider_name"],
                        offering_id=f"{base_id}_blended_io",
                        billing_mode="per_token",
                        gpu_type=None,
                        gpu_count=0,
                        price_per_gpu_hour_usd=None,
                        price_per_1m_tokens_usd=blended,
                        tps_cap=blended_tps_cap,
                        region=pair["input"]["region"],
                        confidence=_lower_confidence(
                            pair["input"]["confidence"],
                            pair["output"]["confidence"],
                        ),
                        notes=(
                            f"Blended input/output price: input={input_price:.6g}, "
                            f"output={output_price:.6g}, output_ratio={output_token_ratio:.2f}"
                        ),
                    )
                )
            elif "input" in pair:
                single_per_token.append(pair["input"])
            elif "output" in pair:
                single_per_token.append(pair["output"])

        for offering in single_per_token:
            configs.append(
                PlannerConfig(
                    provider_id=provider["provider_id"],
                    provider_name=provider["provider_name"],
                    offering_id=offering["offering_id"],
                    billing_mode="per_token",
                    gpu_type=None,
                    gpu_count=0,
                    price_per_gpu_hour_usd=None,
                    price_per_1m_tokens_usd=float(offering["price_per_1m_tokens_usd"]),
                    tps_cap=(
                        float(offering["tps_cap"])
                        if offering.get("tps_cap") is not None
                        else None
                    ),
                    region=offering["region"],
                    confidence=offering["confidence"],
                    notes=offering.get("notes", ""),
                )
            )

    return configs


def enumerate_configs_for_providers(
    model_bucket: str,
    provider_ids: set[str] | None = None,
    output_token_ratio: float = DEFAULT_OUTPUT_TOKEN_RATIO,
) -> list[PlannerConfig]:
    """Enumerate configurations for an optional subset of providers."""
    configs = enumerate_configs(
        model_bucket=model_bucket,
        output_token_ratio=output_token_ratio,
    )
    if provider_ids is None:
        return configs
    return [cfg for cfg in configs if cfg.provider_id in provider_ids]


def get_provider_compatibility(
    model_bucket: str,
    provider_ids: set[str] | None = None,
    output_token_ratio: float = DEFAULT_OUTPUT_TOKEN_RATIO,
) -> list[ProviderCompatibility]:
    """Return compatibility diagnostics for providers against a model bucket."""
    providers = get_mvp_catalog("providers")["providers"]
    all_provider_rows = [
        (str(p["provider_id"]), str(p["provider_name"])) for p in providers
    ]
    if provider_ids is None:
        scoped_rows = all_provider_rows
    else:
        scoped_rows = [row for row in all_provider_rows if row[0] in provider_ids]

    compatible_ids = {
        cfg.provider_id
        for cfg in enumerate_configs_for_providers(
            model_bucket=model_bucket,
            provider_ids={row[0] for row in scoped_rows},
            output_token_ratio=output_token_ratio,
        )
    }
    diagnostics: list[ProviderCompatibility] = []
    for provider_id, provider_name in scoped_rows:
        if provider_id in compatible_ids:
            diagnostics.append(
                ProviderCompatibility(
                    provider_id=provider_id,
                    provider_name=provider_name,
                    compatible=True,
                    reason="Supports selected model bucket",
                )
            )
        else:
            diagnostics.append(
                ProviderCompatibility(
                    provider_id=provider_id,
                    provider_name=provider_name,
                    compatible=False,
                    reason="No offering for selected model bucket",
                )
            )
    return diagnostics


def _is_feasible(
    cfg: PlannerConfig,
    workload: NormalizedWorkload,
    model_bucket: str,
    beta: float,
    capacity_entries: list[dict[str, Any]],
) -> tuple[bool, CapacityEstimate | None]:
    if cfg.billing_mode == "per_token":
        if cfg.tps_cap is None:
            return True, None
        return cfg.tps_cap >= workload.required_capacity_tok_s, None

    assert cfg.gpu_type is not None
    cap = capacity(
        model_bucket=model_bucket,
        gpu_type=cfg.gpu_type,
        gpus=cfg.gpu_count,
        beta=beta,
        capacity_entries=capacity_entries,
    )
    return cap.tok_s_total >= workload.required_capacity_tok_s, cap


def compute_monthly_cost(
    cfg: PlannerConfig,
    workload: NormalizedWorkload,
    cap: CapacityEstimate | None,
    autoscale_inefficiency: float = DEFAULT_AUTOSCALE_INEFFICIENCY,
) -> float:
    """Compute monthly cost under one of the 3 MVP billing modes."""
    if autoscale_inefficiency < 1:
        raise ValueError("autoscale_inefficiency must be >= 1.0")

    if cfg.billing_mode == "per_token":
        assert cfg.price_per_1m_tokens_usd is not None
        return (workload.tokens_per_month / 1_000_000.0) * cfg.price_per_1m_tokens_usd

    assert cfg.price_per_gpu_hour_usd is not None
    if cfg.billing_mode == "dedicated_hourly":
        return cfg.gpu_count * cfg.price_per_gpu_hour_usd * 24 * 30

    if cfg.billing_mode == "autoscale_hourly":
        if cap is None:
            raise ValueError("autoscale_hourly requires capacity estimate")
        gpu_hours = ((workload.tokens_per_month / cap.tok_s_total) / 3600.0) * cfg.gpu_count
        return gpu_hours * cfg.price_per_gpu_hour_usd * autoscale_inefficiency

    raise ValueError(f"Unsupported billing_mode '{cfg.billing_mode}'")


def risk_score(
    cfg: PlannerConfig,
    required_capacity_tok_s: float,
    provided_capacity_tok_s: float | None,
) -> RiskBreakdown:
    """Compute blended risk (overload + configuration complexity)."""
    if cfg.billing_mode == "per_token" and provided_capacity_tok_s is None:
        risk_overload = 0.20
    else:
        if provided_capacity_tok_s is None:
            raise ValueError("provided_capacity_tok_s is required for non-per_token")
        margin = (provided_capacity_tok_s / required_capacity_tok_s) - 1.0
        risk_overload = min(1.0, max(0.0, math.exp(-4.0 * margin)))

    gpu_count_for_complexity = max(1, cfg.gpu_count)
    risk_complexity = (
        0.0 if gpu_count_for_complexity <= 1 else min(1.0, 0.15 * math.log2(gpu_count_for_complexity))
    )
    total = (0.7 * risk_overload) + (0.3 * risk_complexity)
    return RiskBreakdown(
        risk_overload=risk_overload,
        risk_complexity=risk_complexity,
        total_risk=total,
    )


def rank_configs(
    tokens_per_day: float,
    model_bucket: str,
    peak_to_avg: float = DEFAULT_PEAK_TO_AVG,
    util_target: float = DEFAULT_UTIL_TARGET,
    beta: float = DEFAULT_SCALING_BETA,
    alpha: float = DEFAULT_ALPHA,
    autoscale_inefficiency: float = DEFAULT_AUTOSCALE_INEFFICIENCY,
    top_k: int = 10,
    provider_ids: set[str] | None = None,
    output_token_ratio: float = DEFAULT_OUTPUT_TOKEN_RATIO,
) -> list[RankedPlan]:
    """Run full MVP ranking pipeline and return top-k plans."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if not 0 <= output_token_ratio <= 1:
        raise ValueError("output_token_ratio must be between 0 and 1")

    workload = normalize_workload(
        tokens_per_day=tokens_per_day,
        peak_to_avg=peak_to_avg,
        util_target=util_target,
    )

    entries = get_mvp_catalog("capacity_table")["entries"]
    all_configs = enumerate_configs_for_providers(
        model_bucket=model_bucket,
        provider_ids=provider_ids,
        output_token_ratio=output_token_ratio,
    )
    candidates: list[RankedPlan] = []

    for cfg in all_configs:
        feasible, cap = _is_feasible(
            cfg=cfg,
            workload=workload,
            model_bucket=model_bucket,
            beta=beta,
            capacity_entries=entries,
        )
        if not feasible:
            continue

        monthly = compute_monthly_cost(
            cfg=cfg,
            workload=workload,
            cap=cap,
            autoscale_inefficiency=autoscale_inefficiency,
        )
        provided = cap.tok_s_total if cap is not None else cfg.tps_cap
        risk = risk_score(
            cfg=cfg,
            required_capacity_tok_s=workload.required_capacity_tok_s,
            provided_capacity_tok_s=provided,
        )
        score = monthly * (1.0 + alpha * risk.total_risk)

        if provided is None:
            utilization = None
            headroom_pct = None
            why = (
                f"{cfg.billing_mode}; throughput cap unspecified; "
                f"cost-driven ranking with risk {risk.total_risk:.2f}"
            )
        else:
            utilization = workload.peak_tok_s / provided
            headroom_pct = ((provided / workload.required_capacity_tok_s) - 1.0) * 100.0
            why = (
                f"{cfg.billing_mode}; provides {provided:.0f} tok/s vs required "
                f"{workload.required_capacity_tok_s:.0f} tok/s; "
                f"peak util {utilization:.2f}; headroom {headroom_pct:.1f}%"
            )

        candidates.append(
            RankedPlan(
                rank=0,
                provider_id=cfg.provider_id,
                provider_name=cfg.provider_name,
                offering_id=cfg.offering_id,
                billing_mode=cfg.billing_mode,
                gpu_type=cfg.gpu_type,
                gpu_count=cfg.gpu_count,
                confidence=cfg.confidence,
                monthly_cost_usd=monthly,
                score=score,
                utilization_at_peak=utilization,
                headroom_pct=headroom_pct,
                required_capacity_tok_s=workload.required_capacity_tok_s,
                provided_capacity_tok_s=provided,
                risk=risk,
                why=why,
                assumptions={
                    "util_target": util_target,
                    "peak_to_avg": peak_to_avg,
                    "scaling_beta": beta,
                    "alpha": alpha,
                    "output_token_ratio": output_token_ratio,
                },
            )
        )

    if not candidates:
        raise ValueError("No feasible configurations found for this workload/model bucket")

    candidates.sort(key=lambda row: row.score)
    ranked: list[RankedPlan] = []
    for idx, row in enumerate(candidates[:top_k], start=1):
        ranked.append(
            RankedPlan(
                rank=idx,
                provider_id=row.provider_id,
                provider_name=row.provider_name,
                offering_id=row.offering_id,
                billing_mode=row.billing_mode,
                gpu_type=row.gpu_type,
                gpu_count=row.gpu_count,
                confidence=row.confidence,
                monthly_cost_usd=row.monthly_cost_usd,
                score=row.score,
                utilization_at_peak=row.utilization_at_peak,
                headroom_pct=row.headroom_pct,
                required_capacity_tok_s=row.required_capacity_tok_s,
                provided_capacity_tok_s=row.provided_capacity_tok_s,
                risk=row.risk,
                why=row.why,
                assumptions=row.assumptions,
            )
        )
    return ranked
