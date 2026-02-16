"""Streamlit UI for InferenceAtlas planning and pricing catalog browsing."""

from __future__ import annotations

import csv
import io
import inspect
import os
from datetime import datetime, timezone

import streamlit as st

from inference_atlas import (
    get_provider_compatibility,
    get_huggingface_catalog_metadata,
    get_huggingface_models,
    get_pricing_records,
    refresh_huggingface_catalog_cache,
    rank_configs,
)
from inference_atlas.data_loader import get_models
from inference_atlas.huggingface_catalog import fetch_huggingface_models, write_huggingface_catalog
from inference_atlas.llm import LLMRouter, WorkloadSpec
from inference_atlas.workload_types import WorkloadType

st.set_page_config(page_title="InferenceAtlas", layout="centered")
st.title("InferenceAtlas")
st.caption("Category-first workload planning and provider pricing intelligence.")

MODEL_REQUIREMENTS = get_models()
has_llm_key = bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))

pattern_to_label = {"steady": "Steady", "business_hours": "Business Hours", "bursty": "Bursty"}
label_to_pattern = {"Steady": "steady", "Business Hours": "business_hours", "Bursty": "bursty"}
pattern_default_peak_to_avg = {"steady": 1.5, "business_hours": 2.5, "bursty": 3.5}
HF_DEPLOY_PROVIDER_IDS = {
    "baseten",
    "fireworks",
    "modal",
    "replicate",
    "runpod",
    "fal_ai",
    "together_ai",
}


def _model_key_to_bucket(model_key: str) -> str:
    model = MODEL_REQUIREMENTS.get(model_key)
    if not model:
        return "70b"
    param_count = int(model["parameter_count"])
    if param_count <= 9_000_000_000:
        return "7b"
    if param_count <= 20_000_000_000:
        return "13b"
    if param_count <= 50_000_000_000:
        return "34b"
    if param_count <= 120_000_000_000:
        return "70b"
    return "405b"


def _catalog_is_stale(generated_at_utc: str | None, max_age_days: int = 14) -> bool:
    if not generated_at_utc:
        return True
    try:
        generated = datetime.fromisoformat(generated_at_utc.replace("Z", "+00:00"))
    except ValueError:
        return True
    return (datetime.now(timezone.utc) - generated).days > max_age_days


def _rows_to_csv_bytes(rows: list[dict[str, object]]) -> bytes:
    if not rows:
        return b""
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode("utf-8")


def _rank_catalog_offers(
    rows: list[object],
    allowed_providers: set[str],
    unit_name: str | None,
    top_k: int,
) -> list[object]:
    filtered = [row for row in rows if row.provider in allowed_providers]
    if unit_name:
        filtered = [row for row in filtered if row.unit_name == unit_name]
    filtered.sort(key=lambda row: row.unit_price_usd)
    return filtered[:top_k]


workload_options = {
    WorkloadType.LLM.display_name: WorkloadType.LLM,
    WorkloadType.SPEECH_TO_TEXT.display_name: WorkloadType.SPEECH_TO_TEXT,
    WorkloadType.TEXT_TO_SPEECH.display_name: WorkloadType.TEXT_TO_SPEECH,
    WorkloadType.EMBEDDINGS.display_name: WorkloadType.EMBEDDINGS,
    WorkloadType.IMAGE_GENERATION.display_name: WorkloadType.IMAGE_GENERATION,
    WorkloadType.VISION.display_name: WorkloadType.VISION,
}

selected_workload_label = st.selectbox(
    "1. What are you optimizing for?",
    options=list(workload_options),
    index=0,
    help="Choose workload category first. The rest of the flow is filtered by this choice.",
)
selected_workload = workload_options[selected_workload_label]
workload_rows = get_pricing_records(selected_workload)
workload_provider_ids = sorted({row.provider for row in workload_rows})
all_provider_ids = sorted({row.provider for row in get_pricing_records()})

show_all_providers = st.checkbox(
    "Show all providers (including ones without this workload)",
    value=True,
    help="Off = workload-aware providers only. On = full provider list from the catalog.",
)
provider_options = all_provider_ids if show_all_providers else workload_provider_ids

provider_labels = {
    provider: (
        f"{provider} (supports {selected_workload.value})"
        if provider in workload_provider_ids
        else f"{provider} (no {selected_workload.value} rows)"
    )
    for provider in provider_options
}
provider_by_label = {label: provider for provider, label in provider_labels.items()}
provider_label_options = list(provider_by_label)

selected_global_providers = st.multiselect(
    (
        f"2. Providers ({len(provider_options)} shown, "
        f"{len(workload_provider_ids)} support this workload)"
    ),
    options=provider_label_options,
    default=[],
    help="This provider filter is shared across optimizer and catalog views.",
)
selected_global_providers = [provider_by_label[label] for label in selected_global_providers]
if not selected_global_providers:
    st.info("Select one or more providers to continue.")
elif show_all_providers:
    unsupported = sorted(set(selected_global_providers) - set(workload_provider_ids))
    if unsupported:
        st.caption(
            "Selected providers without current workload coverage: "
            + ", ".join(unsupported)
        )

tab_optimize, tab_catalog = st.tabs(["Optimize Workload", "Browse Pricing Catalog"])

with tab_optimize:
    if selected_workload != WorkloadType.LLM:
        st.subheader("Top Offers (Catalog Ranker - Beta)")
        st.caption(
            "This optimizer mode ranks by listed unit price only. "
            "Throughput/SLA-aware optimization is not implemented yet ?"
        )
        available_units = sorted({row.unit_name for row in workload_rows})
        with st.form("optimize_non_llm"):
            non_llm_top_k = st.slider("Top K", min_value=1, max_value=20, value=5)
            selected_unit = st.selectbox(
                "Unit filter",
                options=["All units", *available_units],
                help="Cross-unit normalization is not implemented yet ? Filter by a single unit for clean comparisons.",
            )
            monthly_usage = st.number_input(
                "Monthly usage estimate",
                min_value=0.0,
                value=0.0,
                step=1000.0,
                help="Optional. Cost estimate only applies when a single unit filter is selected.",
            )
            non_llm_submit = st.form_submit_button("Get Top Offers")

        if non_llm_submit:
            if not selected_global_providers:
                st.error("No providers selected. Choose at least one provider.")
            else:
                unit_filter = None if selected_unit == "All units" else selected_unit
                ranked = _rank_catalog_offers(
                    rows=workload_rows,
                    allowed_providers=set(selected_global_providers),
                    unit_name=unit_filter,
                    top_k=int(non_llm_top_k),
                )
                if not ranked:
                    st.warning("No offers matched the selected providers/unit filter.")
                else:
                    table_rows = []
                    for idx, row in enumerate(ranked, start=1):
                        est_cost = None
                        if unit_filter is not None and monthly_usage > 0:
                            est_cost = monthly_usage * row.unit_price_usd
                        table_rows.append(
                            {
                                "rank": idx,
                                "provider": row.provider,
                                "offering": row.sku_name,
                                "billing": row.billing_type,
                                "unit_price": row.unit_price_usd,
                                "unit_name": row.unit_name,
                                "confidence": row.confidence,
                                "monthly_estimate_usd": round(est_cost, 2) if est_cost is not None else None,
                            }
                        )
                    try:
                        st.dataframe(table_rows, use_container_width=True, hide_index=True)
                    except TypeError:
                        st.dataframe(table_rows)
                    if unit_filter is None:
                        st.info("Monthly estimate hidden because multiple units are mixed.")
    else:
        parsed_workload = st.session_state.get("parsed_workload", {})

        with st.expander("Parse workload from plain English"):
            user_text = st.text_area(
                "Describe your LLM deployment needs",
                placeholder="e.g., Chat app with 10k daily users, steady traffic, Llama 70B, need <200ms latency",
                height=100,
                key="ai_parse_input_text",
            )
            if st.button("Parse with AI"):
                if not has_llm_key:
                    st.error("Set OPENAI_API_KEY or ANTHROPIC_API_KEY to use AI parsing.")
                else:
                    try:
                        router = LLMRouter()
                        parsed = router.parse_workload(user_text)
                        st.session_state["parsed_workload"] = {
                            "tokens_per_day": parsed.tokens_per_day,
                            "pattern": parsed.pattern,
                            "model_key": parsed.model_key,
                            "latency_requirement_ms": parsed.latency_requirement_ms,
                        }
                        st.success("Parsed successfully.")
                        st.json(st.session_state["parsed_workload"])
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Parsing failed: {exc}")

        hf_meta: dict[str, object] | None = None
        try:
            hf_meta = get_huggingface_catalog_metadata()
            if _catalog_is_stale(str(hf_meta.get("generated_at_utc") or None)):
                st.warning("Hugging Face catalog is stale. Sync before selecting HF models.")
        except Exception:
            st.warning("Hugging Face catalog unavailable. Sync before selecting HF models.")

        with st.expander("Hugging Face Catalog Status"):
            try:
                meta = hf_meta or get_huggingface_catalog_metadata()
                st.write(f"Last sync (UTC): `{meta.get('generated_at_utc', 'unknown')}`")
                st.write(f"Model count: `{meta.get('model_count', 0)}`")
                st.write(f"Source: `{meta.get('source', 'unknown')}`")
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Catalog status unavailable: {exc}")

            sync_limit = st.slider(
                "Sync model limit",
                min_value=20,
                max_value=500,
                value=100,
                step=20,
                help="Number of top downloaded models to fetch from Hugging Face API.",
            )
            if st.button("Sync Hugging Face models now"):
                try:
                    with st.spinner("Syncing models from Hugging Face API..."):
                        token = os.getenv("HUGGINGFACE_TOKEN")
                        synced = fetch_huggingface_models(limit=sync_limit, token=token)
                        write_huggingface_catalog(synced)
                        refreshed = refresh_huggingface_catalog_cache()
                    st.success(
                        f"Synced {refreshed.get('model_count', 0)} models at "
                        f"{refreshed.get('generated_at_utc', 'unknown')}."
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Hugging Face sync failed: {exc}")

        with st.form("optimize_inputs"):
            model_items = list(MODEL_REQUIREMENTS.items())
            default_model_key = str(parsed_workload.get("model_key", "llama_70b"))
            default_index = next((i for i, (k, _) in enumerate(model_items) if k == default_model_key), 0)
            model_display_names = [v["display_name"] for _, v in model_items]
            model_key_by_display = {v["display_name"]: k for k, v in model_items}

            model_display_name = st.selectbox(
                "Model (curated)",
                options=model_display_names,
                index=default_index,
            )
            model_key = model_key_by_display[model_display_name]

            tokens_per_day = st.number_input(
                "Traffic (tokens/day)",
                min_value=1.0,
                value=float(parsed_workload.get("tokens_per_day", 5_000_000.0)),
                step=100_000.0,
            )
            pattern_label = st.selectbox(
                "Traffic pattern",
                ["Steady", "Business Hours", "Bursty"],
                index=["Steady", "Business Hours", "Bursty"].index(
                    pattern_to_label.get(str(parsed_workload.get("pattern", "steady")), "Steady")
                ),
            )

            source_lane = st.radio(
                "Model source lane",
                options=["Provider Catalog", "Hugging Face"],
                index=0,
                help=(
                    "Provider Catalog: use curated model set from this app. "
                    "Hugging Face: use API-synced OSS model catalog."
                ),
            )

            hf_model_choice = None
            if source_lane == "Hugging Face":
                hf_models = get_huggingface_models(min_downloads=1000, include_gated=False)
                if hf_models:
                    labels = [
                        f"{m['model_id']} (downloads: {m['downloads']:,}, bucket: {m['size_bucket']})"
                        for m in hf_models[:100]
                    ]
                    by_label = {label: hf_models[idx] for idx, label in enumerate(labels)}
                    chosen = st.selectbox("Open-source model (Hugging Face)", options=labels)
                    hf_model_choice = by_label[chosen]
                else:
                    st.caption("No HF models available locally. Sync from panel above.")

            model_bucket_preview = _model_key_to_bucket(model_key)
            if hf_model_choice is not None:
                model_bucket_preview = str(hf_model_choice["size_bucket"])

            lane_scope_provider_ids = (
                [p for p in selected_global_providers if p in HF_DEPLOY_PROVIDER_IDS]
                if source_lane == "Hugging Face"
                else selected_global_providers
            )
            compatibility = get_provider_compatibility(
                model_bucket=model_bucket_preview,
                provider_ids=set(lane_scope_provider_ids),
            )
            compatible_filtered = sorted(
                row.provider_id for row in compatibility if row.compatible
            )
            incompatible_filtered = [row for row in compatibility if not row.compatible]

            selected_provider_ids = st.multiselect(
                f"Providers to include ({len(compatible_filtered)} compatible after filters)",
                options=compatible_filtered,
                default=[],
                help=(
                    "Explicit provider selection is required. "
                    "HF lane only shows OSS deployment-compatible providers."
                ),
            )
            if source_lane == "Hugging Face":
                non_hf_lane = sorted(
                    set(selected_global_providers) - set(lane_scope_provider_ids)
                )
                if non_hf_lane:
                    st.caption(
                        "Ignored for HF lane (not OSS deploy providers): "
                        + ", ".join(non_hf_lane)
                    )
            if incompatible_filtered:
                st.caption(
                    "Excluded by model compatibility: "
                    + ", ".join(
                        f"{row.provider_id} ({row.reason})" for row in incompatible_filtered
                    )
                )

            with st.expander("Advanced tuning"):
                latency_requirement_ms = st.number_input(
                    "Latency requirement (ms, optional)",
                    min_value=0.0,
                    value=float(parsed_workload.get("latency_requirement_ms") or 0.0),
                    step=10.0,
                )
                top_k = st.slider("Top K", min_value=1, max_value=10, value=3)
                peak_to_avg = st.slider(
                    "Peak-to-average",
                    min_value=1.0,
                    max_value=6.0,
                    value=float(pattern_default_peak_to_avg[label_to_pattern[pattern_label]]),
                    step=0.1,
                )
                util_target = st.slider("Util target", min_value=0.50, max_value=0.90, value=0.75, step=0.01)
                beta = st.slider("Scaling beta", min_value=0.01, max_value=0.20, value=0.08, step=0.01)
                alpha = st.slider("Risk alpha", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
                autoscale_inefficiency = st.slider(
                    "Autoscale inefficiency",
                    min_value=1.0,
                    max_value=1.5,
                    value=1.15,
                    step=0.01,
                )
                output_token_ratio = st.slider(
                    "Output token ratio",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.30,
                    step=0.01,
                    help=(
                        "Used when a provider exposes separate Input/Output token prices. "
                        "0.30 means 30% of total billed tokens are treated as output tokens."
                    ),
                )

            submit = st.form_submit_button("Get Recommendations")

        if submit:
            if not selected_provider_ids:
                st.error("No providers selected. Choose at least one provider.")
            else:
                selected_model_key = model_key
                selected_model_bucket = _model_key_to_bucket(model_key)
                if hf_model_choice is not None:
                    selected_model_key = str(hf_model_choice["model_id"])
                    selected_model_bucket = str(hf_model_choice["size_bucket"])

                latency = latency_requirement_ms if latency_requirement_ms > 0 else None
                effective_workload = WorkloadSpec(
                    tokens_per_day=float(tokens_per_day),
                    pattern=label_to_pattern[pattern_label],
                    model_key=selected_model_key,
                    latency_requirement_ms=latency,
                )
                st.session_state["last_workload"] = effective_workload
                st.session_state["last_model_source"] = source_lane
                st.session_state["last_selected_providers"] = selected_provider_ids

                try:
                    rank_kwargs = {
                        "tokens_per_day": effective_workload.tokens_per_day,
                        "model_bucket": selected_model_bucket,
                        "peak_to_avg": float(peak_to_avg),
                        "util_target": float(util_target),
                        "beta": float(beta),
                        "alpha": float(alpha),
                        "autoscale_inefficiency": float(autoscale_inefficiency),
                        "top_k": int(top_k),
                        "provider_ids": set(selected_provider_ids),
                    }
                    if "output_token_ratio" in inspect.signature(rank_configs).parameters:
                        rank_kwargs["output_token_ratio"] = float(output_token_ratio)
                    ranked_plans = rank_configs(
                        **rank_kwargs,
                    )
                except ValueError as exc:
                    st.error(f"{exc}. Try broader provider selection or another model/bucket.")
                    ranked_plans = []
                st.session_state["last_ranked_plans"] = ranked_plans

        ranked = st.session_state.get("last_ranked_plans", [])
        last_workload = st.session_state.get("last_workload")
        if ranked:
            st.subheader("Top Recommendations")
            st.caption(f"Model source: {st.session_state.get('last_model_source', 'unknown')}")
            st.caption(
                f"Providers included: {', '.join(st.session_state.get('last_selected_providers', []))}"
            )
            for plan in ranked:
                with st.container():
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"### {plan.rank}. {plan.provider_name} - {plan.offering_id}")
                        st.caption(plan.why)
                    with c2:
                        st.metric("Monthly Cost", f"${plan.monthly_cost_usd:,.0f}")
                        st.metric("Score", f"{plan.score:,.1f}")
                        st.metric("Confidence", plan.confidence)

                    if plan.utilization_at_peak is not None:
                        st.progress(min(max(plan.utilization_at_peak, 0.0), 1.0))
                        st.caption(f"Peak utilization: {plan.utilization_at_peak * 100:.0f}%")
                    st.caption(
                        f"Risk: overload={plan.risk.risk_overload:.2f}, complexity={plan.risk.risk_complexity:.2f}, "
                        f"total={plan.risk.total_risk:.2f}"
                    )
                st.caption(
                    f"Assumptions: peak_to_avg={plan.assumptions['peak_to_avg']}, "
                    f"util_target={plan.assumptions['util_target']}, "
                    f"beta={plan.assumptions['scaling_beta']}, "
                    f"output_ratio={plan.assumptions.get('output_token_ratio', 0.30)}"
                )
                st.caption("Latency queueing model is not implemented yet ?")
                st.markdown("---")

            explain_disabled = (not has_llm_key) or (last_workload is None)
            if st.button("Explain top recommendation", disabled=explain_disabled):
                if not has_llm_key:
                    st.error("Set OPENAI_API_KEY or ANTHROPIC_API_KEY to use AI explanations.")
                elif last_workload is None:
                    st.error("Run recommendations first.")
                else:
                    try:
                        router = LLMRouter()
                        top = ranked[0]
                        summary = (
                            f"{top.provider_name} - {top.offering_id}, "
                            f"${top.monthly_cost_usd:.0f}/mo, risk={top.risk.total_risk:.2f}"
                        )
                        st.info(router.explain(summary, last_workload))
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Explanation failed: {exc}")
        else:
            st.info("Set LLM inputs and click Get Recommendations.")

with tab_catalog:
    st.subheader("Pricing Catalog")
    st.caption(f"Current category: {selected_workload.display_name}")

    provider_summary = []
    summary_providers = provider_options if show_all_providers else workload_provider_ids
    for provider in summary_providers:
        rows = [row for row in workload_rows if row.provider == provider]
        dates = sorted([row.source_date for row in rows if row.source_date])
        provider_summary.append(
            {
                "provider": provider,
                "offerings": len({row.sku_key for row in rows}),
                "billing_modes": ", ".join(sorted({row.billing_type for row in rows})),
                "confidence": ", ".join(sorted({row.confidence for row in rows if row.confidence})),
                "latest_source_date": dates[-1] if dates else "",
                "supports_workload": "yes" if provider in workload_provider_ids else "no",
            }
        )
    try:
        st.dataframe(provider_summary, use_container_width=True, hide_index=True)
    except TypeError:
        st.dataframe(provider_summary)

    preview_limit = st.selectbox("Preview row limit", options=[50, 100, 200, 500], index=2)
    filtered_rows = [row for row in workload_rows if row.provider in set(selected_global_providers)]
    if not selected_global_providers:
        st.warning("No providers selected. Pick providers from the workload-aware provider selector at the top.")
    preview_rows = [
        {
            "provider": row.provider,
            "offering": row.sku_name,
            "billing": row.billing_type,
            "price": f"{row.unit_price_usd:.6g} / {row.unit_name}",
            "confidence": row.confidence,
        }
        for row in filtered_rows[:preview_limit]
    ]
    st.caption(f"Showing first {len(preview_rows)} of {len(filtered_rows)} rows.")
    try:
        st.dataframe(preview_rows, use_container_width=True, hide_index=True)
    except TypeError:
        st.dataframe(preview_rows)

    export_rows = [
        {
            "provider": row.provider,
            "sku_key": row.sku_key,
            "sku_name": row.sku_name,
            "billing_type": row.billing_type,
            "model_key": row.model_key,
            "unit_price_usd": row.unit_price_usd,
            "unit_name": row.unit_name,
            "region": row.region,
            "source_date": row.source_date,
            "confidence": row.confidence,
            "source_url": row.source_url,
        }
        for row in filtered_rows
    ]
    st.download_button(
        "Download filtered catalog CSV",
        data=_rows_to_csv_bytes(export_rows),
        file_name=f"{selected_workload.value}_pricing_catalog.csv",
        mime="text/csv",
    )
