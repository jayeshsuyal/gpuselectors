"""Streamlit UI for InferenceAtlas planning and pricing catalog browsing."""

from __future__ import annotations

import csv
import io
import os
from datetime import datetime, timezone

import streamlit as st

from inference_atlas import (
    enumerate_configs,
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
st.caption("Workload-aware provider comparison and LLM deployment optimization.")

MODEL_REQUIREMENTS = get_models()
has_llm_key = bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))

pattern_to_label = {
    "steady": "Steady",
    "business_hours": "Business Hours",
    "bursty": "Bursty",
}
label_to_pattern = {
    "Steady": "steady",
    "Business Hours": "business_hours",
    "Bursty": "bursty",
}
pattern_to_peak_to_avg = {
    "steady": 1.5,
    "business_hours": 2.5,
    "bursty": 3.5,
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
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


tab_optimize, tab_catalog = st.tabs(["Optimize LLM Deployment", "Browse Pricing Catalog"])

with tab_optimize:
    parsed_workload = st.session_state.get("parsed_workload", {})

    with st.expander("Describe workload in plain English"):
        user_text = st.text_area(
            "Describe your LLM deployment needs",
            placeholder=(
                "e.g., Chat app with 10k daily users, steady traffic, Llama 70B, need <200ms latency"
            ),
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

    hf_meta_for_banner: dict[str, object] | None = None
    try:
        hf_meta_for_banner = get_huggingface_catalog_metadata()
        if _catalog_is_stale(str(hf_meta_for_banner.get("generated_at_utc") or None)):
            st.warning(
                "Hugging Face model catalog is stale. Open the catalog status panel and sync to refresh."
            )
    except Exception:
        st.warning("Hugging Face model catalog is unavailable. Sync is recommended before using HF models.")

    with st.expander("Hugging Face Catalog Status"):
        try:
            hf_meta = hf_meta_for_banner or get_huggingface_catalog_metadata()
            st.write(f"Last sync (UTC): `{hf_meta.get('generated_at_utc', 'unknown')}`")
            st.write(f"Model count: `{hf_meta.get('model_count', 0)}`")
            st.write(f"Source: `{hf_meta.get('source', 'unknown')}`")
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
                    synced_models = fetch_huggingface_models(limit=sync_limit, token=token)
                    write_huggingface_catalog(synced_models)
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
            "Model",
            options=model_display_names,
            index=default_index,
            help="Choose a curated model from the built-in model catalog.",
        )
        model_key = model_key_by_display[model_display_name]

        tokens_per_day = st.number_input(
            "Traffic (tokens/day)",
            min_value=1.0,
            value=float(parsed_workload.get("tokens_per_day", 5_000_000.0)),
            step=100_000.0,
            help="Total tokens processed per day (input + output).",
        )
        pattern_label = st.selectbox(
            "Traffic pattern",
            ["Steady", "Business Hours", "Bursty"],
            index=["Steady", "Business Hours", "Bursty"].index(
                pattern_to_label.get(str(parsed_workload.get("pattern", "steady")), "Steady")
            ),
            help="Steady: uniform load. Business Hours: office-hour heavy. Bursty: spiky load.",
        )
        model_source = st.radio(
            "Model source",
            options=["Curated models", "Hugging Face models"],
            index=0,
            help=(
                "Curated: built-in model set (Llama/Mistral etc). "
                "Hugging Face: API-synced open-source model catalog."
            ),
        )

        hf_model_choice = None
        if model_source == "Hugging Face models":
            hf_models = get_huggingface_models(min_downloads=1000, include_gated=False)
            if hf_models:
                hf_model_labels = [
                    f"{m['model_id']} (downloads: {m['downloads']:,}, bucket: {m['size_bucket']})"
                    for m in hf_models[:100]
                ]
                by_label = {label: hf_models[idx] for idx, label in enumerate(hf_model_labels)}
                hf_label = st.selectbox(
                    "Open-source model (Hugging Face)",
                    options=hf_model_labels,
                    help="Top downloaded Hugging Face models from local synced catalog.",
                )
                hf_model_choice = by_label[hf_label]
            else:
                st.caption("No HF models available locally. Sync from the panel above.")

        selected_model_bucket_preview = _model_key_to_bucket(model_key)
        if hf_model_choice is not None:
            selected_model_bucket_preview = str(hf_model_choice["size_bucket"])

        compatible_provider_ids = sorted(
            {cfg.provider_id for cfg in enumerate_configs(selected_model_bucket_preview)}
        )
        selected_provider_ids = st.multiselect(
            f"Providers to include ({len(compatible_provider_ids)} compatible)",
            options=compatible_provider_ids,
            default=compatible_provider_ids,
            help="Only providers compatible with selected model bucket are shown.",
        )

        with st.expander("Advanced options"):
            latency_requirement_ms = st.number_input(
                "Latency requirement (ms, optional)",
                min_value=0.0,
                value=float(parsed_workload.get("latency_requirement_ms") or 0.0),
                step=10.0,
                help="Set to 0 to ignore latency in analysis.",
            )
            top_k = st.slider(
                "Top K results",
                min_value=1,
                max_value=10,
                value=3,
                help="Number of ranked configurations to return.",
            )

        submit = st.form_submit_button("Get Recommendations")

    if submit:
        if not selected_provider_ids:
            st.error("No providers selected. Choose at least one provider in the provider selector.")
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

            try:
                ranked_plans = rank_configs(
                    tokens_per_day=effective_workload.tokens_per_day,
                    model_bucket=selected_model_bucket,
                    peak_to_avg=pattern_to_peak_to_avg[effective_workload.pattern],
                    top_k=int(top_k),
                    provider_ids=set(selected_provider_ids),
                )
            except ValueError as exc:
                st.error(f"{exc}. Try a different model source, model, or provider set.")
                ranked_plans = []

            st.session_state["last_ranked_plans"] = ranked_plans
            st.session_state["last_model_source"] = model_source
            st.session_state["last_selected_providers"] = list(selected_provider_ids)

    last_ranked_plans = st.session_state.get("last_ranked_plans", [])
    last_workload = st.session_state.get("last_workload")
    if last_ranked_plans:
        st.subheader("Top Recommendations")
        last_model_source = st.session_state.get("last_model_source")
        if last_model_source:
            st.caption(f"Model source: {last_model_source}")
        selected_providers = st.session_state.get("last_selected_providers", [])
        if selected_providers:
            st.caption(f"Providers included: {', '.join(selected_providers)}")
        if last_model_source == "Hugging Face models":
            try:
                hf_meta = get_huggingface_catalog_metadata()
                st.caption(
                    f"Open-source catalog freshness: {hf_meta.get('generated_at_utc', 'unknown')} UTC "
                    f"(models={hf_meta.get('model_count', 0)})"
                )
            except Exception:
                pass

        for plan in last_ranked_plans:
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
                    ratio = min(max(plan.utilization_at_peak, 0.0), 1.0)
                    st.progress(ratio)
                    st.caption(f"Peak utilization: {plan.utilization_at_peak * 100:.0f}%")
                st.caption(
                    f"Risk: overload={plan.risk.risk_overload:.2f}, "
                    f"complexity={plan.risk.risk_complexity:.2f}, total={plan.risk.total_risk:.2f}"
                )
                st.caption(
                    f"Assumptions: peak_to_avg={plan.assumptions['peak_to_avg']}, "
                    f"util_target={plan.assumptions['util_target']}, beta={plan.assumptions['scaling_beta']}"
                )
                st.markdown("---")

        explain_disabled = (not has_llm_key) or (last_workload is None)
        if st.button("Explain this recommendation", disabled=explain_disabled):
            if not has_llm_key:
                st.error("Set OPENAI_API_KEY or ANTHROPIC_API_KEY to use AI explanations.")
            elif last_workload is None:
                st.error("Run recommendations first to generate an explanation.")
            else:
                try:
                    router = LLMRouter()
                    top_plan = last_ranked_plans[0]
                    summary = (
                        f"{top_plan.provider_name} - {top_plan.offering_id}, "
                        f"${top_plan.monthly_cost_usd:.0f}/mo, risk={top_plan.risk.total_risk:.2f}"
                    )
                    explanation = router.explain(summary, last_workload)
                    st.session_state["last_explanation"] = explanation
                    st.info(explanation)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Explanation failed: {exc}")
        if st.session_state.get("last_explanation"):
            st.info(st.session_state["last_explanation"])
    else:
        st.info("Enter LLM workload details and click Get Recommendations.")

with tab_catalog:
    workload_options = {
        WorkloadType.LLM.display_name: WorkloadType.LLM,
        WorkloadType.SPEECH_TO_TEXT.display_name: WorkloadType.SPEECH_TO_TEXT,
        WorkloadType.TEXT_TO_SPEECH.display_name: WorkloadType.TEXT_TO_SPEECH,
        WorkloadType.EMBEDDINGS.display_name: WorkloadType.EMBEDDINGS,
        WorkloadType.IMAGE_GENERATION.display_name: WorkloadType.IMAGE_GENERATION,
        WorkloadType.VISION.display_name: WorkloadType.VISION,
    }
    selected_workload_label = st.selectbox(
        "Workload type",
        options=list(workload_options),
        index=0,
        help="Browse providers and pricing rows for a specific workload type.",
    )
    selected_workload = workload_options[selected_workload_label]
    workload_rows = get_pricing_records(selected_workload)
    providers_in_workload = sorted({row.provider for row in workload_rows})

    st.markdown("**Provider Coverage**")
    st.caption(
        f"{len(providers_in_workload)} providers found for {selected_workload.display_name.lower()}."
    )
    provider_summary = []
    for provider in providers_in_workload:
        rows = [row for row in workload_rows if row.provider == provider]
        dates = sorted([row.source_date for row in rows if row.source_date])
        provider_summary.append(
            {
                "provider": provider,
                "offerings": len({row.sku_key for row in rows}),
                "billing_modes": ", ".join(sorted({row.billing_type for row in rows})),
                "confidence": ", ".join(sorted({row.confidence for row in rows if row.confidence})),
                "latest_source_date": dates[-1] if dates else "",
            }
        )
    try:
        st.dataframe(provider_summary, use_container_width=True, hide_index=True)
    except TypeError:
        st.dataframe(provider_summary)

    provider_filter = st.multiselect(
        "Filter providers",
        options=providers_in_workload,
        default=providers_in_workload,
    )
    preview_limit = st.selectbox("Preview row limit", options=[50, 100, 200, 500], index=2)
    filtered_rows = [row for row in workload_rows if row.provider in set(provider_filter)]
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

    raw_export_rows = [
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
        data=_rows_to_csv_bytes(raw_export_rows),
        file_name=f"{selected_workload.value}_pricing_catalog.csv",
        mime="text/csv",
    )
