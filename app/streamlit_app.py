"""Streamlit UI for InferenceAtlas planning and pricing catalog browsing."""

from __future__ import annotations

import csv
import io
import os
from datetime import datetime, timezone

import streamlit as st

from inference_atlas.catalog_ranking import build_provider_diagnostics, rank_catalog_offers
from inference_atlas.config import TRAFFIC_PATTERN_LABELS, TRAFFIC_PATTERN_PEAK_TO_AVG_DEFAULT
from inference_atlas.data_loader import get_catalog_v2_metadata, get_catalog_v2_rows, get_models
from inference_atlas.invoice_analyzer import analyze_invoice_csv
from inference_atlas.llm import LLMRouter, RouterConfig, WorkloadSpec
from inference_atlas.mvp_planner import get_provider_compatibility, rank_configs
from inference_atlas.workload_types import WorkloadType

st.set_page_config(page_title="InferenceAtlas", layout="centered")
st.title("InferenceAtlas")
st.caption("Category-first workload planning and provider pricing intelligence.")

MODEL_REQUIREMENTS = get_models()
has_llm_key = bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))

label_to_pattern = {label: token for token, label in TRAFFIC_PATTERN_LABELS.items()}


def _get_ask_ia_router() -> LLMRouter:
    """Build Ask IA router lazily to avoid hard-failing without API keys."""
    return LLMRouter(
        config=RouterConfig(primary_provider="opus_4_6", fallback_provider="gpt_5_2")
    )


def _catalog_freshness_days(generated_at_utc: str | None) -> int | None:
    if not generated_at_utc:
        return None
    try:
        generated = datetime.fromisoformat(str(generated_at_utc).replace("Z", "+00:00"))
    except ValueError:
        return None
    return (datetime.now(timezone.utc) - generated).days


def _model_key_to_bucket(model_key: str) -> str:
    model = MODEL_REQUIREMENTS.get(model_key)
    if not model:
        key = str(model_key).lower()
        if any(token in key for token in ["405b", "400b", "390b"]):
            return "405b"
        if any(token in key for token in ["120b", "110b", "100b", "90b", "80b", "72b", "70b", "65b"]):
            return "70b"
        if any(token in key for token in ["50b", "40b", "34b", "32b", "30b", "27b", "22b", "20b"]):
            return "34b"
        if any(token in key for token in ["19b", "18b", "17b", "16b", "15b", "14b", "13b", "12b", "11b", "10b"]):
            return "13b"
        if any(token in key for token in ["9b", "8b", "7b", "6b", "5b", "4b", "3b", "2b", "1b"]):
            return "7b"
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


def _rows_to_csv_bytes(rows: list[dict[str, object]]) -> bytes:
    if not rows:
        return b""
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode("utf-8")


def _format_model_label(model_key: str) -> str:
    """Return a readable model label without changing the underlying model key."""
    key = str(model_key).strip()
    lower = key.lower()
    if lower.startswith("accounts/fireworks/models/"):
        return f"Fireworks - {key.split('/')[-1]}"
    if lower.startswith("accounts/stability/models/"):
        return f"Stability - {key.split('/')[-1]}"
    if "/" in key:
        org, name = key.split("/", 1)
        return f"{org} - {name}"
    return key


def _build_ai_context_workload() -> WorkloadSpec:
    """Build a safe fallback workload context for AI explanation helper."""
    return WorkloadSpec(
        tokens_per_day=float(st.session_state.get("ia_tokens_per_day", 5_000_000.0)),
        pattern=str(st.session_state.get("ia_pattern", "steady")),
        model_key=str(st.session_state.get("ia_model_key", "llama_70b")),
        latency_requirement_ms=None,
    )


def _build_catalog_context(
    *,
    selected_workload: str | None,
    selected_providers: list[str],
    rows: list[object],
    max_rows: int = 40,
) -> str:
    """Build compact, grounded catalog context for AI prompts."""
    selected_provider_set = set(selected_providers)
    filtered = [
        row
        for row in rows
        if (selected_workload is None or row.workload_type == selected_workload)
        and row.provider in selected_provider_set
    ]
    if not filtered:
        return "No matching catalog rows for current workload/provider filters."

    # Sort by unit price for a stable, compact context.
    filtered.sort(key=lambda row: row.unit_price_usd)
    sample = filtered[:max_rows]
    providers = sorted({row.provider for row in filtered})
    units = sorted({row.unit_name for row in filtered})

    lines = [
        f"workload={selected_workload if selected_workload is not None else 'all'}",
        f"providers={','.join(providers)}",
        f"rows_total={len(filtered)}",
        f"units={','.join(units)}",
        "rows_sample_start",
    ]
    for row in sample:
        lines.append(
            "|".join(
                [
                    row.provider,
                    row.sku_name,
                    row.model_key,
                    row.billing_mode,
                    f"{row.unit_price_usd:.8g}",
                    row.unit_name,
                    row.region,
                    row.confidence,
                    row.source_kind,
                ]
            )
        )
    lines.append("rows_sample_end")
    return "\n".join(lines)


def _infer_workload_from_ai_text(ai_text: str, default_workload: str) -> str:
    """Infer workload intent from free-form AI helper text."""
    text = ai_text.lower()
    keyword_map = {
        "speech_to_text": ["speech to text", "speech-to-text", "stt", "transcription"],
        "text_to_speech": ["text to speech", "text-to-speech", "tts", "voice synthesis"],
        "embeddings": ["embedding", "vector search", "semantic search", "retrieval"],
        "image_generation": ["image generation", "text to image", "text-to-image", "diffusion"],
        "vision": ["vision", "image understanding", "ocr", "visual qa", "multimodal"],
        "video_generation": ["video generation", "text to video", "text-to-video"],
        "moderation": ["moderation", "safety", "content filter"],
        "llm": ["llm", "chat", "completion", "text generation", "inference"],
    }
    for workload, keywords in keyword_map.items():
        if any(keyword in text for keyword in keywords):
            return workload
    return default_workload


all_rows = get_catalog_v2_rows()
available_workloads = sorted({row.workload_type for row in all_rows})
preferred_order = [
    "llm",
    "speech_to_text",
    "text_to_speech",
    "embeddings",
    "image_generation",
    "vision",
    "video_generation",
    "moderation",
]
ordered_workloads = [w for w in preferred_order if w in available_workloads] + [
    w for w in available_workloads if w not in preferred_order
]
known_display = {
    "llm": "LLM Inference",
    "speech_to_text": "Speech-to-Text",
    "text_to_speech": "Text-to-Speech",
    "embeddings": "Embeddings",
    "image_generation": "Image Generation",
    "vision": "Vision",
    "video_generation": "Video Generation",
    "moderation": "Moderation",
}
workload_options = {
    known_display.get(token, token.replace("_", " ").title()): token
    for token in ordered_workloads
}

selected_workload_label = st.selectbox(
    "1. What are you optimizing for?",
    options=list(workload_options),
    index=0,
    help="Choose workload category first. The rest of the flow is filtered by this choice.",
)
selected_workload = workload_options[selected_workload_label]
workload_rows = get_catalog_v2_rows(selected_workload)
workload_provider_ids = sorted({row.provider for row in workload_rows})
selected_global_providers = workload_provider_ids
llm_catalog_model_keys = sorted({row.model_key for row in get_catalog_v2_rows("llm") if row.model_key})

meta = get_catalog_v2_metadata()
generated_at = str(meta.get("generated_at_utc") or "")
freshness_days = _catalog_freshness_days(generated_at)

with st.sidebar:
    st.header("Ask IA AI")

    with st.expander("AI Suggest", expanded=False):
        if not has_llm_key:
            st.caption(
                "Set OPENAI_API_KEY or ANTHROPIC_API_KEY to enable AI parsing, "
                "explanations, and what-if suggestions."
            )
        ai_text = st.text_area(
            "Ask AI to help configure this workload",
            placeholder=(
                "e.g. I need low-cost text-to-speech for 2M chars/month, which providers "
                "should I select and what budget should I set?"
            ),
            height=80,
            key="ai_helper_input_text",
        )
        if st.button(
            "AI: Suggest next steps",
            disabled=not has_llm_key,
        ):
            try:
                ai_workload = _infer_workload_from_ai_text(ai_text, selected_workload)
                ai_providers = selected_global_providers
                if ai_workload != selected_workload:
                    ai_providers = sorted(
                        {
                            row.provider
                            for row in all_rows
                            if row.workload_type == ai_workload
                        }
                    )
                catalog_context = _build_catalog_context(
                    selected_workload=ai_workload,
                    selected_providers=ai_providers,
                    rows=all_rows,
                )
                prompt = (
                    "You are IA AI. Use ONLY the provided catalog context. "
                    "If data is missing, say 'not available in current catalog'. "
                    "Do not invent providers/SKUs/prices.\n\n"
                    f"Context:\n{catalog_context}\n\n"
                    f"User asks: {ai_text}\n"
                    "Answer in concise bullets with provider/sku/price citations from context."
                )
                st.info(_get_ask_ia_router().explain(prompt, _build_ai_context_workload()))
            except Exception as exc:  # noqa: BLE001
                st.error(f"AI assistant failed: {exc}")

    if freshness_days is None:
        st.warning("Catalog freshness unknown. Run sync to ensure data is current.")
    elif freshness_days > 3:
        st.warning(
            f"Catalog is stale ({freshness_days} days old). "
            f"Last sync: {generated_at}. Run daily sync."
        )

opt_tab, catalog_tab, invoice_tab = st.tabs(
    ["Optimize Workload", "Browse Pricing Catalog", "Invoice Analyzer"]
)

with opt_tab:
    if selected_workload != "llm":
        st.subheader("Top Offers (Catalog Ranker - Beta)")
        st.caption(
            "This optimizer mode ranks by listed unit price only. "
            "Throughput/SLA-aware optimization is not implemented yet."
        )
        available_units = sorted({row.unit_name for row in workload_rows})
        available_models = sorted({row.model_key for row in workload_rows if row.model_key})
        with st.form("optimize_non_llm"):
            selected_model = st.selectbox(
                "Model filter",
                options=["All models", *available_models],
                help="Optional: focus ranking to one model/provider family.",
            )
            comparator_mode = "normalized"
            confidence_weighted = True
            selected_unit = st.selectbox(
                "Unit filter",
                options=["All units", *available_units],
                help="Cross-unit normalization is not implemented yet. Filter by a single unit for clean comparisons.",
            )
            monthly_usage = st.number_input(
                "Monthly usage estimate",
                min_value=0.0,
                value=0.0,
                step=1000.0,
                help="Optional. Cost estimate only applies when a single unit filter is selected.",
            )
            monthly_budget_max = st.number_input(
                "Max monthly budget (USD, optional)",
                min_value=0.0,
                value=0.0,
                step=100.0,
            )
            with st.expander("Advanced options", expanded=False):
                comparator_mode = st.selectbox(
                    "Comparator",
                    options=["normalized", "raw"],
                    format_func=lambda value: (
                        "Normalized workload comparator (recommended)"
                        if value == "normalized"
                        else "Raw listed unit price"
                    ),
                    help="Normalized comparator ranks only offers with workload-comparable units.",
                )
                confidence_weighted = st.checkbox(
                    "Confidence-weighted ranking",
                    value=True,
                    help="Apply a penalty to lower-confidence data before ranking.",
                )
            non_llm_submit = st.form_submit_button("Get Top 10 Offers")

        if non_llm_submit:
            unit_filter = None if selected_unit == "All units" else selected_unit
            rows_for_rank = workload_rows
            if selected_model != "All models":
                rows_for_rank = [row for row in workload_rows if row.model_key == selected_model]
            ranked, provider_reasons, excluded_offer_count = rank_catalog_offers(
                rows=rows_for_rank,
                allowed_providers=set(selected_global_providers),
                unit_name=unit_filter,
                top_k=10,
                monthly_budget_max_usd=float(monthly_budget_max),
                comparator_mode=comparator_mode,
                confidence_weighted=confidence_weighted,
                workload_type=selected_workload,
                monthly_usage=float(monthly_usage),
            )
            if excluded_offer_count > 0:
                st.warning(
                    f"{excluded_offer_count} offers were excluded by normalization/budget filters."
                )
            if not ranked:
                st.warning("No offers matched the selected providers/unit/budget filter.")
            else:
                table_rows = []
                for idx, ranked_row in enumerate(ranked, start=1):
                    table_rows.append(
                        {
                            "rank": idx,
                            "provider": ranked_row.provider,
                            "offering": ranked_row.offering,
                            "billing": ranked_row.billing,
                            "listed_unit_price": ranked_row.listed_unit_price,
                            "comparator_price": round(ranked_row.comparator_price, 8),
                            "unit_name": ranked_row.unit_name,
                            "confidence": ranked_row.confidence,
                            "monthly_estimate_usd": (
                                round(ranked_row.monthly_estimate_usd, 2)
                                if ranked_row.monthly_estimate_usd is not None
                                else None
                            ),
                        }
                    )
                try:
                    st.dataframe(table_rows, use_container_width=True, hide_index=True)
                except TypeError:
                    st.dataframe(table_rows)

            diagnostics = build_provider_diagnostics(
                workload_provider_ids=workload_provider_ids,
                selected_global_providers=selected_global_providers,
                provider_reasons=provider_reasons,
            )
            with st.expander("Provider inclusion diagnostics", expanded=False):
                try:
                    st.dataframe(diagnostics, use_container_width=True, hide_index=True)
                except TypeError:
                    st.dataframe(diagnostics)
                st.download_button(
                    "Download diagnostics CSV",
                    data=_rows_to_csv_bytes(diagnostics),
                    file_name=f"{selected_workload}_provider_diagnostics.csv",
                    mime="text/csv",
                )
    else:
        ai_defaults = st.session_state.get("ai_defaults", {})

        with st.form("optimize_inputs"):
            model_options = []
            model_items = list(MODEL_REQUIREMENTS.items())
            for model_key, model_data in model_items:
                model_options.append(
                    {
                        "display": f"{model_data['display_name']} (curated)",
                        "model_key": model_key,
                    }
                )
            curated_model_keys = set(MODEL_REQUIREMENTS)
            for model_key in llm_catalog_model_keys:
                if model_key not in curated_model_keys:
                    model_options.append(
                        {
                            "display": f"{_format_model_label(model_key)} (catalog)",
                            "model_key": model_key,
                        }
                    )
            if not model_options:
                st.error("No LLM models are available in curated or catalog data.")
                st.stop()
            model_key_to_index = {
                option["model_key"]: idx for idx, option in enumerate(model_options)
            }
            default_model_key = str(ai_defaults.get("model_key", model_options[0]["model_key"]))
            default_model_index = model_key_to_index.get(default_model_key, 0)

            model_display_name = st.selectbox(
                "Model",
                options=[option["display"] for option in model_options],
                index=default_model_index,
            )
            model_key = next(
                option["model_key"]
                for option in model_options
                if option["display"] == model_display_name
            )

            tokens_per_day = st.number_input(
                "Traffic (tokens/day)",
                min_value=1.0,
                value=float(ai_defaults.get("tokens_per_day", 5_000_000.0)),
                step=100_000.0,
            )
            st.session_state["ia_tokens_per_day"] = float(tokens_per_day)
            pattern_label = st.selectbox(
                "Traffic pattern",
                options=[TRAFFIC_PATTERN_LABELS[key] for key in TRAFFIC_PATTERN_LABELS],
                index=list(TRAFFIC_PATTERN_LABELS).index(
                    str(ai_defaults.get("pattern", "steady"))
                    if str(ai_defaults.get("pattern", "steady")) in set(TRAFFIC_PATTERN_LABELS)
                    else "steady"
                ),
            )
            st.session_state["ia_pattern"] = label_to_pattern[pattern_label]
            st.session_state["ia_model_key"] = model_key

            model_bucket_preview = _model_key_to_bucket(model_key)
            compatibility = get_provider_compatibility(
                model_bucket=model_bucket_preview,
                provider_ids=set(workload_provider_ids),
            )
            compatibility_by_provider = {row.provider_id: row for row in compatibility}
            llm_provider_options = sorted(selected_global_providers)
            compatible_filtered = sorted(
                row.provider_id
                for row in compatibility
                if row.compatible and row.provider_id in set(selected_global_providers)
            )
            incompatible_filtered = [
                row
                for row in compatibility
                if not row.compatible and row.provider_id in set(selected_global_providers)
            ]
            monthly_budget_max = 0.0

            selected_provider_ids = st.multiselect(
                f"Providers to include ({len(llm_provider_options)} available for LLM)",
                options=llm_provider_options,
                default=[],
                help="Incompatible providers are automatically excluded during ranking and shown in diagnostics.",
            )
            if not selected_provider_ids:
                st.info("No providers selected - pick one or more to continue.")

            with st.expander("Advanced options", expanded=False):
                monthly_budget_max = st.number_input(
                    "Max monthly budget (USD, optional)",
                    min_value=0.0,
                    value=0.0,
                    step=100.0,
                )
                if incompatible_filtered:
                    st.caption(
                        "Excluded by model compatibility: "
                        + ", ".join(
                            f"{row.provider_id} ({row.reason})" for row in incompatible_filtered
                        )
                    )

            submit = st.form_submit_button("Get Top 10 Recommendations")

        if submit:
            diagnostics = []
            selected_set = set(selected_global_providers)
            selected_compatible_set = set(compatible_filtered)
            for provider_id in workload_provider_ids:
                if provider_id not in selected_set:
                    diagnostics.append(
                        {"provider": provider_id, "status": "excluded", "reason": "Not selected by user."}
                    )
                    continue
                diag = compatibility_by_provider.get(provider_id)
                if diag is None:
                    diagnostics.append(
                        {
                            "provider": provider_id,
                            "status": "excluded",
                            "reason": "No compatibility diagnostics available.",
                        }
                    )
                    continue
                if provider_id in selected_compatible_set:
                    diagnostics.append(
                        {
                            "provider": provider_id,
                            "status": "included",
                            "reason": "Model-compatible and selected.",
                        }
                    )
                else:
                    diagnostics.append(
                        {"provider": provider_id, "status": "excluded", "reason": diag.reason}
                    )
            st.session_state["llm_provider_diagnostics"] = diagnostics

            if not selected_provider_ids:
                st.error("No compatible providers selected. Choose at least one provider.")
            else:
                try:
                    ranked_plans = rank_configs(
                        tokens_per_day=float(tokens_per_day),
                        model_bucket=model_bucket_preview,
                        peak_to_avg=float(
                            TRAFFIC_PATTERN_PEAK_TO_AVG_DEFAULT[label_to_pattern[pattern_label]]
                        ),
                        top_k=10,
                        provider_ids=set(selected_provider_ids),
                    )
                    if monthly_budget_max > 0:
                        ranked_plans = [
                            plan for plan in ranked_plans if plan.monthly_cost_usd <= float(monthly_budget_max)
                        ]
                    st.session_state["last_ranked_plans"] = ranked_plans
                    st.session_state["last_selected_providers"] = selected_provider_ids
                    st.session_state["last_budget"] = float(monthly_budget_max)
                    st.session_state["last_workload"] = WorkloadSpec(
                        tokens_per_day=float(tokens_per_day),
                        pattern=label_to_pattern[pattern_label],
                        model_key=model_key,
                        latency_requirement_ms=None,
                    )
                except ValueError as exc:
                    st.error(f"{exc}. Try broader provider selection or another model/bucket.")
                    st.session_state["last_ranked_plans"] = []

        llm_diagnostics = st.session_state.get("llm_provider_diagnostics", [])
        if llm_diagnostics:
            with st.expander("Provider diagnostics", expanded=False):
                try:
                    st.dataframe(llm_diagnostics, use_container_width=True, hide_index=True)
                except TypeError:
                    st.dataframe(llm_diagnostics)

        ranked = st.session_state.get("last_ranked_plans", [])
        if ranked:
            st.subheader("Top Recommendations")
            st.caption(
                f"Providers included: {', '.join(st.session_state.get('last_selected_providers', []))}"
            )
            last_budget = float(st.session_state.get("last_budget", 0.0))
            if last_budget > 0:
                st.caption(f"Budget filter: <= ${last_budget:,.0f}/month")

            for plan in ranked:
                with st.container():
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"### {plan.rank}. {plan.provider_name} - {plan.offering_id}")
                    with c2:
                        st.metric("Monthly Cost", f"${plan.monthly_cost_usd:,.0f}")
                        st.metric("Confidence", plan.confidence)

                    if plan.utilization_at_peak is not None:
                        st.progress(min(max(plan.utilization_at_peak, 0.0), 1.0))
                        st.caption(f"Peak utilization: {plan.utilization_at_peak * 100:.0f}%")

                    if plan.risk.total_risk < 0.3:
                        st.success("Risk: Low")
                    elif plan.risk.total_risk < 0.6:
                        st.warning("Risk: Medium")
                    else:
                        st.error("Risk: High")

                    with st.expander("Why this? / Assumptions", expanded=False):
                        st.caption(plan.why)
                        assumptions_line = ", ".join(
                            f"{key}={value}"
                            for key, value in sorted(plan.assumptions.items())
                        )
                        st.caption(f"Assumptions: {assumptions_line}")
                st.markdown("---")
        else:
            st.info("Set LLM inputs and click Get Top 10 Recommendations.")

with catalog_tab:
    st.subheader("Pricing Catalog")
    st.caption(f"Current category: {selected_workload_label}")

    provider_summary = []
    summary_providers = workload_provider_ids
    for provider in summary_providers:
        rows = [row for row in workload_rows if row.provider == provider]
        dates = sorted([row.source_date for row in rows if row.source_date])
        provider_summary.append(
            {
                "provider": provider,
                "offerings": len({row.sku_key for row in rows}),
                "billing_modes": ", ".join(sorted({row.billing_mode for row in rows})),
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
    preview_rows = [
        {
            "provider": row.provider,
            "offering": row.sku_name,
            "billing": row.billing_mode,
            "price": f"{row.unit_price_usd:.6g} / {row.unit_name}",
            "confidence": row.confidence,
            "source_kind": row.source_kind,
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
            "billing_mode": row.billing_mode,
            "model_key": row.model_key,
            "unit_price_usd": row.unit_price_usd,
            "unit_name": row.unit_name,
            "region": row.region,
            "source_date": row.source_date,
            "confidence": row.confidence,
            "source_kind": row.source_kind,
            "source_url": row.source_url,
        }
        for row in filtered_rows
    ]
    st.download_button(
        "Download filtered catalog CSV",
        data=_rows_to_csv_bytes(export_rows),
        file_name=f"{selected_workload}_pricing_catalog.csv",
        mime="text/csv",
    )

with invoice_tab:
    st.subheader("Invoice Analyzer (Beta)")
    st.caption(
        "Upload invoice CSV and compare effective unit prices against current catalog rows."
    )
    st.caption(
        "Required columns: provider, workload_type, usage_qty, usage_unit, amount_usd"
    )
    uploaded = st.file_uploader(
        "Upload invoice CSV",
        type=["csv"],
        accept_multiple_files=False,
    )
    if uploaded is not None:
        try:
            suggestions, summary = analyze_invoice_csv(uploaded.getvalue(), all_rows)
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Invoice Spend", f"${summary['total_spend_usd']:,.2f}")
            with c2:
                st.metric("Potential Savings", f"${summary['total_estimated_savings_usd']:,.2f}")

            if not suggestions:
                st.info("No savings opportunities found with current catalog match rules.")
            else:
                try:
                    st.dataframe(suggestions[:25], use_container_width=True, hide_index=True)
                except TypeError:
                    st.dataframe(suggestions[:25])
                st.download_button(
                    "Download invoice recommendations CSV",
                    data=_rows_to_csv_bytes(suggestions),
                    file_name="invoice_savings_recommendations.csv",
                    mime="text/csv",
                )
        except ValueError as exc:
            st.error(str(exc))
