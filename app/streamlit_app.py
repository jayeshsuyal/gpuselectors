"""Streamlit UI for InferenceAtlas planning and pricing catalog browsing."""

from __future__ import annotations

import csv
import io
import os
from datetime import datetime, timezone

import streamlit as st

from inference_atlas import (
    analyze_invoice_csv,
    get_catalog_v2_rows,
    get_provider_compatibility,
    rank_configs,
)
from inference_atlas.config import TRAFFIC_PATTERN_LABELS, TRAFFIC_PATTERN_PEAK_TO_AVG_DEFAULT
from inference_atlas.data_loader import get_catalog_v2_metadata, get_models
from inference_atlas.llm import LLMRouter, RouterConfig, WorkloadSpec
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


def _rank_catalog_offers(
    rows: list[object],
    allowed_providers: set[str],
    unit_name: str | None,
    top_k: int,
    monthly_budget_max_usd: float,
) -> list[object]:
    filtered = [row for row in rows if row.provider in allowed_providers]
    if unit_name:
        filtered = [row for row in filtered if row.unit_name == unit_name]
    if monthly_budget_max_usd > 0 and unit_name:
        filtered = [row for row in filtered if row.unit_price_usd <= monthly_budget_max_usd]
    filtered.sort(key=lambda row: row.unit_price_usd)
    return filtered[:top_k]


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
    selected_workload: str,
    selected_providers: list[str],
    rows: list[object],
    max_rows: int = 40,
) -> str:
    """Build compact, grounded catalog context for AI prompts."""
    filtered = [
        row
        for row in rows
        if row.workload_type == selected_workload and row.provider in set(selected_providers)
    ]
    if not filtered:
        return "No matching catalog rows for current workload/provider filters."

    # Sort by unit price for a stable, compact context.
    filtered.sort(key=lambda row: row.unit_price_usd)
    sample = filtered[:max_rows]
    providers = sorted({row.provider for row in filtered})
    units = sorted({row.unit_name for row in filtered})

    lines = [
        f"workload={selected_workload}",
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

meta = get_catalog_v2_metadata()
generated_at = str(meta.get("generated_at_utc") or "")
freshness_days = _catalog_freshness_days(generated_at)
if freshness_days is None:
    st.warning("Catalog freshness unknown. Run sync to ensure data is current.")
elif freshness_days > 3:
    st.warning(
        f"Catalog is stale ({freshness_days} days old). "
        f"Last sync: {generated_at}. Run daily sync."
    )
else:
    st.caption(f"Catalog freshness: {freshness_days} day(s) old. Last sync: {generated_at}")
all_provider_ids = sorted({row.provider for row in all_rows})
all_provider_count = len(all_provider_ids)

coverage_tokens = []
for workload in (
    "llm",
    "embeddings",
    "text_to_speech",
    "speech_to_text",
):
    count = len({row.provider for row in all_rows if row.workload_type == workload})
    coverage_tokens.append(f"{workload}: {count}/{all_provider_count}")

selected_coverage_count = len(workload_provider_ids)
st.caption(
    f"Coverage ({selected_workload}): {selected_coverage_count}/{all_provider_count} providers | "
    + " | ".join(coverage_tokens)
)

page_mode = st.selectbox(
    "2. Choose view",
    options=["Optimize Workload", "Browse Pricing Catalog", "Invoice Analyzer"],
    index=0,
    help="Catalog is hidden unless you explicitly choose it here.",
)

selected_global_providers = st.multiselect(
    f"3. Providers ({len(workload_provider_ids)} available for {selected_workload})",
    options=workload_provider_ids,
    default=workload_provider_ids,
    help="This provider filter is shared across optimizer and catalog views.",
)
if not selected_global_providers:
    st.info("Select one or more providers to continue.")

with st.expander("AI Assistant (optional)"):
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
    if st.button("AI: Suggest next steps", disabled=not has_llm_key):
        try:
            catalog_context = _build_catalog_context(
                selected_workload=selected_workload,
                selected_providers=selected_global_providers,
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

if page_mode == "Optimize Workload":
    if selected_workload != "llm":
        st.subheader("Top Offers (Catalog Ranker - Beta)")
        st.caption(
            "This optimizer mode ranks by listed unit price only. "
            "Throughput/SLA-aware optimization is not implemented yet ?"
        )
        available_units = sorted({row.unit_name for row in workload_rows})
        with st.form("optimize_non_llm"):
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
            monthly_budget_max = st.number_input(
                "Max monthly budget (USD, optional)",
                min_value=0.0,
                value=0.0,
                step=100.0,
            )
            non_llm_submit = st.form_submit_button("Get Top 10 Offers")

        if non_llm_submit:
            if not selected_global_providers:
                st.error("No providers selected. Choose at least one provider.")
            else:
                unit_filter = None if selected_unit == "All units" else selected_unit
                ranked = _rank_catalog_offers(
                    rows=workload_rows,
                    allowed_providers=set(selected_global_providers),
                    unit_name=unit_filter,
                    top_k=10,
                    monthly_budget_max_usd=float(monthly_budget_max),
                )
                if not ranked:
                    st.warning("No offers matched the selected providers/unit/budget filter.")
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
                                "billing": row.billing_mode,
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
        ai_defaults = st.session_state.get("ai_defaults", {})

        with st.form("optimize_inputs"):
            model_items = list(MODEL_REQUIREMENTS.items())
            model_display_names = [v["display_name"] for _, v in model_items]
            model_key_by_display = {v["display_name"]: k for k, v in model_items}
            default_model_key = str(ai_defaults.get("model_key", model_items[0][0]))
            if default_model_key not in MODEL_REQUIREMENTS:
                default_model_key = model_items[0][0]
            default_model_index = next(
                idx for idx, (model_key, _) in enumerate(model_items) if model_key == default_model_key
            )

            model_display_name = st.selectbox(
                "Model (curated)",
                options=model_display_names,
                index=default_model_index,
            )
            model_key = model_key_by_display[model_display_name]

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
            monthly_budget_max = st.number_input(
                "Max monthly budget (USD, optional)",
                min_value=0.0,
                value=0.0,
                step=100.0,
            )

            model_bucket_preview = _model_key_to_bucket(model_key)
            compatibility = get_provider_compatibility(
                model_bucket=model_bucket_preview,
                provider_ids=set(selected_global_providers),
            )
            compatible_filtered = sorted(
                row.provider_id for row in compatibility if row.compatible
            )
            incompatible_filtered = [row for row in compatibility if not row.compatible]

            selected_provider_ids = st.multiselect(
                f"Providers to include ({len(compatible_filtered)} compatible after filters)",
                options=compatible_filtered,
                default=compatible_filtered,
                help="Top 10 recommendations will be ranked across selected compatible providers.",
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
                st.markdown("---")
        else:
            st.info("Set LLM inputs and click Get Top 10 Recommendations.")

if page_mode == "Browse Pricing Catalog":
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
    if not selected_global_providers:
        st.warning("No providers selected. Pick providers from the provider selector at the top.")
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

if page_mode == "Invoice Analyzer":
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

st.markdown("---")
st.subheader("Ask IA AI")
if "ia_chat_history" not in st.session_state:
    st.session_state["ia_chat_history"] = []

for message in st.session_state["ia_chat_history"][-6:]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

chat_prompt = None
if hasattr(st, "chat_input"):
    chat_prompt = st.chat_input("Ask about providers, pricing, workload setup, and trade-offs...")
else:
    fallback_prompt = st.text_input(
        "Ask IA AI",
        value="",
        key="ia_chat_prompt_fallback",
    )
    if st.button("Send", key="ia_chat_send_fallback"):
        chat_prompt = fallback_prompt.strip()

if chat_prompt:
    st.session_state["ia_chat_history"].append({"role": "user", "content": chat_prompt})
    if not has_llm_key:
        answer = "AI is disabled. Set OPENAI_API_KEY or ANTHROPIC_API_KEY to use Ask IA AI."
    else:
        try:
            context = _build_ai_context_workload()
            catalog_context = _build_catalog_context(
                selected_workload=selected_workload,
                selected_providers=selected_global_providers,
                rows=all_rows,
            )
            ranked = st.session_state.get("last_ranked_plans", [])
            ranked_context = ""
            if ranked:
                top = ranked[0]
                ranked_context = (
                    f"\nTop ranked plan: provider={top.provider_id}, offering={top.offering_id}, "
                    f"monthly_cost={top.monthly_cost_usd:.2f}, score={top.score:.4f}"
                )
            prompt = (
                "You are IA AI. Use ONLY the provided catalog/ranking context. "
                "If data is missing, say 'not available in current catalog'. "
                "Do not invent providers/SKUs/prices.\n\n"
                f"Catalog context:\n{catalog_context}\n"
                f"{ranked_context}\n\n"
                f"Current workload={selected_workload}, mode={page_mode}, "
                f"selected_providers={','.join(selected_global_providers)}.\n"
                f"User question: {chat_prompt}\n"
                "Answer with concise bullets and explicit provider/sku/price citations."
            )
            answer = _get_ask_ia_router().explain(prompt, context)
        except Exception as exc:  # noqa: BLE001
            answer = f"AI request failed: {exc}"
    st.session_state["ia_chat_history"].append({"role": "assistant", "content": answer})
    st.rerun()
