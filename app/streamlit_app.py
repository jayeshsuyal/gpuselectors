"""Streamlit UI for InferenceAtlas LLM deployment recommendations."""

from __future__ import annotations

import os

import streamlit as st

from inference_atlas import get_mvp_catalog, get_recommendations, rank_configs
from inference_atlas.data_loader import get_models
from inference_atlas.llm import LLMRouter, WorkloadSpec

st.set_page_config(page_title="InferenceAtlas", layout="centered")

st.title("InferenceAtlas: LLM Deployment Optimizer")
st.caption("Multi-GPU scaling + cost optimization for LLM deployments")

# Load model catalog
MODEL_REQUIREMENTS = get_models()
has_llm_key = bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))

with st.expander("Or describe your workload in plain English"):
    user_text = st.text_area(
        "Describe your LLM deployment needs",
        placeholder=(
            "e.g., Chat app with 10k daily users, steady traffic, Llama 70B, "
            "need <200ms latency"
        ),
        height=100,
        key="ai_parse_input_text",
    )
    if st.button("Parse with AI"):
        if not has_llm_key:
            st.error("Set OPENAI_API_KEY or ANTHROPIC_API_KEY to use AI parsing.")
            st.stop()
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
        except Exception as exc:  # noqa: BLE001 - user-facing parser message
            st.error(f"Parsing failed: {exc}")

parsed_workload = st.session_state.get("parsed_workload", {})
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

with st.form("inputs"):
    model_items = list(MODEL_REQUIREMENTS.items())
    default_model_key = str(parsed_workload.get("model_key", "llama_70b"))
    default_index = next((i for i, (k, _) in enumerate(model_items) if k == default_model_key), 0)
    model_display_names = [v["display_name"] for _, v in model_items]
    model_key_by_display = {v["display_name"]: k for k, v in model_items}

    model_display_name = st.selectbox(
        "Model",
        options=model_display_names,
        index=default_index,
        help="Choose the LLM model you plan to deploy.",
    )
    model_key = model_key_by_display[model_display_name]

    tokens_per_day = st.number_input(
        "Traffic (tokens/day)",
        min_value=1.0,
        value=float(parsed_workload.get("tokens_per_day", 5_000_000.0)),
        step=100_000.0,
        help="Total generated+processed tokens per day.",
    )

    pattern_label = st.selectbox(
        "Traffic Pattern",
        ["Steady", "Business Hours", "Bursty"],
        index=["Steady", "Business Hours", "Bursty"].index(
            pattern_to_label.get(str(parsed_workload.get("pattern", "steady")), "Steady")
        ),
        help="Steady: 24/7 uniform load. Business Hours: 40hrs/week. Bursty: Irregular spikes.",
    )

    latency_requirement_ms = st.number_input(
        "Latency requirement (ms, optional)",
        min_value=0.0,
        value=float(parsed_workload.get("latency_requirement_ms") or 0.0),
        step=10.0,
        help="Set to 0 to ignore latency constraint. <300ms triggers strict latency penalties.",
    )

    submit = st.form_submit_button("Get Recommendations")

if submit:
    latency = latency_requirement_ms if latency_requirement_ms > 0 else None
    effective_workload = WorkloadSpec(
        tokens_per_day=float(tokens_per_day),
        pattern=label_to_pattern[pattern_label],
        model_key=model_key,
        latency_requirement_ms=latency,
    )
    st.session_state["last_workload"] = effective_workload

    try:
        recommendations = get_recommendations(
            tokens_per_day=effective_workload.tokens_per_day,
            pattern=effective_workload.pattern,
            model_key=effective_workload.model_key,
            latency_requirement_ms=effective_workload.latency_requirement_ms,
            top_k=3,
        )
    except ValueError as exc:
        st.error(str(exc))
        recommendations = []
    st.session_state["last_recommendations"] = recommendations

last_recommendations = st.session_state.get("last_recommendations", [])
last_workload = st.session_state.get("last_workload")
if last_recommendations:
    st.subheader("Top 3 Recommendations")
    for rec in last_recommendations:
        with st.container():
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"### {rec.rank}. {rec.platform} - {rec.option}")
                st.caption(rec.reasoning)

            with col2:
                st.metric("Monthly Cost", f"${rec.monthly_cost_usd:,.0f}")
                st.metric("Cost/1M Tokens", f"${rec.cost_per_million_tokens:.2f}")

            if rec.utilization_pct < 60:
                util_label = "Low"
            elif rec.utilization_pct < 80:
                util_label = "Moderate"
            else:
                util_label = "High"
            st.progress(min(max(rec.utilization_pct / 100, 0.0), 1.0))
            st.caption(f"{util_label} utilization: {rec.utilization_pct:.0f}%")

            if rec.idle_waste_pct > 40:
                potential_savings = rec.monthly_cost_usd * rec.idle_waste_pct / 100
                st.warning(
                    f"{rec.idle_waste_pct:.0f}% idle capacity. "
                    f"Consider autoscaling to save about ${potential_savings:,.0f}/mo."
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
                top_rec = last_recommendations[0]
                summary = (
                    f"{top_rec.platform} - {top_rec.option}, "
                    f"${top_rec.monthly_cost_usd:.0f}/mo, "
                    f"{top_rec.utilization_pct:.0f}% util"
                )
                explanation = router.explain(summary, last_workload)
                st.session_state["last_explanation"] = explanation
                st.info(explanation)
            except Exception as exc:  # noqa: BLE001 - user-facing parser message
                st.error(f"Explanation failed: {exc}")
    if st.session_state.get("last_explanation"):
        st.info(st.session_state["last_explanation"])
else:
    st.info("Enter workload details and click Get Recommendations.")

st.markdown("---")
st.subheader("MVP Planner (Schema-Validated Catalogs)")
st.caption("Capacity-first ranking with workload normalization, risk scoring, and cost-adjusted score.")

try:
    mvp_models = get_mvp_catalog("models")["models"]
    bucket_options = sorted({model["size_bucket"] for model in mvp_models if model["size_bucket"] != "other"})
except Exception as exc:  # noqa: BLE001 - user-facing catalog error
    st.error(f"Failed to load MVP catalogs: {exc}")
    bucket_options = ["70b"]

with st.form("mvp_planner_inputs"):
    mvp_tokens_day = st.number_input(
        "MVP Traffic (tokens/day)",
        min_value=1.0,
        value=8_000_000.0,
        step=100_000.0,
        help="Total tokens processed per day (input + output) for this workload.",
    )
    mvp_peak_to_avg = st.number_input(
        "Peak-to-average",
        min_value=1.0,
        value=2.5,
        step=0.1,
        help="Peak traffic divided by average traffic (higher means burstier demand).",
    )
    mvp_model_bucket = st.selectbox(
        "Model bucket",
        options=bucket_options,
        index=bucket_options.index("70b") if "70b" in bucket_options else 0,
        help="Approximate model size class used for capacity lookup (e.g., 7b, 70b).",
    )
    mvp_top_k = st.slider(
        "Top K",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of best-ranked deployment configurations to display.",
    )

    with st.expander("Advanced assumptions"):
        mvp_util_target = st.slider(
            "Util target",
            min_value=0.50,
            max_value=0.90,
            value=0.75,
            step=0.01,
            help="Target max utilization at peak load; lower values reserve more headroom.",
        )
        mvp_beta = st.slider(
            "Scaling beta",
            min_value=0.01,
            max_value=0.20,
            value=0.08,
            step=0.01,
            help="Controls multi-GPU scaling efficiency drop as GPU count increases.",
        )
        mvp_alpha = st.slider(
            "Risk alpha",
            min_value=0.0,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Weight of risk in final score; higher values penalize risky configs more.",
        )
        mvp_autoscale_ineff = st.slider(
            "Autoscale inefficiency",
            min_value=1.0,
            max_value=1.5,
            value=1.15,
            step=0.01,
            help="Multiplier for autoscaling overhead (cold starts, orchestration, idle drift).",
        )

    mvp_submit = st.form_submit_button("Run MVP Planner")

if mvp_submit:
    try:
        mvp_plans = rank_configs(
            tokens_per_day=float(mvp_tokens_day),
            model_bucket=mvp_model_bucket,
            peak_to_avg=float(mvp_peak_to_avg),
            util_target=float(mvp_util_target),
            beta=float(mvp_beta),
            alpha=float(mvp_alpha),
            autoscale_inefficiency=float(mvp_autoscale_ineff),
            top_k=int(mvp_top_k),
        )
        st.session_state["mvp_plans"] = mvp_plans
    except Exception as exc:  # noqa: BLE001 - user-facing planner message
        st.error(f"MVP planner failed: {exc}")
        st.session_state["mvp_plans"] = []

mvp_plans = st.session_state.get("mvp_plans", [])
if mvp_plans:
    table_rows = []
    for plan in mvp_plans:
        table_rows.append(
            {
                "rank": plan.rank,
                "provider": plan.provider_id,
                "billing": plan.billing_mode,
                "gpu": plan.gpu_type or "-",
                "gpus": plan.gpu_count,
                "monthly_usd": round(plan.monthly_cost_usd, 2),
                "score": round(plan.score, 2),
                "risk": round(plan.risk.total_risk, 3),
                "headroom_pct": round(plan.headroom_pct, 1) if plan.headroom_pct is not None else None,
            }
        )
    try:
        st.dataframe(table_rows, use_container_width=True, hide_index=True)
    except TypeError:
        # Backward compatibility for older Streamlit versions.
        st.dataframe(table_rows)

    st.markdown("**Top Explanations**")
    for plan in mvp_plans[:3]:
        with st.container():
            st.markdown(
                f"**{plan.rank}. {plan.provider_name}**  \n"
                f"`{plan.billing_mode}` | `gpu={plan.gpu_type or '-'}` | `count={plan.gpu_count}`"
            )
            st.caption(plan.why)
            col1, col2, col3 = st.columns(3)
            col1.metric("Monthly Cost", f"${plan.monthly_cost_usd:,.0f}")
            col2.metric("Score", f"{plan.score:,.1f}")
            col3.metric("Risk", f"{plan.risk.total_risk:.2f}")
