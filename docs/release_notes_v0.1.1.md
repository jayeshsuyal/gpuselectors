# InferenceAtlas v0.1.1 (Pre-release)

InferenceAtlas v0.1.1 focuses on reliability and UX hardening across ranking, AI routing, and Streamlit deployment behavior.

## Live Demo

https://inferenceatlas-hjcltilq4njm6vrez4o877.streamlit.app/

## Highlights

- Category-first workload planning across LLM, STT, TTS, embeddings, vision, image, video, and moderation.
- Throughput-aware non-LLM ranking path (with safe fallbacks where metadata is missing).
- Grounded AI Suggest routing improved to correctly detect workload intent from natural language.
- Invoice Analyzer remains available in beta.

## What Changed

- Hardened AI intent routing with typo tolerance, negation-aware parsing, and cross-category fallbacks.
- Fixed monthly usage + budget filtering so estimates are computed from listed unit pricing.
- Added same-unit fallback in normalized comparator mode when a specific unit filter is selected.
- Added guardrails for strict capacity mode when throughput metadata is missing.
- Improved non-LLM UX messaging for mixed-unit usage and no-result budget scenarios.
- Improved OpenAI adapter compatibility with fallback from `responses` API to `chat.completions`.
- Added dependency cap: `jsonschema<5.0.0`.
- Updated app/footer and repo links for the `InferenceAtlas` repository migration.

## Notes

- Some non-LLM providers still lack throughput metadata. In those cases, ranking remains cost-first and strict capacity checks are relaxed.
- 2 integration tests remain skip-gated by API keys.

## Validation Snapshot

- Test suite status at release prep: `124 passed, 2 skipped`.

