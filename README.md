# InferenceAtlas

**Category-first AI inference cost comparison and deployment planning.**

InferenceAtlas is a Streamlit-based tool for comparing AI inference pricing across providers. It supports multiple workload types (LLM, speech, embeddings, image generation, and more), uses a unified live catalog updated daily, and provides an Invoice Analyzer to identify savings opportunities against current market rates.

**Live demo:** https://inferenceatlas.us  
Fallback: https://inferenceatlas-hjcltilq4njm6vrez4o877.streamlit.app/

---

## What It Does

| Capability | Status |
|---|---|
| Category-first provider comparison (all workload types) | ✅ Production |
| LLM deployment optimizer (GPU/token cost + capacity modeling) | ✅ Production |
| Pricing catalog browser (filterable, exportable) | ✅ Production |
| Invoice Analyzer — find savings vs. current catalog | ✅ Beta |
| Non-LLM optimizer (demand-aware ranking) | 🧪 Beta — throughput checks when metadata exists |
| AI assistant + Ask IA AI chat | ✅ Optional (API key required) |
| Latency queueing model | ❌ Not implemented |

---

## Quickstart (Local Run)

**Requirements:** Python 3.9+

```bash
# 1. Clone and install
git clone https://github.com/jayeshsuyal/InferenceAtlas.git
cd InferenceAtlas
pip install -e ".[dev]"

# 2. (Optional) Set AI assistant keys
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Run the app
.venv/bin/streamlit run app/streamlit_app.py --server.port 8501 --server.address 127.0.0.1
```

Open [http://localhost:8501](http://localhost:8501).

### Frontend (React v1 workbench)

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

By default, frontend runs in mock mode (`VITE_USE_MOCK_API=true`).  
To use a real backend, set:

```bash
VITE_USE_MOCK_API=false
VITE_API_BASE_URL=http://127.0.0.1:8000
```

---

## Streamlit Usage Walkthrough

The UI is category-first. Every step filters what comes next.

### Step 1 — Select Workload
Choose what you are optimizing. Available workloads are derived from the live catalog and include: `LLM Inference`, `Speech-to-Text`, `Text-to-Speech`, `Embeddings`, `Image Generation`, `Vision`, `Video Generation`, `Moderation`.

A catalog freshness indicator shows how many days since the last sync.

### Step 2 — Choose View
Select one of three views:
- **Optimize Workload** — ranked recommendations (full planner for LLM, price ranker beta for non-LLM)
- **Browse Pricing Catalog** — filterable table of all pricing rows, exportable to CSV
- **Invoice Analyzer** *(beta)* — upload an invoice CSV to find savings

### Step 3 — Optional Provider Filter
Use the provider filter to narrow ranking/browse scope for the selected workload.

### LLM Optimize Flow
1. Choose a curated model (maps to a capacity bucket: 7B / 13B / 34B / 70B / 405B)
2. Set daily token volume and traffic pattern (Steady / Business Hours / Bursty)
3. Optionally set a monthly budget cap
4. The compatible provider list updates dynamically based on the model bucket — incompatible providers are shown with a reason
5. Click **Get Top 10 Recommendations** — results include cost, score, peak utilization, and risk breakdown

### Non-LLM Optimize Flow *(Beta)*
For all non-LLM workloads (speech, images, embeddings, etc.):
- Select a unit filter to compare apples-to-apples (e.g., `audio_hour`, `1k_chars`)
- Optionally set a monthly usage estimate for cost projection
- Optionally set a max monthly budget to filter expensive options
- Results ranked by normalized unit price and monthly estimate
- Throughput-aware replica estimation applies when throughput metadata is present

### AI Assistant
The **AI Suggest** panel uses Claude or GPT-5 to answer questions grounded in current catalog data. See [docs/ai_assistant.md](docs/ai_assistant.md) for grounding rules.

---

## Data Pipeline

InferenceAtlas uses a three-tier ingestion strategy, all normalized into `data/catalog_v2/pricing_catalog.json`.

```
Tier 1: API-first connectors (when secrets configured)
  └── fal.ai pricing API        → FAL_AI_PRICING_API_URL + FAL_KEY
  └── AWS Rekognition pricing   → AWS_REKOGNITION_PRICING_API_URL
  └── Google Cloud pricing      → GOOGLE_CLOUD_PRICING_API_URL + GOOGLE_CLOUD_PRICING_API_TOKEN

Tier 2: Provider CSV fallback
  └── data/providers_csv/       → per-provider CSV files (16 providers, 578 rows)
  └── Normalized into canonical schema on load

Tier 3: Bundled catalog snapshot
  └── data/catalog_v2/pricing_catalog.json  → always-available fallback
```

All tiers produce the same `CanonicalPricingRow` schema: `provider`, `workload_type`, `sku_key`, `sku_name`, `billing_mode`, `unit_price_usd`, `unit_name`, `region`, `source_date`, `confidence`, `source_kind`.

## Ranking Notes

For non-LLM optimizer mode, catalog ranking supports:
- `normalized` comparator: workload-aware unit normalization (for example: `audio_min -> audio_hour`, `1k_chars -> 1m_chars`, `image -> 1k_images` when comparable).
- `raw` comparator: direct listed unit price ranking.
- optional confidence-weighted pricing penalty (higher penalty for lower-confidence rows).

The UI now reports excluded-offer counts (for normalization/budget filters) and provides provider diagnostics CSV export for included/excluded reasoning.

### Current Coverage (catalog snapshot: 2026-02-16)

| Provider | SKUs | Workloads |
|---|---|---|
| Fireworks | 220 | LLM, Vision, Image Generation |
| Together AI | 119 | LLM, Video, Image Generation |
| OpenAI | 55 | LLM, Image, Embedding, STT, TTS, Moderation, Video |
| Anthropic | 24 | LLM |
| ElevenLabs | 16 | TTS, STT |
| Cohere | 15 | LLM, Embedding, Rerank |
| Deepgram | 15 | STT, TTS |
| Voyage AI | 11 | Embedding, Rerank |
| Baseten | 8 | LLM, Image |
| Modal | 8 | GPU pricing |
| AssemblyAI | 7 | STT |
| Replicate | 5 | GPU pricing |
| RunPod | 5 | GPU pricing |
| fal.ai | 22 | LLM, Image Generation |
| AWS Rekognition | 22 | Vision |
| Google Cloud | 26 | Vision |

**Total: 16 providers, 578 rows across 11 raw workload tokens (normalized to 8 workload categories in app).**

---

## Invoice Analyzer CSV Format

Upload a CSV to the Invoice Analyzer with these exact columns:

| Column | Type | Description |
|---|---|---|
| `provider` | string | Provider name (e.g., `openai`, `anthropic`) |
| `workload_type` | string | Workload type (see aliases below) |
| `usage_qty` | float | Units consumed in billing period |
| `usage_unit` | string | Unit of measure (must match catalog `unit_name`) |
| `amount_usd` | float | Total billed amount in USD |

**Supported workload_type aliases:**

| Input | Canonical |
|---|---|
| `llm` | `llm` |
| `transcription`, `stt`, `speech_to_text` | `speech_to_text` |
| `tts`, `text_to_speech` | `text_to_speech` |
| `embeddings`, `embedding`, `rerank` | `embeddings` |
| `image_gen`, `image_generation` | `image_generation` |
| `vision` | `vision` |
| `video_generation` | `video_generation` |
| `moderation` | `moderation` |

**Example row:**

```csv
provider,workload_type,usage_qty,usage_unit,amount_usd
openai,llm,10000000,1m_tokens,150.00
anthropic,llm,5000000,1m_tokens,75.00
elevenlabs,text_to_speech,500000,1k_chars,80.00
```

The analyzer computes your effective unit price and compares it against the cheapest matching catalog offer. Rows where your price is already optimal are omitted from results. See [docs/invoice_analyzer.md](docs/invoice_analyzer.md) for matching logic and limitations.

---

## AI Features

AI features are **optional** and **key-gated**. The app runs fully without AI keys.

| Feature | Trigger | Required Key |
|---|---|---|
| AI Suggest next steps | Button in AI Assistant panel | `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` |
| Ask IA AI chat | Chat input at page bottom | Same |
| LLM optimizer explanation | (Removed in current version) | — |

All AI responses are grounded in current catalog data. The AI is instructed to cite only providers/SKUs/prices from the catalog and to say "not available in current catalog" for unknown data. See [docs/ai_assistant.md](docs/ai_assistant.md).

```bash
# Set keys before running
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
.venv/bin/streamlit run app/streamlit_app.py --server.port 8501 --server.address 127.0.0.1
```

---

## Daily Catalog Sync

A GitHub Actions workflow syncs the catalog at **07:00 UTC daily**.

**Workflow:** [`.github/workflows/daily-catalog-sync.yml`](.github/workflows/daily-catalog-sync.yml)

**What it does:**
1. Runs `python scripts/sync_catalog_v2.py --providers all --fail-on-empty`
2. Commits `data/catalog_v2/pricing_catalog.json` if changed
3. Pushes to main

**Required secrets (in GitHub repo settings):**

| Secret | Purpose |
|---|---|
| `FAL_AI_PRICING_API_URL` | fal.ai live pricing endpoint |
| `FAL_KEY` | fal.ai API auth |
| `AWS_REKOGNITION_PRICING_API_URL` | AWS pricing API URL |
| `GOOGLE_CLOUD_PRICING_API_URL` | GCP pricing API URL |
| `GOOGLE_CLOUD_PRICING_API_TOKEN` | GCP pricing auth |

Providers without API secrets configured fall back to their CSV files in `data/providers_csv/`.

**Manual sync:**
```bash
python scripts/sync_catalog_v2.py --providers all
```

---

## Testing

```bash
# Run all tests (unit + integration markers)
pytest tests/ -v

# Run only fast unit tests (skip integration tests requiring API keys)
pytest tests/ -v -m "not integration"

# Run with coverage report
pytest tests/ --cov=src/inference_atlas --cov-report=term-missing

# Run a specific file
pytest tests/test_mvp_planner.py -v
```

---

## Project Structure

```
InferenceAtlas/
├── app/
│   └── streamlit_app.py          # Streamlit UI (category-first, catalog_v2-backed)
├── src/inference_atlas/
│   ├── __init__.py               # Public API exports
│   ├── mvp_planner.py            # LLM capacity planning engine (rank_configs)
│   ├── data_loader.py            # Data loading: catalog_v2, MVP catalogs, HF
│   ├── catalog_v2/               # Catalog v2 sync and schema
│   │   ├── sync.py               # Connector-based sync pipeline
│   │   └── connectors/           # Per-provider connectors
│   ├── workload_types.py         # WorkloadType enum definitions
│   ├── llm/                      # LLM adapter layer (OpenAI / Anthropic)
│   │   ├── router.py             # LLMRouter with primary/fallback logic
│   │   └── schema.py             # WorkloadSpec, ParseWorkloadResult
│   ├── scaling.py                # Legacy scaling utilities
│   ├── cost_model.py             # Legacy cost model
│   └── recommender.py            # Legacy recommender (deprecated)
├── data/
│   ├── catalog_v2/
│   │   ├── pricing_catalog.json  # Bundled canonical catalog (synced daily)
│   │   └── pricing_catalog.schema.json
│   ├── providers_csv/            # Per-provider CSV files (16 providers)
│   │   └── README.md             # CSV dataset overview
│   ├── providers.json            # MVP planner provider configs
│   ├── models.json               # Model metadata + size buckets
│   └── capacity_table.json       # GPU throughput benchmarks
├── scripts/
│   ├── sync_catalog_v2.py        # Manual/CI catalog sync runner
│   └── sync_huggingface_catalog.py
├── docs/
│   ├── architecture.md           # Pipeline diagram and component map
│   ├── data_freshness.md         # Sync schedule and staleness behavior
│   ├── invoice_analyzer.md       # Invoice Analyzer schema and matching
│   ├── ai_assistant.md           # AI grounding rules and key setup
│   ├── methodology.md            # LLM capacity planning math
│   ├── assumptions.md            # Model assumptions and limitations
│   ├── data_sources.md           # Pricing data provenance
│   └── validation.md             # Test scenarios
├── tests/                        # pytest test suite (57+ tests)
├── .github/
│   └── workflows/
│       └── daily-catalog-sync.yml
└── pyproject.toml
```

---

## Python API

For direct library use:

```python
from inference_atlas import rank_configs

# LLM capacity planning
plans = rank_configs(
    tokens_per_day=5_000_000,
    model_bucket="70b",
    peak_to_avg=2.5,
    top_k=5,
    provider_ids={"anthropic", "openai", "fireworks"},
)

for plan in plans:
    print(f"{plan.rank}. {plan.provider_name} — ${plan.monthly_cost_usd:,.0f}/mo")
    print(f"   Score: {plan.score:.1f}, Risk: {plan.risk.total_risk:.2f}")
    print(f"   {plan.why}")
```

```python
from inference_atlas import get_catalog_v2_rows

# Browse catalog data
rows = get_catalog_v2_rows("speech_to_text")
cheapest = sorted(rows, key=lambda r: r.unit_price_usd)[:5]
for r in cheapest:
    print(f"{r.provider} {r.sku_name}: {r.unit_price_usd:.6g} / {r.unit_name}")
```

---

## Roadmap

- [ ] Throughput-aware ranking for non-LLM workloads (images/min, audio/min)
- [ ] Latency queueing model (M/M/c) for LLM optimizer
- [ ] Pydantic service layer (`PlanningService`) — Phase 2 of current refactor
- [ ] Multi-region cost comparison with latency estimates
- [ ] Provider API connectors for more providers (Replicate live pricing, RunPod)
- [ ] Time-series price tracking (catalog diffing on each sync)
- [ ] Batch scenario comparison (multiple workloads side-by-side)

---

## Limitations

### LLM Optimizer
- Latency modeling is not implemented (queueing model placeholder only)
- GPU memory validation uses heuristic estimates
- Per-token tps_cap is not always available from providers
- Output token ratio defaults to 30% (configurable but not workload-adaptive)

### Non-LLM Optimizer (Beta)
- Demand-aware monthly ranking with optional throughput checks when metadata is available
- Mixed units (e.g., `image` vs `megapixel`) cannot be cross-compared; use the unit filter
- No SLA or reliability comparison

### Invoice Analyzer (Beta)
- Matches on `unit_name` exactly — unit name must match catalog
- Does not account for volume discounts or committed-use pricing
- Does not handle multi-line invoice entries that blend workloads

### Data
- Pricing sourced from public vendor documentation; verify before making purchasing decisions
- Catalog freshness depends on daily sync; data may lag provider changes by up to 24 hours
- Some rates are estimated where not publicly listed (confidence: `estimated`)

---

## Disclaimer

This is a planning tool, not a procurement or SLA guarantee. Pricing changes frequently. Always verify with provider websites before committing to a deployment configuration.

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
