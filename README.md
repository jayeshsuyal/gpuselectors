# InferenceAtlas

**Multi-GPU scaling and cost optimization for LLM deployments.**

InferenceAtlas is a planning model for estimating GPU requirements and monthly infrastructure costs for serving large language models. It models traffic patterns, multi-GPU scaling behavior, and cross-platform cost comparisons to help identify cost-effective deployment configurations.

---

## Problem Statement

Deploying LLMs in production requires answering:
1. How many GPUs do I need for X tokens/day?
2. Which platform (Fireworks, RunPod, Together, etc.) is most cost-effective?
3. Should I use autoscaling or dedicated instances?
4. What is the trade-off between cost and latency headroom?

Manual capacity planning is error-prone. Over-provisioning wastes money. Under-provisioning causes latency spikes.

InferenceAtlas automates this analysis using traffic modeling, utilization forecasting, and penalty-based ranking.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       User Input                            │
│  tokens/day, traffic pattern, model, latency requirement    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  Scaling Engine                             │
│  • Traffic → avg TPS → peak TPS (burst/active ratio)        │
│  • GPU throughput → effective TPS (efficiency × batching)   │
│  • Utilization = peak / effective                           │
│  • Multi-GPU: ceil(utilization / 0.75)                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   Cost Model                                │
│  • Autoscaling: active_hours × rate × gpu_count            │
│  • Dedicated: 720 hours × rate × gpu_count                 │
│  • Per-token: (tokens/mo ÷ 1M) × price_per_m_tokens        │
│  • Idle waste = (720 - active_hours) × rate                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                Penalty Model                                │
│  • Overload: +$20k if utilization > 90%                     │
│  • Scaling: +$50k/GPU if gpu_count > 8                      │
│  • Latency: +$30k if high-risk + strict requirement         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Ranked Recommendations                         │
│  Sort by: monthly_cost + penalties                          │
│  Return top-k options with cost/utilization/waste metrics   │
└─────────────────────────────────────────────────────────────┘
```

---

## How the Math Works

### 1. Traffic → GPU Utilization

```python
# Convert daily volume to average throughput
avg_tps = tokens_per_day / 86_400

# Apply traffic pattern (business hours, bursty, steady)
required_peak_tps = (avg_tps / active_ratio) × burst_factor

# Account for GPU efficiency (batching overhead)
effective_gpu_tps = gpu_tps × efficiency × batch_mult

# Compute utilization
utilization_ratio = required_peak_tps / effective_gpu_tps
```

### 2. Multi-GPU Scaling (75% Target)

```python
# If utilization exceeds 75%, scale to multiple GPUs
gpu_count = ceil(utilization_ratio / 0.75)
utilization_after = utilization_ratio / gpu_count
```

**Why 75%?** Provides 25% headroom for traffic bursts, reducing latency variance.

### 3. Cost Calculation

**Autoscaling** (Fireworks):
```python
monthly_cost = active_hours_per_month × hourly_rate × gpu_count
idle_waste = 0  # Pay only for active hours
```

**Dedicated** (RunPod, Modal):
```python
monthly_cost = 720 × hourly_rate × gpu_count
idle_waste = (720 - active_hours) × hourly_rate × gpu_count
```

**Per-Token** (Together AI):
```python
monthly_cost = (tokens_per_day × 30 / 1_000_000) × price_per_m_tokens
```

---

## Example Output

**Scenario**: 5M tokens/day, Llama 70B, Bursty traffic

```
Top 3 Recommendations:

1. fireworks - NVIDIA H100 80GB
   Reasoning: autoscaling billing; 1 GPU(s); utilization 34%; latency risk low; idle waste 0%
   Monthly cost: $835
   Cost/1M tokens: $5.57
   Utilization: 34%
   Idle waste: 0%

2. vast_ai - NVIDIA A100 80GB
   Reasoning: hourly_variable billing; 1 GPU(s); utilization 42%; latency risk medium; idle waste 60%
   Monthly cost: $1,260
   Cost/1M tokens: $8.40
   Utilization: 42%
   Idle waste: 60%

3. runpod - NVIDIA A100 80GB
   Reasoning: hourly billing; 1 GPU(s); utilization 42%; latency risk medium; idle waste 60%
   Monthly cost: $1,361
   Cost/1M tokens: $9.07
   Utilization: 42%
   Idle waste: 60%
```

**Insight**: Autoscaling (Fireworks) wins for bursty traffic by avoiding idle waste.

---

## Limitations

### Not Modeled
- ❌ Network latency (region, CDN)
- ❌ Cold start overhead (serverless platforms)
- ❌ Tensor parallelism (multi-GPU within single model instance)
- ❌ Quantization benefits (FP8, INT4)
- ❌ Speculative decoding gains
- ❌ Request queueing dynamics (M/M/c theory)
- ❌ Spot pricing volatility
- ❌ KV cache memory constraints

### Assumptions
- **30-day month** (720 hours)
- **FP16 precision** (not quantized)
- **No volume discounts** (on-demand rates only)
- **Simplified traffic patterns** (3 reference profiles)
- **Static efficiency factors** (no dynamic adaptation)

**See [docs/assumptions.md](docs/assumptions.md) for full list.**

---

## Installation

### Requirements
- Python 3.10+
- pip

### Install (Editable Mode)

```bash
# Clone repository
git clone https://github.com/jayeshsuyal/gpuselectors
cd gpuselector

# Install package in editable mode
pip install -e .

# Install dev dependencies (for testing)
pip install -e ".[dev]"
```

---

## Usage

### Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Python API

Canonical planning API: `rank_configs(...)`

```python
from inference_atlas import rank_configs

plans = rank_configs(
    tokens_per_day=5_000_000,
    model_bucket="70b",
    peak_to_avg=3.0,
    top_k=3,
)

for plan in plans:
    print(f"{plan.rank}. {plan.provider_name} - {plan.offering_id}")
    print(f"   Monthly cost: ${plan.monthly_cost_usd:,.0f}")
    print(f"   Risk score: {plan.risk.total_risk:.2f}")
```

Legacy API note: `get_recommendations(...)` is deprecated and retained only for backward compatibility.

### Batch Example Runner

```bash
python scripts/run_examples.py
```

Runs 6 validation scenarios from `examples/`.

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/inference_atlas --cov-report=term-missing

# Run specific test file
pytest tests/test_memory_filtering.py -v
```

---

## Project Structure

```
inference-atlas/
├── src/inference_atlas/      # Core recommendation engine
│   ├── config.py              # Constants (U_TARGET, HOURS_PER_MONTH, patterns)
│   ├── scaling.py             # Multi-GPU utilization modeling
│   ├── cost_model.py          # Monthly cost calculations
│   ├── recommender.py         # Main ranking engine
│   └── data_loader.py         # Platform/model data access
├── app/
│   └── streamlit_app.py       # Streamlit UI
├── data/
│   ├── platforms.py           # GPU pricing catalog
│   └── performance.py         # Model memory requirements
├── docs/
│   ├── methodology.md         # Math walkthrough
│   ├── assumptions.md         # Model limitations
│   ├── data_sources.md        # Pricing/throughput provenance
│   └── validation.md          # Sanity test scenarios
├── examples/
│   └── scenario_*.json        # 6 reference workloads
├── scripts/
│   └── run_examples.py        # Batch runner for examples
├── tests/
│   ├── test_math_engine.py    # Core algorithm tests
│   └── test_memory_filtering.py  # Memory validation tests
├── pyproject.toml             # Package configuration
└── README.md
```

---

## Documentation

- **[Methodology](docs/methodology.md)**: Detailed explanation of traffic modeling, scaling math, and cost formulas
- **[Assumptions](docs/assumptions.md)**: Explicit listing of all model assumptions and limitations
- **[Data Sources](docs/data_sources.md)**: Pricing/throughput data provenance (last verified: Feb 2025)
- **[Validation](docs/validation.md)**: Reference scenarios and expected behavior

---

## Disclaimer

**This is a planning model, not a production SLA guarantee.**

- Pricing changes frequently (validate with provider websites)
- Throughput varies by workload (benchmark your specific use case)
- Real-world latency depends on many factors not modeled here
- Model predictions are directionally accurate (±20% variance expected)

**Always run real-world benchmarks before production deployment.**

---

## Contributing

Contributions welcome! Areas for improvement:
- [ ] Add more platforms (Lambda Labs, GCP, AWS, Azure)
- [ ] Model PagedAttention memory savings
- [ ] Add quantization throughput multipliers (FP8, INT4)
- [ ] Incorporate speculative decoding gains
- [ ] Add region/latency constraints

---

## License

MIT License - see LICENSE file for details.

---

## Contact

- **Author**: Jayesh Suyal
- **Repository**: [github.com/jayeshsuyal/gpuselectors](https://github.com/jayeshsuyal/gpuselectors)
- **Issues**: [github.com/jayeshsuyal/gpuselectors/issues](https://github.com/jayeshsuyal/gpuselectors/issues)
