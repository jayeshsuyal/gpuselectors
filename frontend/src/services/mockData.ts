import type {
  LLMPlanningResponse,
  CatalogRankingResponse,
  CatalogBrowseResponse,
  InvoiceAnalysisResponse,
  AIAssistResponse,
} from './types'

// ─── LLM Planning Mock ────────────────────────────────────────────────────────

export const MOCK_LLM_RESPONSE: LLMPlanningResponse = {
  plans: [
    {
      rank: 1,
      provider_id: 'fireworks',
      provider_name: 'Fireworks AI',
      offering_id: 'fireworks::llama-3-1-70b-instruct::autoscale',
      billing_mode: 'autoscale_hourly',
      confidence: 'high',
      monthly_cost_usd: 892.4,
      score: 0.21,
      utilization_at_peak: 0.72,
      risk: { risk_overload: 0.12, risk_complexity: 0.08, total_risk: 0.20 },
      assumptions: {
        peak_to_avg: 2.5,
        util_target: 0.75,
        scaling_beta: 0.08,
        alpha: 1.0,
        output_token_ratio: 0.30,
        replicas: 2,
      },
      why: 'Autoscale pricing at this volume avoids over-provisioning. High confidence pricing from official API.',
    },
    {
      rank: 2,
      provider_id: 'together',
      provider_name: 'Together AI',
      offering_id: 'together::llama-3-1-70b-instruct::serverless',
      billing_mode: 'per_token',
      confidence: 'high',
      monthly_cost_usd: 1147.2,
      score: 0.28,
      utilization_at_peak: null,
      risk: { risk_overload: 0.05, risk_complexity: 0.03, total_risk: 0.08 },
      assumptions: {
        peak_to_avg: 2.5,
        util_target: 0.75,
        scaling_beta: 0.08,
        alpha: 1.0,
        output_token_ratio: 0.30,
        replicas: 1,
      },
      why: 'Per-token serverless eliminates capacity planning risk. Cost scales linearly with usage.',
    },
    {
      rank: 3,
      provider_id: 'groq',
      provider_name: 'Groq',
      offering_id: 'groq::llama-3-1-70b-versatile::serverless',
      billing_mode: 'per_token',
      confidence: 'high',
      monthly_cost_usd: 1340.8,
      score: 0.31,
      utilization_at_peak: null,
      risk: { risk_overload: 0.05, risk_complexity: 0.03, total_risk: 0.08 },
      assumptions: {
        peak_to_avg: 2.5,
        util_target: 0.75,
        scaling_beta: 0.08,
        alpha: 1.0,
        output_token_ratio: 0.30,
        replicas: 1,
      },
      why: 'Ultra-low latency via custom LPU hardware. Best choice if p99 latency < 200ms is a hard requirement.',
    },
    {
      rank: 4,
      provider_id: 'aws',
      provider_name: 'AWS Bedrock',
      offering_id: 'aws::meta.llama3-1-70b-instruct-v1:0::on_demand',
      billing_mode: 'per_token',
      confidence: 'official',
      monthly_cost_usd: 1628.0,
      score: 0.38,
      utilization_at_peak: null,
      risk: { risk_overload: 0.04, risk_complexity: 0.02, total_risk: 0.06 },
      assumptions: {
        peak_to_avg: 2.5,
        util_target: 0.75,
        scaling_beta: 0.08,
        alpha: 1.0,
        output_token_ratio: 0.30,
        replicas: 1,
      },
      why: 'Official AWS pricing. Higher cost but enterprise SLA, VPC isolation, and compliance certifications included.',
    },
  ],
  provider_diagnostics: [
    { provider: 'fireworks', status: 'included', reason: 'Included (3 rankable offers).' },
    { provider: 'together', status: 'included', reason: 'Included (2 rankable offers).' },
    { provider: 'groq', status: 'included', reason: 'Included (1 rankable offers).' },
    { provider: 'aws', status: 'included', reason: 'Included (1 rankable offers).' },
    { provider: 'azure', status: 'excluded', reason: 'No rankable offers for 70b bucket.' },
    { provider: 'anthropic', status: 'excluded', reason: 'No open-weight 70B equivalent.' },
  ],
  excluded_count: 2,
  warnings: [],
}

// ─── Non-LLM Catalog Mock (STT) ───────────────────────────────────────────────

export const MOCK_STT_RESPONSE: CatalogRankingResponse = {
  offers: [
    {
      rank: 1,
      provider: 'deepgram',
      sku_name: 'deepgram::nova-2::pay_as_you_go',
      billing_mode: 'per_unit',
      unit_price_usd: 0.0043,
      normalized_price: 0.258,
      unit_name: 'audio_min',
      confidence: 'high',
      monthly_estimate_usd: 12.44,
      required_replicas: null,
      capacity_check: 'ok',
    },
    {
      rank: 2,
      provider: 'openai',
      sku_name: 'openai::whisper-1::default',
      billing_mode: 'per_unit',
      unit_price_usd: 0.006,
      normalized_price: 0.36,
      unit_name: 'audio_min',
      confidence: 'official',
      monthly_estimate_usd: 17.28,
      required_replicas: null,
      capacity_check: 'ok',
    },
    {
      rank: 3,
      provider: 'assemblyai',
      sku_name: 'assemblyai::best::pay_as_you_go',
      billing_mode: 'per_unit',
      unit_price_usd: 0.0065,
      normalized_price: 0.39,
      unit_name: 'audio_min',
      confidence: 'high',
      monthly_estimate_usd: 18.72,
      required_replicas: null,
      capacity_check: 'ok',
    },
    {
      rank: 4,
      provider: 'google',
      sku_name: 'google::speech-to-text-v2::standard',
      billing_mode: 'per_unit',
      unit_price_usd: 0.016,
      normalized_price: 0.96,
      unit_name: 'audio_min',
      confidence: 'official',
      monthly_estimate_usd: 46.08,
      required_replicas: null,
      capacity_check: 'ok',
    },
  ],
  provider_diagnostics: [
    { provider: 'deepgram', status: 'included', reason: 'Included (2 rankable offers).' },
    { provider: 'openai', status: 'included', reason: 'Included (1 rankable offers).' },
    { provider: 'assemblyai', status: 'included', reason: 'Included (1 rankable offers).' },
    { provider: 'google', status: 'included', reason: 'Included (1 rankable offers).' },
  ],
  excluded_count: 0,
  warnings: [],
}

// ─── Catalog Browse Mock ──────────────────────────────────────────────────────

export const MOCK_CATALOG_BROWSE: CatalogBrowseResponse = {
  rows: [
    { provider: 'openai', workload_type: 'llm', sku_name: 'gpt-4o', unit_name: '1m_tokens', unit_price_usd: 2.5, billing_mode: 'per_token', confidence: 'official', region: null, model_name: 'gpt-4o', throughput_value: null, throughput_unit: null },
    { provider: 'openai', workload_type: 'llm', sku_name: 'gpt-4o-mini', unit_name: '1m_tokens', unit_price_usd: 0.15, billing_mode: 'per_token', confidence: 'official', region: null, model_name: 'gpt-4o-mini', throughput_value: null, throughput_unit: null },
    { provider: 'anthropic', workload_type: 'llm', sku_name: 'claude-3-5-sonnet', unit_name: '1m_tokens', unit_price_usd: 3.0, billing_mode: 'per_token', confidence: 'official', region: null, model_name: 'claude-3-5-sonnet-20241022', throughput_value: null, throughput_unit: null },
    { provider: 'anthropic', workload_type: 'llm', sku_name: 'claude-3-haiku', unit_name: '1m_tokens', unit_price_usd: 0.25, billing_mode: 'per_token', confidence: 'official', region: null, model_name: 'claude-3-haiku-20240307', throughput_value: null, throughput_unit: null },
    { provider: 'fireworks', workload_type: 'llm', sku_name: 'llama-3-1-70b-instruct', unit_name: '1m_tokens', unit_price_usd: 0.9, billing_mode: 'per_token', confidence: 'high', region: null, model_name: 'llama-3-1-70b-instruct', throughput_value: null, throughput_unit: null },
    { provider: 'groq', workload_type: 'llm', sku_name: 'llama-3-1-70b-versatile', unit_name: '1m_tokens', unit_price_usd: 0.59, billing_mode: 'per_token', confidence: 'high', region: null, model_name: 'llama-3-1-70b-versatile', throughput_value: null, throughput_unit: null },
    { provider: 'together', workload_type: 'llm', sku_name: 'llama-3-1-70b-instruct-turbo', unit_name: '1m_tokens', unit_price_usd: 0.88, billing_mode: 'per_token', confidence: 'high', region: null, model_name: 'Meta Llama 3.1 70B Instruct Turbo', throughput_value: null, throughput_unit: null },
    { provider: 'deepgram', workload_type: 'speech_to_text', sku_name: 'nova-2', unit_name: 'audio_min', unit_price_usd: 0.0043, billing_mode: 'per_unit', confidence: 'high', region: null, model_name: 'nova-2', throughput_value: null, throughput_unit: null },
    { provider: 'openai', workload_type: 'speech_to_text', sku_name: 'whisper-1', unit_name: 'audio_min', unit_price_usd: 0.006, billing_mode: 'per_unit', confidence: 'official', region: null, model_name: 'whisper-1', throughput_value: null, throughput_unit: null },
    { provider: 'assemblyai', workload_type: 'speech_to_text', sku_name: 'best', unit_name: 'audio_min', unit_price_usd: 0.0065, billing_mode: 'per_unit', confidence: 'high', region: null, model_name: 'best', throughput_value: null, throughput_unit: null },
    { provider: 'elevenlabs', workload_type: 'text_to_speech', sku_name: 'starter', unit_name: '1k_chars', unit_price_usd: 0.03, billing_mode: 'per_unit', confidence: 'high', region: null, model_name: null, throughput_value: null, throughput_unit: null },
    { provider: 'openai', workload_type: 'text_to_speech', sku_name: 'tts-1', unit_name: '1k_chars', unit_price_usd: 0.015, billing_mode: 'per_unit', confidence: 'official', region: null, model_name: 'tts-1', throughput_value: null, throughput_unit: null },
    { provider: 'openai', workload_type: 'embeddings', sku_name: 'text-embedding-3-small', unit_name: '1m_tokens', unit_price_usd: 0.02, billing_mode: 'per_token', confidence: 'official', region: null, model_name: 'text-embedding-3-small', throughput_value: null, throughput_unit: null },
    { provider: 'cohere', workload_type: 'embeddings', sku_name: 'embed-v3', unit_name: '1m_tokens', unit_price_usd: 0.1, billing_mode: 'per_token', confidence: 'high', region: null, model_name: 'embed-english-v3.0', throughput_value: null, throughput_unit: null },
    { provider: 'openai', workload_type: 'image_generation', sku_name: 'dall-e-3-standard', unit_name: 'image', unit_price_usd: 0.04, billing_mode: 'per_image', confidence: 'official', region: null, model_name: 'dall-e-3', throughput_value: null, throughput_unit: null },
    { provider: 'stability', workload_type: 'image_generation', sku_name: 'stable-diffusion-xl', unit_name: 'image', unit_price_usd: 0.002, billing_mode: 'per_image', confidence: 'high', region: null, model_name: 'stable-diffusion-xl-1024-v1-0', throughput_value: null, throughput_unit: null },
    { provider: 'openai', workload_type: 'moderation', sku_name: 'omni-moderation-latest', unit_name: '1k_tokens', unit_price_usd: 0.0, billing_mode: 'per_token', confidence: 'official', region: null, model_name: 'omni-moderation-latest', throughput_value: null, throughput_unit: null },
  ],
  total: 17,
}

// ─── Invoice Mock ─────────────────────────────────────────────────────────────

export const MOCK_INVOICE_RESPONSE: InvoiceAnalysisResponse = {
  line_items: [
    { provider: 'openai', workload_type: 'llm', line_item: 'GPT-4o input tokens', quantity: 12_500_000, unit: '1m_tokens', unit_price: 2.5, total: 31.25 },
    { provider: 'openai', workload_type: 'llm', line_item: 'GPT-4o output tokens', quantity: 3_800_000, unit: '1m_tokens', unit_price: 10.0, total: 38.0 },
    { provider: 'openai', workload_type: 'speech_to_text', line_item: 'Whisper transcription', quantity: 2880, unit: 'audio_min', unit_price: 0.006, total: 17.28 },
    { provider: 'anthropic', workload_type: 'llm', line_item: 'Claude 3.5 Sonnet input', quantity: 8_000_000, unit: '1m_tokens', unit_price: 3.0, total: 24.0 },
    { provider: 'anthropic', workload_type: 'llm', line_item: 'Claude 3.5 Sonnet output', quantity: 2_000_000, unit: '1m_tokens', unit_price: 15.0, total: 30.0 },
    { provider: 'elevenlabs', workload_type: 'text_to_speech', line_item: 'TTS characters', quantity: 500_000, unit: '1k_chars', unit_price: 0.03, total: 15.0 },
  ],
  totals_by_provider: {
    openai: 86.53,
    anthropic: 54.0,
    elevenlabs: 15.0,
  },
  grand_total: 155.53,
  detected_workloads: ['llm', 'speech_to_text', 'text_to_speech'],
  warnings: [],
}

// ─── AI Assistant Mock ────────────────────────────────────────────────────────

export const MOCK_AI_RESPONSES: Record<string, AIAssistResponse> = {
  default: {
    reply: 'I can help you compare AI inference costs. Try asking: "What\'s the cheapest STT provider?", "Compare LLM options for 10M tokens/day", or "Explain the risk scores."',
    suggested_action: null,
  },
  cheapest: {
    reply: 'For speech-to-text at typical usage, **Deepgram Nova-2** at $0.0043/min is the clear winner — roughly 30% cheaper than OpenAI Whisper. For LLM inference, cost depends heavily on model size and traffic pattern. At 70B scale, Fireworks AI autoscale typically beats serverless by 20-30% above ~2M tokens/day.',
    suggested_action: 'run_optimize',
  },
  risk: {
    reply: 'Risk scores reflect two factors: **overload risk** (probability of exceeding capacity at peak) and **complexity risk** (ops burden of the billing model). Dedicated instances score higher complexity due to capacity planning requirements. Autoscale reduces overload risk but adds ~15% cost inefficiency.',
    suggested_action: null,
  },
}
