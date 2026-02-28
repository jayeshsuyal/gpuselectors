/**
 * Service layer — all backend calls go through here.
 * Set VITE_USE_MOCK_API=true for mock mode, false for real backend.
 */

import type {
  LLMPlanningRequest,
  LLMPlanningResponse,
  CatalogRankingRequest,
  CatalogRankingResponse,
  CatalogBrowseResponse,
  QualityCatalogResponse,
  InvoiceAnalysisResponse,
  AIAssistRequest,
  AIAssistResponse,
  CopilotTurnRequest,
  CopilotTurnResponse,
  CopilotApplyPayload,
  ReportGenerateRequest,
  ReportGenerateResponse,
  ReportChart,
  ScalingPlanRequest,
  ScalingPlanResponse,
  CostAuditRequest,
  CostAuditResponse,
} from './types'

const USE_MOCK = (import.meta.env.VITE_USE_MOCK_API ?? 'true').toLowerCase() !== 'false'
const BASE = import.meta.env.VITE_API_BASE_URL ?? ''
let mockCatalogRowsCache: CatalogBrowseResponse['rows'] | null = null
let mockModulePromise: Promise<typeof import('./mockData')> | null = null

async function getMockModule(): Promise<typeof import('./mockData')> {
  if (mockModulePromise === null) {
    mockModulePromise = import('./mockData')
  }
  return mockModulePromise
}

const WORKLOAD_ALIASES: Record<string, string> = {
  llm: 'llm',
  transcription: 'speech_to_text',
  speech_to_text: 'speech_to_text',
  stt: 'speech_to_text',
  tts: 'text_to_speech',
  text_to_speech: 'text_to_speech',
  embedding: 'embeddings',
  embeddings: 'embeddings',
  rerank: 'embeddings',
  image_gen: 'image_generation',
  image_generation: 'image_generation',
  vision: 'vision',
  video_generation: 'video_generation',
  moderation: 'moderation',
}

const CONFIDENCE_MULTIPLIER: Record<string, number> = {
  official: 1.0,
  high: 1.0,
  medium: 1.1,
  estimated: 1.1,
  low: 1.25,
  vendor_list: 1.25,
}

function delay(ms = 600): Promise<void> {
  return new Promise((r) => setTimeout(r, ms))
}

function canonicalWorkload(value: string): string {
  const token = value.trim().toLowerCase()
  return WORKLOAD_ALIASES[token] ?? token
}

function confidenceMultiplier(confidence: string): number {
  return CONFIDENCE_MULTIPLIER[confidence.trim().toLowerCase()] ?? 1.3
}

function normalizeUnitPriceForWorkload(
  unitPriceUsd: number,
  unitName: string,
  workloadType: string
): number | null {
  const unit = unitName.trim().toLowerCase()
  const workload = canonicalWorkload(workloadType)

  if (workload === 'llm' || workload === 'embeddings' || workload === 'moderation') {
    return unit === '1m_tokens' ? unitPriceUsd : null
  }
  if (workload === 'speech_to_text') {
    if (unit === 'audio_hour') return unitPriceUsd
    if (unit === 'audio_min' || unit === 'per_minute') return unitPriceUsd * 60
    return null
  }
  if (workload === 'text_to_speech') {
    if (unit === '1m_chars') return unitPriceUsd
    if (unit === '1k_chars') return unitPriceUsd * 1000
    return null
  }
  if (workload === 'image_generation') {
    if (unit === 'image' || unit === 'per_image') return unitPriceUsd * 1000
    return null
  }
  if (workload === 'video_generation') {
    if (unit === 'per_second') return unitPriceUsd * 60
    if (unit === 'video_min') return unitPriceUsd
    return null
  }
  if (workload === 'vision') {
    if (unit === '1k_images') return unitPriceUsd
    return null
  }
  return null
}

function throughputToPerHour(value: number, unit: string | null): number | null {
  if (!unit) return null
  const token = unit.trim().toLowerCase()
  if (token === 'per_hour') return value
  if (token === 'per_minute') return value * 60
  if (token === 'per_second') return value * 3600
  if (token === 'audio_min_per_minute') return value * 60
  if (token === 'audio_hour_per_hour') return value
  return null
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const text = await res.text().catch(() => 'Unknown error')
    throw new Error(`${res.status} ${res.statusText}: ${text}`)
  }
  return res.json() as Promise<T>
}

export interface DownloadReportResult {
  blob: Blob
  filename: string
  contentType: string
}

async function get<T>(
  path: string,
  params?: Record<string, string>,
  signal?: AbortSignal
): Promise<T> {
  const url = new URL(`${BASE}${path}`, window.location.origin)
  if (params) {
    Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, v))
  }
  const res = await fetch(url.toString(), { signal })
  if (!res.ok) {
    const text = await res.text().catch(() => 'Unknown error')
    throw new Error(`${res.status} ${res.statusText}: ${text}`)
  }
  return res.json() as Promise<T>
}

async function getMockCatalogRows(): Promise<CatalogBrowseResponse['rows']> {
  if (mockCatalogRowsCache !== null) {
    return mockCatalogRowsCache
  }
  try {
    const res = await fetch('/pricing_catalog.json')
    if (!res.ok) {
      throw new Error(`Failed to load pricing_catalog.json: ${res.status}`)
    }
    const payload = (await res.json()) as {
      rows?: Array<{
        provider: string
        workload_type: string
        sku_name: string
        unit_name: string
        unit_price_usd: number
        billing_mode: string
        confidence: string
        region?: string | null
        model_key?: string | null
        throughput_value?: number | null
        throughput_unit?: string | null
        previous_unit_price_usd?: number | null
        price_change_abs_usd?: number | null
        price_change_pct?: number | null
      }>
    }
    const rows =
      payload.rows?.map((r) => ({
        provider: r.provider,
        workload_type: canonicalWorkload(r.workload_type),
        sku_name: r.sku_name,
        unit_name: r.unit_name,
        unit_price_usd: Number(r.unit_price_usd),
        billing_mode: r.billing_mode as CatalogBrowseResponse['rows'][number]['billing_mode'],
        confidence: r.confidence as CatalogBrowseResponse['rows'][number]['confidence'],
        region: r.region ?? null,
        model_name: r.model_key ?? null,
        throughput_value: r.throughput_value ?? null,
        throughput_unit: r.throughput_unit ?? null,
        previous_unit_price_usd: r.previous_unit_price_usd ?? null,
        price_change_abs_usd: r.price_change_abs_usd ?? null,
        price_change_pct: r.price_change_pct ?? null,
      })) ?? []
    mockCatalogRowsCache = rows
    return rows
  } catch {
    const mock = await getMockModule()
    mockCatalogRowsCache = mock.MOCK_CATALOG_BROWSE.rows
    return mockCatalogRowsCache
  }
}

// ─── LLM Planning ─────────────────────────────────────────────────────────────

export async function planLLMWorkload(req: LLMPlanningRequest): Promise<LLMPlanningResponse> {
  if (USE_MOCK) {
    await delay(900)
    const allRows = await getMockCatalogRows()
    const allowedProviders = new Set(req.provider_ids)
    const rows = allRows.filter(
      (r) =>
        canonicalWorkload(r.workload_type) === 'llm' &&
        r.unit_name === '1m_tokens' &&
        (allowedProviders.size === 0 || allowedProviders.has(r.provider))
    )
    if (rows.length === 0) {
      return {
        plans: [],
        provider_diagnostics: req.provider_ids.map((provider) => ({
          provider,
          status: 'excluded',
          reason: 'No LLM token-priced offers found in catalog for this provider.',
        })),
        excluded_count: 0,
        warnings: ['No LLM token offers matched current provider filter.'],
      }
    }

    const monthlyTokensM = (req.tokens_per_day * 30) / 1_000_000
    const byProvider = new Map<string, typeof rows>()
    for (const row of rows) {
      const bucket = byProvider.get(row.provider) ?? []
      bucket.push(row)
      byProvider.set(row.provider, bucket)
    }

    const plans = Array.from(byProvider.entries())
      .map(([provider, providerRows], index) => {
        const sorted = [...providerRows].sort((a, b) => a.unit_price_usd - b.unit_price_usd)
        const best = sorted[0]
        const monthlyCost = monthlyTokensM * best.unit_price_usd
        const utilization = Math.min(0.98, (req.peak_to_avg / Math.max(req.util_target, 0.1)) * 0.2)
        const riskOverload = Math.min(1, 0.08 + utilization * 0.25)
        const riskComplexity = 0.05 + (index % 4) * 0.03
        const totalRisk = Math.min(1, 0.7 * riskOverload + 0.3 * riskComplexity)
        const score = monthlyCost * (1 + req.alpha * totalRisk)
        return {
          rank: 0,
          provider_id: provider,
          provider_name: provider,
          offering_id: best.sku_name,
          billing_mode: best.billing_mode,
          confidence: best.confidence,
          monthly_cost_usd: monthlyCost,
          score,
          utilization_at_peak: Number(utilization.toFixed(3)),
          risk: {
            risk_overload: Number(riskOverload.toFixed(3)),
            risk_complexity: Number(riskComplexity.toFixed(3)),
            total_risk: Number(totalRisk.toFixed(3)),
          },
          assumptions: {
            peak_to_avg: req.peak_to_avg,
            util_target: req.util_target,
            scaling_beta: req.beta,
            alpha: req.alpha,
            output_token_ratio: req.output_token_ratio,
            autoscale_inefficiency: req.autoscale_inefficiency,
            monthly_budget_max_usd: req.monthly_budget_max_usd,
          },
          why: `Lowest available token price for ${provider} in current catalog. Estimated monthly spend uses ${monthlyTokensM.toFixed(2)}M tokens/month at ${best.unit_price_usd.toFixed(4)} USD per 1M tokens.`,
        }
      })
      .filter((plan) => req.monthly_budget_max_usd <= 0 || plan.monthly_cost_usd <= req.monthly_budget_max_usd)
      .sort((a, b) => a.score - b.score)
      .slice(0, req.top_k)
      .map((plan, idx) => ({ ...plan, rank: idx + 1 }))

    const providerDiagnostics = req.provider_ids.map((provider) => {
      const providerRows = byProvider.get(provider) ?? []
      if (providerRows.length === 0) {
        return {
          provider,
          status: 'excluded' as const,
          reason: 'No LLM 1m_tokens offers found in current catalog.',
        }
      }
      return {
        provider,
        status: 'included' as const,
        reason: `Included (${providerRows.length} LLM offers).`,
      }
    })

    return {
      plans,
      provider_diagnostics: providerDiagnostics,
      excluded_count: Math.max(0, rows.length - plans.length),
      warnings: plans.length === 0 ? ['All candidate plans were filtered by budget.'] : [],
    }
  }
  return post<LLMPlanningResponse>('/api/v1/plan/llm', req)
}

// ─── Catalog Ranking (Non-LLM) ────────────────────────────────────────────────

export async function rankCatalogOffers(req: CatalogRankingRequest): Promise<CatalogRankingResponse> {
  if (USE_MOCK) {
    await delay(700)
    const allRows = await getMockCatalogRows()
    const workload = canonicalWorkload(req.workload_type)
    const allowed =
      req.allowed_providers.length > 0 ? new Set(req.allowed_providers) : null
    const rows = allRows.filter(
      (r) =>
        canonicalWorkload(r.workload_type) === workload &&
        (allowed === null || allowed.has(r.provider))
    )

    const providerSet = new Set(
      (allowed ? Array.from(allowed) : rows.map((r) => r.provider)).sort()
    )
    const providerReasons: Record<string, string> = {}
    let excludedCount = 0

    const rankable = rows
      .filter((r) => {
        if (req.unit_name && r.unit_name !== req.unit_name) return false
        return true
      })
      .map((r) => {
        let normalized = normalizeUnitPriceForWorkload(r.unit_price_usd, r.unit_name, workload)
        if (req.comparator_mode === 'normalized' && normalized === null && req.unit_name) {
          normalized = r.unit_price_usd
        }
        if (req.comparator_mode === 'normalized' && normalized === null) {
          excludedCount += 1
          return null
        }
        const comparator =
          req.comparator_mode === 'normalized' ? (normalized ?? r.unit_price_usd) : r.unit_price_usd
        const weighted = req.confidence_weighted
          ? comparator * confidenceMultiplier(r.confidence)
          : comparator
        let monthlyEstimate = req.monthly_usage > 0 ? req.monthly_usage * r.unit_price_usd : null
        let requiredReplicas: number | null = null
        let capacityCheck: CatalogRankingResponse['offers'][number]['capacity_check'] = 'unknown'

        if (req.throughput_aware && req.monthly_usage > 0) {
          const requiredPeakPerHour = (req.monthly_usage / (30 * 24)) * Math.max(req.peak_to_avg, 1)
          const requiredCapacityPerHour = requiredPeakPerHour / Math.max(req.util_target, 1e-6)
          const rowCapacity = throughputToPerHour(r.throughput_value ?? 0, r.throughput_unit)
          if (rowCapacity && rowCapacity > 0) {
            requiredReplicas = Math.max(1, Math.ceil(requiredCapacityPerHour / rowCapacity))
            capacityCheck = 'ok'
            if (monthlyEstimate !== null) monthlyEstimate *= requiredReplicas
          } else {
            capacityCheck = 'unknown'
          }
          if (req.strict_capacity_check && capacityCheck !== 'ok') {
            excludedCount += 1
            return null
          }
        } else {
          capacityCheck = 'ok'
        }

        if (req.monthly_budget_max_usd > 0) {
          const budgetValue = monthlyEstimate ?? weighted
          if (budgetValue > req.monthly_budget_max_usd) {
            excludedCount += 1
            return null
          }
        }

        return {
          provider: r.provider,
          sku_name: r.sku_name,
          billing_mode: r.billing_mode,
          unit_price_usd: r.unit_price_usd,
          normalized_price: req.comparator_mode === 'normalized' ? weighted : normalized,
          unit_name: r.unit_name,
          confidence: r.confidence,
          monthly_estimate_usd: monthlyEstimate,
          required_replicas: requiredReplicas,
          capacity_check: capacityCheck,
          previous_unit_price_usd: r.previous_unit_price_usd,
          price_change_abs_usd: r.price_change_abs_usd,
          price_change_pct: r.price_change_pct,
          _sort_price: weighted,
        }
      })
      .filter((row): row is NonNullable<typeof row> => row !== null)
      .sort((a, b) => {
        if (a.monthly_estimate_usd !== null && b.monthly_estimate_usd !== null) {
          if (a.monthly_estimate_usd !== b.monthly_estimate_usd) {
            return a.monthly_estimate_usd - b.monthly_estimate_usd
          }
        }
        return a._sort_price - b._sort_price
      })

    for (const provider of providerSet) {
      const providerRows = rankable.filter((r) => r.provider === provider)
      if (providerRows.length === 0) {
        providerReasons[provider] = 'No rankable offers after unit/comparator/budget filters.'
      } else {
        providerReasons[provider] = `Included (${providerRows.length} rankable offers).`
      }
    }

    const offers = rankable.slice(0, req.top_k).map((row, idx) => ({
      rank: idx + 1,
      provider: row.provider,
      sku_name: row.sku_name,
      billing_mode: row.billing_mode,
      unit_price_usd: row.unit_price_usd,
      normalized_price: row.normalized_price,
      unit_name: row.unit_name,
      confidence: row.confidence,
      monthly_estimate_usd: row.monthly_estimate_usd,
      required_replicas: row.required_replicas,
      capacity_check: row.capacity_check,
      previous_unit_price_usd: row.previous_unit_price_usd,
      price_change_abs_usd: row.price_change_abs_usd,
      price_change_pct: row.price_change_pct,
    }))

    const diagnostics = Array.from(providerSet).map((provider) => {
      const reason = providerReasons[provider] ?? 'No matching offers.'
      return {
        provider,
        status: reason.startsWith('Included') ? 'included' as const : 'excluded' as const,
        reason,
      }
    })

    return {
      offers,
      provider_diagnostics: diagnostics,
      excluded_count: excludedCount,
      warnings: offers.length === 0 ? ['No offers matched the selected filters.'] : [],
    }
  }
  return post<CatalogRankingResponse>('/api/v1/rank/catalog', req)
}

// ─── Catalog Browse ───────────────────────────────────────────────────────────

export async function browseCatalog(filters?: {
  workload_type?: string
  provider?: string
  unit_name?: string
  signal?: AbortSignal
}): Promise<CatalogBrowseResponse> {
  if (USE_MOCK) {
    await delay(400)
    const allRows = await getMockCatalogRows()
    const rows = allRows.filter((r) => {
      if (filters?.workload_type && r.workload_type !== filters.workload_type) return false
      if (filters?.provider && r.provider !== filters.provider) return false
      if (filters?.unit_name && r.unit_name !== filters.unit_name) return false
      return true
    })
    return { rows, total: rows.length }
  }
  return get<CatalogBrowseResponse>('/api/v1/catalog', {
    ...(filters?.workload_type ? { workload_type: filters.workload_type } : {}),
    ...(filters?.provider ? { provider: filters.provider } : {}),
    ...(filters?.unit_name ? { unit_name: filters.unit_name } : {}),
  }, filters?.signal)
}

// ─── Quality Catalog (v2.0) ─────────────────────────────────────────────────

function mockQualityScore(modelKey: string, workload: string): number | null {
  const token = `${modelKey} ${workload}`.toLowerCase()
  if (
    token.includes('claude-opus') ||
    token.includes('gpt-4o') ||
    token.includes('gemini-2.5-pro')
  ) {
    return 90
  }
  if (token.includes('deepseek') || token.includes('llama')) {
    return 80
  }
  if (token.includes('whisper') || token.includes('nova-2')) {
    return 84
  }
  if (token.includes('embedding') || token.includes('voyage')) {
    return 78
  }
  return null
}

export async function getQualityCatalog(filters?: {
  workload_type?: string
  provider?: string
  model_key_query?: string
  mapped_only?: boolean
  signal?: AbortSignal
}): Promise<QualityCatalogResponse> {
  if (USE_MOCK) {
    await delay(400)
    const allRows = await getMockCatalogRows()
    const query = filters?.model_key_query?.trim().toLowerCase() ?? ''
    const mappedOnly = filters?.mapped_only ?? false

    const rows = allRows
      .filter((r) => {
        if (filters?.workload_type && r.workload_type !== filters.workload_type) return false
        if (filters?.provider && r.provider !== filters.provider) return false
        if (
          query &&
          !((r.model_name ?? '').toLowerCase().includes(query) || r.sku_name.toLowerCase().includes(query))
        ) {
          return false
        }
        return true
      })
      .map((r) => {
        const score = mockQualityScore(r.model_name ?? '', r.workload_type)
        const mapped = score !== null
        return {
          provider: r.provider,
          workload_type: r.workload_type,
          model_key: r.model_name ?? '',
          sku_name: r.sku_name,
          billing_mode: r.billing_mode,
          unit_price_usd: r.unit_price_usd,
          unit_name: r.unit_name,
          quality_mapped: mapped,
          quality_model_id: mapped ? (r.model_name ?? '') : null,
          quality_score_0_100: score,
          quality_score_adjusted_0_100: mapped ? Number((50 + ((score ?? 50) - 50) * 0.8).toFixed(3)) : null,
          quality_confidence: mapped ? 'medium' : null,
          quality_confidence_weight: mapped ? 0.8 : null,
          quality_matched_by: mapped ? 'alias' : null,
        }
      })
      .filter((r) => (mappedOnly ? r.quality_mapped : true))
      .sort((a, b) => {
        const am = a.quality_mapped ? 0 : 1
        const bm = b.quality_mapped ? 0 : 1
        if (am !== bm) return am - bm
        const as = a.quality_score_adjusted_0_100 ?? -1
        const bs = b.quality_score_adjusted_0_100 ?? -1
        if (as !== bs) return bs - as
        if (a.unit_price_usd !== b.unit_price_usd) return a.unit_price_usd - b.unit_price_usd
        return a.provider.localeCompare(b.provider)
      })

    const mappedCount = rows.filter((r) => r.quality_mapped).length
    return {
      rows,
      total: rows.length,
      mapped_count: mappedCount,
      unmapped_count: rows.length - mappedCount,
    }
  }

  return get<QualityCatalogResponse>(
    '/api/v1/quality/catalog',
    {
      ...(filters?.workload_type ? { workload_type: filters.workload_type } : {}),
      ...(filters?.provider ? { provider: filters.provider } : {}),
      ...(filters?.model_key_query ? { model_key_query: filters.model_key_query } : {}),
      ...(typeof filters?.mapped_only === 'boolean' ? { mapped_only: String(filters.mapped_only) } : {}),
    },
    filters?.signal
  )
}

// ─── Invoice Analysis ─────────────────────────────────────────────────────────

export async function analyzeInvoice(file: File): Promise<InvoiceAnalysisResponse> {
  if (USE_MOCK) {
    await delay(1200)
    const mock = await getMockModule()
    return mock.MOCK_INVOICE_RESPONSE
  }
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE}/api/v1/invoice/analyze`, {
    method: 'POST',
    body: form,
  })
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json() as Promise<InvoiceAnalysisResponse>
}

// ─── AI Assistant ─────────────────────────────────────────────────────────────

// ─── AI Copilot (guided config) ───────────────────────────────────────────────

export async function nextCopilotTurn(req: CopilotTurnRequest): Promise<CopilotTurnResponse> {
  if (USE_MOCK) {
    await delay(700)
    const assistantTurns = req.history.filter((m) => m.role === 'assistant').length
    const msg = req.message.toLowerCase()
    const workload = canonicalWorkload(req.workload_type)
    const isLLM = workload === 'llm'

    // Turn 0 ─ greet + ask core question
    if (assistantTurns === 0) {
      return isLLM
        ? {
            reply:
              "Got it — LLM Inference optimization. A couple of quick questions to narrow the plan:\n- What's your approximate daily token volume?\n- Do you prefer a smaller model for speed/cost, or a larger one for quality?",
            extracted_spec: { workload_type: req.workload_type },
            missing_fields: ['daily token volume', 'model size', 'traffic pattern'],
            follow_up_questions: [
              '~5M tokens/day (small team)',
              '~50M tokens/day (production)',
              '~500M tokens/day (high volume)',
            ],
            apply_payload: null,
            is_ready: false,
          }
        : {
            reply: `Got it — ${workload.replace(/_/g, ' ')} pricing. Two quick questions:\n- What's your monthly usage volume?\n- Do you have a monthly budget cap?`,
            extracted_spec: { workload_type: req.workload_type },
            missing_fields: ['monthly usage volume', 'budget cap'],
            follow_up_questions: [
              'Low volume (~1K units/month)',
              'Medium volume (~10K units/month)',
              'No budget limit',
            ],
            apply_payload: null,
            is_ready: false,
          }
    }

    // Turn 1 ─ partial extraction + one more question
    if (assistantTurns === 1) {
      const big = msg.includes('500m') || msg.includes('high volume')
      if (isLLM) {
        const tokens_per_day = big ? 500_000_000 : msg.includes('50m') || msg.includes('production') ? 50_000_000 : 5_000_000
        const model_bucket = msg.includes('small') || msg.includes('speed') || msg.includes('7b') ? '7b' : '70b'
        return {
          reply:
            'Almost there! Which providers should I include — cheap open-weight inference, proprietary APIs, or cloud/enterprise?',
          extracted_spec: { workload_type: req.workload_type, tokens_per_day, model_bucket, traffic_pattern: 'business_hours' },
          missing_fields: ['provider preferences'],
          follow_up_questions: [
            'Open-weight (Fireworks, Together, Groq)',
            'Mix of open + proprietary',
            'Cloud / enterprise (AWS, Azure)',
          ],
          apply_payload: null,
          is_ready: false,
        }
      } else {
        const monthly_usage = msg.includes('10k') || msg.includes('medium') ? 10_000 : 1_000
        const monthly_budget_max_usd = msg.includes('no budget') || msg.includes('no limit') ? 0 : 0
        return {
          reply: 'Good. Should I factor in throughput capacity when ranking, or optimize purely by unit price?',
          extracted_spec: { workload_type: req.workload_type, monthly_usage, monthly_budget_max_usd },
          missing_fields: ['throughput requirements'],
          follow_up_questions: [
            'Rank by price only',
            'Factor in throughput capacity',
            'Strict capacity check',
          ],
          apply_payload: null,
          is_ready: false,
        }
      }
    }

    // Turn 2+ ─ ready with full apply_payload
    const cheap = msg.includes('cheap') || msg.includes('open-weight') || msg.includes('price only') || msg.includes('fireworks')
    const reliable = msg.includes('cloud') || msg.includes('enterprise') || msg.includes('strict') || msg.includes('aws')
    const throughput = msg.includes('throughput') || msg.includes('capacity') || msg.includes('strict')

    const preset: 'cheap' | 'balanced' | 'reliable' = cheap ? 'cheap' : reliable ? 'reliable' : 'balanced'
    const presetLabel = { cheap: 'Cost-optimized', balanced: 'Balanced', reliable: 'Reliability-focused' }[preset]

    const payload: CopilotApplyPayload = isLLM
      ? {
          tokens_per_day: 5_000_000,
          model_bucket: cheap ? '7b' : '70b',
          provider_ids: cheap
            ? ['fireworks', 'together_ai', 'groq']
            : reliable
            ? ['openai', 'anthropic', 'aws']
            : ['fireworks', 'together_ai', 'openai', 'anthropic'],
          traffic_pattern: 'business_hours',
          util_target: reliable ? 0.7 : 0.75,
          top_k: reliable ? 3 : 5,
          monthly_budget_max_usd: 0,
        }
      : {
          monthly_usage: 5_000,
          monthly_budget_max_usd: 0,
          confidence_weighted: true,
          comparator_mode: 'normalized',
          throughput_aware: throughput,
          strict_capacity_check: reliable,
          top_k: 5,
        }

    return {
      reply: `**${presetLabel}** configuration ready. Click **Apply to Config** to prefill the form, then review and submit.`,
      extracted_spec: { workload_type: req.workload_type, ...payload },
      missing_fields: [],
      follow_up_questions: [],
      apply_payload: payload,
      is_ready: true,
    }
  }
  return post<CopilotTurnResponse>('/api/v1/ai/copilot', req)
}

export async function askAI(req: AIAssistRequest): Promise<AIAssistResponse> {
  if (USE_MOCK) {
    await delay(800)
    const mock = await getMockModule()
    const msg = req.message.toLowerCase()
    const rows = await getMockCatalogRows()
    const workload = req.context.workload_type ? canonicalWorkload(req.context.workload_type) : null
    const scoped = rows.filter((r) => !workload || canonicalWorkload(r.workload_type) === workload)
    if (
      workload &&
      (msg.includes('cheap') || msg.includes('low cost') || msg.includes('best price'))
    ) {
      const cheapest = [...scoped].sort((a, b) => a.unit_price_usd - b.unit_price_usd).slice(0, 3)
      if (cheapest.length > 0) {
        const summary = cheapest
          .map(
            (r, i) =>
              `${i + 1}. ${r.provider} — ${r.sku_name} at ${r.unit_price_usd.toFixed(6)} USD/${r.unit_name}`
          )
          .join('\n')
        return {
          reply: `For workload **${workload}**, current lowest-priced catalog options are:\n\n${summary}\n\nThese are list-price comparisons from the current catalog snapshot.`,
          suggested_action: 'run_optimize',
        }
      }
    }
    if (msg.includes('cheap') || msg.includes('cost') || msg.includes('price')) {
      return mock.MOCK_AI_RESPONSES.cheapest!
    }
    if (msg.includes('risk') || msg.includes('score') || msg.includes('confidence')) {
      return mock.MOCK_AI_RESPONSES.risk!
    }
    return mock.MOCK_AI_RESPONSES.default!
  }
  return post<AIAssistResponse>('/api/v1/ai/assist', req)
}

// ─── Scaling Planner ──────────────────────────────────────────────────────────

export async function planScaling(req: ScalingPlanRequest): Promise<ScalingPlanResponse> {
  if (USE_MOCK) {
    await delay(550)
    const isLLM = req.mode === 'llm'
    const plans = req.llm_planning?.plans ?? []
    const offers = req.catalog_ranking?.offers ?? []

    const topRisk = plans.length > 0 ? plans[0].risk.total_risk : 0.25
    const riskBand: ScalingPlanResponse['risk_band'] =
      topRisk < 0.3 ? 'low' : topRisk < 0.6 ? 'medium' : 'high'
    const gpuCount = isLLM ? Math.max(1, Math.ceil(plans.length * 0.8)) : 0

    return {
      mode: req.mode,
      deployment_mode: isLLM ? 'autoscale' : 'serverless',
      estimated_gpu_count: gpuCount,
      suggested_gpu_type: isLLM ? 'A100 80GB' : null,
      projected_utilization: isLLM ? 0.72 : null,
      utilization_target: isLLM ? 0.75 : null,
      risk_band: riskBand,
      capacity_check: 'ok',
      rationale: isLLM
        ? `Autoscale deployment recommended across ${plans.length} provider plan${plans.length !== 1 ? 's' : ''}. GPU sizing targets 72% utilization at peak with a 20% headroom buffer. Overall risk profile is ${riskBand}.`
        : `Serverless billing is optimal for ${offers.length} ranked catalog offer${offers.length !== 1 ? 's' : ''}. No sustained GPU reservation is required — scale-to-zero eliminates idle cost.`,
      assumptions: isLLM
        ? [
            'GPU sizing derived from peak-to-avg ratio in form parameters',
            'A100 80GB selected for frontier-class LLM model compatibility',
            'Autoscale overhead: 20% buffer above projected peak load',
            'Utilization target (75%) applies to steady-state traffic',
          ]
        : [
            'Serverless billing model assumed for all selected providers',
            'No dedicated GPU reservation required',
            'Cost is usage-proportional; no idle overhead',
          ],
    }
  }
  return post<ScalingPlanResponse>('/api/v1/plan/scaling', req)
}

export async function generateReport(req: ReportGenerateRequest): Promise<ReportGenerateResponse> {
  if (USE_MOCK) {
    await delay(500)
    const catalogRows = await getMockCatalogRows()
    const providersSynced = new Set(catalogRows.map((r) => r.provider)).size
    const now = new Date().toISOString()
    const mode = req.mode
    const title = req.title ?? 'InferenceAtlas Optimization Report'
    const outputFormat = req.output_format ?? 'markdown'
    const includeCharts = req.include_charts !== false
    const includeCsv = req.include_csv_exports !== false
    const includeNarrative = req.include_narrative === true

    const llmPlans = req.llm_planning?.plans ?? []
    const catalogOffers = req.catalog_ranking?.offers ?? []
    const topLLM = llmPlans[0]
    const topCatalog = catalogOffers[0]

    const llmChartData = {
      cost_by_rank: llmPlans.map((plan) => ({
        rank: plan.rank,
        provider_id: plan.provider_id,
        provider_name: plan.provider_name,
        monthly_cost_usd: plan.monthly_cost_usd,
        score: plan.score,
        total_risk: plan.risk.total_risk,
      })),
      risk_breakdown: llmPlans.map((plan) => ({
        rank: plan.rank,
        provider_id: plan.provider_id,
        risk_overload: plan.risk.risk_overload,
        risk_complexity: plan.risk.risk_complexity,
        total_risk: plan.risk.total_risk,
      })),
      confidence_distribution: llmPlans.reduce<Record<string, number>>((acc, row) => {
        acc[row.confidence] = (acc[row.confidence] ?? 0) + 1
        return acc
      }, {}),
    }

    const catalogChartData = {
      cost_by_rank: catalogOffers.map((offer) => ({
        rank: offer.rank,
        provider: offer.provider,
        sku_name: offer.sku_name,
        monthly_estimate_usd: offer.monthly_estimate_usd,
        unit_price_usd: offer.unit_price_usd,
        unit_name: offer.unit_name,
      })),
      exclusion_breakdown: req.catalog_ranking?.exclusion_breakdown ?? {},
      relaxation_trace: req.catalog_ranking?.relaxation_steps ?? [],
      confidence_distribution: catalogOffers.reduce<Record<string, number>>((acc, row) => {
        acc[row.confidence] = (acc[row.confidence] ?? 0) + 1
        return acc
      }, {}),
    }

    const charts: ReportChart[] =
      mode === 'llm'
        ? [
            {
              id: 'cost_comparison',
              title: 'Cost Comparison',
              type: 'bar',
              x_label: 'Rank',
              y_label: 'Monthly Cost (USD)',
              sort_key: 'rank',
              legend: ['Monthly Cost'],
              series: [
                {
                  id: 'monthly_cost_usd',
                  label: 'Monthly Cost',
                  unit: 'usd',
                  points: llmPlans.map((plan) => ({
                    rank: plan.rank,
                    provider_name: plan.provider_name,
                    value: plan.monthly_cost_usd,
                  })),
                },
              ],
            },
            {
              id: 'risk_comparison',
              title: 'Risk Comparison',
              type: 'stacked_bar',
              x_label: 'Rank',
              y_label: 'Risk Score',
              sort_key: 'risk',
              legend: ['Overload Risk', 'Complexity Risk'],
              series: [
                {
                  id: 'risk_overload',
                  label: 'Overload Risk',
                  unit: 'risk_score',
                  points: llmPlans.map((plan) => ({
                    rank: plan.rank,
                    provider_name: plan.provider_name,
                    value: plan.risk.risk_overload,
                  })),
                },
                {
                  id: 'risk_complexity',
                  label: 'Complexity Risk',
                  unit: 'risk_score',
                  points: llmPlans.map((plan) => ({
                    rank: plan.rank,
                    provider_name: plan.provider_name,
                    value: plan.risk.risk_complexity,
                  })),
                },
              ],
            },
          ]
        : [
            {
              id: 'cost_comparison',
              title: 'Cost Comparison',
              type: 'bar',
              x_label: 'Rank',
              y_label: 'Monthly Estimate (USD)',
              sort_key: 'rank',
              legend: ['Monthly Estimate'],
              series: [
                {
                  id: 'monthly_estimate_usd',
                  label: 'Monthly Estimate',
                  unit: 'usd',
                  points: catalogOffers.map((offer) => ({
                    rank: offer.rank,
                    provider: offer.provider,
                    value: offer.monthly_estimate_usd ?? 0,
                  })),
                },
              ],
            },
            {
              id: 'fallback_trace',
              title: 'Fallback Trace',
              type: 'step_line',
              x_label: 'Relaxation Step',
              y_label: 'Offers Returned',
              sort_key: 'rank',
              legend: ['Offers Returned'],
              series: [
                {
                  id: 'offers_returned',
                  label: 'Offers Returned',
                  unit: 'count',
                  points: (req.catalog_ranking?.relaxation_steps ?? []).map((step, idx) => ({
                    step_index: idx,
                    step: String(step.step ?? `step_${idx}`),
                    value: Number(step.offers_returned ?? 0),
                  })),
                },
              ],
            },
          ]

    const sections =
      mode === 'llm'
        ? [
            {
              title: 'Executive Summary',
              bullets: [
                topLLM
                  ? `Top plan: ${topLLM.provider_name} (${topLLM.offering_id}) at $${topLLM.monthly_cost_usd.toFixed(2)}/mo.`
                  : 'No feasible LLM plans returned.',
                `Plans included: ${llmPlans.length}.`,
              ],
            },
            {
              title: 'Top Recommendations',
              bullets: llmPlans.slice(0, 5).map(
                (plan) =>
                  `#${plan.rank} ${plan.provider_name} · $${plan.monthly_cost_usd.toFixed(2)} · risk ${plan.risk.total_risk.toFixed(2)}`
              ),
            },
            {
              title: 'Diagnostics',
              bullets: [
                `Included providers: ${req.llm_planning?.provider_diagnostics.filter((d) => d.status === 'included').length ?? 0}.`,
                ...(req.llm_planning?.warnings ?? []).slice(0, 3),
              ],
            },
          ]
        : [
            {
              title: 'Executive Summary',
              bullets: [
                topCatalog
                  ? `Top offer: ${topCatalog.provider} (${topCatalog.sku_name}) at $${(topCatalog.monthly_estimate_usd ?? 0).toFixed(2)}/mo.`
                  : 'No matching catalog offers returned.',
                `Offers included: ${catalogOffers.length}.`,
              ],
            },
            {
              title: 'Top Recommendations',
              bullets: catalogOffers.slice(0, 10).map(
                (offer) =>
                  `#${offer.rank} ${offer.provider} · $${(offer.monthly_estimate_usd ?? 0).toFixed(2)} · ${offer.confidence}`
              ),
            },
            {
              title: 'Filter Diagnostics',
              bullets: [
                `Excluded offers count: ${req.catalog_ranking?.excluded_count ?? 0}.`,
                ...(req.catalog_ranking?.warnings ?? []).slice(0, 4),
              ],
            },
          ]

    const markdown = [
      `# ${title}`,
      '',
      `- Mode: \`${mode}\``,
      `- Generated at (UTC): \`${now}\``,
      '',
      '## Executive Summary',
      ...(sections[0]?.bullets ?? []).map((bullet) => `- ${bullet}`),
      '',
      ...(sections.slice(1).flatMap((section) => [
        `## ${section.title}`,
        ...(section.bullets.length > 0 ? section.bullets.map((bullet) => `- ${bullet}`) : ['- n/a']),
        '',
      ])),
      '## Notes',
      '- Generated in frontend mock mode from live run payload.',
      '',
    ].join('\n')

    const rankedCsv =
      mode === 'llm'
        ? [
            'rank,provider_id,provider_name,offering_id,monthly_cost_usd,total_risk',
            ...llmPlans.map(
              (plan) =>
                `${plan.rank},${plan.provider_id},${plan.provider_name},${plan.offering_id},${plan.monthly_cost_usd},${plan.risk.total_risk}`
            ),
          ].join('\n') + '\n'
        : [
            'rank,provider,sku_name,monthly_estimate_usd,unit_price_usd,unit_name,confidence',
            ...catalogOffers.map(
              (offer) =>
                `${offer.rank},${offer.provider},${offer.sku_name},${offer.monthly_estimate_usd ?? ''},${offer.unit_price_usd},${offer.unit_name},${offer.confidence}`
            ),
          ].join('\n') + '\n'

    const diagnosticsCsv =
      mode === 'llm'
        ? [
            'provider,status,reason',
            ...(req.llm_planning?.provider_diagnostics ?? []).map(
              (row) => `${row.provider},${row.status},${row.reason}`
            ),
          ].join('\n') + '\n'
        : [
            'provider,status,reason',
            ...(req.catalog_ranking?.provider_diagnostics ?? []).map(
              (row) => `${row.provider},${row.status},${row.reason}`
            ),
          ].join('\n') + '\n'

    return {
      report_id: `rep_mock_${Date.now().toString(36)}`,
      generated_at_utc: now,
      title,
      mode,
      sections,
      charts: includeCharts ? charts : [],
      chart_data: includeCharts ? (mode === 'llm' ? llmChartData : catalogChartData) : {},
      metadata: {
        chart_schema_version: 'v1.2-mock',
        catalog_generated_at_utc: now,
        catalog_schema_version: 'mock',
        catalog_row_count: catalogRows.length,
        catalog_providers_synced_count: providersSynced,
      },
      output_format: outputFormat,
      narrative: includeNarrative
        ? mode === 'llm'
          ? topLLM
            ? `Primary recommendation: ${topLLM.provider_name} is rank #1 at $${topLLM.monthly_cost_usd.toFixed(2)}/month with risk ${topLLM.risk.total_risk.toFixed(2)}.`
            : 'Primary recommendation unavailable because no LLM plans were returned.'
          : topCatalog
          ? `Primary recommendation: ${topCatalog.provider} ${topCatalog.sku_name} is rank #1 at $${(topCatalog.monthly_estimate_usd ?? 0).toFixed(2)}/month.`
          : 'Primary recommendation unavailable because no catalog offers were returned.'
        : null,
      csv_exports: includeCsv
        ? {
            'ranked_results.csv': rankedCsv,
            'provider_diagnostics.csv': diagnosticsCsv,
          }
        : {},
      markdown,
      html:
        outputFormat === 'html' || outputFormat === 'pdf'
          ? `<!doctype html><html><body><h1>${title}</h1><pre>${markdown}</pre></body></html>`
          : null,
      pdf_base64:
        outputFormat === 'pdf'
          ? 'JVBERi0xLjQKJcTl8uXrCg=='
          : null,
    }
  }
  return post<ReportGenerateResponse>('/api/v1/report/generate', req)
}

export async function downloadReport(
  req: ReportGenerateRequest
): Promise<DownloadReportResult> {
  if (USE_MOCK) {
    const report = await generateReport(req)
    const format = report.output_format ?? req.output_format ?? 'markdown'
    const dateToken = new Date(report.generated_at_utc).toISOString().slice(0, 10)
    if (format === 'html') {
      const htmlContent = report.html ?? '<!doctype html><html><body></body></html>'
      return {
        blob: new Blob([htmlContent], { type: 'text/html;charset=utf-8' }),
        filename: `${report.mode}_report_${dateToken}.html`,
        contentType: 'text/html;charset=utf-8',
      }
    }
    if (format === 'pdf') {
      const bytes = Uint8Array.from(atob(report.pdf_base64 ?? ''), (c) => c.charCodeAt(0))
      return {
        blob: new Blob([bytes], { type: 'application/pdf' }),
        filename: `${report.mode}_report_${dateToken}.pdf`,
        contentType: 'application/pdf',
      }
    }
    return {
      blob: new Blob([report.markdown], { type: 'text/markdown;charset=utf-8' }),
      filename: `${report.mode}_report_${dateToken}.md`,
      contentType: 'text/markdown;charset=utf-8',
    }
  }

  const res = await fetch(`${BASE}/api/v1/report/download`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!res.ok) {
    const text = await res.text().catch(() => 'Unknown error')
    throw new Error(`${res.status} ${res.statusText}: ${text}`)
  }
  const blob = await res.blob()
  const contentType = res.headers.get('content-type') ?? 'application/octet-stream'
  const cd = res.headers.get('content-disposition') ?? ''
  const match = cd.match(/filename=\"([^\"]+)\"/)
  const filename = match?.[1] ?? 'report.bin'
  return { blob, filename, contentType }
}

// ─── Cost Audit ────────────────────────────────────────────────────────────────

export async function auditCost(req: CostAuditRequest): Promise<CostAuditResponse> {
  if (USE_MOCK) {
    await delay(950)
    const isGpu = req.pricing_model !== 'token_api'
    const lowUtil = typeof req.avg_utilization === 'number' && req.avg_utilization < 0.5
    return {
      efficiency_score: 72,
      recommendations: [
        {
          recommendation_type: 'quantization',
          title: 'Enable lower-precision inference',
          rationale: 'FP8/INT8 can increase throughput 1.5–2× on H100-class GPUs, directly reducing per-token cost.',
          estimated_savings_pct: 28,
          priority: 'high',
        },
        {
          recommendation_type: isGpu ? 'procurement' : 'pricing_model_switch',
          title: isGpu ? 'Negotiate reserved instance pricing' : 'Evaluate dedicated GPU deployment',
          rationale: isGpu
            ? 'Steady traffic justifies a 1-year reserved commitment at a 30–40 % discount over on-demand.'
            : 'Token volume may have crossed the dedicated-GPU break-even threshold (est. 3–4 months payback).',
          estimated_savings_pct: 18,
          priority: 'medium',
        },
      ],
      hardware_recommendation: {
        tier: isGpu ? 'single_gpu' : 'serverless',
        gpu_family: isGpu ? 'H100-class' : null,
        deployment_shape: req.pricing_model,
        reasoning: isGpu
          ? 'Single H100 80 GB covers a typical 70 B model at moderate QPS with headroom for bursts.'
          : 'Serverless token API appropriate while traffic remains unpredictable.',
      },
      pricing_model_verdict: {
        current_model: req.pricing_model,
        verdict: req.pricing_model === 'token_api' ? 'consider_switch' : 'appropriate',
        reason: req.pricing_model === 'token_api'
          ? 'Token volume is high enough to reach dedicated-GPU break-even within 3–4 months.'
          : 'Dedicated GPU deployment aligns well with your steady traffic pattern.',
      },
      red_flags: lowUtil
        ? ['Low GPU utilisation detected — on-demand costs likely exceed committed pricing.']
        : [],
      estimated_monthly_savings: {
        low_usd: 400,
        high_usd: 1100,
        basis: 'combined_savings_pct=35; top_recommendations=2; current_spend estimated from token volume.',
      },
      score_breakdown: {
        base_score: 100,
        penalty_points: 32,
        bonus_points: 4,
        pre_cap_score: 72,
        post_cap_score: 72,
        major_flags: lowUtil ? 2 : 1,
        caps_applied: lowUtil ? ['major_flags_cap'] : [],
        combined_savings_pct: 35,
      },
      assumptions: [
        'Savings are directional estimates based on pricing priors.',
        'GPU recommendations assume H100 80 GB PCIe baseline.',
      ],
      data_gaps: [],
      data_gaps_detailed: [],
      pricing_source: 'provider_csv',
      pricing_source_provider: 'anthropic',
      pricing_source_gpu: isGpu ? req.gpu_type ?? null : null,
      per_modality_audits: req.pricing_model === 'mixed'
        ? [
            { modality: 'llm' as const,        efficiency_score: 68, top_recommendation: 'Enable quantization to reduce per-token cost.', red_flags: [] },
            { modality: 'embeddings' as const,  efficiency_score: 81, top_recommendation: null, red_flags: [] },
            { modality: 'asr' as const,  efficiency_score: 74, top_recommendation: 'Batch requests to improve GPU utilisation.', red_flags: ['Idle GPU hours detected on off-peak periods.'] },
          ]
        : undefined,
    }
  }
  const rawTokensPerDay = (req as unknown as { tokens_per_day?: number | null }).tokens_per_day
  const monthlyTokens = typeof rawTokensPerDay === 'number' && Number.isFinite(rawTokensPerDay)
    ? Math.max(0, rawTokensPerDay * 30)
    : 0

  const modalityMap: Record<string, string> = {
    llm: 'llm',
    asr: 'asr',
    speech_to_text: 'asr',
    tts: 'tts',
    text_to_speech: 'tts',
    embeddings: 'embeddings',
    image_gen: 'image_gen',
    image_generation: 'image_gen',
    vision: 'image_gen',
    video_gen: 'video_gen',
    video_generation: 'video_gen',
    mixed: 'mixed',
  }

  const payload = {
    modality: modalityMap[req.modality] ?? 'llm',
    model_name: req.model_name,
    pricing_model: req.pricing_model,
    monthly_input_tokens: req.pricing_model === 'token_api' ? monthlyTokens : null,
    monthly_output_tokens: req.pricing_model === 'token_api' ? monthlyTokens * 0.3 : null,
    gpu_type: req.gpu_type ?? null,
    gpu_count: req.gpu_count ?? null,
    traffic_pattern: req.traffic_pattern ?? 'unknown',
    caching_enabled: req.has_caching ? 'yes' : 'no',
    quantization_applied: req.has_quantization ? 'yes' : 'no',
    autoscaling: req.has_autoscaling ? 'yes' : 'no',
    monthly_ai_spend_usd: req.monthly_ai_spend_usd ?? null,
  }
  return post<CostAuditResponse>('/api/v1/audit/cost', payload)
}
