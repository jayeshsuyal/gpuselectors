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
  InvoiceAnalysisResponse,
  AIAssistRequest,
  AIAssistResponse,
} from './types'

import {
  MOCK_CATALOG_BROWSE,
  MOCK_INVOICE_RESPONSE,
  MOCK_AI_RESPONSES,
} from './mockData'

const USE_MOCK = (import.meta.env.VITE_USE_MOCK_API ?? 'true').toLowerCase() !== 'false'
const BASE = import.meta.env.VITE_API_BASE_URL ?? ''
let mockCatalogRowsCache: CatalogBrowseResponse['rows'] | null = null

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
      })) ?? []
    mockCatalogRowsCache = rows
    return rows
  } catch {
    mockCatalogRowsCache = MOCK_CATALOG_BROWSE.rows
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

// ─── Invoice Analysis ─────────────────────────────────────────────────────────

export async function analyzeInvoice(file: File): Promise<InvoiceAnalysisResponse> {
  if (USE_MOCK) {
    await delay(1200)
    return MOCK_INVOICE_RESPONSE
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

export async function askAI(req: AIAssistRequest): Promise<AIAssistResponse> {
  if (USE_MOCK) {
    await delay(800)
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
      return MOCK_AI_RESPONSES.cheapest!
    }
    if (msg.includes('risk') || msg.includes('score') || msg.includes('confidence')) {
      return MOCK_AI_RESPONSES.risk!
    }
    return MOCK_AI_RESPONSES.default!
  }
  return post<AIAssistResponse>('/api/v1/ai/assist', req)
}
