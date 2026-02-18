/**
 * Service layer — all backend calls go through here.
 * Swap USE_MOCK = false and set VITE_API_BASE_URL to connect a real API.
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
  MOCK_LLM_RESPONSE,
  MOCK_STT_RESPONSE,
  MOCK_CATALOG_BROWSE,
  MOCK_INVOICE_RESPONSE,
  MOCK_AI_RESPONSES,
} from './mockData'

const USE_MOCK = true
const BASE = import.meta.env.VITE_API_BASE_URL ?? ''

function delay(ms = 600): Promise<void> {
  return new Promise((r) => setTimeout(r, ms))
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

async function get<T>(path: string, params?: Record<string, string>): Promise<T> {
  const url = new URL(`${BASE}${path}`, window.location.origin)
  if (params) {
    Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, v))
  }
  const res = await fetch(url.toString())
  if (!res.ok) {
    const text = await res.text().catch(() => 'Unknown error')
    throw new Error(`${res.status} ${res.statusText}: ${text}`)
  }
  return res.json() as Promise<T>
}

// ─── LLM Planning ─────────────────────────────────────────────────────────────

export async function planLLMWorkload(req: LLMPlanningRequest): Promise<LLMPlanningResponse> {
  if (USE_MOCK) {
    await delay(900)
    return MOCK_LLM_RESPONSE
  }
  return post<LLMPlanningResponse>('/api/v1/plan/llm', req)
}

// ─── Catalog Ranking (Non-LLM) ────────────────────────────────────────────────

export async function rankCatalogOffers(req: CatalogRankingRequest): Promise<CatalogRankingResponse> {
  if (USE_MOCK) {
    await delay(700)
    return MOCK_STT_RESPONSE
  }
  return post<CatalogRankingResponse>('/api/v1/rank/catalog', req)
}

// ─── Catalog Browse ───────────────────────────────────────────────────────────

export async function browseCatalog(filters?: {
  workload_type?: string
  provider?: string
  unit_name?: string
}): Promise<CatalogBrowseResponse> {
  if (USE_MOCK) {
    await delay(400)
    const rows = MOCK_CATALOG_BROWSE.rows.filter((r) => {
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
  })
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
