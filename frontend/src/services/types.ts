// ─── Shared ──────────────────────────────────────────────────────────────────

export type ConfidenceLevel =
  | 'high'
  | 'official'
  | 'medium'
  | 'estimated'
  | 'low'
  | 'vendor_list'

export type CapacityCheck = 'ok' | 'insufficient' | 'unknown'

export type BillingMode =
  | 'per_token'
  | 'dedicated_hourly'
  | 'autoscale_hourly'
  | 'per_unit'
  | 'hourly'
  | 'per_image'
  | 'per_minute'
  | 'per_character'

// ─── LLM Planning ────────────────────────────────────────────────────────────

export interface RiskBreakdown {
  risk_overload: number
  risk_complexity: number
  total_risk: number
}

export interface RankedPlan {
  rank: number
  provider_id: string
  provider_name: string
  offering_id: string
  billing_mode: BillingMode
  confidence: ConfidenceLevel
  monthly_cost_usd: number
  score: number
  utilization_at_peak: number | null
  risk: RiskBreakdown
  assumptions: Record<string, number>
  why: string
}

export interface ProviderDiagnostic {
  provider: string
  status: 'included' | 'excluded' | 'not_selected'
  reason: string
}

export interface LLMPlanningRequest {
  tokens_per_day: number
  model_bucket: string
  provider_ids: string[]
  peak_to_avg: number
  util_target: number
  output_token_ratio: number
  top_k: number
}

export interface LLMPlanningResponse {
  plans: RankedPlan[]
  provider_diagnostics: ProviderDiagnostic[]
  excluded_count: number
  warnings: string[]
}

// ─── Catalog / Non-LLM ───────────────────────────────────────────────────────

export interface RankedCatalogOffer {
  rank: number
  provider: string
  sku_name: string
  billing_mode: BillingMode
  unit_price_usd: number
  normalized_price: number | null
  unit_name: string
  confidence: ConfidenceLevel
  monthly_estimate_usd: number | null
  required_replicas: number | null
  capacity_check: CapacityCheck
}

export interface CatalogRankingRequest {
  workload_type: string
  allowed_providers: string[]
  unit_name: string | null
  monthly_usage: number
  monthly_budget_max_usd: number
  top_k: number
  confidence_weighted: boolean
  comparator_mode: 'normalized' | 'listed'
  throughput_aware: boolean
  peak_to_avg: number
  util_target: number
  strict_capacity_check: boolean
}

export interface CatalogRankingResponse {
  offers: RankedCatalogOffer[]
  provider_diagnostics: ProviderDiagnostic[]
  excluded_count: number
  warnings: string[]
}

// ─── Catalog Browse ───────────────────────────────────────────────────────────

export interface CatalogRow {
  provider: string
  workload_type: string
  sku_name: string
  unit_name: string
  unit_price_usd: number
  billing_mode: BillingMode
  confidence: ConfidenceLevel
  region: string | null
  model_name: string | null
  throughput_value: number | null
  throughput_unit: string | null
}

export interface CatalogBrowseResponse {
  rows: CatalogRow[]
  total: number
}

// ─── Invoice Analysis ─────────────────────────────────────────────────────────

export interface InvoiceLineItem {
  provider: string
  workload_type: string
  line_item: string
  quantity: number
  unit: string
  unit_price: number
  total: number
}

export interface InvoiceAnalysisResponse {
  line_items: InvoiceLineItem[]
  totals_by_provider: Record<string, number>
  grand_total: number
  detected_workloads: string[]
  warnings: string[]
}

// ─── AI Assistant ─────────────────────────────────────────────────────────────

export interface AIMessage {
  role: 'user' | 'assistant'
  content: string
  timestamp: number
}

export interface AIAssistRequest {
  message: string
  context: {
    workload_type: string | null
    providers: string[]
    recent_results: RankedCatalogOffer[] | RankedPlan[] | null
  }
}

export interface AIAssistResponse {
  reply: string
  suggested_action: string | null
}
