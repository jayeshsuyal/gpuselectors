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
  beta: number
  alpha: number
  autoscale_inefficiency: number
  monthly_budget_max_usd: number
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
  previous_unit_price_usd?: number | null
  price_change_abs_usd?: number | null
  price_change_pct?: number | null
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
  relaxation_applied?: boolean
  relaxation_steps?: Array<Record<string, unknown>>
  exclusion_breakdown?: Record<string, number>
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
  previous_unit_price_usd?: number | null
  price_change_abs_usd?: number | null
  price_change_pct?: number | null
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

// ─── AI Copilot ───────────────────────────────────────────────────────────────

export interface CopilotMessage {
  role: 'user' | 'assistant'
  content: string
  timestamp: number
}

export interface CopilotExtractedSpec {
  workload_type?: string
  tokens_per_day?: number
  model_bucket?: string
  provider_ids?: string[]
  traffic_pattern?: string
  monthly_budget_max_usd?: number
  unit_name?: string | null
  monthly_usage?: number
}

/** Flat union of all form fields — Optimize.tsx casts to the appropriate subset. */
export type CopilotApplyPayload = {
  // LLM
  tokens_per_day?: number
  model_bucket?: string
  provider_ids?: string[]
  traffic_pattern?: string
  peak_to_avg?: number
  util_target?: number
  beta?: number
  alpha?: number
  autoscale_inefficiency?: number
  monthly_budget_max_usd?: number
  output_token_ratio?: number
  top_k?: number
  // Non-LLM
  unit_name?: string | null
  monthly_usage?: number
  confidence_weighted?: boolean
  comparator_mode?: 'normalized' | 'listed'
  throughput_aware?: boolean
  strict_capacity_check?: boolean
}

export interface CopilotTurnRequest {
  message: string
  history: CopilotMessage[]
  workload_type: string
}

export interface CopilotTurnResponse {
  reply: string
  extracted_spec: CopilotExtractedSpec
  missing_fields: string[]
  follow_up_questions: string[]
  apply_payload: CopilotApplyPayload | null
  is_ready: boolean
}

// ─── Report Generation ────────────────────────────────────────────────────────

export interface ReportSection {
  title: string
  bullets: string[]
}

export interface ReportGenerateRequest {
  mode: 'llm' | 'catalog'
  title?: string
  include_charts?: boolean
  llm_planning?: LLMPlanningResponse
  catalog_ranking?: CatalogRankingResponse
}

export interface ReportGenerateResponse {
  report_id: string
  generated_at_utc: string
  title: string
  mode: 'llm' | 'catalog'
  sections: ReportSection[]
  chart_data: Record<string, unknown>
  metadata: Record<string, unknown>
  markdown: string
}
