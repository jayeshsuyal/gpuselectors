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

// ─── Quality Catalog (v2.0) ─────────────────────────────────────────────────

export interface QualityCatalogRow {
  provider: string
  workload_type: string
  model_key: string
  sku_name: string
  billing_mode: BillingMode
  unit_price_usd: number
  unit_name: string
  quality_mapped: boolean
  quality_model_id?: string | null
  quality_score_0_100?: number | null
  quality_score_adjusted_0_100?: number | null
  quality_confidence?: string | null
  quality_confidence_weight?: number | null
  quality_matched_by?: string | null
}

export interface QualityCatalogResponse {
  rows: QualityCatalogRow[]
  total: number
  mapped_count: number
  unmapped_count: number
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

// ─── Scaling Planner ─────────────────────────────────────────────────────────

export type DeploymentMode = 'serverless' | 'dedicated' | 'autoscale' | 'unknown'
export type RiskBand = 'low' | 'medium' | 'high' | 'unknown'

export interface ScalingPlanRequest {
  mode: 'llm' | 'catalog'
  llm_planning?: LLMPlanningResponse | null
  catalog_ranking?: CatalogRankingResponse | null
}

export interface ScalingPlanResponse {
  mode: 'llm' | 'catalog'
  deployment_mode: DeploymentMode
  estimated_gpu_count: number
  suggested_gpu_type: string | null
  projected_utilization: number | null
  utilization_target: number | null
  risk_band: RiskBand
  capacity_check: 'ok' | 'insufficient' | 'unknown'
  rationale: string
  assumptions: string[]
}

// ─── Report Charts ────────────────────────────────────────────────────────────

export type ChartType = 'bar' | 'line' | 'stacked_bar' | 'step_line'
export type ChartUnit = 'usd' | '%' | 'count' | 'risk_score'

export interface ReportChartPoint {
  x?: string | number
  y?: number
  value?: number
  rank?: number
  step?: string
  step_index?: number
  confidence?: string
  provider?: string
  provider_name?: string
  sku_name?: string
  label?: string
}

export interface ReportChartSeries {
  id: string
  label?: string
  name?: string
  points?: ReportChartPoint[]
  data?: ReportChartPoint[]
  unit?: ChartUnit | string | null
  color?: string
}

export interface ReportChart {
  id: string
  type: ChartType
  title: string
  x_label?: string
  y_label?: string
  legend?: string[]
  meta?: Record<string, unknown>
  description?: string
  series: ReportChartSeries[]
  sort_key?: 'rank' | 'cost' | 'risk'
}

// ─── Cost Audit ───────────────────────────────────────────────────────────────

export type CostAuditModality =
  | 'llm'
  | 'asr'
  | 'tts'
  | 'embeddings'
  | 'image_gen'
  | 'video_gen'
  | 'mixed'

export type CostAuditPricingModel = 'token_api' | 'dedicated_gpu' | 'mixed'
export type CostAuditTrafficPattern = 'steady' | 'business_hours' | 'bursty'
export type CostAuditVerdict = 'appropriate' | 'consider_switch' | 'suboptimal' | 'unknown'
export type CostAuditHardwareTier = 'serverless' | 'single_gpu' | 'multi_gpu' | 'hybrid' | 'unknown'
export type CostAuditPriority = 'high' | 'medium' | 'low'
export type CostAuditDataGapImpact = 'high' | 'medium' | 'low'
export type CostAuditRecommendationType =
  | 'pricing_model_switch'
  | 'procurement'
  | 'quantization'
  | 'autoscaling'
  | 'caching'
  | 'hardware_match'
  | 'other'
export type CostAuditPricingSource = 'provider_csv' | 'heuristic_prior' | 'unknown'

export interface CostAuditRequest {
  modality: CostAuditModality
  model_name: string
  pricing_model: CostAuditPricingModel
  // Frontend convenience input; translated to monthly token fields in service layer.
  tokens_per_day?: number | null
  monthly_ai_spend_usd?: number | null
  gpu_type?: string | null
  gpu_count?: number | null
  avg_utilization?: number | null
  traffic_pattern?: CostAuditTrafficPattern | null
  has_caching?: boolean | null
  has_quantization?: boolean | null
  has_autoscaling?: boolean | null
}

export interface CostAuditRecommendation {
  recommendation_type: CostAuditRecommendationType
  title: string
  rationale: string
  estimated_savings_pct: number
  priority: CostAuditPriority
}

export interface CostAuditHardwareRecommendation {
  tier: CostAuditHardwareTier
  gpu_family: string | null
  deployment_shape: string
  reasoning: string
}

export interface CostAuditPricingVerdict {
  current_model: CostAuditPricingModel
  verdict: CostAuditVerdict
  reason: string
}

export interface CostAuditSavingsEstimate {
  low_usd: number
  high_usd: number
  basis: string
}

export interface CostAuditScoreBreakdown {
  base_score: number
  penalty_points: number
  bonus_points: number
  pre_cap_score: number
  post_cap_score: number
  major_flags: number
  caps_applied: string[]
  combined_savings_pct: number
}

export interface CostAuditDataGap {
  field: string
  impact: CostAuditDataGapImpact
  why_it_matters: string
}

export interface CostAuditAlternative {
  provider: string
  gpu_type?: string | null
  deployment_mode: 'serverless' | 'dedicated' | 'autoscale'
  estimated_monthly_cost_usd: number
  savings_vs_current_usd: number
  savings_vs_current_pct: number
  confidence: 'high' | 'medium' | 'low'
  source: 'provider_csv' | 'heuristic_prior' | 'current_baseline'
  rationale: string
}

/** Per-modality sub-audit leg for mixed-pipeline responses */
export interface CostAuditModalityLeg {
  modality: CostAuditModality
  efficiency_score: number
  top_recommendation?: string | null
  red_flags?: string[]
}

export interface CostAuditResponse {
  efficiency_score: number
  recommendations: CostAuditRecommendation[]
  hardware_recommendation: CostAuditHardwareRecommendation
  pricing_model_verdict: CostAuditPricingVerdict
  red_flags: string[]
  estimated_monthly_savings: CostAuditSavingsEstimate
  /** Optional for resilience — backend may not always populate */
  score_breakdown?: CostAuditScoreBreakdown
  assumptions: string[]
  data_gaps: string[]
  data_gaps_detailed: CostAuditDataGap[]
  /** Upcoming pricing source fields — absent until backend ships them */
  pricing_source?: CostAuditPricingSource
  pricing_source_provider?: string | null
  pricing_source_gpu?: string | null
  recommended_options?: CostAuditAlternative[]
  /** Mixed-pipeline only — per-modality sub-audit legs */
  per_modality_audits?: CostAuditModalityLeg[]
}

// ─── Report Generation ────────────────────────────────────────────────────────

export interface ReportSection {
  title: string
  bullets: string[]
}

export interface ReportGenerateRequest {
  mode: 'llm' | 'catalog'
  title?: string
  output_format?: 'markdown' | 'html' | 'pdf'
  include_charts?: boolean
  include_csv_exports?: boolean
  include_narrative?: boolean
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
  charts?: ReportChart[]
  metadata: Record<string, unknown>
  output_format: 'markdown' | 'html' | 'pdf'
  narrative?: string | null
  csv_exports: Record<string, string>
  markdown: string
  html?: string | null
  pdf_base64?: string | null
}
