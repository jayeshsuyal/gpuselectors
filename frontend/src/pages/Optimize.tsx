import { useState } from 'react'
import { ArrowLeft, Download, Loader2, Sparkles, SlidersHorizontal, X } from 'lucide-react'
import { useAIContext } from '@/context/AIContext'
import { WorkloadSelector } from '@/components/WorkloadSelector'
import { LLMForm } from '@/components/optimize/LLMForm'
import { NonLLMForm } from '@/components/optimize/NonLLMForm'
import { CopilotPanel } from '@/components/optimize/CopilotPanel'
import { ResultsTable } from '@/components/optimize/ResultsTable'
import { SkeletonCard } from '@/components/ui/skeleton'
import { generateReport, planLLMWorkload, rankCatalogOffers } from '@/services/api'
import type { LLMFormValues, NonLLMFormValues } from '@/schemas/forms'
import type {
  LLMPlanningResponse,
  CatalogRankingResponse,
  CopilotApplyPayload,
  ReportGenerateRequest,
  ReportGenerateResponse,
} from '@/services/types'
import type { WorkloadTypeId } from '@/lib/constants'
import { WORKLOAD_TYPES } from '@/lib/constants'

type Step = 'select' | 'configure'
type ConfigMode = 'copilot' | 'guided'

export function OptimizePage() {
  const [step, setStep] = useState<Step>('select')
  const [mode, setMode] = useState<ConfigMode>('copilot')
  const [workload, setWorkload] = useState<WorkloadTypeId | null>(null)
  const [initialValues, setInitialValues] = useState<CopilotApplyPayload | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [llmResult, setLlmResult] = useState<LLMPlanningResponse | null>(null)
  const [catalogResult, setCatalogResult] = useState<CatalogRankingResponse | null>(null)
  const [reportLoading, setReportLoading] = useState(false)
  const [reportError, setReportError] = useState<string | null>(null)
  const [reportData, setReportData] = useState<ReportGenerateResponse | null>(null)
  const [downloadSuccess, setDownloadSuccess] = useState(false)
  const { setAIContext } = useAIContext()

  function handleWorkloadSelect(id: WorkloadTypeId) {
    setWorkload(id)
    setMode('copilot')
    setStep('configure')
    setInitialValues(null)
    setLlmResult(null)
    setCatalogResult(null)
    setError(null)
    setReportError(null)
    setReportData(null)
    setDownloadSuccess(false)
    setAIContext({ workload_type: id, providers: [] })
  }

  function handleApply(payload: CopilotApplyPayload) {
    setInitialValues(payload)
    setMode('guided')
    setError(null)
  }

  function handleBack() {
    setStep('select')
    setWorkload(null)
    setInitialValues(null)
    setLlmResult(null)
    setCatalogResult(null)
    setError(null)
    setReportError(null)
    setReportData(null)
    setDownloadSuccess(false)
    setAIContext({ workload_type: null, providers: [] })
  }

  async function handleLLMSubmit(values: LLMFormValues) {
    setLoading(true)
    setError(null)
    try {
      const res = await planLLMWorkload({
        tokens_per_day: values.tokens_per_day,
        model_bucket: values.model_bucket,
        provider_ids: values.provider_ids,
        peak_to_avg: values.peak_to_avg,
        util_target: values.util_target,
        beta: values.beta,
        alpha: values.alpha,
        autoscale_inefficiency: values.autoscale_inefficiency,
        monthly_budget_max_usd: values.monthly_budget_max_usd,
        output_token_ratio: values.output_token_ratio,
        top_k: values.top_k,
      })
      setLlmResult(res)
      setReportError(null)
      setReportData(null)
      setAIContext({
        workload_type: workload,
        providers: [...new Set(res.plans.map((p) => p.provider_id))],
      })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Planning failed')
    } finally {
      setLoading(false)
    }
  }

  async function handleNonLLMSubmit(values: NonLLMFormValues) {
    setLoading(true)
    setError(null)
    try {
      const res = await rankCatalogOffers({
        workload_type: values.workload_type,
        allowed_providers: values.provider_ids,
        unit_name: values.unit_name,
        monthly_usage: values.monthly_usage,
        monthly_budget_max_usd: values.monthly_budget_max_usd,
        top_k: values.top_k,
        confidence_weighted: values.confidence_weighted,
        comparator_mode: values.comparator_mode,
        throughput_aware: values.throughput_aware,
        peak_to_avg: values.peak_to_avg,
        util_target: values.util_target,
        strict_capacity_check: values.strict_capacity_check,
      })
      setCatalogResult(res)
      setReportError(null)
      setReportData(null)
      setAIContext({
        workload_type: workload,
        providers: [...new Set(res.offers.map((o) => o.provider))],
      })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Ranking failed')
    } finally {
      setLoading(false)
    }
  }

  const isLLM = workload === 'llm'
  const hasResults = isLLM ? llmResult !== null : catalogResult !== null
  const workloadMeta = WORKLOAD_TYPES.find((w) => w.id === workload)

  const chartData = reportData?.chart_data ?? {}
  const costByRankCount = Array.isArray(chartData['cost_by_rank'])
    ? (chartData['cost_by_rank'] as unknown[]).length : null
  const riskBreakdownCount = Array.isArray(chartData['risk_breakdown'])
    ? (chartData['risk_breakdown'] as unknown[]).length : null
  const exclusionKeys = (typeof chartData['exclusion_breakdown'] === 'object' &&
    chartData['exclusion_breakdown'] !== null &&
    !Array.isArray(chartData['exclusion_breakdown']))
    ? Object.keys(chartData['exclusion_breakdown'] as Record<string, unknown>).length : null

  function downloadReportMarkdown(report: ReportGenerateResponse) {
    const blob = new Blob([report.markdown], { type: 'text/markdown;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    const dateToken = new Date(report.generated_at_utc).toISOString().slice(0, 10)
    anchor.href = url
    anchor.download = `${report.mode}_report_${dateToken}.md`
    document.body.appendChild(anchor)
    anchor.click()
    anchor.remove()
    URL.revokeObjectURL(url)
    setDownloadSuccess(true)
    window.setTimeout(() => setDownloadSuccess(false), 3000)
  }

  async function handleGenerateReport(autoDownload = false) {
    if (!workload) return
    setReportLoading(true)
    setReportError(null)
    try {
      const req: ReportGenerateRequest = isLLM
        ? {
            mode: 'llm',
            title: `${workloadMeta?.label ?? 'LLM'} Optimization Report`,
            include_charts: true,
            llm_planning: llmResult ?? undefined,
          }
        : {
            mode: 'catalog',
            title: `${workloadMeta?.label ?? 'Catalog'} Optimization Report`,
            include_charts: true,
            catalog_ranking: catalogResult ?? undefined,
          }
      const report = await generateReport(req)
      setReportData(report)
      if (autoDownload) {
        downloadReportMarkdown(report)
      }
    } catch (e) {
      setReportError(e instanceof Error ? e.message : 'Report generation failed')
    } finally {
      setReportLoading(false)
    }
  }

  return (
    <div className="max-w-3xl mx-auto px-4 py-8 sm:px-6">
      {/* ── Page header (select step only) ── */}
      {step === 'select' && (
        <div className="mb-8 page-section">
          <div className="eyebrow mb-2">Cost Intelligence</div>
          <h1 className="text-2xl font-bold tracking-tight mb-2">
            <span className="text-gradient">Optimize</span>{' '}
            <span style={{ color: 'var(--text-primary)' }}>Workload</span>
          </h1>
          <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
            Pick a workload category and we'll rank the most cost-effective inference options
            across every major AI provider.
          </p>
        </div>
      )}

      {/* Back */}
      {step !== 'select' && (
        <button
          onClick={handleBack}
          className="flex items-center gap-1.5 text-xs mb-6 transition-colors hover:text-zinc-200"
          style={{ color: 'var(--text-tertiary)' }}
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          Change workload
        </button>
      )}

      {/* Step indicator */}
      <div className="flex items-center gap-2 mb-6">
        {(['select', 'configure'] as Step[]).map((s, i) => {
          const isActive = step === s
          const isDone = step === 'configure' && s === 'select'
          return (
            <div key={s} className="flex items-center gap-2">
              <div
                className="w-5 h-5 rounded-full text-[10px] flex items-center justify-center font-bold transition-all duration-200"
                style={
                  isActive
                    ? { background: 'var(--brand-gradient)', color: '#fff', boxShadow: 'var(--shadow-glow-sm)' }
                    : isDone
                    ? { background: 'rgba(124,92,252,0.15)', border: '1px solid rgba(124,92,252,0.35)', color: 'var(--brand-hover)' }
                    : { background: 'rgba(255,255,255,0.05)', border: '1px solid var(--border-default)', color: 'var(--text-disabled)' }
                }
              >
                {i + 1}
              </div>
              <span
                className="text-xs transition-colors"
                style={{ color: isActive ? 'var(--text-primary)' : 'var(--text-disabled)', fontWeight: isActive ? 600 : 400 }}
              >
                {s === 'select' ? 'Category' : 'Configure'}
              </span>
              {i < 1 && <div className="w-8 subtle-divider" />}
            </div>
          )
        })}
      </div>

      {/* ── Step: Select ── */}
      {step === 'select' && (
        <WorkloadSelector selected={workload} onSelect={handleWorkloadSelect} />
      )}

      {/* ── Step: Configure ── */}
      {step === 'configure' && workload && (
        <div className="space-y-5 animate-enter">
          {/* Workload header */}
          <div>
            <div className="eyebrow mb-1">{workloadMeta?.label ?? workload.replace(/_/g, ' ')}</div>
            <h2 className="text-lg font-bold tracking-tight" style={{ color: 'var(--text-primary)' }}>
              Cost Optimization
            </h2>
            <p className="text-xs mt-0.5" style={{ color: 'var(--text-tertiary)' }}>
              {isLLM
                ? 'Capacity-aware ranking with GPU scaling, traffic modeling, and risk scoring'
                : 'Price catalog ranking with optional throughput and budget constraints'}
            </p>
          </div>

          {/* Mode tabs */}
          <div className="flex border-b border-white/[0.06] -mb-px">
            {[
              { id: 'copilot' as ConfigMode, icon: <Sparkles className="h-3.5 w-3.5" />, label: 'Ask IA AI' },
              { id: 'guided' as ConfigMode, icon: <SlidersHorizontal className="h-3.5 w-3.5" />, label: 'Guided Config' },
            ].map((tab) => (
              <button
                key={tab.id}
                type="button"
                onClick={() => setMode(tab.id)}
                className="flex items-center gap-1.5 px-4 py-2.5 text-xs font-medium border-b-2 transition-all duration-200"
                style={
                  mode === tab.id
                    ? { borderColor: 'var(--brand)', color: 'var(--brand-hover)' }
                    : { borderColor: 'transparent', color: 'var(--text-tertiary)' }
                }
              >
                {tab.icon}
                {tab.label}
              </button>
            ))}
          </div>

          {/* ── Copilot mode ── */}
          {mode === 'copilot' && (
            <CopilotPanel
              workloadType={workload}
              isLLM={isLLM}
              onApply={handleApply}
            />
          )}

          {/* ── Guided mode ── */}
          {mode === 'guided' && (
            <>
              {error && (
                <div
                  className="rounded-lg px-3 py-2.5 text-xs flex items-start justify-between gap-2 border"
                  style={{ borderColor: 'var(--danger-border)', background: 'var(--danger-bg)', color: 'var(--danger-text)' }}
                >
                  <span>{error}</span>
                  <button
                    onClick={() => setError(null)}
                    className="flex-shrink-0 transition-colors hover:text-white mt-0.5"
                    aria-label="Dismiss error"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </div>
              )}

              {isLLM ? (
                <LLMForm
                  key={`llm-${JSON.stringify(initialValues)}`}
                  onSubmit={handleLLMSubmit}
                  loading={loading}
                  initialValues={initialValues as Partial<LLMFormValues> | undefined}
                />
              ) : (
                <NonLLMForm
                  key={`non-llm-${JSON.stringify(initialValues)}`}
                  workloadType={workload}
                  onSubmit={handleNonLLMSubmit}
                  loading={loading}
                  initialValues={initialValues as Partial<NonLLMFormValues> | undefined}
                />
              )}
            </>
          )}

          {/* Loading skeleton */}
          {loading && (
            <div className="space-y-3 pt-2">
              <SkeletonCard />
              <SkeletonCard />
              <SkeletonCard />
            </div>
          )}

          {/* ── Shared results panel ── */}
          {hasResults && !loading && (
            <div className="space-y-4 pt-5 border-t border-white/[0.06] animate-enter">
              <div>
                <div className="eyebrow mb-1">Results</div>
                <h2 className="text-base font-semibold tracking-tight" style={{ color: 'var(--text-primary)' }}>
                  Ranked Configurations
                </h2>
                <p className="text-xs mt-0.5" style={{ color: 'var(--text-tertiary)' }}>
                  {isLLM
                    ? 'Capacity-optimized plans sorted by total cost score'
                    : 'Sorted by normalized unit price'}
                </p>
              </div>
              <div className="space-y-2">
                {/* Format selector */}
                <div className="flex items-center gap-2" role="group" aria-label="Report format">
                  <span className="text-[11px]" style={{ color: 'var(--text-tertiary)' }}>Format</span>
                  <button
                    type="button"
                    aria-pressed={true}
                    className="rounded px-2 py-1 text-[11px] border focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--brand)] focus-visible:ring-offset-1 focus-visible:ring-offset-[var(--bg-base)]"
                    style={{ borderColor: 'var(--brand-border)', background: 'rgba(124,92,252,0.08)', color: 'var(--brand-hover)' }}
                  >
                    Markdown (.md)
                  </button>
                  <button
                    type="button"
                    disabled
                    aria-label="PDF format — coming soon"
                    className="rounded px-2 py-1 text-[11px] border disabled:opacity-40 disabled:cursor-not-allowed"
                    style={{ borderColor: 'rgba(255,255,255,0.08)', background: 'var(--bg-elevated)', color: 'var(--text-disabled)' }}
                  >
                    PDF <span className="text-[9px] opacity-70">soon</span>
                  </button>
                </div>

                {/* Generate + Download */}
                <div className="flex items-center gap-2 flex-wrap">
                  <button
                    type="button"
                    onClick={() => void handleGenerateReport(false)}
                    disabled={reportLoading}
                    className="inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs transition-colors disabled:opacity-60 disabled:cursor-not-allowed focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--brand)] focus-visible:ring-offset-1 focus-visible:ring-offset-[var(--bg-base)]"
                    style={{ borderColor: 'var(--border-default)', color: 'var(--text-secondary)', background: 'var(--bg-elevated)' }}
                  >
                    {reportLoading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
                    {reportLoading ? 'Generating…' : 'Generate Report'}
                  </button>
                  <button
                    type="button"
                    aria-label="Download report as Markdown file"
                    onClick={() => {
                      if (reportData) {
                        downloadReportMarkdown(reportData)
                      } else {
                        void handleGenerateReport(true)
                      }
                    }}
                    disabled={reportLoading || (!reportData && !hasResults)}
                    className="inline-flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-xs transition-colors disabled:opacity-60 disabled:cursor-not-allowed focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--brand)] focus-visible:ring-offset-1 focus-visible:ring-offset-[var(--bg-base)]"
                    style={{ borderColor: 'var(--border-default)', color: 'var(--text-secondary)', background: 'var(--bg-elevated)' }}
                  >
                    {reportLoading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Download className="h-3.5 w-3.5" />}
                    {reportLoading ? 'Working…' : 'Download .md'}
                  </button>
                  {downloadSuccess && (
                    <span className="text-[11px]" style={{ color: '#22c55e' }} aria-live="polite">
                      ✓ Downloaded
                    </span>
                  )}
                </div>

                {reportError && (
                  <p className="text-xs" style={{ color: 'var(--danger)' }}>
                    {reportError}
                  </p>
                )}
              </div>

              {isLLM ? (
                <ResultsTable
                  mode="llm"
                  plans={llmResult?.plans ?? []}
                  diagnostics={llmResult?.provider_diagnostics ?? []}
                  warnings={llmResult?.warnings ?? []}
                  excludedCount={llmResult?.excluded_count ?? 0}
                />
              ) : (
                <ResultsTable
                  mode="non-llm"
                  offers={catalogResult?.offers ?? []}
                  diagnostics={catalogResult?.provider_diagnostics ?? []}
                  warnings={catalogResult?.warnings ?? []}
                  excludedCount={catalogResult?.excluded_count ?? 0}
                />
              )}

              {reportData && (
                <div
                  className="rounded-lg border p-4 space-y-3"
                  style={{ borderColor: 'var(--border-default)', background: 'var(--bg-elevated)' }}
                >
                  <div className="flex items-center justify-between gap-2">
                    <div>
                      <div className="eyebrow mb-1">Report Preview</div>
                      <h3 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
                        {reportData.title}
                      </h3>
                      <p className="text-[11px]" style={{ color: 'var(--text-tertiary)' }}>
                        Generated at {new Date(reportData.generated_at_utc).toLocaleString()}
                      </p>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                    <div className="rounded border p-2" style={{ borderColor: 'var(--border-subtle)' }}>
                      <div className="micro-label mb-1">Mode</div>
                      <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>{reportData.mode}</div>
                    </div>
                    <div className="rounded border p-2" style={{ borderColor: 'var(--border-subtle)' }}>
                      <div className="micro-label mb-1">Catalog Rows</div>
                      <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                        {String(reportData.metadata?.catalog_row_count ?? 'n/a')}
                      </div>
                    </div>
                    <div className="rounded border p-2" style={{ borderColor: 'var(--border-subtle)' }}>
                      <div className="micro-label mb-1">Providers Synced</div>
                      <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                        {String(reportData.metadata?.catalog_providers_synced_count ?? 'n/a')}
                      </div>
                    </div>
                    <div className="rounded border p-2" style={{ borderColor: 'var(--border-subtle)' }}>
                      <div className="micro-label mb-1">Schema</div>
                      <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                        {String(reportData.metadata?.catalog_schema_version ?? 'n/a')}
                      </div>
                    </div>
                  </div>

                  {/* Chart data availability */}
                  <div>
                    <div className="micro-label mb-1.5">Chart data</div>
                    <div className="grid grid-cols-2 gap-2">
                      <div className="rounded border p-2" style={{ borderColor: 'var(--border-subtle)' }}>
                        <div className="micro-label mb-0.5">Cost by rank</div>
                        <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                          {costByRankCount !== null ? `${costByRankCount} entries` : 'n/a'}
                        </div>
                      </div>
                      {isLLM ? (
                        <div className="rounded border p-2" style={{ borderColor: 'var(--border-subtle)' }}>
                          <div className="micro-label mb-0.5">Risk breakdown</div>
                          <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                            {riskBreakdownCount !== null ? `${riskBreakdownCount} entries` : 'n/a'}
                          </div>
                        </div>
                      ) : (
                        <div className="rounded border p-2" style={{ borderColor: 'var(--border-subtle)' }}>
                          <div className="micro-label mb-0.5">Exclusion breakdown</div>
                          <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                            {exclusionKeys !== null ? `${exclusionKeys} keys` : 'n/a'}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {Array.isArray(reportData.sections) && reportData.sections.length > 0 && (
                    <div className="space-y-2">
                      {reportData.sections.slice(0, 3).map((section) => (
                        <div key={section.title}>
                          <div className="text-xs font-semibold" style={{ color: 'var(--text-primary)' }}>
                            {section.title}
                          </div>
                          <ul className="mt-1 space-y-1">
                            {section.bullets.slice(0, 2).map((bullet) => (
                              <li key={bullet} className="text-[11px]" style={{ color: 'var(--text-tertiary)' }}>
                                • {bullet}
                              </li>
                            ))}
                          </ul>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
