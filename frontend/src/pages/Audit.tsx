import { useState } from 'react'
import { ShieldCheck, Loader2 } from 'lucide-react'
import { Link } from 'react-router-dom'
import { auditCost } from '@/services/api'
import { AuditResultCard } from '@/components/audit/AuditResultCard'
import type {
  CostAuditResponse,
  CostAuditModality,
  CostAuditPricingModel,
  CostAuditTrafficPattern,
} from '@/services/types'

// ─── Form state ────────────────────────────────────────────────────────────────

interface AuditFormState {
  modality: CostAuditModality
  model_name: string
  pricing_model: CostAuditPricingModel
  tokens_per_day: string
  monthly_ai_spend_usd: string
  gpu_type: string
  gpu_count: string
  avg_utilization: string
  traffic_pattern: CostAuditTrafficPattern | ''
  has_caching: boolean
  has_quantization: boolean
  has_autoscaling: boolean
}

const DEFAULT_FORM: AuditFormState = {
  modality: 'llm',
  model_name: '',
  pricing_model: 'token_api',
  tokens_per_day: '',
  monthly_ai_spend_usd: '',
  gpu_type: '',
  gpu_count: '',
  avg_utilization: '',
  traffic_pattern: '',
  has_caching: false,
  has_quantization: false,
  has_autoscaling: false,
}

// ─── Option lists ──────────────────────────────────────────────────────────────

const MODALITY_OPTIONS: { value: CostAuditModality; label: string }[] = [
  { value: 'llm',       label: 'LLM' },
  { value: 'asr',       label: 'Speech-to-Text' },
]

const PRICING_MODEL_OPTIONS: { value: CostAuditPricingModel; label: string }[] = [
  { value: 'token_api',    label: 'Token API' },
  { value: 'dedicated_gpu', label: 'Dedicated GPU' },
  { value: 'mixed',        label: 'Mixed' },
]

const TRAFFIC_OPTIONS: { value: CostAuditTrafficPattern; label: string }[] = [
  { value: 'steady',          label: 'Steady' },
  { value: 'business_hours',  label: 'Business Hours' },
  { value: 'bursty',          label: 'Bursty' },
]

// ─── Shared input classes ──────────────────────────────────────────────────────

const INPUT_CLASS =
  'w-full rounded-md border px-3 py-2 text-xs transition-colors bg-transparent ' +
  'focus:outline-none focus:ring-1 focus:ring-[var(--brand)] focus:border-[var(--brand-border)]'

const INPUT_STYLE = { borderColor: 'var(--border-default)', color: 'var(--text-primary)' }

// ─── Page ─────────────────────────────────────────────────────────────────────

export function AuditPage() {
  const [form, setForm] = useState<AuditFormState>(DEFAULT_FORM)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<CostAuditResponse | null>(null)

  function set<K extends keyof AuditFormState>(key: K, value: AuditFormState[K]) {
    setForm((f) => ({ ...f, [key]: value }))
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      const res = await auditCost({
        modality:             form.modality,
        model_name:           form.model_name,
        pricing_model:        form.pricing_model,
        tokens_per_day:       form.tokens_per_day       ? Number(form.tokens_per_day)       : null,
        monthly_ai_spend_usd: form.monthly_ai_spend_usd ? Number(form.monthly_ai_spend_usd) : null,
        gpu_type:             form.gpu_type             || null,
        gpu_count:            form.gpu_count            ? Number(form.gpu_count)            : null,
        avg_utilization:      form.avg_utilization      ? Number(form.avg_utilization) / 100 : null,
        traffic_pattern:      form.traffic_pattern      || null,
        has_caching:          form.has_caching,
        has_quantization:     form.has_quantization,
        has_autoscaling:      form.has_autoscaling,
      })
      setResult(res)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Audit failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const isGpu = form.pricing_model !== 'token_api'
  const compareWorkload = form.modality === 'asr' ? 'speech_to_text' : 'llm'
  const compareTo = `/?from=audit&workload=${encodeURIComponent(compareWorkload)}${
    form.tokens_per_day ? `&tokens_per_day=${encodeURIComponent(form.tokens_per_day)}` : ''
  }${
    form.monthly_ai_spend_usd ? `&monthly_budget=${encodeURIComponent(form.monthly_ai_spend_usd)}` : ''
  }`

  return (
    <div className="max-w-3xl mx-auto px-4 py-8 sm:px-6">
      {/* ── Page header ── */}
      <div className="mb-8 page-section">
        <div className="eyebrow mb-2 flex items-center gap-1.5">
          <ShieldCheck className="h-3 w-3" />
          Cost Intelligence
        </div>
        <h1 className="text-2xl font-bold tracking-tight mb-2">
          <span className="text-gradient">Cost</span>{' '}
          <span style={{ color: 'var(--text-primary)' }}>Audit</span>
        </h1>
        <p className="text-sm" style={{ color: 'var(--text-secondary)', maxWidth: '42rem' }}>
          Describe your current AI deployment. Get an efficiency score, actionable
          recommendations, and a full score breakdown explaining every deduction.
        </p>
      </div>

      {/* ── Form ── */}
      <form onSubmit={handleSubmit} className="page-section section-delay-1 space-y-6">
        <div
          className="rounded-lg border p-5 space-y-5"
          style={{ borderColor: 'var(--border-default)', background: 'var(--bg-elevated)' }}
        >
          {/* Row 1 — modality + model name */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="space-y-1.5">
              <label className="micro-label" htmlFor="audit-modality">Modality</label>
              <select
                id="audit-modality"
                value={form.modality}
                onChange={(e) => set('modality', e.target.value as CostAuditModality)}
                className={INPUT_CLASS}
                style={INPUT_STYLE}
              >
                {MODALITY_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value} style={{ background: 'var(--bg-base)' }}>
                    {o.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="space-y-1.5">
              <label className="micro-label" htmlFor="audit-model-name">Model name</label>
              <input
                id="audit-model-name"
                type="text"
                placeholder="e.g. claude-3-haiku, gpt-4o"
                value={form.model_name}
                onChange={(e) => set('model_name', e.target.value)}
                className={INPUT_CLASS}
                style={INPUT_STYLE}
                required
              />
            </div>
          </div>

          {/* Row 2 — pricing model + traffic */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="space-y-1.5">
              <label className="micro-label" htmlFor="audit-pricing-model">Current pricing model</label>
              <select
                id="audit-pricing-model"
                value={form.pricing_model}
                onChange={(e) => set('pricing_model', e.target.value as CostAuditPricingModel)}
                className={INPUT_CLASS}
                style={INPUT_STYLE}
              >
                {PRICING_MODEL_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value} style={{ background: 'var(--bg-base)' }}>
                    {o.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="space-y-1.5">
              <label className="micro-label" htmlFor="audit-traffic">Traffic pattern</label>
              <select
                id="audit-traffic"
                value={form.traffic_pattern}
                onChange={(e) => set('traffic_pattern', e.target.value as CostAuditTrafficPattern | '')}
                className={INPUT_CLASS}
                style={INPUT_STYLE}
              >
                <option value="" style={{ background: 'var(--bg-base)' }}>— optional —</option>
                {TRAFFIC_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value} style={{ background: 'var(--bg-base)' }}>
                    {o.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Row 3 — tokens + spend */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="space-y-1.5">
              <label className="micro-label" htmlFor="audit-tokens">Tokens / day</label>
              <input
                id="audit-tokens"
                type="number"
                min={0}
                placeholder="e.g. 5000000"
                value={form.tokens_per_day}
                onChange={(e) => set('tokens_per_day', e.target.value)}
                className={INPUT_CLASS}
                style={INPUT_STYLE}
              />
            </div>
            <div className="space-y-1.5">
              <label className="micro-label" htmlFor="audit-spend">Monthly AI spend (USD)</label>
              <input
                id="audit-spend"
                type="number"
                min={0}
                placeholder="e.g. 3000"
                value={form.monthly_ai_spend_usd}
                onChange={(e) => set('monthly_ai_spend_usd', e.target.value)}
                className={INPUT_CLASS}
                style={INPUT_STYLE}
              />
            </div>
          </div>

          {/* Row 4 — GPU fields (visible when GPU model) */}
          {isGpu && (
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div className="space-y-1.5">
                <label className="micro-label" htmlFor="audit-gpu-type">GPU type</label>
                <input
                  id="audit-gpu-type"
                  type="text"
                  placeholder="e.g. A100, H100"
                  value={form.gpu_type}
                  onChange={(e) => set('gpu_type', e.target.value)}
                  className={INPUT_CLASS}
                  style={INPUT_STYLE}
                />
              </div>
              <div className="space-y-1.5">
                <label className="micro-label" htmlFor="audit-gpu-count">GPU count</label>
                <input
                  id="audit-gpu-count"
                  type="number"
                  min={1}
                  placeholder="e.g. 4"
                  value={form.gpu_count}
                  onChange={(e) => set('gpu_count', e.target.value)}
                  className={INPUT_CLASS}
                  style={INPUT_STYLE}
                />
              </div>
              <div className="space-y-1.5">
                <label className="micro-label" htmlFor="audit-util">Avg utilisation (%)</label>
                <input
                  id="audit-util"
                  type="number"
                  min={0}
                  max={100}
                  placeholder="e.g. 70"
                  value={form.avg_utilization}
                  onChange={(e) => set('avg_utilization', e.target.value)}
                  className={INPUT_CLASS}
                  style={INPUT_STYLE}
                />
              </div>
            </div>
          )}

          {/* Row 5 — optimisation toggles */}
          <div className="space-y-1.5">
            <div className="micro-label mb-2">Optimisations already enabled</div>
            <div className="flex flex-wrap gap-2">
              {(
                [
                  { key: 'has_caching',      label: 'Caching' },
                  { key: 'has_quantization', label: 'Quantization' },
                  { key: 'has_autoscaling',  label: 'Autoscaling' },
                ] as const
              ).map(({ key, label }) => (
                <button
                  key={key}
                  type="button"
                  aria-pressed={form[key]}
                  onClick={() => set(key, !form[key])}
                  className="rounded-md border px-3 py-1.5 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[var(--brand)]"
                  style={
                    form[key]
                      ? { borderColor: 'var(--brand-border)', background: 'rgba(124,92,252,0.10)', color: 'var(--brand-hover)' }
                      : { borderColor: 'var(--border-default)', background: 'transparent', color: 'var(--text-tertiary)' }
                  }
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Submit */}
        <button
          type="submit"
          disabled={loading || !form.model_name.trim()}
          className="ai-send-btn w-full flex items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-sm font-semibold transition-all disabled:opacity-50 disabled:cursor-not-allowed focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--brand)]"
          style={{ background: 'var(--brand)', color: '#fff' }}
        >
          {loading ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              Analysing…
            </>
          ) : (
            <>
              <ShieldCheck className="h-4 w-4" />
              Run Cost Audit
            </>
          )}
        </button>

        {error && (
          <p className="text-xs" style={{ color: 'var(--danger-text)' }} role="alert">
            {error}
          </p>
        )}
      </form>

      {/* ── Result ── */}
      {result && (
        <div className="mt-8 page-section section-delay-2 space-y-3">
          <AuditResultCard data={result} />
          <div className="flex justify-end">
            <Link
              to={compareTo}
              className="rounded-md px-3 py-1.5 text-xs font-medium border transition-colors"
              style={{ borderColor: 'var(--brand-border)', background: 'rgba(34,211,238,0.08)', color: 'var(--brand-hover)' }}
            >
              Compare alternatives now
            </Link>
          </div>
        </div>
      )}
    </div>
  )
}
