import { useState } from 'react'
import { ChevronDown, ChevronUp, AlertTriangle, TrendingDown } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { ScoreBreakdownCard } from './ScoreBreakdownCard'
import { PricingSourceBadge } from './PricingSourceBadge'
import type {
  CostAuditResponse,
  CostAuditPriority,
  CostAuditVerdict,
  CostAuditDataGapImpact,
  CostAuditModalityLeg,
} from '@/services/types'

// ─── Helpers ──────────────────────────────────────────────────────────────────

function scoreBadgeVariant(score: number): 'ok' | 'medium' | 'risk_high' {
  if (score >= 70) return 'ok'
  if (score >= 40) return 'medium'
  return 'risk_high'
}

function priorityVariant(p: CostAuditPriority) {
  return p === 'high' ? 'risk_high' : p === 'medium' ? 'medium' : ('unknown' as const)
}

function verdictVariant(v: CostAuditVerdict) {
  if (v === 'appropriate') return 'ok' as const
  if (v === 'suboptimal')  return 'risk_high' as const
  if (v === 'consider_switch') return 'medium' as const
  return 'unknown' as const
}

function gapImpactVariant(i: CostAuditDataGapImpact) {
  if (i === 'high')   return 'risk_high' as const
  if (i === 'medium') return 'medium' as const
  return 'unknown' as const
}

function formatUSD(n: number) {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(n)
}

function formatPct(n: number) {
  return `${n.toFixed(1)}%`
}

function humanizePricingModel(m: string) {
  return { token_api: 'Token API', dedicated_gpu: 'Dedicated GPU', mixed: 'Mixed' }[m] ?? m
}

function humanizeTier(t: string) {
  return { serverless: 'Serverless', single_gpu: 'Single GPU', multi_gpu: 'Multi-GPU', hybrid: 'Hybrid', unknown: 'Unknown' }[t] ?? t
}

// ─── Sub-sections ─────────────────────────────────────────────────────────────

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="eyebrow mb-2">{children}</div>
  )
}

// ─── Per-modality legs table ───────────────────────────────────────────────────

const MODALITY_LABEL: Record<string, string> = {
  llm:              'LLM',
  asr:              'Speech-to-Text',
  tts:              'Text-to-Speech',
  image_gen:        'Image Gen',
  video_gen:        'Video Gen',
  mixed:            'Mixed',
  // Legacy aliases kept for backward-compatible display from older payloads.
  speech_to_text:   'Speech-to-Text',
  text_to_speech:   'Text-to-Speech',
  embeddings:       'Embeddings',
  vision:           'Vision',
  image_generation: 'Image Gen',
  video_generation: 'Video Gen',
  moderation:       'Moderation',
}

function ModalityLegsTable({ legs }: { legs: CostAuditModalityLeg[] }) {
  return (
    <div
      className="rounded-lg border p-4 space-y-3"
      style={{ borderColor: 'var(--border-default)', background: 'var(--bg-elevated)' }}
      aria-label="Per-modality audit breakdown"
    >
      <SectionLabel>Per-Modality Breakdown</SectionLabel>
      <div className="overflow-x-auto -mx-1">
        <table className="w-full text-[11px] border-separate" style={{ borderSpacing: '0 2px' }}>
          <thead>
            <tr>
              {['Modality', 'Score', 'Top Recommendation', 'Flags'].map((h) => (
                <th
                  key={h}
                  className="text-left px-2 pb-1 font-medium"
                  style={{ color: 'var(--text-disabled)' }}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {legs.map((leg) => {
              const scoreVariant = scoreBadgeVariant(leg.efficiency_score)
              const flags = leg.red_flags ?? []
              return (
                <tr
                  key={leg.modality}
                  className="rounded"
                  style={{ background: 'var(--bg-base)' }}
                >
                  <td className="px-2 py-2 rounded-l font-medium" style={{ color: 'var(--text-secondary)' }}>
                    {MODALITY_LABEL[leg.modality] ?? leg.modality}
                  </td>
                  <td className="px-2 py-2">
                    <Badge variant={scoreVariant} className="tabular-nums">
                      {leg.efficiency_score}
                    </Badge>
                  </td>
                  <td className="px-2 py-2" style={{ color: 'var(--text-tertiary)', maxWidth: '18rem' }}>
                    {leg.top_recommendation ?? <span style={{ color: 'var(--text-disabled)' }}>—</span>}
                  </td>
                  <td className="px-2 py-2 rounded-r">
                    {flags.length > 0
                      ? <span style={{ color: '#f87171' }}>{flags.length} flag{flags.length !== 1 ? 's' : ''}</span>
                      : <span style={{ color: 'var(--text-disabled)' }}>—</span>
                    }
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ─── Main ─────────────────────────────────────────────────────────────────────

interface AuditResultCardProps {
  data: CostAuditResponse
}

export function AuditResultCard({ data }: AuditResultCardProps) {
  const [gapsOpen, setGapsOpen] = useState(false)

  const scoreVariant = scoreBadgeVariant(data.efficiency_score)
  const isTokenApiCurrent = data.pricing_model_verdict.current_model === 'token_api'
  const showPricingSource = !isTokenApiCurrent || data.pricing_model_verdict.verdict === 'consider_switch'
  const topScoreDrivers = [
    ...data.red_flags.slice(0, 2),
    ...data.recommendations
      .slice(0, 2)
      .map((rec) => `${rec.title} (est. ${rec.estimated_savings_pct.toFixed(0)}%)`),
  ].slice(0, 3)

  return (
    <div className="space-y-4 animate-enter">

      {/* ── Score + metadata header ── */}
      <div
        className="rounded-lg border p-4 space-y-3"
        style={{ borderColor: 'var(--border-default)', background: 'var(--bg-elevated)' }}
      >
        {/* Top row */}
        <div className="flex items-start justify-between gap-3 flex-wrap">
          <div>
            <div className="eyebrow mb-0.5">Cost Audit Result</div>
            <h3 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
              Efficiency Score
            </h3>
          </div>
          <div className="flex items-center gap-2 flex-wrap justify-end">
            <Badge variant={scoreVariant} className="text-sm font-bold px-3 py-1">
              {data.efficiency_score} / 100
            </Badge>
            <Badge variant={verdictVariant(data.pricing_model_verdict.verdict)}>
              {data.pricing_model_verdict.verdict.replace(/_/g, ' ')}
            </Badge>
          </div>
        </div>

        {/* Metadata grid */}
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
          {/* Current pricing model */}
          <div className="rounded border p-2" style={{ borderColor: 'var(--border-subtle)' }}>
            <div className="micro-label mb-0.5">Current Model</div>
            <div className="text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>
              {humanizePricingModel(data.pricing_model_verdict.current_model)}
            </div>
          </div>
          {/* HW tier */}
          <div className="rounded border p-2" style={{ borderColor: 'var(--border-subtle)' }}>
            <div className="micro-label mb-0.5">HW Tier</div>
            <div className="text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>
              {humanizeTier(data.hardware_recommendation.tier)}
              {data.hardware_recommendation.gpu_family && ` · ${data.hardware_recommendation.gpu_family}`}
            </div>
          </div>
          {/* Savings range */}
          <div className="rounded border p-2" style={{ borderColor: 'var(--border-subtle)' }}>
            <div className="micro-label mb-0.5">Est. Monthly Savings</div>
            <div className="text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>
              {formatUSD(data.estimated_monthly_savings.low_usd)} – {formatUSD(data.estimated_monthly_savings.high_usd)}
            </div>
          </div>
        </div>

        {/* Pricing verdict reason */}
        <p className="text-[11px] leading-relaxed" style={{ color: 'var(--text-tertiary)' }}>
          {data.pricing_model_verdict.reason}
        </p>

        {/* Pricing source */}
        {showPricingSource ? (
          <PricingSourceBadge
            source={data.pricing_source}
            provider={data.pricing_source_provider}
            gpu={data.pricing_source_gpu}
          />
        ) : (
          <p className="text-[11px]" style={{ color: 'var(--text-disabled)' }}>
            Pricing source is abstracted for token-API/serverless audits.
          </p>
        )}
      </div>

      {/* ── Red flags ── */}
      {data.red_flags.length > 0 && (
        <div
          className="rounded-lg border p-3 space-y-1.5"
          style={{ borderColor: 'rgba(248,113,113,0.28)', background: 'rgba(248,113,113,0.06)' }}
          role="alert"
          aria-label="Red flags"
        >
          <div className="flex items-center gap-1.5">
            <AlertTriangle className="h-3.5 w-3.5 shrink-0" style={{ color: '#f87171' }} />
            <span className="text-[11px] font-semibold" style={{ color: '#f87171' }}>
              Red flags ({data.red_flags.length})
            </span>
          </div>
          <ul className="space-y-0.5" role="list">
            {data.red_flags.map((flag, i) => (
              <li key={i} className="text-[11px]" style={{ color: 'var(--text-tertiary)' }}>
                · {flag}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* ── Recommendations ── */}
      {data.recommendations.length > 0 && (
        <div
          className="rounded-lg border p-4 space-y-3"
          style={{ borderColor: 'var(--border-default)', background: 'var(--bg-elevated)' }}
        >
          <SectionLabel>Recommendations</SectionLabel>
          <div className="space-y-3">
            {data.recommendations.map((rec, i) => (
              <div
                key={i}
                className="rounded border p-3 space-y-1.5"
                style={{ borderColor: 'var(--border-subtle)', background: 'var(--bg-base)' }}
              >
                <div className="flex items-start justify-between gap-2 flex-wrap">
                  <div className="flex items-center gap-1.5 flex-wrap">
                    <TrendingDown className="h-3 w-3 shrink-0" style={{ color: 'var(--brand-hover)' }} />
                    <span className="text-xs font-semibold" style={{ color: 'var(--text-primary)' }}>
                      {rec.title}
                    </span>
                  </div>
                  <div className="flex items-center gap-1.5 flex-shrink-0">
                    <Badge variant={priorityVariant(rec.priority)}>
                      {rec.priority}
                    </Badge>
                    <span
                      className="text-[11px] font-semibold tabular-nums"
                      style={{ color: '#4ade80' }}
                    >
                      −{rec.estimated_savings_pct.toFixed(0)} %
                    </span>
                  </div>
                </div>
                <p className="text-[11px] leading-relaxed" style={{ color: 'var(--text-tertiary)' }}>
                  {rec.rationale}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {data.recommended_options && data.recommended_options.length > 0 && (
        <div
          className="rounded-lg border p-4 space-y-3"
          style={{ borderColor: 'var(--border-default)', background: 'var(--bg-elevated)' }}
        >
          <SectionLabel>Recommended Alternatives</SectionLabel>
          <div className="space-y-2">
            {data.recommended_options.map((opt, i) => (
              <div
                key={`${opt.provider}-${opt.gpu_type ?? 'none'}-${opt.deployment_mode}-${i}`}
                className="rounded border p-3 space-y-1.5"
                style={{ borderColor: 'var(--border-subtle)', background: 'var(--bg-base)' }}
              >
                <div className="flex items-start justify-between gap-2 flex-wrap">
                  <div className="flex items-center gap-1.5 flex-wrap">
                    <span className="text-xs font-semibold" style={{ color: 'var(--text-primary)' }}>
                      {opt.provider}
                    </span>
                    <Badge variant="default">{opt.deployment_mode}</Badge>
                    {opt.gpu_type && <Badge variant="violet">{opt.gpu_type}</Badge>}
                  </div>
                  <div className="text-right">
                    <div className="text-xs font-semibold" style={{ color: 'var(--text-primary)' }}>
                      {formatUSD(opt.estimated_monthly_cost_usd)}
                    </div>
                    <div className="text-[10px]" style={{ color: '#4ade80' }}>
                      Save {formatUSD(opt.savings_vs_current_usd)} ({formatPct(opt.savings_vs_current_pct)})
                    </div>
                  </div>
                </div>
                <p className="text-[11px] leading-relaxed" style={{ color: 'var(--text-tertiary)' }}>
                  {opt.rationale}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Hardware recommendation ── */}
      <div
        className="rounded-lg border p-4 space-y-2"
        style={{ borderColor: 'var(--border-default)', background: 'var(--bg-elevated)' }}
      >
        <SectionLabel>Hardware Recommendation</SectionLabel>
        <div className="flex items-center gap-1.5 flex-wrap">
          <Badge variant="default">{humanizeTier(data.hardware_recommendation.tier)}</Badge>
          {data.hardware_recommendation.gpu_family && (
            <Badge variant="violet">{data.hardware_recommendation.gpu_family}</Badge>
          )}
          <Badge variant="default">
            {humanizePricingModel(data.hardware_recommendation.deployment_shape)}
          </Badge>
        </div>
        <p className="text-[11px] leading-relaxed" style={{ color: 'var(--text-tertiary)' }}>
          {data.hardware_recommendation.reasoning}
        </p>
      </div>

      {/* ── Per-modality legs (mixed pipelines) ── */}
      {data.per_modality_audits && data.per_modality_audits.length > 0 && (
        <ModalityLegsTable legs={data.per_modality_audits} />
      )}

      {/* ── Score breakdown (explainability) ── */}
      <ScoreBreakdownCard breakdown={data.score_breakdown} />

      {topScoreDrivers.length > 0 && (
        <div
          className="rounded-lg border p-4 space-y-2"
          style={{ borderColor: 'var(--border-default)', background: 'var(--bg-elevated)' }}
        >
          <SectionLabel>Top Score Drivers</SectionLabel>
          <ul className="space-y-1" role="list">
            {topScoreDrivers.map((driver, idx) => (
              <li key={`${idx}-${driver}`} className="text-[11px]" style={{ color: 'var(--text-tertiary)' }}>
                • {driver}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* ── Data gaps (collapsible) ── */}
      {data.data_gaps_detailed.length > 0 && (
        <div
          className="rounded-lg border p-4 space-y-2"
          style={{ borderColor: 'var(--border-default)', background: 'var(--bg-elevated)' }}
        >
          <button
            type="button"
            aria-expanded={gapsOpen}
            aria-controls="data-gaps-list"
            onClick={() => setGapsOpen((o) => !o)}
            className="flex items-center gap-1 text-[11px] transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[var(--brand)] rounded w-full text-left"
            style={{ color: gapsOpen ? 'var(--text-secondary)' : 'var(--text-disabled)' }}
          >
            {gapsOpen ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
            <span>
              {gapsOpen ? 'Hide' : 'Show'} data gaps ({data.data_gaps_detailed.length})
            </span>
          </button>
          {gapsOpen && (
            <ul id="data-gaps-list" className="space-y-2" role="list">
              {data.data_gaps_detailed.map((gap, i) => (
                <li key={i} className="flex items-start gap-2">
                  <Badge variant={gapImpactVariant(gap.impact)} className="mt-0.5 shrink-0">
                    {gap.impact}
                  </Badge>
                  <div>
                    <div className="text-[11px] font-medium" style={{ color: 'var(--text-secondary)' }}>
                      {gap.field.replace(/_/g, ' ')}
                    </div>
                    <div className="text-[11px]" style={{ color: 'var(--text-tertiary)' }}>
                      {gap.why_it_matters}
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {/* ── Assumptions ── */}
      {data.assumptions.length > 0 && (
        <div className="px-1">
          <p className="text-[10px]" style={{ color: 'var(--text-disabled)' }}>
            Assumptions: {data.assumptions.join(' · ')}
          </p>
        </div>
      )}
    </div>
  )
}
