import { lazy, Suspense, useState } from 'react'
import { ChevronDown, ChevronUp, Info } from 'lucide-react'
import { cn, formatUSD, formatPercent, riskLevel, confidenceLabel, billingModeLabel, capacityLabel } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'
import { ProviderLogo, providerDisplayName } from '@/components/ui/provider-logo'
import type { RankedPlan, RankedCatalogOffer, ProviderDiagnostic } from '@/services/types'
import { ASSUMPTION_LABELS } from '@/lib/constants'

const InsightsCharts = lazy(() =>
  import('@/components/optimize/InsightsCharts').then((module) => ({
    default: module.InsightsCharts,
  }))
)

// ─── Semantic badges ──────────────────────────────────────────────────────────

function ConfidenceBadge({ confidence }: { confidence: string }) {
  const variant = confidence as 'high' | 'official' | 'medium' | 'estimated' | 'low' | 'vendor_list'
  return <Badge variant={variant}>{confidenceLabel(confidence)}</Badge>
}

function RiskBadge({ totalRisk }: { totalRisk: number }) {
  const level = riskLevel(totalRisk)
  const variant = `risk_${level}` as 'risk_low' | 'risk_medium' | 'risk_high'
  const labels = { low: 'Low risk', medium: 'Med risk', high: 'High risk' }
  return <Badge variant={variant}>{labels[level]}</Badge>
}

function CapacityBadge({ check }: { check: string }) {
  const variant = check as 'ok' | 'insufficient' | 'unknown'
  return <Badge variant={variant}>{capacityLabel(check)}</Badge>
}

// ─── LLM result card ──────────────────────────────────────────────────────────

function LLMResultCard({ plan, isFirst, index }: { plan: RankedPlan; isFirst: boolean; index: number }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div
      style={{
        animationDelay: `${index * 50}ms`,
        ...(isFirst
          ? {
              borderColor: 'rgba(124,92,252,0.28)',
              background: 'rgba(124,92,252,0.04)',
            }
          : {
              borderColor: 'rgba(255,255,255,0.06)',
              background: 'var(--bg-elevated)',
            }),
      }}
      className={cn(
        'relative rounded-lg border overflow-hidden transition-all duration-200 animate-enter',
        'hover:-translate-y-px',
        !isFirst && 'hover:border-white/[0.11]'
      )}
    >
      {/* Brand gradient top accent for #1 */}
      {isFirst && (
        <div className="h-[1.5px]" style={{ background: 'var(--brand-gradient)' }} />
      )}

      <div className="p-5">
        {/* Header: rank label + provider + cost */}
        <div className="flex items-start justify-between gap-4 mb-4">
          <div className="flex-1 min-w-0">
            {/* Rank + best-value tag */}
            <div className="flex items-center gap-2 mb-2">
              <span
                className="text-[10px] font-mono font-bold tracking-[0.12em]"
                style={{ color: isFirst ? 'var(--brand)' : 'var(--text-disabled)' }}
              >
                #{String(plan.rank).padStart(2, '0')}
              </span>
              {isFirst && (
                <span
                  className="text-[9px] font-bold uppercase tracking-widest px-1.5 py-0.5 rounded"
                  style={{ background: 'rgba(124,92,252,0.15)', color: 'var(--brand-hover)' }}
                >
                  Best value
                </span>
              )}
            </div>
            {/* Provider */}
            <div className="flex items-center gap-2">
              <ProviderLogo provider={plan.provider_id} size="sm" />
              <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
                {providerDisplayName(plan.provider_id) || plan.provider_name}
              </span>
            </div>
            <div className="text-[10px] font-mono mt-1 truncate" style={{ color: 'var(--text-disabled)' }}>
              {plan.offering_id}
            </div>
          </div>

          {/* Cost — hero number */}
          <div className="text-right flex-shrink-0">
            <div className="micro-label mb-1">est. monthly</div>
            <div
              className="font-bold font-numeric leading-none"
              style={{
                fontSize: isFirst ? '1.875rem' : '1.375rem',
                letterSpacing: '-0.035em',
                color: isFirst ? 'var(--brand-hover)' : 'var(--text-primary)',
              }}
            >
              {formatUSD(plan.monthly_cost_usd, 0)}
            </div>
          </div>
        </div>

        {/* Badges */}
        <div className="flex flex-wrap gap-1.5">
          <Badge variant="default">{billingModeLabel(plan.billing_mode)}</Badge>
          <ConfidenceBadge confidence={plan.confidence} />
          <RiskBadge totalRisk={plan.risk.total_risk} />
          {plan.utilization_at_peak !== null && (
            <Badge variant="default">{formatPercent(plan.utilization_at_peak)} util</Badge>
          )}
        </div>

        {/* Why text */}
        <p className="text-xs leading-relaxed mt-3" style={{ color: 'var(--text-secondary)' }}>
          {plan.why}
        </p>

        {/* Expand trigger */}
        <button
          onClick={() => setExpanded(!expanded)}
          aria-expanded={expanded}
          aria-controls={`assumptions-${plan.offering_id}`}
          className="flex items-center gap-1.5 mt-3 text-[11px] transition-colors hover:opacity-100"
          style={{ color: 'var(--text-disabled)' }}
        >
          <Info className="h-3 w-3" />
          Assumptions
          {expanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
        </button>
      </div>

      {/* Assumptions panel */}
      {expanded && (
        <div
          id={`assumptions-${plan.offering_id}`}
          className="px-5 pb-4 pt-3 animate-enter-fast"
          style={{ borderTop: '1px solid var(--border-subtle)' }}
        >
          <div className="grid grid-cols-3 gap-x-4 gap-y-3">
            {Object.entries(plan.assumptions).map(([key, val]) => (
              <div key={key}>
                <div className="micro-label mb-0.5">{ASSUMPTION_LABELS[key] ?? key}</div>
                <div className="text-xs font-mono" style={{ color: 'var(--text-secondary)' }}>{val}</div>
              </div>
            ))}
          </div>
          <div
            className="mt-3 pt-3 grid grid-cols-2 gap-x-4"
            style={{ borderTop: '1px solid var(--border-subtle)' }}
          >
            <div>
              <div className="micro-label mb-0.5">Overload risk</div>
              <div className="text-xs font-mono" style={{ color: 'var(--text-secondary)' }}>
                {formatPercent(plan.risk.risk_overload)}
              </div>
            </div>
            <div>
              <div className="micro-label mb-0.5">Complexity risk</div>
              <div className="text-xs font-mono" style={{ color: 'var(--text-secondary)' }}>
                {formatPercent(plan.risk.risk_complexity)}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// ─── Non-LLM result row ───────────────────────────────────────────────────────

function NonLLMResultRow({ offer, isFirst, index }: { offer: RankedCatalogOffer; isFirst: boolean; index: number }) {
  return (
    <div
      style={{
        animationDelay: `${index * 40}ms`,
        ...(isFirst
          ? {
              borderColor: 'rgba(124,92,252,0.28)',
              background: 'rgba(124,92,252,0.04)',
            }
          : {
              borderColor: 'rgba(255,255,255,0.06)',
              background: 'var(--bg-elevated)',
            }),
      }}
      className={cn(
        'relative flex items-center gap-4 px-4 py-3.5 rounded-lg border overflow-hidden',
        'transition-all duration-200 animate-enter hover:-translate-y-px',
        !isFirst && 'hover:border-white/[0.11]'
      )}
    >
      {/* Left accent strip for #1 */}
      {isFirst && (
        <div
          className="absolute left-0 top-0 bottom-0 w-[2px]"
          style={{ background: 'var(--brand-gradient)' }}
        />
      )}

      {/* Rank */}
      <span
        className="text-[10px] font-mono font-bold tracking-[0.12em] flex-shrink-0 w-7"
        style={{ color: isFirst ? 'var(--brand)' : 'var(--text-disabled)' }}
      >
        #{String(offer.rank).padStart(2, '0')}
      </span>

      {/* Provider + SKU */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <ProviderLogo provider={offer.provider} size="sm" />
          <span className="text-xs font-semibold" style={{ color: 'var(--text-primary)' }}>
            {providerDisplayName(offer.provider)}
          </span>
        </div>
        <div className="text-[10px] font-mono truncate mt-0.5" style={{ color: 'var(--text-disabled)' }}>
          {offer.sku_name}
        </div>
      </div>

      {/* Unit price */}
      <div className="text-right hidden sm:block">
        <div className="micro-label mb-0.5">unit price</div>
        <div className="text-xs font-mono" style={{ color: 'var(--text-secondary)' }}>
          {formatUSD(offer.unit_price_usd, 4)}/{offer.unit_name}
        </div>
      </div>

      {/* Monthly estimate — hero number */}
      <div className="text-right flex-shrink-0">
        <div className="micro-label mb-0.5">monthly est.</div>
        <div
          className="font-bold font-numeric leading-none"
          style={{
            fontSize: isFirst ? '1.375rem' : '1rem',
            letterSpacing: '-0.025em',
            color: isFirst ? 'var(--brand-hover)' : 'var(--text-primary)',
          }}
        >
          {offer.monthly_estimate_usd !== null ? formatUSD(offer.monthly_estimate_usd) : '—'}
        </div>
      </div>

      {/* Badges */}
      <div className="flex flex-col gap-1 items-end">
        <ConfidenceBadge confidence={offer.confidence} />
        <CapacityBadge check={offer.capacity_check} />
      </div>

      {offer.required_replicas !== null && (
        <div className="text-[10px] font-mono hidden md:block flex-shrink-0" style={{ color: 'var(--text-disabled)' }}>
          {offer.required_replicas}× rep
        </div>
      )}
    </div>
  )
}

// ─── Provider diagnostics ─────────────────────────────────────────────────────

function DiagnosticsPanel({ diagnostics }: { diagnostics: ProviderDiagnostic[] }) {
  const [open, setOpen] = useState(false)
  const excluded = diagnostics.filter((d) => d.status !== 'included')

  if (excluded.length === 0) return null

  return (
    <div className="border border-white/[0.06] rounded-lg overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        aria-controls="diagnostics-list"
        className="w-full flex items-center justify-between px-4 py-2.5 text-xs transition-colors hover:bg-white/[0.02]"
        style={{ color: 'var(--text-tertiary)' }}
      >
        <span>
          {excluded.length} provider{excluded.length !== 1 ? 's' : ''} excluded or not selected
        </span>
        {open ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
      </button>
      {open && (
        <div
          id="diagnostics-list"
          className="divide-y animate-enter-fast"
          style={{ borderTop: '1px solid var(--border-subtle)', borderColor: 'var(--border-subtle)' }}
        >
          {diagnostics.map((d) => (
            <div key={d.provider} className="flex items-center gap-3 px-4 py-2">
              <Badge
                variant={
                  d.status === 'included' ? 'emerald' :
                  d.status === 'not_selected' ? 'default' : 'amber'
                }
              >
                {d.status}
              </Badge>
              <div className="w-36 flex items-center gap-2">
                <ProviderLogo provider={d.provider} size="sm" />
                <span className="text-xs font-medium truncate" style={{ color: 'var(--text-secondary)' }}>
                  {providerDisplayName(d.provider)}
                </span>
              </div>
              <span className="text-[11px] flex-1" style={{ color: 'var(--text-tertiary)' }}>{d.reason}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ─── Main export ──────────────────────────────────────────────────────────────

interface ResultsTableProps {
  mode: 'llm' | 'non-llm'
  plans?: RankedPlan[]
  offers?: RankedCatalogOffer[]
  diagnostics?: ProviderDiagnostic[]
  warnings?: string[]
  excludedCount?: number
}

export function ResultsTable({
  mode,
  plans = [],
  offers = [],
  diagnostics = [],
  warnings = [],
  excludedCount = 0,
}: ResultsTableProps) {
  if (mode === 'llm' && plans.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-center">
        <div className="text-sm mb-1" style={{ color: 'var(--text-tertiary)' }}>
          No feasible configurations found
        </div>
        <div className="text-xs leading-relaxed max-w-xs" style={{ color: 'var(--text-disabled)' }}>
          Try adding more providers, selecting a smaller model size, or reducing your daily token volume.
        </div>
      </div>
    )
  }

  if (mode === 'non-llm' && offers.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-center">
        <div className="text-sm mb-1" style={{ color: 'var(--text-tertiary)' }}>
          No offers found for this workload
        </div>
        <div className="text-xs leading-relaxed max-w-xs" style={{ color: 'var(--text-disabled)' }}>
          Try removing provider filters, switching to &ldquo;All units&rdquo; (browse mode), or setting budget to 0.
        </div>
      </div>
    )
  }

  const count = mode === 'llm' ? plans.length : offers.length

  return (
    <div className="space-y-3">
      {/* Warnings */}
      {warnings.map((w, i) => (
        <div
          key={i}
          className="flex gap-2 items-start rounded-lg border px-3 py-2"
          style={{ borderColor: 'var(--warning-border)', background: 'var(--warning-bg)' }}
        >
          <span className="text-xs" style={{ color: 'var(--warning)' }}>⚠</span>
          <span className="text-xs" style={{ color: 'var(--warning-text)' }}>{w}</span>
        </div>
      ))}

      {/* Summary line */}
      <div className="flex items-center gap-2 text-[11px]" style={{ color: 'var(--text-disabled)' }}>
        <span style={{ color: 'var(--text-tertiary)' }}>{count}</span>
        {' '}result{count !== 1 ? 's' : ''}
        {excludedCount > 0 && <span>· {excludedCount} excluded</span>}
      </div>

      {/* Insights charts */}
      <Suspense
        fallback={
          <div className="rounded-lg border p-4 text-xs" style={{ borderColor: 'var(--border-default)', color: 'var(--text-disabled)' }}>
            Loading insights…
          </div>
        }
      >
        <InsightsCharts mode={mode} plans={plans} offers={offers} />
      </Suspense>

      {/* Result cards */}
      {mode === 'llm'
        ? plans.map((plan, i) => (
            <LLMResultCard key={plan.offering_id} plan={plan} isFirst={i === 0} index={i} />
          ))
        : offers.map((offer, i) => (
            <NonLLMResultRow key={offer.sku_name} offer={offer} isFirst={i === 0} index={i} />
          ))}

      {/* Provider diagnostics */}
      {diagnostics.length > 0 && <DiagnosticsPanel diagnostics={diagnostics} />}
    </div>
  )
}
