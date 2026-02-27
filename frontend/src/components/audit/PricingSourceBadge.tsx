import type { CostAuditPricingSource } from '@/services/types'

// ─── Badge config ─────────────────────────────────────────────────────────────

const SOURCE_CONFIG: Record<
  CostAuditPricingSource,
  { label: string; color: string; bg: string; border: string }
> = {
  provider_csv:    { label: 'Provider CSV',    color: '#4ade80', bg: 'rgba(74,222,128,0.08)',  border: 'rgba(74,222,128,0.25)'  },
  heuristic_prior: { label: 'Heuristic Prior', color: '#fbbf24', bg: 'rgba(251,191,36,0.08)',  border: 'rgba(251,191,36,0.25)'  },
  unknown:         { label: 'Unknown',         color: 'var(--text-disabled)', bg: 'rgba(255,255,255,0.04)', border: 'rgba(255,255,255,0.09)' },
}

// ─── Component ────────────────────────────────────────────────────────────────

interface PricingSourceBadgeProps {
  source?: CostAuditPricingSource
  provider?: string | null
  gpu?: string | null
}

export function PricingSourceBadge({ source, provider, gpu }: PricingSourceBadgeProps) {
  const cfg = SOURCE_CONFIG[source ?? 'unknown']

  const hasDetails = (provider ?? null) !== null || (gpu ?? null) !== null

  return (
    <div className="space-y-1.5">
      {/* Source badge */}
      <div className="flex items-center gap-2 flex-wrap">
        <span
          className="text-[11px]"
          style={{ color: 'var(--text-disabled)' }}
        >
          Pricing source
        </span>
        <span
          className="inline-flex items-center gap-1 text-[11px] font-medium rounded-md px-2 py-0.5 border"
          aria-label={`Pricing source: ${cfg.label}`}
          style={{ color: cfg.color, background: cfg.bg, borderColor: cfg.border }}
        >
          {/* Status dot */}
          <span
            className="w-1.5 h-1.5 rounded-full shrink-0"
            style={{ background: cfg.color }}
            aria-hidden="true"
          />
          {cfg.label}
        </span>
      </div>

      {/* Optional provider / GPU detail lines */}
      {hasDetails && (
        <div className="flex flex-wrap gap-x-4 gap-y-0.5 pl-0.5">
          {provider != null && (
            <span className="text-[11px]" style={{ color: 'var(--text-tertiary)' }}>
              Provider:{' '}
              <span style={{ color: 'var(--text-secondary)' }}>{provider}</span>
            </span>
          )}
          {gpu != null && (
            <span className="text-[11px]" style={{ color: 'var(--text-tertiary)' }}>
              GPU:{' '}
              <span style={{ color: 'var(--text-secondary)' }}>{gpu}</span>
            </span>
          )}
        </div>
      )}
    </div>
  )
}
