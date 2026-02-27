import { useState } from 'react'
import { ChevronDown, ChevronUp } from 'lucide-react'
import type { CostAuditScoreBreakdown } from '@/services/types'

// ─── Cap chip label map ────────────────────────────────────────────────────────

const CAP_LABELS: Record<string, string> = {
  major_flags_cap:          'Major Flags Cap',
  high_switch_savings_cap:  'High Switch Savings Cap',
}

function capLabel(key: string): string {
  return CAP_LABELS[key] ?? key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

// ─── Row ──────────────────────────────────────────────────────────────────────

function Row({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className="flex items-center justify-between gap-4 py-1.5 border-b last:border-0"
      style={{ borderColor: 'var(--border-subtle)' }}>
      <span className="text-[11px]" style={{ color: 'var(--text-tertiary)' }}>{label}</span>
      <span
        className="text-[11px] font-medium tabular-nums"
        style={{ color: highlight ? 'var(--brand-hover)' : 'var(--text-secondary)' }}
      >
        {value}
      </span>
    </div>
  )
}

// ─── Main ─────────────────────────────────────────────────────────────────────

interface ScoreBreakdownCardProps {
  breakdown: CostAuditScoreBreakdown | undefined
}

export function ScoreBreakdownCard({ breakdown }: ScoreBreakdownCardProps) {
  const [open, setOpen] = useState(false)

  return (
    <div
      className="rounded-lg border p-4 space-y-3"
      style={{ borderColor: 'var(--border-default)', background: 'var(--bg-elevated)' }}
    >
      {/* Header */}
      <div className="flex items-center justify-between gap-2">
        <div>
          <div className="eyebrow mb-0.5">Explainability</div>
          <h3 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
            Why this score
          </h3>
        </div>
        {breakdown && (
          <button
            type="button"
            aria-expanded={open}
            aria-controls="score-breakdown-body"
            onClick={() => setOpen((o) => !o)}
            className="flex items-center gap-1 text-[11px] transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[var(--brand)] rounded"
            style={{ color: open ? 'var(--text-secondary)' : 'var(--text-disabled)' }}
          >
            {open ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
            {open ? 'Hide' : 'Show'} details
          </button>
        )}
      </div>

      {/* Body */}
      {!breakdown ? (
        <p className="text-[11px]" style={{ color: 'var(--text-disabled)' }}>
          Score breakdown unavailable for this run.
        </p>
      ) : (
        <>
          {/* Always-visible summary row */}
          <div className="flex items-center gap-3 flex-wrap">
            <span
              className="text-[11px] rounded-md px-2 py-0.5 border"
              style={{
                color: 'var(--text-secondary)',
                borderColor: 'var(--border-subtle)',
                background: 'var(--bg-base)',
              }}
            >
              −{breakdown.penalty_points} penalties
            </span>
            {breakdown.bonus_points > 0 && (
              <span
                className="text-[11px] rounded-md px-2 py-0.5 border"
                style={{
                  color: '#4ade80',
                  borderColor: 'rgba(74,222,128,0.25)',
                  background: 'rgba(74,222,128,0.07)',
                }}
              >
                +{breakdown.bonus_points} bonuses
              </span>
            )}
            {breakdown.major_flags > 0 && (
              <span
                className="text-[11px] rounded-md px-2 py-0.5 border"
                style={{
                  color: '#f87171',
                  borderColor: 'rgba(248,113,113,0.25)',
                  background: 'rgba(248,113,113,0.07)',
                }}
              >
                {breakdown.major_flags} major flag{breakdown.major_flags !== 1 ? 's' : ''}
              </span>
            )}
          </div>

          {/* Cap chips */}
          {breakdown.caps_applied.length > 0 && (
            <div className="flex flex-wrap gap-1.5" role="list" aria-label="Score caps applied">
              {breakdown.caps_applied.map((cap) => (
                <span
                  key={cap}
                  role="listitem"
                  aria-label={`Cap applied: ${capLabel(cap)}`}
                  className="inline-flex items-center text-[10px] font-medium rounded px-1.5 py-0.5 border"
                  style={{
                    color: '#fbbf24',
                    borderColor: 'rgba(251,191,36,0.28)',
                    background: 'rgba(251,191,36,0.08)',
                  }}
                >
                  {capLabel(cap)}
                </span>
              ))}
            </div>
          )}

          {/* Collapsible detail rows */}
          {open && (
            <div id="score-breakdown-body" className="space-y-0 mt-1">
              <Row label="Base score"         value={String(breakdown.base_score)} />
              <Row label="Penalty points"     value={`−${breakdown.penalty_points}`} />
              <Row label="Bonus points"       value={`+${breakdown.bonus_points}`} />
              <Row label="Pre-cap score"      value={String(breakdown.pre_cap_score)} />
              <Row label="Post-cap score"     value={String(breakdown.post_cap_score)} highlight />
              <Row label="Major flags"        value={String(breakdown.major_flags)} />
              <Row
                label="Combined savings"
                value={`${breakdown.combined_savings_pct.toFixed(1)} %`}
              />
            </div>
          )}
        </>
      )}
    </div>
  )
}
