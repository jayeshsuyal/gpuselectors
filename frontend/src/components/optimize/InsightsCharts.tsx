import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { RankedCatalogOffer, RankedPlan } from '@/services/types'
import { formatUSD } from '@/lib/utils'

interface InsightsChartsProps {
  mode: 'llm' | 'non-llm'
  plans: RankedPlan[]
  offers: RankedCatalogOffer[]
}

const PROVIDER_COLORS = ['#6366f1', '#14b8a6', '#f59e0b', '#ef4444', '#a855f7', '#22c55e']

function providerColor(name: string): string {
  let hash = 0
  for (let i = 0; i < name.length; i += 1) {
    hash = (hash * 31 + name.charCodeAt(i)) | 0
  }
  return PROVIDER_COLORS[Math.abs(hash) % PROVIDER_COLORS.length]
}

function ProviderCostChart({ mode, plans, offers }: InsightsChartsProps) {
  const source = mode === 'llm'
    ? plans.map((plan) => ({
        provider: plan.provider_name,
        monthly_cost: Number(plan.monthly_cost_usd.toFixed(2)),
      }))
    : offers
        .filter((offer) => offer.monthly_estimate_usd !== null)
        .map((offer) => ({
          provider: offer.provider,
          monthly_cost: Number((offer.monthly_estimate_usd ?? 0).toFixed(2)),
        }))

  const byProvider = new Map<string, number>()
  for (const row of source) {
    const current = byProvider.get(row.provider)
    if (current === undefined || row.monthly_cost < current) {
      byProvider.set(row.provider, row.monthly_cost)
    }
  }
  const data = Array.from(byProvider.entries())
    .map(([provider, monthly_cost]) => ({ provider, monthly_cost }))
    .sort((a, b) => a.monthly_cost - b.monthly_cost)
    .slice(0, 8)

  if (data.length === 0) {
    return null
  }

  return (
    <div className="rounded-lg border p-4" style={{ borderColor: 'var(--border-default)', background: 'var(--bg-elevated)' }}>
      <h3 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>Provider Cost Comparison</h3>
      <p className="text-[11px] mt-0.5" style={{ color: 'var(--text-tertiary)' }}>
        Cheapest monthly estimate per provider (top 8).
      </p>
      <div className="h-64 mt-3">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
            <XAxis dataKey="provider" tick={{ fill: 'var(--text-tertiary)', fontSize: 11 }} />
            <YAxis tick={{ fill: 'var(--text-tertiary)', fontSize: 11 }} />
            <Tooltip
              formatter={(value: number) => [formatUSD(value), 'Monthly cost']}
              contentStyle={{
                background: 'var(--bg-surface)',
                border: '1px solid var(--border-default)',
                borderRadius: '8px',
                color: 'var(--text-primary)',
              }}
              labelStyle={{ color: 'var(--text-secondary)' }}
            />
            <Bar dataKey="monthly_cost" radius={[4, 4, 0, 0]}>
              {data.map((row, i) => (
                <Cell key={i} fill={providerColor(row.provider)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function ChartTakeaway({ mode, plans, offers }: InsightsChartsProps) {
  if (mode === 'llm') {
    if (plans.length === 0) return null
    const bestValue = plans[0]
    const lowestRisk = [...plans].sort((a, b) => a.risk.total_risk - b.risk.total_risk)[0]
    return (
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className="text-[11px]" style={{ color: 'var(--text-tertiary)' }}>
          <span style={{ color: 'var(--brand-hover)' }}>Best value</span>
          {' '}{bestValue.provider_name} at {formatUSD(bestValue.monthly_cost_usd, 0)}/mo
        </span>
        <span className="text-[11px]" style={{ color: 'var(--border-default)' }}>·</span>
        <span className="text-[11px]" style={{ color: 'var(--text-tertiary)' }}>
          <span style={{ color: 'var(--brand-hover)' }}>Lowest risk</span>
          {' '}{lowestRisk.provider_name} ({lowestRisk.risk.total_risk.toFixed(2)})
        </span>
      </div>
    )
  }
  // non-LLM
  if (offers.length === 0) return null
  const bestValue = offers[0]
  const bestCostStr = bestValue.monthly_estimate_usd !== null
    ? `${formatUSD(bestValue.monthly_estimate_usd, 0)}/mo`
    : `${formatUSD(bestValue.unit_price_usd, 4)}/${bestValue.unit_name}`
  const withDelta = offers.filter((o) => typeof o.price_change_pct === 'number')
  const largestDrop = withDelta.length > 0
    ? [...withDelta].sort((a, b) => (a.price_change_pct ?? 0) - (b.price_change_pct ?? 0))[0]
    : null
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
      <span className="text-[11px]" style={{ color: 'var(--text-tertiary)' }}>
        <span style={{ color: 'var(--brand-hover)' }}>Best value</span>
        {' '}{bestValue.provider} at {bestCostStr}
      </span>
      {largestDrop && (
        <>
          <span className="text-[11px]" style={{ color: 'var(--border-default)' }}>·</span>
          <span className="text-[11px]" style={{ color: 'var(--text-tertiary)' }}>
            <span style={{ color: 'var(--brand-hover)' }}>Largest drop</span>{' '}
            {largestDrop.provider} ({(largestDrop.price_change_pct ?? 0).toFixed(1)}%)
          </span>
        </>
      )}
    </div>
  )
}

export function InsightsCharts(props: InsightsChartsProps) {
  return (
    <div className="space-y-3">
      <ChartTakeaway {...props} />
      <ProviderCostChart {...props} />
    </div>
  )
}
