import { useState, useMemo, useEffect } from 'react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { Search, ArrowUpDown, ArrowUp, ArrowDown } from 'lucide-react'
import { cn, formatUSD, confidenceLabel, billingModeLabel, workloadDisplayName } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { ProviderLogo, providerDisplayName } from '@/components/ui/provider-logo'
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from '@/components/ui/select'
import { catalogFilterSchema, type CatalogFilterValues } from '@/schemas/forms'
import type { CatalogRow } from '@/services/types'
import { WORKLOAD_TYPES } from '@/lib/constants'

type SortKey = keyof Pick<CatalogRow, 'provider' | 'unit_price_usd' | 'workload_type' | 'unit_name'>
type SortDir = 'asc' | 'desc'

function SortIcon({ col, active, dir }: { col: string; active: string; dir: SortDir }) {
  if (col !== active) return <ArrowUpDown className="h-3 w-3 opacity-30" />
  return dir === 'asc'
    ? <ArrowUp className="h-3 w-3" style={{ color: 'var(--brand)' }} />
    : <ArrowDown className="h-3 w-3" style={{ color: 'var(--brand)' }} />
}

interface CatalogTableProps {
  rows: CatalogRow[]
  loading: boolean
}

export function CatalogTable({ rows, loading }: CatalogTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>('unit_price_usd')
  const [sortDir, setSortDir] = useState<SortDir>('asc')
  const [browseMode, setBrowseMode] = useState<'workload_first' | 'provider_first'>('workload_first')
  const [scopeNotice, setScopeNotice] = useState('')

  const { register, watch, setValue } = useForm<CatalogFilterValues>({
    resolver: zodResolver(catalogFilterSchema),
    defaultValues: { workload_type: '', provider: '', model_name: '', unit_name: '', search: '' },
  })

  const filters = watch()
  const isWorkloadFirst = browseMode === 'workload_first'
  const hasPrimarySelection = isWorkloadFirst ? Boolean(filters.workload_type) : Boolean(filters.provider)

  function handleSort(key: SortKey) {
    if (key === sortKey) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      setSortDir('asc')
    }
  }

  const filtered = useMemo(() => {
    let out = [...rows]
    if (filters.workload_type) out = out.filter((r) => r.workload_type === filters.workload_type)
    if (filters.provider) out = out.filter((r) => r.provider === filters.provider)
    if (filters.model_name) out = out.filter((r) => (r.model_name ?? '') === filters.model_name)
    if (filters.unit_name) out = out.filter((r) => r.unit_name === filters.unit_name)
    if (filters.search) {
      const q = filters.search.toLowerCase()
      out = out.filter((r) =>
        r.sku_name.toLowerCase().includes(q) ||
        r.provider.toLowerCase().includes(q) ||
        (r.model_name ?? '').toLowerCase().includes(q)
      )
    }
    out.sort((a, b) => {
      const av: string | number | null = a[sortKey] ?? null
      const bv: string | number | null = b[sortKey] ?? null
      if (av === null && bv === null) return 0
      if (av === null) return 1
      if (bv === null) return -1
      if (av < bv) return sortDir === 'asc' ? -1 : 1
      if (av > bv) return sortDir === 'asc' ? 1 : -1
      return 0
    })
    return out
  }, [rows, filters, sortKey, sortDir])

  const allProviders = useMemo(() => [...new Set(rows.map((r) => r.provider))].sort(), [rows])
  const allWorkloads = useMemo(() => [...new Set(rows.map((r) => r.workload_type))].sort(), [rows])

  // Options follow selected browse mode:
  // workload_first: workload is primary filter, provider options depend on workload.
  // provider_first: provider is primary filter, workload options depend on provider.
  const uniqueProviders = useMemo(() => {
    if (browseMode === 'provider_first') return allProviders
    if (!filters.workload_type) return allProviders
    return [...new Set(rows
      .filter((r) => r.workload_type === filters.workload_type)
      .map((r) => r.provider))]
      .sort()
  }, [allProviders, rows, browseMode, filters.workload_type])

  const uniqueWorkloads = useMemo(() => {
    if (browseMode === 'workload_first') return allWorkloads
    if (!filters.provider) return allWorkloads
    return [...new Set(rows
      .filter((r) => r.provider === filters.provider)
      .map((r) => r.workload_type))]
      .sort()
  }, [allWorkloads, rows, browseMode, filters.provider])
  const uniqueModels = useMemo(() => {
    let s = rows
    if (filters.workload_type) s = s.filter((r) => r.workload_type === filters.workload_type)
    if (filters.provider) s = s.filter((r) => r.provider === filters.provider)
    return [...new Set(s.map((r) => r.model_name).filter(Boolean) as string[])].sort()
  }, [rows, filters.workload_type, filters.provider])
  const uniqueUnits = useMemo(() => {
    let s = rows
    if (filters.workload_type) s = s.filter((r) => r.workload_type === filters.workload_type)
    if (filters.provider) s = s.filter((r) => r.provider === filters.provider)
    if (filters.model_name) s = s.filter((r) => (r.model_name ?? '') === filters.model_name)
    return [...new Set(s.map((r) => r.unit_name))].sort()
  }, [rows, filters.workload_type, filters.provider, filters.model_name])

  // Keep dependent filters valid as users switch browse direction or parent filters.
  useEffect(() => {
    if (filters.provider && !uniqueProviders.includes(filters.provider)) {
      setValue('provider', '')
      setScopeNotice('Filters updated to match current browse scope.')
    }
  }, [filters.provider, uniqueProviders, setValue])

  useEffect(() => {
    if (filters.workload_type && !uniqueWorkloads.includes(filters.workload_type)) {
      setValue('workload_type', '')
      setScopeNotice('Filters updated to match current browse scope.')
    }
  }, [filters.workload_type, uniqueWorkloads, setValue])

  useEffect(() => {
    if (filters.model_name && !uniqueModels.includes(filters.model_name)) {
      setValue('model_name', '')
      setScopeNotice('Filters updated to match current browse scope.')
    }
  }, [filters.model_name, uniqueModels, setValue])

  useEffect(() => {
    if (filters.unit_name && !uniqueUnits.includes(filters.unit_name)) {
      setValue('unit_name', '')
      setScopeNotice('Filters updated to match current browse scope.')
    }
  }, [filters.unit_name, uniqueUnits, setValue])

  useEffect(() => {
    if (!scopeNotice) return
    const timer = window.setTimeout(() => setScopeNotice(''), 2200)
    return () => window.clearTimeout(timer)
  }, [scopeNotice])

  // Primary selector drives scope; clear dependent filters when it changes.
  useEffect(() => {
    if (!isWorkloadFirst) return
    setValue('provider', '')
    setValue('model_name', '')
    setValue('unit_name', '')
  }, [filters.workload_type, isWorkloadFirst, setValue])

  useEffect(() => {
    if (isWorkloadFirst) return
    setValue('workload_type', '')
    setValue('model_name', '')
    setValue('unit_name', '')
  }, [filters.provider, isWorkloadFirst, setValue])

  const COLS = [
    { key: 'provider' as SortKey, label: 'Provider' },
    { key: 'workload_type' as SortKey, label: 'Workload' },
    { key: null, label: 'SKU / Model' },
    { key: 'unit_name' as SortKey, label: 'Unit' },
    { key: 'unit_price_usd' as SortKey, label: 'Price' },
    { key: null, label: 'Billing' },
    { key: null, label: 'Confidence' },
  ]

  const primaryLabel = isWorkloadFirst ? 'Primary: Workload' : 'Primary: Provider'
  const dependentLabel = isWorkloadFirst ? 'Dependent: Provider' : 'Dependent: Workload'

  const workloadSelect = (
    <Select
      value={filters.workload_type || '__all__'}
      onValueChange={(v) => setValue('workload_type', v === '__all__' ? '' : v)}
    >
      <SelectTrigger disabled={!isWorkloadFirst && !filters.provider}>
        <SelectValue placeholder={isWorkloadFirst ? 'All workloads' : 'Select provider first'} />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="__all__">All workloads</SelectItem>
        {WORKLOAD_TYPES.filter((w) => uniqueWorkloads.includes(w.id)).map((w) => (
          <SelectItem key={w.id} value={w.id}>{w.label}</SelectItem>
        ))}
      </SelectContent>
    </Select>
  )

  const providerSelect = (
    <Select
      value={filters.provider || '__all__'}
      onValueChange={(v) => setValue('provider', v === '__all__' ? '' : v)}
    >
      <SelectTrigger disabled={isWorkloadFirst && !filters.workload_type}>
        <SelectValue placeholder={isWorkloadFirst ? 'Select workload first' : 'All providers'} />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="__all__">All providers</SelectItem>
        {uniqueProviders.map((p) => (
          <SelectItem key={p} value={p}>
            <div className="flex items-center gap-2">
              <ProviderLogo provider={p} size="sm" />
              <span>{providerDisplayName(p)}</span>
            </div>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )

  return (
    <div className="space-y-4">
      {/* Browse mode toggle */}
      <div className="flex items-center gap-2">
        <span className="text-[11px]" style={{ color: 'var(--text-tertiary)' }}>Browse mode</span>
        {[
          { value: 'workload_first' as const, label: 'Workload → Provider' },
          { value: 'provider_first' as const, label: 'Provider → Workload' },
        ].map((opt) => (
          <button
            key={opt.value}
            type="button"
            onClick={() => {
              setBrowseMode(opt.value)
              // Reset dependent filters on mode switch to avoid stale intersections.
              setValue('provider', '')
              setValue('workload_type', '')
              setValue('model_name', '')
              setValue('unit_name', '')
            }}
            className={cn(
              'rounded-full px-2.5 py-1 text-[11px] border transition-all duration-200',
              browseMode !== opt.value && 'border-white/[0.07] text-zinc-500 hover:text-zinc-300 hover:border-white/[0.12]'
            )}
            style={browseMode === opt.value ? {
              background: 'rgba(124,92,252,0.10)',
              borderColor: 'rgba(124,92,252,0.30)',
              color: 'var(--brand-hover)',
            } : {}}
          >
            {opt.label}
          </button>
        ))}
        <span
          className="rounded-full px-2.5 py-1 text-[10px] uppercase tracking-[0.08em] border"
          style={{
            background: 'rgba(124,92,252,0.12)',
            borderColor: 'rgba(124,92,252,0.35)',
            color: 'var(--brand-hover)',
          }}
        >
          Driver: {isWorkloadFirst ? 'Workload' : 'Provider'}
        </span>
      </div>

      {/* Filters */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-2.5">
        <div className="space-y-1">
          <div className="text-[10px] uppercase tracking-[0.08em]" style={{ color: 'var(--text-tertiary)' }}>
            {primaryLabel}
          </div>
          {isWorkloadFirst ? workloadSelect : providerSelect}
        </div>

        <div className="space-y-1">
          <div className="text-[10px] uppercase tracking-[0.08em]" style={{ color: 'var(--text-tertiary)' }}>
            {dependentLabel}
          </div>
          {isWorkloadFirst ? providerSelect : workloadSelect}
        </div>

        <div className="space-y-1">
          <div className="text-[10px] uppercase tracking-[0.08em]" style={{ color: 'var(--text-tertiary)' }}>
            Model
          </div>
          <Select
            value={filters.model_name || '__all__'}
            onValueChange={(v) => setValue('model_name', v === '__all__' ? '' : v)}
          >
            <SelectTrigger>
              <SelectValue placeholder="All models" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__all__">All models</SelectItem>
              {uniqueModels.map((m) => (
                <SelectItem key={m} value={m}>{m}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-1">
          <div className="text-[10px] uppercase tracking-[0.08em]" style={{ color: 'var(--text-tertiary)' }}>
            Unit
          </div>
          <Select
            value={filters.unit_name || '__all__'}
            onValueChange={(v) => setValue('unit_name', v === '__all__' ? '' : v)}
          >
            <SelectTrigger>
              <SelectValue placeholder="All units" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__all__">All units</SelectItem>
              {uniqueUnits.map((u) => (
                <SelectItem key={u} value={u}>{u}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-1">
          <div className="text-[10px] uppercase tracking-[0.08em]" style={{ color: 'var(--text-tertiary)' }}>
            Search
          </div>
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5" style={{ color: 'var(--text-tertiary)' }} />
            <Input {...register('search')} className="pl-8" placeholder="SKU or model…" />
          </div>
        </div>
      </div>

      {/* Count summary */}
      <div className="text-[11px]" style={{ color: 'var(--text-tertiary)' }}>
        {loading ? (
          <span className="shimmer inline-block w-24 h-3 rounded" />
        ) : (
          <>
            <span style={{ color: 'var(--text-secondary)' }}>{filtered.length.toLocaleString()}</span>
            {' '}of {rows.length.toLocaleString()} entries
          </>
        )}
      </div>
      {!hasPrimarySelection && (
        <div className="text-[11px]" style={{ color: 'var(--text-tertiary)' }}>
          {isWorkloadFirst ? 'Choose a workload first to narrow providers.' : 'Choose a provider first to narrow workloads.'}
        </div>
      )}
      {scopeNotice && (
        <div
          role="status"
          aria-live="polite"
          className="scope-notice text-[11px] rounded-md px-2.5 py-1.5 inline-block"
          style={{ color: 'var(--text-tertiary)', background: 'var(--bg-elevated)', border: '1px solid var(--border-default)' }}
        >
          {scopeNotice}
        </div>
      )}

      {/* Table — sticky header inside scroll container */}
      <div
        className="rounded-lg overflow-hidden"
        style={{ border: '1px solid var(--border-default)' }}
      >
        <div className="overflow-auto max-h-[calc(100vh-340px)]">
          <table className="w-full text-sm border-collapse">
            <thead className="sticky top-0 z-10">
              <tr style={{
                background: 'rgba(14,14,20,0.92)',
                backdropFilter: 'blur(12px)',
                WebkitBackdropFilter: 'blur(12px)',
                borderBottom: '1px solid var(--border-default)',
              }}>
                {COLS.map(({ key, label }, i) => (
                  <th
                    key={i}
                    scope="col"
                    tabIndex={key ? 0 : undefined}
                    role={key ? 'button' : undefined}
                    className={cn(
                      'px-4 py-3 text-left text-[11px] font-semibold whitespace-nowrap select-none',
                      key ? 'cursor-pointer transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[var(--brand)] focus-visible:ring-inset' : ''
                    )}
                    style={{ color: key === sortKey ? 'var(--brand-hover)' : 'var(--text-tertiary)' }}
                    onClick={key ? () => handleSort(key) : undefined}
                    onKeyDown={key ? (e) => { if (e.key === 'Enter' || e.key === ' ') handleSort(key) } : undefined}
                  >
                    <div className="flex items-center gap-1.5">
                      {label}
                      {key && <SortIcon col={key} active={sortKey} dir={sortDir} />}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {loading ? (
                Array.from({ length: 8 }).map((_, i) => (
                  <tr
                    key={i}
                    style={{ borderBottom: '1px solid var(--border-subtle)', animationDelay: `${i * 60}ms` }}
                  >
                    {Array.from({ length: 7 }).map((_, j) => (
                      <td key={j} className="px-4 py-3">
                        <div
                          className="shimmer h-3 rounded"
                          style={{ width: `${40 + (j * 12) % 45}%` }}
                        />
                      </td>
                    ))}
                  </tr>
                ))
              ) : filtered.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-4 py-12 text-center text-sm" style={{ color: 'var(--text-tertiary)' }}>
                    No entries match current filters
                  </td>
                </tr>
              ) : (
                filtered.map((row, i) => (
                  <tr
                    key={i}
                    className="catalog-row group"
                    data-workload-card={row.workload_type}
                    style={{
                      borderBottom: '1px solid var(--border-subtle)',
                      // Left accent line appears on hover via CSS
                    }}
                  >
                    {/* Provider */}
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <ProviderLogo provider={row.provider} size="sm" />
                        <span className="text-xs font-medium" style={{ color: 'var(--text-primary)' }}>
                          {providerDisplayName(row.provider)}
                        </span>
                      </div>
                    </td>
                    {/* Workload */}
                    <td className="px-4 py-3">
                      <Badge variant="default" className="text-[10px]">
                        {workloadDisplayName(row.workload_type)}
                      </Badge>
                    </td>
                    {/* SKU / Model */}
                    <td className="px-4 py-3 max-w-[200px]">
                      <div className="text-xs truncate" style={{ color: 'var(--text-secondary)' }}>
                        {row.sku_name}
                      </div>
                      {row.model_name && (
                        <div className="text-[10px] font-mono truncate mt-0.5" style={{ color: 'var(--text-disabled)' }}>
                          {row.model_name}
                        </div>
                      )}
                    </td>
                    {/* Unit */}
                    <td className="px-4 py-3 text-[11px] font-mono" style={{ color: 'var(--text-tertiary)' }}>
                      {row.unit_name}
                    </td>
                    {/* Price — visually dominant */}
                    <td className="px-4 py-3">
                      <span className="text-sm font-bold font-numeric" style={{ color: 'var(--text-primary)' }}>
                        {formatUSD(row.unit_price_usd, row.unit_price_usd < 0.01 ? 6 : 4)}
                      </span>
                    </td>
                    {/* Billing */}
                    <td className="px-4 py-3">
                      <Badge variant="default" className="text-[10px]">
                        {billingModeLabel(row.billing_mode)}
                      </Badge>
                    </td>
                    {/* Confidence */}
                    <td className="px-4 py-3">
                      <Badge
                        variant={row.confidence as 'high' | 'official' | 'medium' | 'estimated' | 'low' | 'vendor_list'}
                        className="text-[10px]"
                      >
                        {confidenceLabel(row.confidence)}
                      </Badge>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
