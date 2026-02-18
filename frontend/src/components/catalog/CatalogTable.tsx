import { useState, useMemo } from 'react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { Search, ArrowUpDown, ArrowUp, ArrowDown } from 'lucide-react'
import { cn, formatUSD, confidenceLabel, billingModeLabel, workloadDisplayName } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from '@/components/ui/select'
import { catalogFilterSchema, type CatalogFilterValues } from '@/schemas/forms'
import type { CatalogRow } from '@/services/types'
import { WORKLOAD_TYPES } from '@/lib/constants'

type SortKey = keyof Pick<CatalogRow, 'provider' | 'unit_price_usd' | 'workload_type' | 'unit_name'>
type SortDir = 'asc' | 'desc'

function SortIcon({ col, active, dir }: { col: string; active: string; dir: SortDir }) {
  if (col !== active) return <ArrowUpDown className="h-3 w-3 text-zinc-600" />
  return dir === 'asc'
    ? <ArrowUp className="h-3 w-3 text-indigo-400" />
    : <ArrowDown className="h-3 w-3 text-indigo-400" />
}

interface CatalogTableProps {
  rows: CatalogRow[]
  loading: boolean
}

export function CatalogTable({ rows, loading }: CatalogTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>('unit_price_usd')
  const [sortDir, setSortDir] = useState<SortDir>('asc')

  const { register, watch, setValue } = useForm<CatalogFilterValues>({
    resolver: zodResolver(catalogFilterSchema),
    defaultValues: { workload_type: '', provider: '', unit_name: '', search: '' },
  })

  const filters = watch()

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

    if (filters.workload_type) {
      out = out.filter((r) => r.workload_type === filters.workload_type)
    }
    if (filters.provider) {
      out = out.filter((r) => r.provider.toLowerCase().includes(filters.provider.toLowerCase()))
    }
    if (filters.unit_name) {
      out = out.filter((r) => r.unit_name === filters.unit_name)
    }
    if (filters.search) {
      const q = filters.search.toLowerCase()
      out = out.filter(
        (r) =>
          r.sku_name.toLowerCase().includes(q) ||
          r.provider.toLowerCase().includes(q) ||
          (r.model_name ?? '').toLowerCase().includes(q)
      )
    }

    out.sort((a, b) => {
      let av: string | number = a[sortKey]
      let bv: string | number = b[sortKey]
      if (av < bv) return sortDir === 'asc' ? -1 : 1
      if (av > bv) return sortDir === 'asc' ? 1 : -1
      return 0
    })

    return out
  }, [rows, filters, sortKey, sortDir])

  const uniqueProviders = useMemo(() => [...new Set(rows.map((r) => r.provider))].sort(), [rows])
  const uniqueUnits = useMemo(() => [...new Set(rows.map((r) => r.unit_name))].sort(), [rows])

  return (
    <div className="space-y-4">
      {/* Filters */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="md:col-span-1">
          <Select
            value={filters.workload_type || '__all__'}
            onValueChange={(v) => setValue('workload_type', v === '__all__' ? '' : v)}
          >
            <SelectTrigger>
              <SelectValue placeholder="All workloads" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__all__">All workloads</SelectItem>
              {WORKLOAD_TYPES.map((w) => (
                <SelectItem key={w.id} value={w.id}>{w.label}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="md:col-span-1">
          <Select
            value={filters.provider || '__all__'}
            onValueChange={(v) => setValue('provider', v === '__all__' ? '' : v)}
          >
            <SelectTrigger>
              <SelectValue placeholder="All providers" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__all__">All providers</SelectItem>
              {uniqueProviders.map((p) => (
                <SelectItem key={p} value={p}>{p}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="md:col-span-1">
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

        <div className="relative md:col-span-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-zinc-500" />
          <Input
            {...register('search')}
            className="pl-8"
            placeholder="Search SKU or model…"
          />
        </div>
      </div>

      {/* Summary */}
      <div className="text-[11px] text-zinc-500">
        {loading ? 'Loading…' : `${filtered.length} of ${rows.length} entries`}
      </div>

      {/* Table */}
      <div className="rounded-lg border border-zinc-800 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-zinc-800 bg-zinc-900/50">
                {[
                  { key: 'provider' as SortKey, label: 'Provider' },
                  { key: 'workload_type' as SortKey, label: 'Workload' },
                  { key: null, label: 'SKU / Model' },
                  { key: 'unit_name' as SortKey, label: 'Unit' },
                  { key: 'unit_price_usd' as SortKey, label: 'Price' },
                  { key: null, label: 'Billing' },
                  { key: null, label: 'Confidence' },
                ].map(({ key, label }, i) => (
                  <th
                    key={i}
                    className={cn(
                      'px-4 py-2.5 text-left text-[11px] font-semibold text-zinc-400',
                      key ? 'cursor-pointer hover:text-zinc-200' : ''
                    )}
                    onClick={key ? () => handleSort(key) : undefined}
                  >
                    <div className="flex items-center gap-1.5">
                      {label}
                      {key && <SortIcon col={key} active={sortKey} dir={sortDir} />}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800">
              {loading ? (
                Array.from({ length: 5 }).map((_, i) => (
                  <tr key={i} className="animate-pulse">
                    {Array.from({ length: 7 }).map((_, j) => (
                      <td key={j} className="px-4 py-3">
                        <div className="h-3 bg-zinc-800 rounded" style={{ width: `${50 + (j * 10) % 40}%` }} />
                      </td>
                    ))}
                  </tr>
                ))
              ) : filtered.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-4 py-10 text-center text-xs text-zinc-500">
                    No entries match current filters
                  </td>
                </tr>
              ) : (
                filtered.map((row, i) => (
                  <tr key={i} className="hover:bg-zinc-900/50 transition-colors">
                    <td className="px-4 py-3 text-xs font-medium text-zinc-200">{row.provider}</td>
                    <td className="px-4 py-3">
                      <Badge variant="default" className="text-[10px]">
                        {workloadDisplayName(row.workload_type)}
                      </Badge>
                    </td>
                    <td className="px-4 py-3 max-w-[200px]">
                      <div className="text-xs text-zinc-300 truncate">{row.sku_name}</div>
                      {row.model_name && (
                        <div className="text-[10px] text-zinc-600 truncate">{row.model_name}</div>
                      )}
                    </td>
                    <td className="px-4 py-3 text-[11px] text-zinc-400 font-mono">{row.unit_name}</td>
                    <td className="px-4 py-3 text-xs font-bold text-zinc-100 font-numeric">
                      {formatUSD(row.unit_price_usd, row.unit_price_usd < 0.01 ? 6 : 4)}
                    </td>
                    <td className="px-4 py-3">
                      <Badge variant="default" className="text-[10px]">
                        {billingModeLabel(row.billing_mode)}
                      </Badge>
                    </td>
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
