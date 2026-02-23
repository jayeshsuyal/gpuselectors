import { useState, useEffect, useCallback } from 'react'
import { RefreshCw, Database } from 'lucide-react'
import { CatalogTable } from '@/components/catalog/CatalogTable'
import { browseCatalog } from '@/services/api'
import type { CatalogRow } from '@/services/types'

export function CatalogPage() {
  const [rows, setRows] = useState<CatalogRow[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchCatalog = useCallback((signal?: AbortSignal) => {
    setLoading(true)
    setError(null)
    browseCatalog({ signal })
      .then((res) => { setRows(res.rows); setLoading(false) })
      .catch((e: unknown) => {
        if (e instanceof DOMException && e.name === 'AbortError') return
        setError(e instanceof Error ? e.message : 'Failed to load catalog')
        setLoading(false)
      })
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    fetchCatalog(controller.signal)
    return () => controller.abort()
  }, [fetchCatalog])

  const providerCount = new Set(rows.map((r) => r.provider)).size
  const workloadCount = new Set(rows.map((r) => r.workload_type)).size

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 sm:px-6">
      {/* ── Page header ── */}
      <div className="mb-8 page-section">
        <div className="eyebrow mb-2">Pricing Intelligence</div>
        <h1 className="text-2xl font-bold tracking-tight mb-2">
          <span className="text-gradient">Pricing</span>{' '}
          <span style={{ color: 'var(--text-primary)' }}>Catalog</span>
        </h1>
        <p className="text-sm mb-4" style={{ color: 'var(--text-secondary)' }}>
          Browse raw inference pricing across all providers and workload types.
          Filter, sort, and compare.
        </p>

        {/* Stats strip */}
        {!loading && rows.length > 0 && (
          <div className="flex flex-wrap items-center gap-2 animate-enter section-delay-1">
            <span className="stat-chip">
              <Database className="h-3 w-3 opacity-60" />
              {rows.length.toLocaleString()} entries
            </span>
            <span className="stat-chip">{providerCount} providers</span>
            <span className="stat-chip">{workloadCount} workload types</span>
          </div>
        )}
      </div>

      {/* ── Content ── */}
      {error ? (
        <div
          className="rounded-lg border px-4 py-3 text-sm flex items-center justify-between animate-enter"
          style={{ borderColor: 'var(--danger-border)', background: 'var(--danger-bg)', color: 'var(--danger-text)' }}
        >
          <span>{error}</span>
          <button
            onClick={() => fetchCatalog()}
            className="flex items-center gap-1.5 text-xs ml-4 flex-shrink-0 transition-colors hover:text-white"
            style={{ color: 'var(--danger-text)' }}
          >
            <RefreshCw className="h-3 w-3" />
            Retry
          </button>
        </div>
      ) : (
        <div className="page-section section-delay-1">
          <CatalogTable rows={rows} loading={loading} />
        </div>
      )}
    </div>
  )
}
