import { useState, useEffect, useCallback } from 'react'
import { RefreshCw } from 'lucide-react'
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
        if (e instanceof DOMException && e.name === 'AbortError') {
          return
        }
        setError(e instanceof Error ? e.message : 'Failed to load catalog')
        setLoading(false)
      })
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    fetchCatalog(controller.signal)
    return () => controller.abort()
  }, [fetchCatalog])

  return (
    <div className="max-w-6xl mx-auto px-4 py-6 sm:px-6">
      <div className="mb-6">
        <h1 className="text-base font-semibold text-zinc-100">Pricing Catalog</h1>
        <p className="text-xs text-zinc-500 mt-0.5">
          Browse raw pricing data across all providers and workload types. Filter, sort, and export.
        </p>
      </div>

      {error ? (
        <div className="rounded-md border border-red-800 bg-red-950/30 px-4 py-3 text-sm text-red-300 flex items-center justify-between">
          <span>{error}</span>
          <button
            onClick={() => fetchCatalog()}
            className="flex items-center gap-1.5 text-xs text-red-400 hover:text-red-200 transition-colors ml-4 flex-shrink-0"
          >
            <RefreshCw className="h-3 w-3" />
            Retry
          </button>
        </div>
      ) : (
        <CatalogTable rows={rows} loading={loading} />
      )}
    </div>
  )
}
