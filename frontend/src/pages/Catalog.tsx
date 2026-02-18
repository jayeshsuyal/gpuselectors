import { useState, useEffect } from 'react'
import { CatalogTable } from '@/components/catalog/CatalogTable'
import { browseCatalog } from '@/services/api'
import type { CatalogRow } from '@/services/types'

export function CatalogPage() {
  const [rows, setRows] = useState<CatalogRow[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    browseCatalog()
      .then((res) => {
        if (!cancelled) {
          setRows(res.rows)
          setLoading(false)
        }
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : 'Failed to load catalog')
          setLoading(false)
        }
      })
    return () => { cancelled = true }
  }, [])

  return (
    <div className="max-w-6xl mx-auto px-4 py-6 sm:px-6">
      <div className="mb-6">
        <h1 className="text-base font-semibold text-zinc-100">Pricing Catalog</h1>
        <p className="text-xs text-zinc-500 mt-0.5">
          Browse raw pricing data across all providers and workload types. Filter, sort, and export.
        </p>
      </div>

      {error ? (
        <div className="rounded-md border border-red-800 bg-red-950/30 px-4 py-3 text-sm text-red-300">
          {error}
        </div>
      ) : (
        <CatalogTable rows={rows} loading={loading} />
      )}
    </div>
  )
}
