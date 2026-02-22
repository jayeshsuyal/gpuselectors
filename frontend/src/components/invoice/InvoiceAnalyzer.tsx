import { useState, useRef } from 'react'
import { Upload, FileText, AlertTriangle, X, Download } from 'lucide-react'
import { cn, formatUSD, workloadDisplayName } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { analyzeInvoice } from '@/services/api'
import type { InvoiceAnalysisResponse } from '@/services/types'

function DropZone({
  onFile,
  file,
  onClear,
}: {
  onFile: (f: File) => void
  file: File | null
  onClear: () => void
}) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)

  function handleDrop(e: React.DragEvent) {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f && f.name.endsWith('.csv')) onFile(f)
  }

  return (
    <div
      className={cn(
        'relative rounded-lg border-2 border-dashed p-8 text-center transition-colors cursor-pointer',
        !dragging && 'border-zinc-700 bg-zinc-900/30 hover:border-zinc-600 hover:bg-zinc-900/50'
      )}
      style={dragging ? { borderColor: 'rgba(124,92,252,0.45)', background: 'rgba(124,92,252,0.06)' } : {}}
      tabIndex={file ? undefined : 0}
      role={!file ? 'button' : undefined}
      aria-label="Upload CSV invoice file"
      onKeyDown={(e) => {
        if ((e.key === 'Enter' || e.key === ' ') && !file) { e.preventDefault(); inputRef.current?.click() }
      }}
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => !file && inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".csv"
        className="hidden"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f) }}
      />

      {file ? (
        <div className="flex items-center justify-center gap-3">
          <FileText className="h-5 w-5" style={{ color: 'var(--brand-hover)' }} />
          <div className="text-left">
            <div className="text-sm font-medium text-zinc-200">{file.name}</div>
            <div className="text-[11px] text-zinc-500">{(file.size / 1024).toFixed(1)} KB</div>
          </div>
          <button
            onClick={(e) => { e.stopPropagation(); onClear() }}
            aria-label="Remove file"
            className="ml-2 p-1 rounded hover:bg-zinc-700 text-zinc-400 hover:text-zinc-200"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      ) : (
        <div className="space-y-2">
          <Upload className="h-8 w-8 text-zinc-500 mx-auto" />
          <div className="text-sm text-zinc-400">Drop a CSV invoice here or click to browse</div>
          <div className="text-xs text-zinc-600">Supports AWS, GCP, Azure, and OpenAI billing exports</div>
        </div>
      )}
    </div>
  )
}

interface InvoiceAnalyzerProps {}

export function InvoiceAnalyzer(_props: InvoiceAnalyzerProps) {
  const [file, setFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<InvoiceAnalysisResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function handleAnalyze() {
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      const res = await analyzeInvoice(file)
      setResult(res)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Analysis failed')
    } finally {
      setLoading(false)
    }
  }

  function handleClear() {
    setFile(null)
    setResult(null)
    setError(null)
  }

  function downloadCSV() {
    if (!result) return
    const headers = ['Provider', 'Workload', 'Line Item', 'Quantity', 'Unit', 'Unit Price', 'Total']
    const rows = result.line_items.map((li) => [
      li.provider, li.workload_type, li.line_item,
      li.quantity, li.unit, li.unit_price, li.total,
    ])
    const csv = [headers, ...rows].map((r) => r.join(',')).join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'invoice_analysis.csv'
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="space-y-6 max-w-3xl">
      {/* Upload */}
      <DropZone file={file} onFile={setFile} onClear={handleClear} />

      {/* Actions */}
      <div className="flex gap-3">
        <Button
          onClick={handleAnalyze}
          disabled={!file || loading}
          className="flex-1 sm:flex-none"
        >
          {loading ? 'Analyzing…' : 'Analyze invoice'}
        </Button>
        {result && (
          <Button variant="outline" onClick={downloadCSV} className="flex items-center gap-2">
            <Download className="h-3.5 w-3.5" />
            Export CSV
          </Button>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="flex gap-2 items-start rounded-md border border-red-800 bg-red-950/30 px-3 py-2.5">
          <AlertTriangle className="h-4 w-4 text-red-400 flex-shrink-0 mt-0.5" />
          <span className="text-xs text-red-300">{error}</span>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-5 animate-fade-in">
          {/* Summary cards */}
          <div className="grid grid-cols-3 gap-3">
            <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
              <div className="text-[11px] text-zinc-500 mb-1">Grand total</div>
              <div className="text-2xl font-bold text-zinc-100 font-numeric">
                {formatUSD(result.grand_total)}
              </div>
            </div>
            <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
              <div className="text-[11px] text-zinc-500 mb-1">Providers</div>
              <div className="text-2xl font-bold text-zinc-100">
                {Object.keys(result.totals_by_provider).length}
              </div>
            </div>
            <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
              <div className="text-[11px] text-zinc-500 mb-1">Workloads</div>
              <div className="text-2xl font-bold text-zinc-100">
                {result.detected_workloads.length}
              </div>
            </div>
          </div>

          {/* By provider */}
          <div>
            <h3 className="text-xs font-semibold text-zinc-400 mb-2 uppercase tracking-wider">
              Cost by provider
            </h3>
            <div className="space-y-2">
              {Object.entries(result.totals_by_provider)
                .sort(([, a], [, b]) => b - a)
                .map(([provider, total]) => {
                  const pct = total / result.grand_total
                  return (
                    <div key={provider} className="flex items-center gap-3">
                      <div className="w-24 text-xs text-zinc-300 font-medium">{provider}</div>
                      <div className="flex-1 bg-zinc-800 rounded-full h-2 overflow-hidden">
                        <div
                          className="h-full rounded-full"
                          style={{ width: `${pct * 100}%`, background: 'var(--brand)' }}
                        />
                      </div>
                      <div className="w-16 text-right text-xs font-bold text-zinc-100 font-numeric">
                        {formatUSD(total)}
                      </div>
                    </div>
                  )
                })}
            </div>
          </div>

          {/* Detected workloads */}
          {result.detected_workloads.length > 0 && (
            <div>
              <h3 className="text-xs font-semibold text-zinc-400 mb-2 uppercase tracking-wider">
                Detected workloads
              </h3>
              <div className="flex flex-wrap gap-1.5">
                {result.detected_workloads.map((w) => (
                  <Badge key={w} variant="indigo">
                    {workloadDisplayName(w)}
                  </Badge>
                ))}
              </div>
            </div>
          )}

          {/* Line items */}
          <div>
            <h3 className="text-xs font-semibold text-zinc-400 mb-2 uppercase tracking-wider">
              Line items
            </h3>
            <div className="rounded-lg border border-zinc-800 overflow-hidden">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-zinc-800 bg-zinc-900/50">
                    <th className="px-4 py-2.5 text-left text-[11px] font-semibold text-zinc-400">Provider</th>
                    <th className="px-4 py-2.5 text-left text-[11px] font-semibold text-zinc-400">Workload</th>
                    <th className="px-4 py-2.5 text-left text-[11px] font-semibold text-zinc-400">Line item</th>
                    <th className="px-4 py-2.5 text-right text-[11px] font-semibold text-zinc-400">Qty</th>
                    <th className="px-4 py-2.5 text-right text-[11px] font-semibold text-zinc-400">Unit $</th>
                    <th className="px-4 py-2.5 text-right text-[11px] font-semibold text-zinc-400">Total</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-800">
                  {result.line_items.map((li, i) => (
                    <tr key={i} className="hover:bg-zinc-900/30">
                      <td className="px-4 py-2.5 font-medium text-zinc-200">{li.provider}</td>
                      <td className="px-4 py-2.5">
                        <Badge variant="default" className="text-[10px]">
                          {workloadDisplayName(li.workload_type)}
                        </Badge>
                      </td>
                      <td className="px-4 py-2.5 text-zinc-400 max-w-[200px] truncate">{li.line_item}</td>
                      <td className="px-4 py-2.5 text-right font-mono text-zinc-300">
                        {li.quantity.toLocaleString()}
                      </td>
                      <td className="px-4 py-2.5 text-right font-mono text-zinc-300">
                        {formatUSD(li.unit_price, 4)}
                      </td>
                      <td className="px-4 py-2.5 text-right font-bold text-zinc-100 font-numeric">
                        {formatUSD(li.total)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Warnings */}
          {result.warnings.map((w, i) => (
            <div key={i} className="flex gap-2 items-start rounded-md border border-amber-800 bg-amber-950/30 px-3 py-2">
              <AlertTriangle className="h-4 w-4 text-amber-400 flex-shrink-0" />
              <span className="text-xs text-amber-300">{w}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
