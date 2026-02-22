import { FileText } from 'lucide-react'
import { InvoiceAnalyzer } from '@/components/invoice/InvoiceAnalyzer'

const SUPPORTED_PROVIDERS = ['OpenAI', 'AWS', 'GCP', 'Azure', 'Anthropic']

export function InvoicePage() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-8 sm:px-6">
      {/* ── Page header ── */}
      <div className="mb-8 page-section">
        <div className="eyebrow mb-2 flex items-center gap-1.5">
          <FileText className="h-3 w-3" />
          Cost Breakdown
        </div>
        <h1 className="text-2xl font-bold tracking-tight mb-2">
          <span className="text-gradient">Invoice</span>{' '}
          <span style={{ color: 'var(--text-primary)' }}>Analyzer</span>
        </h1>
        <p className="text-sm mb-4" style={{ color: 'var(--text-secondary)', maxWidth: '42rem' }}>
          Upload a CSV billing export from any major AI provider. Detect workloads,
          break down costs by provider, and surface optimization opportunities.
        </p>

        {/* Supported providers as pills */}
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-[11px]" style={{ color: 'var(--text-disabled)' }}>Supports:</span>
          {SUPPORTED_PROVIDERS.map((p) => (
            <span key={p} className="stat-chip">{p}</span>
          ))}
        </div>
      </div>

      {/* ── Content ── */}
      <div className="page-section section-delay-1">
        <InvoiceAnalyzer />
      </div>
    </div>
  )
}
