import { InvoiceAnalyzer } from '@/components/invoice/InvoiceAnalyzer'

export function InvoicePage() {
  return (
    <div className="max-w-4xl mx-auto px-4 py-6 sm:px-6">
      <div className="mb-6">
        <h1 className="text-base font-semibold text-zinc-100">Invoice Analyzer</h1>
        <p className="text-xs text-zinc-500 mt-0.5">
          Upload a CSV billing export from any major AI provider. Detect workloads, break down costs by provider, and identify optimization opportunities.
        </p>
        <div className="mt-2 flex flex-wrap gap-2">
          {['OpenAI', 'AWS', 'GCP', 'Azure', 'Anthropic'].map((p) => (
            <span
              key={p}
              className="inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-medium bg-zinc-800 text-zinc-400 ring-1 ring-zinc-700"
            >
              {p}
            </span>
          ))}
        </div>
      </div>
      <InvoiceAnalyzer />
    </div>
  )
}
