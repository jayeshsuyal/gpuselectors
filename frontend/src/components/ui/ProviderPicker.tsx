/**
 * Shared multi-select provider picker used by both LLMForm and NonLLMForm.
 * Renders a pill grid; clicking toggles selection.
 */
import { cn } from '@/lib/utils'
import { PROVIDERS } from '@/lib/constants'
import { ProviderLogo, providerDisplayName } from '@/components/ui/provider-logo'

interface ProviderPickerProps {
  value: string[]
  onChange: (next: string[]) => void
  /** Optionally restrict which providers appear */
  allowedProviders?: readonly string[]
  /** Helper note rendered below the pills */
  helperText?: string
}

export function ProviderPicker({
  value,
  onChange,
  allowedProviders = PROVIDERS,
  helperText,
}: ProviderPickerProps) {
  function toggle(id: string) {
    if (value.includes(id)) {
      onChange(value.filter((p) => p !== id))
    } else {
      onChange([...value, id])
    }
  }

  return (
    <div className="space-y-1.5">
      <div className="flex flex-wrap gap-1.5">
        {allowedProviders.map((p) => {
          const isSelected = value.includes(p)
          return (
            <button
              key={p}
              type="button"
              aria-pressed={isSelected}
              onClick={() => toggle(p)}
              className={cn(
                'rounded-full px-2 py-1 text-[11px] font-medium border transition-colors inline-flex items-center gap-1.5',
                'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--brand)] focus-visible:ring-offset-1 focus-visible:ring-offset-[var(--bg-base)]',
                isSelected
                  ? 'text-[color:var(--brand-hover)]'
                  : 'border-white/[0.08] bg-surface text-zinc-500 hover:border-white/[0.12] hover:text-zinc-300'
              )}
              style={isSelected ? { borderColor: 'rgba(124,92,252,0.35)', background: 'rgba(124,92,252,0.08)' } : {}}
            >
              <ProviderLogo provider={p} size="sm" />
              <span>{providerDisplayName(p)}</span>
            </button>
          )
        })}
      </div>
      {helperText && (
        <p className="text-[10px] text-zinc-500">{helperText}</p>
      )}
    </div>
  )
}
