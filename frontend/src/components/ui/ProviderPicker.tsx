/**
 * Shared multi-select provider picker used by both LLMForm and NonLLMForm.
 * Renders a pill grid; clicking toggles selection.
 */
import { cn } from '@/lib/utils'
import { PROVIDERS } from '@/lib/constants'

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
                'rounded-full px-2.5 py-1 text-[11px] font-medium border transition-colors',
                isSelected
                  ? 'border-indigo-600 bg-indigo-950/60 text-indigo-300'
                  : 'border-zinc-700 bg-zinc-900 text-zinc-500 hover:border-zinc-600 hover:text-zinc-400'
              )}
            >
              {p}
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
