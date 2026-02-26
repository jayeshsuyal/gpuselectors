import {
  Brain, Mic, Volume2, Layers, Eye, ImageIcon,
  Film, Shield,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { WORKLOAD_TYPES, type WorkloadTypeId } from '@/lib/constants'

const ICON_MAP: Record<string, React.ComponentType<{ className?: string }>> = {
  Brain, Mic, Volume2, Layers, Eye, ImageIcon, Film, Shield,
}

interface WorkloadSelectorProps {
  selected: WorkloadTypeId | null
  onSelect: (id: WorkloadTypeId) => void
}

export function WorkloadSelector({ selected, onSelect }: WorkloadSelectorProps) {
  return (
    <div className="space-y-4 animate-enter">
      <div>
        <h2 className="text-base font-semibold" style={{ color: 'var(--text-primary)' }}>
          Select workload category
        </h2>
        <p className="text-xs mt-0.5" style={{ color: 'var(--text-tertiary)' }}>
          Choose the type of AI workload you want to cost-optimize
        </p>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2">
        {WORKLOAD_TYPES.map((w, i) => {
          const Icon = ICON_MAP[w.icon]
          const isSelected = selected === w.id

          return (
            <button
              key={w.id}
              onClick={() => onSelect(w.id)}
              // data-workload-card drives --w-* token cascade from index.css
              data-workload-card={w.id}
              // data-active signals selected state to CSS (hover-like but persistent)
              data-active={isSelected ? '' : undefined}
              style={{ animationDelay: `${i * 25}ms` }}
              className={cn(
                // workload-card: all hover/active/watermark states handled by CSS
                'workload-card group relative flex flex-col gap-2.5 rounded-lg border p-3.5 text-left cursor-pointer animate-enter',
                'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--brand)] focus-visible:ring-offset-1 focus-visible:ring-offset-[var(--bg-base)]',
                !isSelected && 'border-white/[0.07] bg-surface-elevated'
              )}
            >
              {/* Top accent strip — opacity toggled by .workload-card CSS */}
              <div
                className="card-accent-strip absolute top-0 inset-x-0 h-[1.5px] rounded-t-lg"
                aria-hidden="true"
              />

              {/* Watermark — inner clip container so the button's box-shadow is never clipped */}
              {Icon && (
                <div
                  className="absolute inset-0 overflow-hidden rounded-lg pointer-events-none"
                  aria-hidden="true"
                >
                  <div
                    className={cn(
                      'card-watermark absolute bottom-0 right-0 select-none',
                      isSelected && 'watermark-drift'
                    )}
                    style={{ transform: 'translate(18%, 18%)' }}
                  >
                    <Icon className="h-16 w-16" />
                  </div>
                </div>
              )}

              {/* Foreground icon — color driven by .card-icon CSS rule */}
              {Icon && (
                <span className="card-icon inline-flex relative z-10">
                  <Icon className="h-5 w-5 flex-shrink-0" />
                </span>
              )}

              {/* Label + description */}
              <div className="relative z-10">
                <div
                  className="card-label text-xs font-semibold leading-tight"
                  style={{ color: 'var(--text-secondary)' }}
                >
                  {w.label}
                </div>
                <div className="text-[10px] mt-0.5 leading-tight" style={{ color: 'var(--text-disabled)' }}>
                  {w.description}
                </div>
              </div>
            </button>
          )
        })}
      </div>
    </div>
  )
}
