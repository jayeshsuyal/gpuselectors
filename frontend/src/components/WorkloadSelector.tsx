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
              style={{
                animationDelay: `${i * 25}ms`,
                ...(isSelected
                  ? {
                      borderColor: 'rgba(124,92,252,0.45)',
                      background: 'rgba(124,92,252,0.06)',
                      boxShadow: 'var(--shadow-glow-sm)',
                    }
                  : {}),
              }}
              className={cn(
                'group relative flex flex-col gap-2.5 rounded-lg border p-3.5 text-left cursor-pointer',
                'transition-all duration-200 ease-out animate-enter',
                'hover:-translate-y-0.5 active:translate-y-0 active:scale-[0.98]',
                isSelected
                  ? 'ring-1 ring-[rgba(124,92,252,0.18)]'
                  : 'border-white/[0.07] bg-surface-elevated hover:border-white/[0.13] hover:bg-white/[0.02]'
              )}
            >
              {/* Top gradient accent strip when selected */}
              {isSelected && (
                <div
                  className="absolute top-0 inset-x-0 h-[1.5px] rounded-t-lg"
                  style={{ background: 'var(--brand-gradient)' }}
                />
              )}

              {Icon && (
                <span
                  className="transition-all duration-200 group-hover:scale-105 inline-flex"
                  style={{ color: isSelected ? 'var(--brand-hover)' : 'var(--text-tertiary)' }}
                >
                  <Icon className="h-5 w-5 flex-shrink-0" />
                </span>
              )}

              <div>
                <div
                  className="text-xs font-semibold leading-tight"
                  style={{ color: isSelected ? 'var(--text-primary)' : 'var(--text-secondary)' }}
                >
                  {w.label}
                </div>
                <div
                  className="text-[10px] mt-0.5 leading-tight"
                  style={{ color: 'var(--text-disabled)' }}
                >
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
