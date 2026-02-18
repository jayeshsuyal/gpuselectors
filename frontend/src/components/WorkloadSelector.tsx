import {
  Brain, Mic, Volume2, Layers, Eye, ImageIcon,
  Film, Shield,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { WORKLOAD_TYPES, type WorkloadTypeId } from '@/lib/constants'

const ICON_MAP: Record<string, React.ComponentType<{ className?: string }>> = {
  Brain, Mic, Volume2, Layers, Eye, ImageIcon, Film, Shield,
}

const COLOR_MAP: Record<string, string> = {
  indigo: 'border-indigo-800 bg-indigo-950/40 hover:border-indigo-600 hover:bg-indigo-950/70 text-indigo-300',
  emerald: 'border-emerald-800 bg-emerald-950/40 hover:border-emerald-600 hover:bg-emerald-950/70 text-emerald-300',
  sky: 'border-sky-800 bg-sky-950/40 hover:border-sky-600 hover:bg-sky-950/70 text-sky-300',
  violet: 'border-violet-800 bg-violet-950/40 hover:border-violet-600 hover:bg-violet-950/70 text-violet-300',
  amber: 'border-amber-800 bg-amber-950/40 hover:border-amber-600 hover:bg-amber-950/70 text-amber-300',
  rose: 'border-rose-800 bg-rose-950/40 hover:border-rose-600 hover:bg-rose-950/70 text-rose-300',
  orange: 'border-orange-800 bg-orange-950/40 hover:border-orange-600 hover:bg-orange-950/70 text-orange-300',
  teal: 'border-teal-800 bg-teal-950/40 hover:border-teal-600 hover:bg-teal-950/70 text-teal-300',
}

const SELECTED_MAP: Record<string, string> = {
  indigo: 'border-indigo-500 bg-indigo-950/80 ring-1 ring-indigo-500',
  emerald: 'border-emerald-500 bg-emerald-950/80 ring-1 ring-emerald-500',
  sky: 'border-sky-500 bg-sky-950/80 ring-1 ring-sky-500',
  violet: 'border-violet-500 bg-violet-950/80 ring-1 ring-violet-500',
  amber: 'border-amber-500 bg-amber-950/80 ring-1 ring-amber-500',
  rose: 'border-rose-500 bg-rose-950/80 ring-1 ring-rose-500',
  orange: 'border-orange-500 bg-orange-950/80 ring-1 ring-orange-500',
  teal: 'border-teal-500 bg-teal-950/80 ring-1 ring-teal-500',
}

interface WorkloadSelectorProps {
  selected: WorkloadTypeId | null
  onSelect: (id: WorkloadTypeId) => void
}

export function WorkloadSelector({ selected, onSelect }: WorkloadSelectorProps) {
  return (
    <div className="space-y-3">
      <div>
        <h2 className="text-base font-semibold text-zinc-100">Select workload category</h2>
        <p className="text-xs text-zinc-500 mt-0.5">Choose the type of AI workload you want to cost-optimize</p>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
        {WORKLOAD_TYPES.map((w) => {
          const Icon = ICON_MAP[w.icon]
          const isSelected = selected === w.id
          return (
            <button
              key={w.id}
              onClick={() => onSelect(w.id)}
              className={cn(
                'group flex flex-col gap-2 rounded-lg border p-4 text-left transition-all duration-150 cursor-pointer',
                isSelected ? SELECTED_MAP[w.color] : COLOR_MAP[w.color]
              )}
            >
              {Icon && (
                <Icon
                  className={cn(
                    'h-5 w-5 transition-transform group-hover:scale-110',
                    isSelected ? 'scale-110' : ''
                  )}
                />
              )}
              <div>
                <div className="text-xs font-semibold text-zinc-100">{w.label}</div>
                <div className="text-[11px] text-zinc-400 mt-0.5 leading-tight">{w.description}</div>
              </div>
            </button>
          )
        })}
      </div>
    </div>
  )
}
