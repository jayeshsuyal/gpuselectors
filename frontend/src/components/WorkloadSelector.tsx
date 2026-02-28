import { useState } from 'react'
import {
  Brain, Mic, Volume2, Layers, Eye, ImageIcon,
  Film, Shield,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { DEMO_WORKLOAD_IDS, WORKLOAD_TYPES, type WorkloadTypeId } from '@/lib/constants'

const ICON_MAP: Record<string, React.ComponentType<{ className?: string }>> = {
  Brain, Mic, Volume2, Layers, Eye, ImageIcon, Film, Shield,
}

// Per-workload accent palettes — each with a unique colour identity
const PALETTE: Record<string, {
  accent: string
  border: string
  bg: string
  glow: string
}> = {
  llm:              { accent: '#818cf8', border: 'rgba(129,140,248,0.50)', bg: 'rgba(99,102,241,0.07)',  glow: '0 0 18px rgba(99,102,241,0.32), 0 0 36px rgba(99,102,241,0.12)' },
  speech_to_text:   { accent: '#34d399', border: 'rgba(52,211,153,0.50)',  bg: 'rgba(16,185,129,0.07)', glow: '0 0 18px rgba(16,185,129,0.32), 0 0 36px rgba(16,185,129,0.12)' },
  text_to_speech:   { accent: '#38bdf8', border: 'rgba(56,189,248,0.50)',  bg: 'rgba(14,165,233,0.07)', glow: '0 0 18px rgba(14,165,233,0.32), 0 0 36px rgba(14,165,233,0.12)' },
  embeddings:       { accent: '#a78bfa', border: 'rgba(167,139,250,0.50)', bg: 'rgba(139,92,246,0.07)', glow: '0 0 18px rgba(139,92,246,0.32), 0 0 36px rgba(139,92,246,0.12)' },
  vision:           { accent: '#fbbf24', border: 'rgba(251,191,36,0.50)',  bg: 'rgba(245,158,11,0.07)', glow: '0 0 18px rgba(245,158,11,0.32), 0 0 36px rgba(245,158,11,0.12)' },
  image_generation: { accent: '#f472b6', border: 'rgba(244,114,182,0.50)', bg: 'rgba(236,72,153,0.07)', glow: '0 0 18px rgba(236,72,153,0.32), 0 0 36px rgba(236,72,153,0.12)' },
  video_generation: { accent: '#fb923c', border: 'rgba(251,146,60,0.50)',  bg: 'rgba(249,115,22,0.07)', glow: '0 0 18px rgba(249,115,22,0.32), 0 0 36px rgba(249,115,22,0.12)' },
  moderation:       { accent: '#2dd4bf', border: 'rgba(45,212,191,0.50)',  bg: 'rgba(20,184,166,0.07)', glow: '0 0 18px rgba(20,184,166,0.32), 0 0 36px rgba(20,184,166,0.12)' },
}

interface WorkloadSelectorProps {
  selected: WorkloadTypeId | null
  onSelect: (id: WorkloadTypeId) => void
}

export function WorkloadSelector({ selected, onSelect }: WorkloadSelectorProps) {
  const [hoveredId, setHoveredId] = useState<string | null>(null)
  const workloadOptions = WORKLOAD_TYPES.filter((w) => DEMO_WORKLOAD_IDS.includes(w.id))

  return (
    <div className="space-y-4 animate-enter">
      <div>
        <h2 className="text-base font-semibold" style={{ color: 'var(--text-primary)' }}>
          Select workload category
        </h2>
        <p className="text-xs mt-0.5" style={{ color: 'var(--text-tertiary)' }}>
          Demo scope is focused on LLM setup comparison and savings analysis
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
        {workloadOptions.map((w, i) => {
          const Icon = ICON_MAP[w.icon]
          const pal = PALETTE[w.id]
          const isHovered = hoveredId === w.id
          const isSelected = selected === w.id
          const isActive = isHovered || isSelected

          return (
            <button
              key={w.id}
              onClick={() => onSelect(w.id)}
              onMouseEnter={() => setHoveredId(w.id)}
              onMouseLeave={() => setHoveredId(null)}
              style={{
                animationDelay: `${i * 25}ms`,
                ...(isActive && pal
                  ? {
                      borderColor: pal.border,
                      background: pal.bg,
                      boxShadow: pal.glow,
                    }
                  : {}),
              }}
              className={cn(
                'group relative flex flex-col gap-2.5 rounded-lg border p-3.5 text-left cursor-pointer',
                'transition-all duration-200 ease-out animate-enter',
                'active:translate-y-0 active:scale-[0.98]',
                isActive
                  ? '-translate-y-0.5'
                  : 'border-white/[0.07] bg-surface-elevated hover:border-white/[0.12]',
              )}
            >
              {/* Watermark icon — large, centred, drifts slowly */}
              {Icon && (
                <div
                  className="absolute inset-0 overflow-hidden rounded-lg pointer-events-none"
                  aria-hidden="true"
                >
                  <div
                    style={{
                      position: 'absolute',
                      top: '55%',
                      left: '60%',
                      color: pal?.accent ?? 'var(--brand-hover)',
                      opacity: isActive ? 0.07 : 0,
                      transition: 'opacity 300ms ease',
                      animation: isActive ? 'watermarkDrift 7s ease-in-out infinite' : 'none',
                    }}
                  >
                    <Icon className="h-20 w-20 -translate-x-1/2 -translate-y-1/2" />
                  </div>
                </div>
              )}

              {/* Top gradient accent strip */}
              <div
                className="absolute top-0 inset-x-0 h-[1px] rounded-t-lg transition-opacity duration-200"
                style={{
                  background: pal
                    ? `linear-gradient(90deg, transparent, ${pal.accent}99, transparent)`
                    : 'var(--brand-gradient)',
                  opacity: isActive ? 1 : 0,
                }}
              />

              {/* Icon */}
              {Icon && (
                <span
                  className="transition-all duration-200 group-hover:scale-105 inline-flex relative z-10"
                  style={{ color: isActive && pal ? pal.accent : 'var(--text-tertiary)' }}
                >
                  <Icon className="h-5 w-5 flex-shrink-0" />
                </span>
              )}

              {/* Label + description */}
              <div className="relative z-10">
                <div
                  className="text-xs font-semibold leading-tight transition-colors duration-200"
                  style={{ color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)' }}
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
