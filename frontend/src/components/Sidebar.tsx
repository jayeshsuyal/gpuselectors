import { NavLink } from 'react-router-dom'
import { BarChart3, BookOpen, Receipt, ShieldCheck, Github, ChevronRight } from 'lucide-react'
import { cn } from '@/lib/utils'
import { AIAssistantPanel } from './AIAssistantPanel'
import { Separator } from './ui/separator'
import { useAIContext } from '@/context/AIContext'

const NAV_ITEMS = [
  { to: '/',        label: 'Optimize Workload', icon: BarChart3,   end: true },
  { to: '/catalog', label: 'Browse Catalog',    icon: BookOpen },
  { to: '/invoice', label: 'Invoice Analyzer',  icon: Receipt },
  { to: '/audit',   label: 'Cost Audit',        icon: ShieldCheck },
]

interface SidebarProps {
  className?: string
}

export function Sidebar({ className }: SidebarProps) {
  const aiCtx = useAIContext()
  return (
    <aside
      className={cn('relative flex flex-col h-full border-r border-white/[0.06] sidebar-halo overflow-hidden', className)}
      style={{ background: 'var(--bg-surface)' }}
    >
      {/* ── Brand header ── */}
      <div className="relative z-10 px-4 py-4 border-b border-white/[0.05]">
        <div className="flex items-center gap-3">
          {/* Logo mark with brand gradient */}
          <div className="relative w-8 h-8 flex-shrink-0">
            <div
              className="absolute inset-0 rounded-lg shadow-glow-sm"
              style={{ background: 'var(--brand-gradient)' }}
            />
            {/* Inner grid detail */}
            <div className="absolute inset-0 rounded-lg overflow-hidden opacity-20"
              style={{
                backgroundImage: 'linear-gradient(rgba(255,255,255,0.3) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.3) 1px, transparent 1px)',
                backgroundSize: '4px 4px',
              }}
            />
            <div className="absolute inset-0 rounded-lg flex items-center justify-center">
              <span className="text-white text-xs font-bold tracking-tight" style={{ textShadow: '0 1px 2px rgba(0,0,0,0.3)' }}>IA</span>
            </div>
          </div>
          <div>
            <div className="text-sm font-semibold tracking-tight" style={{ color: 'var(--text-primary)', letterSpacing: '-0.02em' }}>
              InferenceAtlas
            </div>
            <div className="text-[10px] font-mono mt-0.5" style={{ color: 'var(--text-disabled)' }}>
              v0.1 · pre-release
            </div>
          </div>
        </div>
      </div>

      {/* ── Navigation ── */}
      <nav className="relative z-10 px-2 py-3 space-y-0.5" aria-label="Main navigation">
        {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) =>
              cn(
                'group flex items-center gap-2.5 px-3 py-2 rounded-md text-sm font-medium transition-all duration-200 border',
                isActive
                  ? 'border-[rgba(124,92,252,0.28)] text-[#c4b5fd]'
                  : 'text-zinc-400 hover:text-zinc-200 hover:bg-white/[0.04] border-transparent'
              )
            }
            style={({ isActive }) =>
              isActive
                ? { background: 'rgba(124,92,252,0.10)', boxShadow: 'var(--shadow-glow-sm)' }
                : {}
            }
          >
            {({ isActive }) => (
              <>
                <Icon
                  className={cn(
                    'h-4 w-4 flex-shrink-0 transition-colors',
                    isActive ? '' : 'text-zinc-500 group-hover:text-zinc-300'
                  )}
                  style={isActive ? { color: 'var(--brand-hover)' } : {}}
                />
                <span>{label}</span>
                {isActive && (
                  <span className="ml-auto active-dot" />
                )}
              </>
            )}
          </NavLink>
        ))}
      </nav>

      <Separator className="relative z-10 bg-white/[0.05] h-px" />

      {/* AI Panel */}
      <div className="relative z-10 flex-1 min-h-0 flex flex-col">
        <AIAssistantPanel context={aiCtx} />
      </div>

      {/* ── Footer ── */}
      <div className="relative z-10 px-4 py-3 border-t border-white/[0.05] space-y-2">
        <a
          href="https://github.com/jayeshsuyal/inference-atlas"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 text-[11px] transition-colors group"
          style={{ color: 'var(--text-tertiary)' }}
        >
          <Github className="h-3.5 w-3.5 flex-shrink-0" />
          <span className="group-hover:text-zinc-200 transition-colors">Star on GitHub</span>
          <ChevronRight className="h-3 w-3 ml-auto opacity-40 group-hover:opacity-70 transition-opacity" />
        </a>
        <details className="group/details">
          <summary
            className="flex items-center gap-2 text-[11px] hover:text-zinc-300 cursor-pointer list-none transition-colors select-none"
            style={{ color: 'var(--text-disabled)' }}
          >
            <span>ℹ︎ About & Roadmap</span>
            <ChevronRight className="h-3 w-3 ml-auto group-open/details:rotate-90 transition-transform duration-200" />
          </summary>
          <div className="mt-2 space-y-2 pl-1 animate-enter-fast">
            <p className="text-[10px] leading-relaxed" style={{ color: 'var(--text-disabled)' }}>
              Early build — expect rough edges and breaking changes before v1.
            </p>
            <div className="text-[10px] space-y-0.5" style={{ color: 'var(--text-disabled)' }}>
              <div className="font-medium mb-1" style={{ color: 'var(--text-tertiary)' }}>v1 Roadmap</div>
              {[
                'Fine-tuning cost estimation',
                'GPU cluster planning',
                'Real-time price sync API',
                'Shareable result links',
                'REST API + widget',
              ].map((item) => (
                <div key={item} className="flex items-start gap-1.5">
                  <span className="opacity-40 mt-0.5">▸</span>
                  <span>{item}</span>
                </div>
              ))}
            </div>
            <a
              href="mailto:inferenceatlas@gmail.com"
              className="block text-[10px] transition-colors"
              style={{ color: 'var(--brand)' }}
            >
              Ideas for v1? → inferenceatlas@gmail.com
            </a>
          </div>
        </details>
      </div>
    </aside>
  )
}
