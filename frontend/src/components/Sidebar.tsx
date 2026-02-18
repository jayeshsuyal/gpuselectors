import { NavLink } from 'react-router-dom'
import { BarChart3, BookOpen, Receipt, Github, ChevronRight } from 'lucide-react'
import { cn } from '@/lib/utils'
import { AIAssistantPanel } from './AIAssistantPanel'
import { Separator } from './ui/separator'

const NAV_ITEMS = [
  { to: '/', label: 'Optimize Workload', icon: BarChart3, end: true },
  { to: '/catalog', label: 'Browse Catalog', icon: BookOpen },
  { to: '/invoice', label: 'Invoice Analyzer', icon: Receipt },
]

interface SidebarProps {
  className?: string
}

export function Sidebar({ className }: SidebarProps) {
  return (
    <aside
      className={cn(
        'flex flex-col h-full bg-zinc-950 border-r border-zinc-800',
        className
      )}
    >
      {/* Logo */}
      <div className="px-4 py-4 border-b border-zinc-800">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-md bg-indigo-600 flex items-center justify-center flex-shrink-0">
            <span className="text-white text-xs font-bold">IA</span>
          </div>
          <div>
            <div className="text-sm font-bold text-zinc-100 leading-none">InferenceAtlas</div>
            <div className="text-[10px] text-zinc-500 mt-0.5">v0.1 · pre-release</div>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="px-2 py-3 space-y-0.5">
        {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-2.5 px-3 py-2 rounded-md text-sm font-medium transition-colors',
                isActive
                  ? 'bg-zinc-800 text-zinc-100'
                  : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900'
              )
            }
          >
            <Icon className="h-4 w-4 flex-shrink-0" />
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>

      <Separator />

      {/* AI Panel — takes remaining space */}
      <div className="flex-1 min-h-0 flex flex-col">
        <AIAssistantPanel />
      </div>

      {/* Footer */}
      <div className="px-4 py-3 border-t border-zinc-800 space-y-2">
        <a
          href="https://github.com/jayeshsuyal/inference-atlas"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 text-[11px] text-zinc-400 hover:text-zinc-200 transition-colors"
        >
          <Github className="h-3.5 w-3.5" />
          <span>Star on GitHub</span>
          <ChevronRight className="h-3 w-3 ml-auto opacity-50" />
        </a>
        <details className="group">
          <summary className="flex items-center gap-2 text-[11px] text-zinc-500 hover:text-zinc-300 cursor-pointer list-none transition-colors select-none">
            <span>ℹ︎ About & Roadmap</span>
            <ChevronRight className="h-3 w-3 ml-auto group-open:rotate-90 transition-transform" />
          </summary>
          <div className="mt-2 space-y-2 pl-1">
            <p className="text-[10px] text-zinc-500 leading-relaxed">
              Early build — expect rough edges and breaking changes before v1.
            </p>
            <div className="text-[10px] text-zinc-500 space-y-0.5">
              <div className="text-zinc-400 font-medium mb-1">v1 Roadmap</div>
              {[
                'Fine-tuning cost estimation',
                'GPU cluster planning',
                'Real-time price sync API',
                'Shareable result links',
                'REST API + widget',
              ].map((item) => (
                <div key={item} className="flex items-start gap-1.5">
                  <span className="text-zinc-600 mt-0.5">▸</span>
                  <span>{item}</span>
                </div>
              ))}
            </div>
            <a
              href="mailto:jksuyal@gmail.com"
              className="block text-[10px] text-indigo-400 hover:text-indigo-300 transition-colors"
            >
              Ideas for v1? → jksuyal@gmail.com
            </a>
          </div>
        </details>
      </div>
    </aside>
  )
}
