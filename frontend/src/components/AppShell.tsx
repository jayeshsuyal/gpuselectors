import { useState } from 'react'
import { Outlet } from 'react-router-dom'
import { Menu, X } from 'lucide-react'
import { Sidebar } from './Sidebar'
import { Button } from './ui/button'
import { AIContextProvider } from '@/context/AIContext'

export function AppShell() {
  const [mobileOpen, setMobileOpen] = useState(false)

  return (
    <AIContextProvider>
    <div className="relative flex h-screen w-screen overflow-hidden" style={{ background: 'var(--bg-base)' }}>
      {/* ── Ambient aurora orbs — behind everything ── */}
      <div aria-hidden className="pointer-events-none fixed inset-0 z-0 overflow-hidden">
        {/* Primary violet orb — top right */}
        <div
          className="aurora-orb-primary"
          style={{
            top: '-15%',
            right: '-8%',
            width: '55%',
            height: '55%',
            opacity: 0.06,
          }}
        />
        {/* Secondary green orb — bottom left */}
        <div
          className="aurora-orb-secondary"
          style={{
            bottom: '-12%',
            left: '-6%',
            width: '38%',
            height: '38%',
            opacity: 0.035,
          }}
        />
        {/* Subtle mid accent */}
        <div
          className="aurora-orb-primary"
          style={{
            top: '50%',
            left: '30%',
            width: '20%',
            height: '20%',
            opacity: 0.018,
            transform: 'translateY(-50%)',
          }}
        />
      </div>

      {/* Desktop sidebar */}
      <div className="hidden lg:flex lg:w-64 xl:w-72 flex-col h-full flex-shrink-0 relative z-10">
        <Sidebar className="h-full" />
      </div>

      {/* Mobile sidebar overlay */}
      {mobileOpen && (
        <div className="fixed inset-0 z-50 lg:hidden animate-fade-in">
          <div
            className="absolute inset-0 bg-black/75 backdrop-blur-sm"
            onClick={() => setMobileOpen(false)}
          />
          <div className="absolute left-0 top-0 bottom-0 w-72 z-10 animate-enter">
            <Sidebar className="h-full" />
          </div>
        </div>
      )}

      {/* Main content */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden relative z-10">
        {/* Mobile header */}
        <header className="lg:hidden flex items-center gap-3 px-4 py-3 border-b border-white/[0.06] backdrop-blur-sm" style={{ background: 'var(--bg-surface)' }}>
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => setMobileOpen(!mobileOpen)}
            aria-label={mobileOpen ? 'Close menu' : 'Open menu'}
          >
            {mobileOpen ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
          </Button>
          <div className="flex items-center gap-2.5">
            {/* Brand mark */}
            <div className="relative w-6 h-6 flex-shrink-0">
              <div className="absolute inset-0 rounded-md" style={{ background: 'var(--brand-gradient)' }} />
              <div className="absolute inset-0 rounded-md flex items-center justify-center">
                <span className="text-white text-[9px] font-bold tracking-tight">IA</span>
              </div>
            </div>
            <span className="text-sm font-semibold tracking-tight" style={{ color: 'var(--text-primary)' }}>
              InferenceAtlas
            </span>
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-y-auto">
          <Outlet />
        </main>
      </div>
    </div>
    </AIContextProvider>
  )
}
