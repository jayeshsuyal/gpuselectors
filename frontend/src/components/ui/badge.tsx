import * as React from 'react'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '@/lib/utils'

const badgeVariants = cva(
  'inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium ring-1 ring-inset transition-colors leading-tight',
  {
    variants: {
      variant: {
        // Base
        default: 'bg-zinc-800/80 text-zinc-300 ring-white/[0.08]',
        // Named color variants
        indigo: 'bg-indigo-950/70 text-indigo-300 ring-indigo-700/40',
        emerald: 'bg-emerald-950/70 text-emerald-300 ring-emerald-700/40',
        amber: 'bg-amber-950/70 text-amber-300 ring-amber-700/40',
        red: 'bg-red-950/70 text-red-300 ring-red-800/40',
        sky: 'bg-sky-950/70 text-sky-300 ring-sky-700/40',
        violet: 'bg-violet-950/70 text-violet-300 ring-violet-700/40',
        // ── Confidence ──
        high: 'bg-emerald-950/70 text-emerald-300 ring-emerald-700/40',
        official: 'bg-emerald-950/70 text-emerald-300 ring-emerald-700/40',
        medium: 'bg-amber-950/70 text-amber-300 ring-amber-700/40',
        estimated: 'bg-amber-950/70 text-amber-300 ring-amber-700/40',
        low: 'bg-zinc-800/60 text-zinc-400 ring-white/[0.06]',
        vendor_list: 'bg-zinc-800/60 text-zinc-400 ring-white/[0.06]',
        // ── Risk ──
        risk_low: 'bg-emerald-950/70 text-emerald-300 ring-emerald-700/40',
        risk_medium: 'bg-amber-950/70 text-amber-300 ring-amber-700/40',
        risk_high: 'bg-red-950/70 text-red-300 ring-red-800/40',
        // ── Capacity ──
        ok: 'bg-emerald-950/70 text-emerald-300 ring-emerald-700/40',
        insufficient: 'bg-red-950/70 text-red-300 ring-red-800/40',
        unknown: 'bg-zinc-800/60 text-zinc-400 ring-white/[0.06]',
      },
    },
    defaultVariants: {
      variant: 'default',
    },
  }
)

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

export function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  )
}
