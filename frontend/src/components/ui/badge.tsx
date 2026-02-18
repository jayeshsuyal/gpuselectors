import * as React from 'react'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '@/lib/utils'

const badgeVariants = cva(
  'inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset transition-colors',
  {
    variants: {
      variant: {
        default: 'bg-zinc-800 text-zinc-300 ring-zinc-700',
        indigo: 'bg-indigo-950 text-indigo-300 ring-indigo-800',
        emerald: 'bg-emerald-950 text-emerald-300 ring-emerald-800',
        amber: 'bg-amber-950 text-amber-300 ring-amber-800',
        red: 'bg-red-950 text-red-300 ring-red-900',
        sky: 'bg-sky-950 text-sky-300 ring-sky-800',
        violet: 'bg-violet-950 text-violet-300 ring-violet-800',
        // Confidence
        high: 'bg-emerald-950 text-emerald-300 ring-emerald-800',
        official: 'bg-emerald-950 text-emerald-300 ring-emerald-800',
        medium: 'bg-amber-950 text-amber-300 ring-amber-800',
        estimated: 'bg-amber-950 text-amber-300 ring-amber-800',
        low: 'bg-zinc-800 text-zinc-400 ring-zinc-700',
        vendor_list: 'bg-zinc-800 text-zinc-400 ring-zinc-700',
        // Risk
        risk_low: 'bg-emerald-950 text-emerald-300 ring-emerald-800',
        risk_medium: 'bg-amber-950 text-amber-300 ring-amber-800',
        risk_high: 'bg-red-950 text-red-300 ring-red-900',
        // Capacity
        ok: 'bg-emerald-950 text-emerald-300 ring-emerald-800',
        insufficient: 'bg-red-950 text-red-300 ring-red-900',
        unknown: 'bg-zinc-800 text-zinc-400 ring-zinc-700',
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
