import * as React from 'react'
import { Slot } from '@radix-ui/react-slot'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '@/lib/utils'

const buttonVariants = cva(
  [
    'inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium',
    'transition-all duration-200 ease-out',
    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand focus-visible:ring-offset-2 focus-visible:ring-offset-[var(--bg-base)]',
    'disabled:pointer-events-none disabled:opacity-40',
    '[&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0',
  ].join(' '),
  {
    variants: {
      variant: {
        // Primary — brand violet with glow on hover
        default:
          'bg-brand text-white shadow-sm shadow-brand/20 hover:bg-brand-hover hover:shadow-glow-sm active:brightness-90 active:scale-[0.98]',
        // Secondary — glass surface
        secondary:
          'bg-surface-elevated text-zinc-100 border border-white/[0.07] shadow-inner-highlight hover:bg-white/[0.06] hover:border-white/[0.11] active:bg-surface active:scale-[0.98]',
        // Ghost — no background until hover
        ghost:
          'text-zinc-400 hover:text-zinc-100 hover:bg-white/[0.06] active:bg-white/[0.04]',
        // Outline — transparent with border
        outline:
          'border border-white/[0.10] bg-transparent text-zinc-300 hover:bg-white/[0.05] hover:border-white/[0.16] hover:text-zinc-100 active:scale-[0.98]',
        // Destructive
        destructive:
          'bg-red-600 text-white hover:bg-red-500 shadow-sm active:bg-red-700 active:scale-[0.98]',
        // Link
        link:
          'text-[var(--brand-hover)] underline-offset-4 hover:underline hover:text-[var(--brand)] p-0 h-auto',
        // Glass — translucent panel button
        glass:
          'glass-card text-zinc-200 hover:border-white/[0.14] hover:text-white card-hover active:scale-[0.98]',
      },
      size: {
        default: 'h-9 px-4 py-2',
        sm: 'h-7 px-3 text-xs rounded-md',
        lg: 'h-10 px-6 text-sm',
        icon: 'h-9 w-9',
        'icon-sm': 'h-7 w-7',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : 'button'
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    )
  }
)
Button.displayName = 'Button'

export { buttonVariants }
