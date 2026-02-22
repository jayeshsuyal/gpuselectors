import * as React from 'react'
import { cn } from '@/lib/utils'

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {}

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, ...props }, ref) => (
    <input
      type={type}
      className={cn(
        'flex h-9 w-full rounded-md px-3 py-1 text-sm text-zinc-100',
        'bg-surface border border-white/[0.08] shadow-xs',
        'placeholder:text-zinc-500',
        'transition-all duration-200',
        'hover:border-white/[0.13]',
        'focus:outline-none focus:border-brand/60 focus:ring-2 focus:ring-brand/25 focus:ring-offset-0',
        'disabled:cursor-not-allowed disabled:opacity-40',
        'file:border-0 file:bg-transparent file:text-sm file:font-medium',
        className
      )}
      ref={ref}
      {...props}
    />
  )
)
Input.displayName = 'Input'
