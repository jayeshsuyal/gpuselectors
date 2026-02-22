import * as React from 'react'
import { cn } from '@/lib/utils'

export interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {}

export const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, ...props }, ref) => (
    <textarea
      className={cn(
        'flex min-h-[80px] w-full rounded-md border border-white/[0.08] bg-surface px-3 py-2 text-sm text-zinc-100 shadow-sm transition-colors',
        'placeholder:text-zinc-500',
        'focus:outline-none focus:ring-2 focus:ring-brand focus:ring-offset-1 focus:ring-offset-[var(--bg-base)]',
        'disabled:cursor-not-allowed disabled:opacity-50',
        'resize-none',
        className
      )}
      ref={ref}
      {...props}
    />
  )
)
Textarea.displayName = 'Textarea'
