/**
 * Conversational AI copilot for guided workload configuration.
 * Maintains multi-turn chat, extracts a spec progressively, and emits
 * an apply_payload the user can push directly into the Guided Config form.
 */
import { useState, useRef, useEffect } from 'react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { Send, Loader2, Sparkles, CircleDot, CheckCircle2 } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { aiChatSchema, type AIChatValues } from '@/schemas/forms'
import { nextCopilotTurn } from '@/services/api'
import type { CopilotMessage, CopilotTurnResponse, CopilotApplyPayload } from '@/services/types'

// ─── Inline markdown renderer ─────────────────────────────────────────────────

function renderInline(text: string): React.ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*)/).map((part, i) =>
    part.startsWith('**') && part.endsWith('**') ? (
      <strong key={i} className="font-semibold text-zinc-100">{part.slice(2, -2)}</strong>
    ) : (
      <span key={i}>{part}</span>
    )
  )
}

function renderMarkdown(content: string): React.ReactNode {
  return (
    <>
      {content.split('\n').map((line, i) => {
        if (line.startsWith('- ') || line.startsWith('* ')) {
          return (
            <div key={i} className="flex gap-1.5 mt-0.5 first:mt-0">
              <span className="text-zinc-500 flex-shrink-0 select-none">•</span>
              <span>{renderInline(line.slice(2))}</span>
            </div>
          )
        }
        if (line.trim() === '') return <div key={i} className="h-1.5" />
        return <div key={i}>{renderInline(line)}</div>
      })}
    </>
  )
}

// ─── Extracted spec card ──────────────────────────────────────────────────────

const SPEC_LABELS: Record<string, string> = {
  workload_type: 'Workload',
  tokens_per_day: 'Daily tokens',
  model_bucket: 'Model size',
  provider_ids: 'Providers',
  traffic_pattern: 'Traffic pattern',
  monthly_budget_max_usd: 'Budget cap',
  unit_name: 'Pricing unit',
  monthly_usage: 'Monthly usage',
  throughput_aware: 'Throughput-aware',
  comparator_mode: 'Comparator',
}

function formatSpecValue(key: string, value: unknown): string {
  if (key === 'tokens_per_day' && typeof value === 'number') {
    return value >= 1_000_000_000
      ? `${(value / 1e9).toFixed(0)}B/day`
      : value >= 1_000_000
      ? `${(value / 1e6).toFixed(0)}M/day`
      : `${(value / 1e3).toFixed(0)}K/day`
  }
  if (key === 'model_bucket' && typeof value === 'string') return value.toUpperCase() + ' params'
  if (key === 'monthly_budget_max_usd') return value === 0 ? 'No limit' : `$${value}/mo`
  if (key === 'workload_type' && typeof value === 'string') return value.replace(/_/g, ' ')
  if (key === 'monthly_usage' && typeof value === 'number') return `${value.toLocaleString()} units`
  if (Array.isArray(value)) return value.join(', ') || '—'
  if (typeof value === 'boolean') return value ? 'Yes' : 'No'
  if (value === null || value === undefined || value === '') return '—'
  return String(value)
}

function ExtractedSpecCard({ spec }: { spec: CopilotTurnResponse['extracted_spec'] }) {
  const entries = Object.entries(spec).filter(
    ([k, v]) => v !== undefined && v !== null && k !== 'workload_type'
  )
  if (entries.length === 0) return null

  return (
    <div
      className="rounded-md border px-3 py-2.5"
      style={{ borderColor: 'var(--border-default)', background: 'var(--bg-elevated)' }}
    >
      <div className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider mb-2 flex items-center gap-1.5">
        <CheckCircle2 className="h-3 w-3 text-emerald-500" />
        Understood so far
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1.5">
        {entries.map(([k, v]) => (
          <div key={k}>
            <div className="text-[10px] text-zinc-600">{SPEC_LABELS[k] ?? k}</div>
            <div className="text-xs text-zinc-200 font-mono truncate">{formatSpecValue(k, v)}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ─── Preset buttons ───────────────────────────────────────────────────────────

type Preset = 'cheap' | 'balanced' | 'reliable'

const PRESET_META: Record<Preset, { label: string; icon: string; description: string }> = {
  cheap:    { label: 'Cheap',    icon: '💰', description: 'Minimize cost' },
  balanced: { label: 'Balanced', icon: '⚖️', description: 'Cost + reliability' },
  reliable: { label: 'Reliable', icon: '🛡️', description: 'Enterprise-grade' },
}

const LLM_PRESETS: Record<Preset, CopilotApplyPayload> = {
  cheap: {
    tokens_per_day: 5_000_000,
    model_bucket: '7b',
    provider_ids: ['fireworks', 'together_ai', 'groq'],
    traffic_pattern: 'business_hours',
    top_k: 5,
    monthly_budget_max_usd: 0,
  },
  balanced: {
    tokens_per_day: 5_000_000,
    model_bucket: '70b',
    provider_ids: ['fireworks', 'together_ai', 'openai', 'anthropic'],
    traffic_pattern: 'business_hours',
    top_k: 5,
    monthly_budget_max_usd: 0,
  },
  reliable: {
    tokens_per_day: 5_000_000,
    model_bucket: '70b',
    provider_ids: ['openai', 'anthropic', 'aws'],
    traffic_pattern: 'business_hours',
    util_target: 0.7,
    top_k: 3,
    monthly_budget_max_usd: 0,
  },
}

const NON_LLM_PRESETS: Record<Preset, CopilotApplyPayload> = {
  cheap: {
    monthly_usage: 5_000,
    monthly_budget_max_usd: 0,
    confidence_weighted: true,
    comparator_mode: 'normalized',
    throughput_aware: false,
    strict_capacity_check: false,
    top_k: 5,
  },
  balanced: {
    monthly_usage: 5_000,
    monthly_budget_max_usd: 0,
    confidence_weighted: true,
    comparator_mode: 'normalized',
    throughput_aware: false,
    strict_capacity_check: false,
    top_k: 5,
  },
  reliable: {
    monthly_usage: 5_000,
    monthly_budget_max_usd: 0,
    confidence_weighted: true,
    comparator_mode: 'normalized',
    throughput_aware: true,
    strict_capacity_check: true,
    top_k: 5,
  },
}

// ─── Message bubble ───────────────────────────────────────────────────────────

function MessageBubble({ msg }: { msg: CopilotMessage }) {
  const isUser = msg.role === 'user'
  return (
    <div className={cn('flex gap-2', isUser ? 'justify-end' : 'justify-start')}>
      {!isUser && (
        <div
          className="flex-shrink-0 w-5 h-5 rounded-full flex items-center justify-center mt-0.5"
          style={{ background: 'rgba(124,92,252,0.15)', border: '1px solid rgba(124,92,252,0.35)' }}
        >
          <Sparkles className="w-3 h-3" style={{ color: 'var(--brand-hover)' }} />
        </div>
      )}
      <div
        className="rounded-lg px-3 py-2 text-xs leading-relaxed max-w-[88%]"
        style={isUser
          ? { background: 'var(--brand)', color: '#fff' }
          : { background: 'var(--bg-elevated)', color: 'var(--text-secondary)', border: '1px solid var(--border-default)' }
        }
      >
        {renderMarkdown(msg.content)}
      </div>
    </div>
  )
}

// ─── Panel ────────────────────────────────────────────────────────────────────

interface CopilotPanelProps {
  workloadType: string
  isLLM: boolean
  onApply: (payload: CopilotApplyPayload) => void
}

export function CopilotPanel({ workloadType, isLLM, onApply }: CopilotPanelProps) {
  const [messages, setMessages] = useState<CopilotMessage[]>([])
  const [lastResponse, setLastResponse] = useState<CopilotTurnResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [chatError, setChatError] = useState<string | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  const { register, handleSubmit, reset, formState: { errors } } = useForm<AIChatValues>({
    resolver: zodResolver(aiChatSchema),
  })

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages])

  async function sendMessage(text: string) {
    const userMsg: CopilotMessage = { role: 'user', content: text, timestamp: Date.now() }
    setMessages((prev) => [...prev, userMsg])
    reset()
    setChatError(null)
    setLoading(true)
    try {
      const res = await nextCopilotTurn({
        message: text,
        history: messages,
        workload_type: workloadType,
      })
      setLastResponse(res)
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: res.reply, timestamp: Date.now() },
      ])
    } catch {
      setChatError('Something went wrong — try again.')
    } finally {
      setLoading(false)
    }
  }

  function onSubmit(values: AIChatValues) {
    void sendMessage(values.message)
  }

  function applyPreset(preset: Preset) {
    onApply(isLLM ? LLM_PRESETS[preset] : NON_LLM_PRESETS[preset])
  }

  const hasExtractedData =
    lastResponse !== null &&
    Object.keys(lastResponse.extracted_spec).filter((k) => k !== 'workload_type').length > 0

  return (
    <div className="space-y-4">
      {/* Chat area */}
      <div
        ref={scrollRef}
        aria-live="polite"
        aria-label="Conversation"
        className={cn('space-y-3 overflow-y-auto', messages.length > 0 && 'max-h-56')}
      >
        {messages.length === 0 ? (
          <p className="text-xs text-zinc-500">
            Describe your workload in plain language, or start with a preset below.
          </p>
        ) : (
          messages.map((msg, i) => <MessageBubble key={i} msg={msg} />)
        )}
        {loading && (
          <div className="flex gap-2 justify-start">
            <div
              className="flex-shrink-0 w-5 h-5 rounded-full flex items-center justify-center"
              style={{ background: 'rgba(124,92,252,0.15)', border: '1px solid rgba(124,92,252,0.35)' }}
            >
              <Sparkles className="w-3 h-3" style={{ color: 'var(--brand-hover)' }} />
            </div>
            <div
              className="rounded-lg px-3 py-2"
              style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-default)' }}
            >
              <Loader2 className="w-3.5 h-3.5 text-zinc-400 animate-spin" />
            </div>
          </div>
        )}
      </div>

      {/* Extracted spec */}
      {hasExtractedData && <ExtractedSpecCard spec={lastResponse!.extracted_spec} />}

      {/* Missing fields */}
      {lastResponse?.missing_fields && lastResponse.missing_fields.length > 0 && (
        <div className="flex items-start gap-2">
          <CircleDot className="h-3.5 w-3.5 text-amber-400 flex-shrink-0 mt-0.5" />
          <p className="text-[11px] text-zinc-500">
            <span className="text-zinc-400">Still needed: </span>
            {lastResponse.missing_fields.join(' · ')}
          </p>
        </div>
      )}

      {/* Follow-up question chips */}
      {lastResponse?.follow_up_questions && lastResponse.follow_up_questions.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {lastResponse.follow_up_questions.map((q) => (
            <button
              key={q}
              type="button"
              onClick={() => void sendMessage(q)}
              disabled={loading}
              className="text-[11px] px-2.5 py-1 rounded-full border border-zinc-700 text-zinc-400 hover:text-zinc-200 hover:border-zinc-600 hover:bg-zinc-800/50 transition-colors disabled:opacity-40"
            >
              {q}
            </button>
          ))}
        </div>
      )}

      {/* Presets */}
      <div>
        <div className="text-[10px] text-zinc-600 uppercase tracking-wider mb-2">
          {messages.length === 0 ? 'Quick-start with a preset' : 'Or jump to a preset'}
        </div>
        <div className="grid grid-cols-3 gap-2">
          {(['cheap', 'balanced', 'reliable'] as Preset[]).map((preset) => {
            const meta = PRESET_META[preset]
            return (
              <button
                key={preset}
                type="button"
                onClick={() => applyPreset(preset)}
                aria-label={`${meta.label}: ${meta.description}`}
                className="flex flex-col items-center gap-1 rounded-md border px-2 py-2.5 transition-colors text-center focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--brand)] focus-visible:ring-offset-1 focus-visible:ring-offset-[var(--bg-base)]"
                style={{ borderColor: 'var(--border-subtle)', background: 'var(--bg-elevated)' }}
                onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.borderColor = 'var(--border-default)' }}
                onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.borderColor = 'var(--border-subtle)' }}
              >
                <span className="text-base leading-none">{meta.icon}</span>
                <span className="text-[11px] font-medium text-zinc-300">{meta.label}</span>
                <span className="text-[10px] text-zinc-600">{meta.description}</span>
              </button>
            )
          })}
        </div>
      </div>

      {/* Text input */}
      <form onSubmit={handleSubmit(onSubmit)} className="relative">
        <Textarea
          {...register('message')}
          placeholder={
            messages.length === 0
              ? `e.g. "I need cheap ${workloadType.replace(/_/g, ' ')} for ~10K units/day"`
              : 'Continue the conversation…'
          }
          className="min-h-[56px] max-h-[120px] text-xs pr-9 resize-none"
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              void handleSubmit(onSubmit)()
            }
          }}
        />
        <Button
          type="submit"
          size="icon-sm"
          variant="ghost"
          disabled={loading}
          aria-label="Send message"
          className="absolute right-2 bottom-2 h-6 w-6 ai-send-btn"
        >
          <Send className="h-3 w-3" />
        </Button>
      </form>
      {(errors.message || chatError) && (
        <p className="text-[10px] text-red-400">{errors.message?.message ?? chatError}</p>
      )}

      {/* Apply to Config */}
      {lastResponse?.apply_payload && (
        <Button
          type="button"
          className="w-full"
          onClick={() => onApply(lastResponse.apply_payload!)}
        >
          <CheckCircle2 className="h-4 w-4 mr-2" />
          Apply to Config
        </Button>
      )}
    </div>
  )
}
