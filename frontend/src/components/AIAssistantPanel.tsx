import { useState, useRef, useEffect } from 'react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { Send, Bot, Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { aiChatSchema, type AIChatValues } from '@/schemas/forms'
import { askAI } from '@/services/api'
import type { AIMessage } from '@/services/types'

// ─── Markdown rendering ───────────────────────────────────────────────────────

function renderInline(text: string): React.ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*)/).map((part, i) =>
    part.startsWith('**') && part.endsWith('**') ? (
      <strong key={i} className="font-semibold text-white">
        {part.slice(2, -2)}
      </strong>
    ) : (
      <span key={i}>{part}</span>
    )
  )
}

function renderMarkdown(content: string): React.ReactNode {
  const lines = content.split('\n')
  return (
    <>
      {lines.map((line, i) => {
        if (line.startsWith('- ') || line.startsWith('* ')) {
          return (
            <div key={i} className="flex gap-1.5 mt-0.5 first:mt-0">
              <span className="text-zinc-500 flex-shrink-0 select-none">•</span>
              <span>{renderInline(line.slice(2))}</span>
            </div>
          )
        }
        if (line.trim() === '') {
          return <div key={i} className="h-1.5" />
        }
        return <div key={i}>{renderInline(line)}</div>
      })}
    </>
  )
}

// ─── Message bubble ───────────────────────────────────────────────────────────

function MessageBubble({ msg }: { msg: AIMessage }) {
  const isUser = msg.role === 'user'
  return (
    <div className={cn('flex gap-2', isUser ? 'justify-end' : 'justify-start')}>
      {!isUser && (
        <div
          className="flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center mt-0.5"
          style={{ background: 'rgba(124,92,252,0.15)', border: '1px solid rgba(124,92,252,0.35)' }}
        >
          <Bot className="w-3.5 h-3.5" style={{ color: 'var(--brand-hover)' }} />
        </div>
      )}
      <div
        className={cn('rounded-lg px-3 py-2 text-xs leading-relaxed max-w-[85%]')}
        style={isUser
          ? { background: 'var(--brand)', color: '#fff' }
          : {
              background: 'var(--bg-elevated)',
              color: 'var(--text-secondary)',
              border: '1px solid var(--border-default)',
              borderLeft: '2px solid rgba(124,92,252,0.35)',
            }
        }
      >
        {renderMarkdown(msg.content)}
      </div>
    </div>
  )
}

// ─── Workload-aware starter prompts ──────────────────────────────────────────

const WORKLOAD_STARTERS: Record<string, [string, string, string]> = {
  llm: [
    'Cheapest 70B provider for business hours?',
    'Per-token vs dedicated GPU pricing?',
    'Explain risk scores',
  ],
  speech_to_text: [
    'Cheapest STT for batch transcription?',
    'Deepgram vs OpenAI Whisper cost?',
    'Real-time vs batch STT pricing',
  ],
  text_to_speech: [
    'Best value TTS under $0.02 / 1K chars?',
    'ElevenLabs vs OpenAI TTS pricing?',
    'Per-character vs per-minute billing',
  ],
  embeddings: [
    'Cheapest embeddings for 1B tokens/month?',
    'OpenAI vs Cohere embedding cost?',
    'Voyage AI vs Cohere for semantic search?',
  ],
  image_generation: [
    'Cheapest per-image generation provider?',
    'Per-image vs GPU-hour billing?',
    'DALL-E vs Stable Diffusion pricing?',
  ],
  vision: [
    'Best value for high-volume image analysis?',
    'AWS Rekognition vs Google Vision cost?',
    'Per-image vs per-1K image pricing',
  ],
  video_generation: [
    'Most cost-effective video generation?',
    'Per-second vs per-minute pricing?',
    'What affects video generation cost?',
  ],
  moderation: [
    'Cheapest content moderation API?',
    'Free tier vs paid moderation options?',
    'AWS vs OpenAI moderation pricing?',
  ],
}

const DEFAULT_STARTERS: [string, string, string] = [
  'Cheapest STT provider?',
  'Explain risk scores',
  'LLM vs dedicated GPU pricing',
]

// ─── Panel ────────────────────────────────────────────────────────────────────

interface AIAssistantPanelProps {
  context?: {
    workload_type: string | null
    providers: string[]
  }
}

export function AIAssistantPanel({ context }: AIAssistantPanelProps) {
  const [messages, setMessages] = useState<AIMessage[]>([])
  const [loading, setLoading] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)

  const { register, handleSubmit, reset, formState: { errors } } = useForm<AIChatValues>({
    resolver: zodResolver(aiChatSchema),
  })

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages])

  async function onSubmit(values: AIChatValues) {
    const userMsg: AIMessage = { role: 'user', content: values.message, timestamp: Date.now() }
    setMessages((prev) => [...prev, userMsg])
    reset()
    setLoading(true)

    try {
      const res = await askAI({
        message: values.message,
        context: {
          workload_type: context?.workload_type ?? null,
          providers: context?.providers ?? [],
          recent_results: null,
        },
      })
      const assistantMsg: AIMessage = {
        role: 'assistant',
        content: res.reply,
        timestamp: Date.now(),
      }
      setMessages((prev) => [...prev, assistantMsg])
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: 'Sorry, something went wrong. Try again.', timestamp: Date.now() },
      ])
    } finally {
      setLoading(false)
    }
  }

  function sendStarter(prompt: string) {
    void onSubmit({ message: prompt })
  }

  const starters: [string, string, string] =
    context?.workload_type
      ? (WORKLOAD_STARTERS[context.workload_type] ?? DEFAULT_STARTERS)
      : DEFAULT_STARTERS

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div
        className="px-3 py-2.5 flex items-center gap-2"
        style={{ borderBottom: '1px solid var(--border-subtle)' }}
      >
        <div
          className="w-5 h-5 rounded-full flex items-center justify-center"
          style={{ background: 'rgba(124,92,252,0.15)', border: '1px solid rgba(124,92,252,0.35)' }}
        >
          <Bot className="w-3 h-3" style={{ color: 'var(--brand-hover)' }} />
        </div>
        <span className="text-xs font-semibold" style={{ color: 'var(--text-secondary)' }}>Ask IA</span>
        <span className="ml-auto micro-label">AI</span>
      </div>

      {/* Messages */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto p-3 space-y-3 min-h-0"
      >
        {messages.length === 0 && (
          <div className="space-y-3">
            <p className="text-[11px] leading-relaxed" style={{ color: 'var(--text-tertiary)' }}>
              Ask anything about inference pricing, risk scores, or provider comparisons.
            </p>
            <div className="space-y-1.5">
              {starters.map((p) => (
                <button
                  key={p}
                  onClick={() => sendStarter(p)}
                  className="w-full text-left text-[11px] px-2.5 py-1.5 rounded-md ai-starter-btn"
                >
                  {p}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((msg, i) => (
          <MessageBubble key={i} msg={msg} />
        ))}
        {loading && (
          <div className="flex gap-2 justify-start">
            <div
              className="flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center"
              style={{ background: 'rgba(124,92,252,0.15)', border: '1px solid rgba(124,92,252,0.35)' }}
            >
              <Bot className="w-3.5 h-3.5" style={{ color: 'var(--brand-hover)' }} />
            </div>
            <div className="rounded-lg px-3 py-2" style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-default)' }}>
              <Loader2 className="w-3.5 h-3.5 animate-spin" style={{ color: 'var(--text-disabled)' }} />
            </div>
          </div>
        )}
      </div>

      {/* Input */}
      <div className="p-3" style={{ borderTop: '1px solid var(--border-subtle)' }}>
        <form onSubmit={handleSubmit(onSubmit)} className="relative">
          <Textarea
            {...register('message')}
            placeholder="Ask about pricing…"
            className="min-h-[60px] max-h-[120px] text-xs pr-9 resize-none"
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
            className="absolute right-2 bottom-2 h-6 w-6 ai-send-btn"
          >
            <Send className="h-3 w-3" />
          </Button>
        </form>
        {errors.message && (
          <p className="text-[10px] text-red-400 mt-1">{errors.message.message}</p>
        )}
      </div>
    </div>
  )
}
