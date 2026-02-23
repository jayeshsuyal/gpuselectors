import { useMemo, useState } from 'react'
import { cn } from '@/lib/utils'

const PROVIDER_LABELS: Record<string, string> = {
  anthropic: 'Anthropic',
  assemblyai: 'AssemblyAI',
  aws: 'AWS',
  aws_rekognition: 'AWS Rekognition',
  baseten: 'Baseten',
  cohere: 'Cohere',
  deepgram: 'Deepgram',
  elevenlabs: 'ElevenLabs',
  fal_ai: 'fal',
  fireworks: 'Fireworks',
  google_cloud: 'Google Cloud',
  groq: 'Groq',
  modal: 'Modal',
  openai: 'OpenAI',
  replicate: 'Replicate',
  runpod: 'RunPod',
  together_ai: 'Together AI',
  voyage_ai: 'Voyage AI',
}

const PROVIDER_ALIASES: Record<string, string> = {
  'aws bedrock': 'aws',
  'google cloud': 'google_cloud',
  together: 'together_ai',
  voyage: 'voyage_ai',
}

function normalizeProviderToken(raw: string): string {
  const token = raw.trim().toLowerCase().replace(/[\s-]+/g, '_')
  return PROVIDER_ALIASES[token] ?? token
}

function initialsFromLabel(label: string): string {
  const words = label
    .replace(/[^a-z0-9 ]/gi, ' ')
    .split(/\s+/)
    .filter(Boolean)
  if (words.length === 0) return 'IA'
  if (words.length === 1) {
    return words[0].slice(0, 2).toUpperCase()
  }
  return (words[0][0] + words[1][0]).toUpperCase()
}

const FALLBACK_BG: string[] = [
  'bg-indigo-950/60 text-indigo-300 ring-indigo-700/40',
  'bg-sky-950/60 text-sky-300 ring-sky-700/40',
  'bg-emerald-950/60 text-emerald-300 ring-emerald-700/40',
  'bg-violet-950/60 text-violet-300 ring-violet-700/40',
  'bg-amber-950/60 text-amber-300 ring-amber-700/40',
]

function colorFromToken(token: string): string {
  let hash = 0
  for (let i = 0; i < token.length; i += 1) {
    hash = (hash * 31 + token.charCodeAt(i)) >>> 0
  }
  return FALLBACK_BG[hash % FALLBACK_BG.length]
}

type ProviderLogoSize = 'sm' | 'md' | 'lg'

const SIZE_CLASS: Record<ProviderLogoSize, string> = {
  sm: 'h-5 w-5 text-[9px]',
  md: 'h-6 w-6 text-[10px]',
  lg: 'h-7 w-7 text-[11px]',
}

interface ProviderLogoProps {
  provider: string
  size?: ProviderLogoSize
  className?: string
}

const missingLogoProviders = new Set<string>()

export function ProviderLogo({ provider, size = 'md', className }: ProviderLogoProps) {
  const providerKey = normalizeProviderToken(provider)
  const label = PROVIDER_LABELS[providerKey] ?? provider
  const initials = useMemo(() => initialsFromLabel(label), [label])
  const colorClass = useMemo(() => colorFromToken(providerKey), [providerKey])
  const [hasImage, setHasImage] = useState(!missingLogoProviders.has(providerKey))
  const imageSrc = `/logos/providers/${providerKey}.svg`

  if (hasImage) {
    return (
      <img
        src={imageSrc}
        alt={`${label} logo`}
        className={cn(
          'rounded-md object-contain ring-1 ring-white/[0.08] bg-zinc-900/80 p-0.5',
          SIZE_CLASS[size],
          className
        )}
        onError={() => {
          missingLogoProviders.add(providerKey)
          setHasImage(false)
        }}
        loading="lazy"
      />
    )
  }

  return (
    <div
      className={cn(
        'inline-flex items-center justify-center rounded-md ring-1 font-semibold uppercase tracking-tight',
        'ring-white/[0.08]',
        colorClass,
        SIZE_CLASS[size],
        className
      )}
      aria-label={label}
      title={label}
    >
      {initials}
    </div>
  )
}

export function providerDisplayName(provider: string): string {
  const key = normalizeProviderToken(provider)
  return PROVIDER_LABELS[key] ?? provider
}
