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

// Crystalline conic-gradient palettes — each provider gets a unique gem-like fingerprint
const GRADIENT_PALETTES = [
  // Indigo gem — brand-adjacent deep violet
  {
    bg: 'conic-gradient(from 120deg at 25% 30%, #1e1b4b, #312e81, #0f0e1f, #1e1b4b)',
    accent: '#818cf8',
    ring: 'rgba(99,102,241,0.38)',
  },
  // Teal gem — electric deep sea
  {
    bg: 'conic-gradient(from 200deg at 70% 25%, #0c2435, #134e4a, #071218, #0c2435)',
    accent: '#2dd4bf',
    ring: 'rgba(20,184,166,0.32)',
  },
  // Amber gem — molten gold
  {
    bg: 'conic-gradient(from 45deg at 35% 65%, #2c1810, #78350f, #180d06, #2c1810)',
    accent: '#fbbf24',
    ring: 'rgba(245,158,11,0.34)',
  },
  // Violet gem — warm deep purple
  {
    bg: 'conic-gradient(from 300deg at 60% 40%, #1a0535, #4c1d95, #100220, #1a0535)',
    accent: '#c084fc',
    ring: 'rgba(168,85,247,0.34)',
  },
  // Emerald gem — forest crystal
  {
    bg: 'conic-gradient(from 160deg at 40% 20%, #0a1f0f, #14532d, #051008, #0a1f0f)',
    accent: '#4ade80',
    ring: 'rgba(34,197,94,0.30)',
  },
  // Rose gem — deep crimson facet
  {
    bg: 'conic-gradient(from 270deg at 50% 70%, #200710, #881337, #14030a, #200710)',
    accent: '#fb7185',
    ring: 'rgba(244,63,94,0.30)',
  },
] as const

type GradientPalette = (typeof GRADIENT_PALETTES)[number]

function paletteFromToken(token: string): GradientPalette {
  let hash = 0
  for (let i = 0; i < token.length; i += 1) {
    hash = (hash * 31 + token.charCodeAt(i)) >>> 0
  }
  return GRADIENT_PALETTES[hash % GRADIENT_PALETTES.length]
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
  const palette = useMemo(() => paletteFromToken(providerKey), [providerKey])
  const [hasImage, setHasImage] = useState(!missingLogoProviders.has(providerKey))
  const imageSrc = `/logos/providers/${providerKey}.svg`

  if (hasImage) {
    return (
      <img
        src={imageSrc}
        alt={`${label} logo`}
        className={cn('rounded-md object-contain p-0.5', SIZE_CLASS[size], className)}
        style={{
          background: 'rgba(13,13,19,0.9)',
          boxShadow:
            '0 0 0 1px rgba(255,255,255,0.09), inset 0 1px 0 rgba(255,255,255,0.06), 0 2px 6px rgba(0,0,0,0.45)',
        }}
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
        'inline-flex items-center justify-center rounded-md font-semibold uppercase',
        SIZE_CLASS[size],
        className
      )}
      style={{
        background: palette.bg,
        color: palette.accent,
        boxShadow: `0 0 0 1px ${palette.ring}, inset 0 1px 0 rgba(255,255,255,0.10), 0 2px 6px rgba(0,0,0,0.50)`,
        letterSpacing: '0.02em',
        textShadow: '0 1px 2px rgba(0,0,0,0.5)',
      }}
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
