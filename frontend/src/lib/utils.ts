import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatUSD(value: number, decimals = 2): string {
  if (value === 0) return '$0'
  if (value < 0.01) return `$${value.toFixed(6)}`
  if (value < 1) return `$${value.toFixed(4)}`
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value)
}

export function formatNumber(value: number, decimals = 0): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`
  return value.toFixed(decimals)
}

export function formatPercent(value: number, decimals = 0): string {
  return `${(value * 100).toFixed(decimals)}%`
}

export function riskLevel(totalRisk: number): 'low' | 'medium' | 'high' {
  if (totalRisk < 0.3) return 'low'
  if (totalRisk < 0.6) return 'medium'
  return 'high'
}

export function confidenceLabel(confidence: string): string {
  const map: Record<string, string> = {
    high: 'High',
    official: 'Official',
    medium: 'Medium',
    estimated: 'Estimated',
    low: 'Low',
    vendor_list: 'Vendor List',
  }
  return map[confidence] ?? confidence
}

export function billingModeLabel(mode: string): string {
  const map: Record<string, string> = {
    per_token: 'Per Token',
    dedicated_hourly: 'Dedicated',
    autoscale_hourly: 'Autoscale',
    per_unit: 'Per Unit',
    hourly: 'Hourly',
    per_image: 'Per Image',
    per_minute: 'Per Minute',
    per_character: 'Per Character',
  }
  return map[mode] ?? mode
}

export function workloadDisplayName(workloadType: string): string {
  const map: Record<string, string> = {
    llm: 'LLM Inference',
    speech_to_text: 'Speech to Text',
    text_to_speech: 'Text to Speech',
    embeddings: 'Embeddings',
    vision: 'Vision',
    image_generation: 'Image Generation',
    video_generation: 'Video Generation',
    moderation: 'Moderation',
  }
  return map[workloadType] ?? workloadType
}

export function capacityLabel(check: string): string {
  const map: Record<string, string> = {
    ok: 'OK',
    insufficient: 'Insufficient',
    unknown: 'Unknown',
  }
  return map[check] ?? check
}
