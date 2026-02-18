export const WORKLOAD_TYPES = [
  {
    id: 'llm',
    label: 'LLM Inference',
    description: 'Chat, completion, reasoning models',
    icon: 'Brain',
    color: 'indigo',
  },
  {
    id: 'speech_to_text',
    label: 'Speech to Text',
    description: 'Transcription, real-time STT',
    icon: 'Mic',
    color: 'emerald',
  },
  {
    id: 'text_to_speech',
    label: 'Text to Speech',
    description: 'Voice synthesis, narration',
    icon: 'Volume2',
    color: 'sky',
  },
  {
    id: 'embeddings',
    label: 'Embeddings',
    description: 'Vector search, semantic similarity',
    icon: 'Layers',
    color: 'violet',
  },
  {
    id: 'vision',
    label: 'Vision',
    description: 'Image understanding, OCR, captioning',
    icon: 'Eye',
    color: 'amber',
  },
  {
    id: 'image_generation',
    label: 'Image Generation',
    description: 'Text-to-image, inpainting',
    icon: 'ImageIcon',
    color: 'rose',
  },
  {
    id: 'video_generation',
    label: 'Video Generation',
    description: 'Text-to-video, frame interpolation',
    icon: 'Film',
    color: 'orange',
  },
  {
    id: 'moderation',
    label: 'Moderation',
    description: 'Content filtering, safety classification',
    icon: 'Shield',
    color: 'teal',
  },
] as const

export type WorkloadTypeId = (typeof WORKLOAD_TYPES)[number]['id']

export const MODEL_BUCKETS = [
  { id: '7b', label: '7B params', description: 'Mistral 7B, Llama 3.1 8B' },
  { id: '13b', label: '13B params', description: 'Llama 2 13B, CodeLlama 13B' },
  { id: '34b', label: '34B params', description: 'CodeLlama 34B, Yi 34B' },
  { id: '70b', label: '70B params', description: 'Llama 3.1 70B, Mixtral 8x7B' },
  { id: '405b', label: '405B params', description: 'Llama 3.1 405B, frontier scale' },
] as const

export const TRAFFIC_PATTERNS = [
  { id: 'steady', label: 'Steady', description: 'Flat load 24/7', peakToAvg: 1.5 },
  { id: 'business_hours', label: 'Business Hours', description: '9-5 weekday spikes', peakToAvg: 2.5 },
  { id: 'bursty', label: 'Bursty', description: 'Heavy spikes, low baseline', peakToAvg: 4.0 },
] as const

export const PROVIDERS = [
  'anthropic', 'openai', 'google', 'aws', 'azure', 'fireworks',
  'together', 'groq', 'deepgram', 'assemblyai', 'elevenlabs',
  'stability', 'replicate', 'cohere', 'mistral', 'deepinfra',
] as const

export const UNIT_NAMES = [
  { id: 'audio_min', label: 'Audio Minute' },
  { id: 'audio_hour', label: 'Audio Hour' },
  { id: '1k_chars', label: '1K Characters' },
  { id: '1m_chars', label: '1M Characters' },
  { id: 'image', label: 'Per Image' },
  { id: '1k_tokens', label: '1K Tokens' },
  { id: '1m_tokens', label: '1M Tokens' },
  { id: 'request', label: 'Per Request' },
  { id: 'second', label: 'Per Second' },
  { id: 'minute', label: 'Per Minute' },
  { id: 'hour', label: 'Per Hour' },
] as const

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? ''
