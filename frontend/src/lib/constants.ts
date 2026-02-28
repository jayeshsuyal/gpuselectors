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

// Demo mode: keep the landing flow tightly scoped to LLM path.
export const DEMO_WORKLOAD_IDS: WorkloadTypeId[] = ['llm', 'speech_to_text']

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
  'anthropic',
  'assemblyai',
  'aws_rekognition',
  'baseten',
  'cohere',
  'deepgram',
  'elevenlabs',
  'fal_ai',
  'fireworks',
  'google_cloud',
  'modal',
  'openai',
  'replicate',
  'runpod',
  'together_ai',
  'voyage_ai',
] as const

export const UNIT_NAMES = [
  { id: 'audio_min', label: 'Audio Minute' },
  { id: 'audio_hour', label: 'Audio Hour' },
  { id: '1k_chars', label: '1K Characters' },
  { id: '1m_chars', label: '1M Characters' },
  { id: 'image', label: 'Per Image' },
  { id: '1k_images', label: '1K Images' },
  { id: '1k_tokens', label: '1K Tokens' },
  { id: '1m_tokens', label: '1M Tokens' },
  { id: 'request', label: 'Per Request' },
  { id: 'second', label: 'Per Second' },
  { id: 'minute', label: 'Per Minute' },
  { id: 'hour', label: 'Per Hour' },
  { id: 'per_minute', label: 'Per Minute (Alt)' },
  { id: 'per_second', label: 'Per Second (Alt)' },
  { id: 'per_image', label: 'Per Image (Alt)' },
  { id: 'megapixel', label: 'Per Megapixel' },
  { id: 'gpu_hour', label: 'GPU Hour' },
  { id: 'gpu_second', label: 'GPU Second' },
  { id: 'free', label: 'Free' },
  { id: 'video', label: 'Video Unit' },
  { id: 'video_min', label: 'Video Minute' },
  { id: 'generation', label: 'Generation' },
] as const

export const WORKLOAD_UNIT_OPTIONS: Record<WorkloadTypeId, string[]> = {
  llm: ['1m_tokens', 'gpu_hour', 'gpu_second'],
  speech_to_text: ['audio_min', 'audio_hour', 'per_minute'],
  text_to_speech: ['1k_chars', '1m_chars', 'audio_min', 'generation', 'per_minute'],
  embeddings: ['1m_tokens'],
  vision: ['1k_images', '1m_tokens', 'request'],
  image_generation: ['image', 'per_image', 'megapixel', 'gpu_hour', 'gpu_second'],
  video_generation: ['video_min', 'per_second', 'video'],
  moderation: ['1m_tokens', 'free'],
}

export const ASSUMPTION_LABELS: Record<string, string> = {
  peak_to_avg: 'Peak-to-avg',
  util_target: 'Util target',
  scaling_beta: 'Beta',
  alpha: 'Alpha',
  output_token_ratio: 'Output ratio',
  replicas: 'Replicas',
}

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? ''
