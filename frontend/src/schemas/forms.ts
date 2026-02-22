import { z } from 'zod'

// ─── LLM Form ─────────────────────────────────────────────────────────────────

export const llmFormSchema = z.object({
  tokens_per_day: z
    .number({ invalid_type_error: 'Required' })
    .positive('Must be greater than 0')
    .max(100_000_000_000, 'Value too large'),
  model_bucket: z.enum(['7b', '13b', '34b', '70b', '405b'], {
    required_error: 'Select a model size',
  }),
  provider_ids: z.array(z.string()).min(1, 'Select at least one provider'),
  traffic_pattern: z.enum(['steady', 'business_hours', 'bursty']).default('business_hours'),
  // Advanced
  peak_to_avg: z.number().min(1).max(10).default(2.5),
  util_target: z.number().min(0.1).max(0.99).default(0.75),
  beta: z.number().min(0).max(1).default(0.08),
  alpha: z.number().min(0).max(5).default(1.0),
  autoscale_inefficiency: z.number().min(1).max(2).default(1.15),
  monthly_budget_max_usd: z.number().min(0).default(0),
  output_token_ratio: z.number().min(0).max(1).default(0.3),
  top_k: z.number().int().min(1).max(20).default(5),
})

export type LLMFormValues = z.infer<typeof llmFormSchema>

// ─── Non-LLM / Catalog Form ───────────────────────────────────────────────────

export const nonLLMFormSchema = z
  .object({
    workload_type: z.string().min(1),
    provider_ids: z.array(z.string()).default([]),
    unit_name: z.string().nullable().default(null),
    monthly_usage: z
      .number({ invalid_type_error: 'Required' })
      .positive('Must be positive')
      .default(1000),
    monthly_budget_max_usd: z.number().min(0).default(0),
    top_k: z.number().int().min(1).max(20).default(5),
    confidence_weighted: z.boolean().default(true),
    comparator_mode: z.enum(['normalized', 'listed']).default('normalized'),
    // Throughput
    throughput_aware: z.boolean().default(false),
    peak_to_avg: z.number().min(1).max(10).default(2.5),
    util_target: z.number().min(0.1).max(0.99).default(0.75),
    strict_capacity_check: z.boolean().default(false),
  })
  .refine(
    (data) => {
      // If monthly_usage or budget is set, unit must be specified
      if (data.monthly_budget_max_usd > 0 && data.unit_name === null) return false
      return true
    },
    {
      message: 'Select a specific unit when setting a budget',
      path: ['unit_name'],
    }
  )

export type NonLLMFormValues = z.infer<typeof nonLLMFormSchema>

// ─── AI Chat ──────────────────────────────────────────────────────────────────

export const aiChatSchema = z.object({
  message: z.string().min(1, 'Type a message').max(500, 'Message too long'),
})

export type AIChatValues = z.infer<typeof aiChatSchema>

// ─── Catalog Filters ──────────────────────────────────────────────────────────

export const catalogFilterSchema = z.object({
  workload_type: z.string().default(''),
  provider: z.string().default(''),
  model_name: z.string().default(''),
  unit_name: z.string().default(''),
  search: z.string().default(''),
})

export type CatalogFilterValues = z.infer<typeof catalogFilterSchema>
