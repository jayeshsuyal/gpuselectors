import { useState } from 'react'
import { useForm, Controller } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { ChevronDown, Info } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { ProviderPicker } from '@/components/ui/ProviderPicker'
import { llmFormSchema, type LLMFormValues } from '@/schemas/forms'
import { MODEL_BUCKETS, PROVIDERS, TRAFFIC_PATTERNS } from '@/lib/constants'

interface LLMFormProps {
  onSubmit: (values: LLMFormValues) => void
  loading: boolean
}

export function LLMForm({ onSubmit, loading }: LLMFormProps) {
  const [showAdvanced, setShowAdvanced] = useState(false)

  const {
    register,
    handleSubmit,
    control,
    watch,
    formState: { errors },
  } = useForm<LLMFormValues>({
    resolver: zodResolver(llmFormSchema),
    defaultValues: {
      tokens_per_day: 5_000_000,
      model_bucket: '70b',
      provider_ids: ['fireworks', 'together_ai', 'openai', 'anthropic'],
      traffic_pattern: 'business_hours',
      peak_to_avg: 2.5,
      util_target: 0.75,
      beta: 0.08,
      alpha: 1.0,
      autoscale_inefficiency: 1.15,
      monthly_budget_max_usd: 0,
      output_token_ratio: 0.3,
      top_k: 5,
    },
  })

  const trafficPattern = watch('traffic_pattern')
  const patternInfo = TRAFFIC_PATTERNS.find((p) => p.id === trafficPattern)
  const selectedBucket = watch('model_bucket')

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="space-y-5">
      {/* Tokens per day */}
      <div className="space-y-1.5">
        <Label htmlFor="tokens_per_day">Daily token volume</Label>
        <Input
          id="tokens_per_day"
          type="number"
          {...register('tokens_per_day', { valueAsNumber: true })}
          placeholder="5000000"
        />
        <p className="text-[10px] text-zinc-500">
          Total input + output tokens per day. 1M tokens/day ≈ 35 active users.
        </p>
        {errors.tokens_per_day && (
          <p className="text-[11px] text-red-400">{errors.tokens_per_day.message}</p>
        )}
      </div>

      {/* Model bucket */}
      <div className="space-y-1.5">
        <Label>Model size</Label>
        <Controller
          name="model_bucket"
          control={control}
          render={({ field }) => (
            <div className="grid grid-cols-5 gap-1.5">
              {MODEL_BUCKETS.map((b) => (
                <button
                  key={b.id}
                  type="button"
                  onClick={() => field.onChange(b.id)}
                  className={cn(
                    'rounded-md border px-2 py-2 text-xs font-medium text-center transition-colors',
                    field.value === b.id
                      ? 'border-indigo-500 bg-indigo-950/60 text-indigo-300'
                      : 'border-zinc-700 bg-zinc-900 text-zinc-400 hover:border-zinc-600 hover:text-zinc-300'
                  )}
                >
                  {b.id.toUpperCase()}
                </button>
              ))}
            </div>
          )}
        />
        <p className="text-[10px] text-zinc-500">
          {MODEL_BUCKETS.find((b) => b.id === selectedBucket)?.description ?? 'Select a model size above'}
        </p>
        {errors.model_bucket && (
          <p className="text-[11px] text-red-400">{errors.model_bucket.message}</p>
        )}
      </div>

      {/* Traffic pattern */}
      <div className="space-y-1.5">
        <Label>Traffic pattern</Label>
        <Controller
          name="traffic_pattern"
          control={control}
          render={({ field }) => (
            <div className="grid grid-cols-3 gap-1.5">
              {TRAFFIC_PATTERNS.map((p) => (
                <button
                  key={p.id}
                  type="button"
                  onClick={() => field.onChange(p.id)}
                  className={cn(
                    'rounded-md border px-2 py-2 text-xs font-medium text-center transition-colors',
                    field.value === p.id
                      ? 'border-indigo-500 bg-indigo-950/60 text-indigo-300'
                      : 'border-zinc-700 bg-zinc-900 text-zinc-400 hover:border-zinc-600 hover:text-zinc-300'
                  )}
                >
                  {p.label}
                </button>
              ))}
            </div>
          )}
        />
        {patternInfo && (
          <p className="text-[10px] text-zinc-500">
            {patternInfo.description} · sets peak-to-avg to {patternInfo.peakToAvg}×
          </p>
        )}
      </div>

      {/* Providers */}
      <div className="space-y-1.5">
        <Label>Providers <span className="text-zinc-500 font-normal">(select at least one)</span></Label>
        <Controller
          name="provider_ids"
          control={control}
          render={({ field }) => (
            <ProviderPicker
              value={field.value}
              onChange={field.onChange}
              allowedProviders={PROVIDERS}
              helperText={
                field.value.length === 0
                  ? 'No providers selected — select at least one to rank'
                  : `${field.value.length} selected · only providers with catalog data for this model size will appear in results`
              }
            />
          )}
        />
        {errors.provider_ids && (
          <p className="text-[11px] text-red-400">{errors.provider_ids.message}</p>
        )}
      </div>

      {/* Advanced */}
      <div className="border border-zinc-800 rounded-lg overflow-hidden">
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          aria-expanded={showAdvanced}
          className="w-full flex items-center justify-between px-4 py-2.5 text-xs font-medium text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900 transition-colors"
        >
          <div className="flex items-center gap-2">
            <Info className="h-3.5 w-3.5" />
            Advanced parameters
          </div>
          <ChevronDown className={cn('h-3.5 w-3.5 transition-transform', showAdvanced && 'rotate-180')} />
        </button>

        {showAdvanced && (
          <div className="px-4 py-3 space-y-3 border-t border-zinc-800 bg-zinc-900/50">
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1">
                <Label htmlFor="peak_to_avg">Peak-to-avg ratio</Label>
                <Input id="peak_to_avg" type="number" step="0.1" {...register('peak_to_avg', { valueAsNumber: true })} />
                <p className="text-[10px] text-zinc-600">Ratio of peak to average load. Drives replica count.</p>
              </div>
              <div className="space-y-1">
                <Label htmlFor="util_target">Utilization target</Label>
                <Input id="util_target" type="number" step="0.05" {...register('util_target', { valueAsNumber: true })} />
                <p className="text-[10px] text-zinc-600">Max GPU utilization before scaling out (0–0.99).</p>
              </div>
              <div className="space-y-1">
                <Label htmlFor="output_token_ratio">Output token ratio</Label>
                <Input id="output_token_ratio" type="number" step="0.05" {...register('output_token_ratio', { valueAsNumber: true })} />
                <p className="text-[10px] text-zinc-600">Share of tokens that are outputs. Affects blended price.</p>
              </div>
              <div className="space-y-1">
                <Label htmlFor="beta">Scaling beta</Label>
                <Input id="beta" type="number" step="0.01" {...register('beta', { valueAsNumber: true })} />
                <p className="text-[10px] text-zinc-600">Risk penalty coefficient for overload probability.</p>
              </div>
              <div className="space-y-1">
                <Label htmlFor="alpha">Risk alpha</Label>
                <Input id="alpha" type="number" step="0.1" {...register('alpha', { valueAsNumber: true })} />
                <p className="text-[10px] text-zinc-600">Ops complexity weight in the composite score.</p>
              </div>
              <div className="space-y-1">
                <Label htmlFor="autoscale_inefficiency">Autoscale overhead</Label>
                <Input id="autoscale_inefficiency" type="number" step="0.01" {...register('autoscale_inefficiency', { valueAsNumber: true })} />
                <p className="text-[10px] text-zinc-600">Cost multiplier for autoscale billing inefficiency (e.g. 1.15 = +15%).</p>
              </div>
              <div className="space-y-1">
                <Label htmlFor="monthly_budget_max_usd">Monthly budget max</Label>
                <Input id="monthly_budget_max_usd" type="number" step="1" {...register('monthly_budget_max_usd', { valueAsNumber: true })} />
                <p className="text-[10px] text-zinc-600">0 = no limit. Plans above this cost are excluded.</p>
              </div>
              <div className="space-y-1">
                <Label htmlFor="top_k">Top K results</Label>
                <Input id="top_k" type="number" {...register('top_k', { valueAsNumber: true })} />
                <p className="text-[10px] text-zinc-600">Maximum number of plans to return (1–20).</p>
              </div>
            </div>
          </div>
        )}
      </div>

      <Button type="submit" className="w-full" disabled={loading}>
        {loading ? 'Ranking providers…' : 'Optimize workload'}
      </Button>
    </form>
  )
}
