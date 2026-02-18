import { useForm, Controller } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { AlertTriangle, ChevronDown } from 'lucide-react'
import { useState } from 'react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Checkbox } from '@/components/ui/checkbox'
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from '@/components/ui/select'
import { nonLLMFormSchema, type NonLLMFormValues } from '@/schemas/forms'
import { PROVIDERS, UNIT_NAMES } from '@/lib/constants'

interface NonLLMFormProps {
  workloadType: string
  onSubmit: (values: NonLLMFormValues) => void
  loading: boolean
}

export function NonLLMForm({ workloadType, onSubmit, loading }: NonLLMFormProps) {
  const [showAdvanced, setShowAdvanced] = useState(false)

  const {
    register,
    handleSubmit,
    control,
    watch,
    formState: { errors },
  } = useForm<NonLLMFormValues>({
    resolver: zodResolver(nonLLMFormSchema),
    defaultValues: {
      workload_type: workloadType,
      provider_ids: [],
      unit_name: null,
      monthly_usage: 1000,
      monthly_budget_max_usd: 0,
      top_k: 5,
      confidence_weighted: true,
      comparator_mode: 'normalized',
      throughput_aware: false,
      peak_to_avg: 2.5,
      util_target: 0.75,
      strict_capacity_check: false,
    },
  })

  const unitName = watch('unit_name')
  const budget = watch('monthly_budget_max_usd')
  const throughputAware = watch('throughput_aware')
  const selectedProviders = watch('provider_ids')

  // Warn: budget set but unit is null
  const showUnitWarning = budget > 0 && unitName === null

  function toggleProvider(id: string, current: string[], onChange: (v: string[]) => void) {
    if (current.includes(id)) {
      onChange(current.filter((p) => p !== id))
    } else {
      onChange([...current, id])
    }
  }

  return (
    <form
      onSubmit={handleSubmit((v) => onSubmit({ ...v, workload_type: workloadType }))}
      className="space-y-5"
    >
      {/* Unit */}
      <div className="space-y-1.5">
        <Label>Pricing unit</Label>
        <Controller
          name="unit_name"
          control={control}
          render={({ field }) => (
            <Select
              value={field.value ?? '__all__'}
              onValueChange={(v) => field.onChange(v === '__all__' ? null : v)}
            >
              <SelectTrigger>
                <SelectValue placeholder="All units (browse mode)" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="__all__">All units (browse mode)</SelectItem>
                {UNIT_NAMES.map((u) => (
                  <SelectItem key={u.id} value={u.id}>{u.label}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        />
        {errors.unit_name && (
          <p className="text-[11px] text-red-400">{errors.unit_name.message}</p>
        )}
      </div>

      {/* Unit warning */}
      {showUnitWarning && (
        <div className="flex gap-2 items-start rounded-md border border-amber-800 bg-amber-950/30 px-3 py-2.5">
          <AlertTriangle className="h-4 w-4 text-amber-400 flex-shrink-0 mt-0.5" />
          <div className="text-xs text-amber-300">
            <strong>Select a specific unit</strong> to use monthly budget filtering.
            Budget cannot be applied when "All units" is selected.
          </div>
        </div>
      )}

      {/* Monthly usage */}
      <div className="space-y-1.5">
        <Label htmlFor="monthly_usage">
          Monthly usage
          {unitName ? (
            <span className="text-zinc-500 font-normal ml-1">
              ({UNIT_NAMES.find((u) => u.id === unitName)?.label ?? unitName})
            </span>
          ) : null}
        </Label>
        <Input
          id="monthly_usage"
          type="number"
          {...register('monthly_usage', { valueAsNumber: true })}
          placeholder="1000"
        />
        {errors.monthly_usage && (
          <p className="text-[11px] text-red-400">{errors.monthly_usage.message}</p>
        )}
      </div>

      {/* Budget */}
      <div className="space-y-1.5">
        <Label htmlFor="monthly_budget_max_usd">Monthly budget max (USD)</Label>
        <Input
          id="monthly_budget_max_usd"
          type="number"
          step="0.01"
          {...register('monthly_budget_max_usd', { valueAsNumber: true })}
          placeholder="0 = no limit"
        />
      </div>

      {/* Providers */}
      <div className="space-y-1.5">
        <Label>Providers <span className="text-zinc-500 font-normal">(optional)</span></Label>
        <Controller
          name="provider_ids"
          control={control}
          render={({ field }) => (
            <div className="flex flex-wrap gap-1.5">
              {PROVIDERS.map((p) => {
                const isSelected = field.value.includes(p)
                return (
                  <button
                    key={p}
                    type="button"
                    onClick={() => toggleProvider(p, field.value, field.onChange)}
                    className={cn(
                      'rounded-full px-2.5 py-1 text-[11px] font-medium border transition-colors',
                      isSelected
                        ? 'border-indigo-600 bg-indigo-950/60 text-indigo-300'
                        : 'border-zinc-700 bg-zinc-900 text-zinc-500 hover:border-zinc-600 hover:text-zinc-400'
                    )}
                  >
                    {p}
                  </button>
                )
              })}
            </div>
          )}
        />
        <p className="text-[10px] text-zinc-500">
          {selectedProviders.length === 0 ? 'All providers' : `${selectedProviders.length} selected`}
        </p>
      </div>

      {/* Throughput toggle */}
      <div className="flex items-start gap-3">
        <Controller
          name="throughput_aware"
          control={control}
          render={({ field }) => (
            <Checkbox
              id="throughput_aware"
              checked={field.value}
              onCheckedChange={field.onChange}
            />
          )}
        />
        <div>
          <Label htmlFor="throughput_aware" className="cursor-pointer">Throughput-aware ranking</Label>
          <p className="text-[10px] text-zinc-500 mt-0.5">
            Account for provider throughput limits when ranking. Requires providers to publish throughput data.
          </p>
        </div>
      </div>

      {/* Advanced */}
      <div className="border border-zinc-800 rounded-lg overflow-hidden">
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="w-full flex items-center justify-between px-4 py-2.5 text-xs font-medium text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900 transition-colors"
        >
          Advanced parameters
          <ChevronDown className={cn('h-3.5 w-3.5 transition-transform', showAdvanced && 'rotate-180')} />
        </button>

        {showAdvanced && (
          <div className="px-4 py-3 space-y-3 border-t border-zinc-800 bg-zinc-900/50">
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1">
                <Label htmlFor="adv_peak_to_avg">Peak-to-avg ratio</Label>
                <Input
                  id="adv_peak_to_avg"
                  type="number"
                  step="0.1"
                  disabled={!throughputAware}
                  {...register('peak_to_avg', { valueAsNumber: true })}
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="adv_util">Utilization target</Label>
                <Input
                  id="adv_util"
                  type="number"
                  step="0.05"
                  disabled={!throughputAware}
                  {...register('util_target', { valueAsNumber: true })}
                />
              </div>
            </div>

            <Controller
              name="comparator_mode"
              control={control}
              render={({ field }) => (
                <div className="space-y-1.5">
                  <Label>Comparator mode</Label>
                  <div className="grid grid-cols-2 gap-2">
                    {(['normalized', 'listed'] as const).map((m) => (
                      <button
                        key={m}
                        type="button"
                        onClick={() => field.onChange(m)}
                        className={cn(
                          'rounded-md border px-3 py-1.5 text-xs font-medium transition-colors',
                          field.value === m
                            ? 'border-indigo-500 bg-indigo-950/60 text-indigo-300'
                            : 'border-zinc-700 bg-zinc-900 text-zinc-400'
                        )}
                      >
                        {m === 'normalized' ? 'Normalized' : 'Listed price'}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            />

            <div className="flex items-center gap-3">
              <Controller
                name="confidence_weighted"
                control={control}
                render={({ field }) => (
                  <Checkbox
                    id="confidence_weighted"
                    checked={field.value}
                    onCheckedChange={field.onChange}
                  />
                )}
              />
              <Label htmlFor="confidence_weighted" className="cursor-pointer">
                Confidence-weighted ranking
              </Label>
            </div>

            {throughputAware && (
              <div className="flex items-center gap-3">
                <Controller
                  name="strict_capacity_check"
                  control={control}
                  render={({ field }) => (
                    <Checkbox
                      id="strict_capacity_check"
                      checked={field.value}
                      onCheckedChange={field.onChange}
                    />
                  )}
                />
                <div>
                  <Label htmlFor="strict_capacity_check" className="cursor-pointer">
                    Strict capacity check
                  </Label>
                  <p className="text-[10px] text-zinc-500 mt-0.5">
                    Exclude providers with unknown throughput
                  </p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      <Button type="submit" className="w-full" disabled={loading}>
        {loading ? 'Ranking offers…' : 'Compare pricing'}
      </Button>
    </form>
  )
}
