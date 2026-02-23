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
import { ProviderPicker } from '@/components/ui/ProviderPicker'
import { nonLLMFormSchema, type NonLLMFormValues } from '@/schemas/forms'
import { UNIT_NAMES, WORKLOAD_UNIT_OPTIONS } from '@/lib/constants'

interface NonLLMFormProps {
  workloadType: string
  onSubmit: (values: NonLLMFormValues) => void
  loading: boolean
  initialValues?: Partial<NonLLMFormValues>
}

export function NonLLMForm({ workloadType, onSubmit, loading, initialValues }: NonLLMFormProps) {
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
      ...initialValues,
    },
  })

  const unitName = watch('unit_name')
  const budget = watch('monthly_budget_max_usd')
  const throughputAware = watch('throughput_aware')
  const selectedProviders = watch('provider_ids')

  // Only show units relevant to this workload type
  const workloadUnitIds = WORKLOAD_UNIT_OPTIONS[workloadType as keyof typeof WORKLOAD_UNIT_OPTIONS] ?? []
  const workloadUnits = UNIT_NAMES.filter((u) => workloadUnitIds.includes(u.id))

  // Warn when budget is set but no unit selected (can't filter without a unit)
  const showUnitWarning = budget > 0 && unitName === null

  const selectedUnitLabel = UNIT_NAMES.find((u) => u.id === unitName)?.label

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
                <SelectValue placeholder="All units for this workload" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="__all__">All units (browse mode)</SelectItem>
                {workloadUnits.map((u) => (
                  <SelectItem key={u.id} value={u.id}>{u.label}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        />
        <p className="text-[10px] text-zinc-500">
          {unitName === null
            ? 'Browse mode — shows all offers without normalizing prices across units.'
            : `Ranked by normalized ${selectedUnitLabel} price. Monthly estimate requires usage below.`}
        </p>
        {errors.unit_name && (
          <p className="text-[11px] text-red-400">{errors.unit_name.message}</p>
        )}
      </div>

      {/* Unit + budget mismatch warning */}
      {showUnitWarning && (
        <div className="flex gap-2 items-start rounded-md border border-amber-800 bg-amber-950/30 px-3 py-2.5">
          <AlertTriangle className="h-4 w-4 text-amber-400 flex-shrink-0 mt-0.5" />
          <div className="text-xs text-amber-300">
            <strong>Select a specific unit above</strong> to use budget filtering.
            Budget filtering requires a single comparable unit — it cannot apply across mixed units.
          </div>
        </div>
      )}

      {/* Monthly usage */}
      <div className="space-y-1.5">
        <Label htmlFor="monthly_usage">
          Monthly usage
          {selectedUnitLabel && (
            <span className="text-zinc-500 font-normal ml-1">({selectedUnitLabel})</span>
          )}
        </Label>
        <Input
          id="monthly_usage"
          type="number"
          {...register('monthly_usage', { valueAsNumber: true })}
          placeholder="1000"
        />
        <p className="text-[10px] text-zinc-500">
          Used to calculate the monthly cost estimate for each offer.
        </p>
        {errors.monthly_usage && (
          <p className="text-[11px] text-red-400">{errors.monthly_usage.message}</p>
        )}
      </div>

      {/* Budget */}
      <div className="space-y-1.5">
        <Label htmlFor="monthly_budget_max_usd">Monthly budget max <span className="text-zinc-500 font-normal">(USD)</span></Label>
        <Input
          id="monthly_budget_max_usd"
          type="number"
          step="0.01"
          {...register('monthly_budget_max_usd', { valueAsNumber: true })}
          placeholder="0 = no limit"
        />
        <p className="text-[10px] text-zinc-500">
          0 = no limit. Offers with estimated monthly cost above this are excluded.
          Requires a specific unit to be selected.
        </p>
      </div>

      {/* Providers */}
      <div className="space-y-1.5">
        <Label>Providers <span className="text-zinc-500 font-normal">(optional — all included if none selected)</span></Label>
        <Controller
          name="provider_ids"
          control={control}
          render={({ field }) => (
            <ProviderPicker
              value={field.value}
              onChange={field.onChange}
              helperText={
                selectedProviders.length === 0
                  ? 'All providers with data for this workload will be included.'
                  : `${selectedProviders.length} selected`
              }
            />
          )}
        />
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
            Factors in provider throughput limits when ranking. Providers without published
            throughput data will be ranked last or excluded (with strict mode).
          </p>
        </div>
      </div>

      {/* Advanced */}
      <div className="border rounded-lg overflow-hidden" style={{ borderColor: 'var(--border-subtle)' }}>
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          aria-expanded={showAdvanced}
          className="w-full flex items-center justify-between px-4 py-2.5 text-xs font-medium text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900 transition-colors"
        >
          Advanced parameters
          <ChevronDown className={cn('h-3.5 w-3.5 transition-transform', showAdvanced && 'rotate-180')} />
        </button>

        {showAdvanced && (
          <div className="px-4 py-3 space-y-3 border-t" style={{ borderColor: 'var(--border-subtle)', background: 'var(--bg-elevated)' }}>
            {/* Throughput params — only meaningful when throughput_aware is on */}
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
                {!throughputAware && (
                  <p className="text-[10px] text-zinc-600">Enable throughput-aware to use.</p>
                )}
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
                {!throughputAware && (
                  <p className="text-[10px] text-zinc-600">Enable throughput-aware to use.</p>
                )}
              </div>
            </div>

            {/* Comparator mode */}
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
                        aria-pressed={field.value === m}
                        onClick={() => field.onChange(m)}
                        className="rounded-md border px-3 py-1.5 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--brand)] focus-visible:ring-offset-1 focus-visible:ring-offset-[var(--bg-base)]"
                        style={field.value === m
                          ? { borderColor: 'var(--brand-border)', background: 'rgba(124,92,252,0.08)', color: 'var(--brand-hover)' }
                          : { borderColor: 'rgba(255,255,255,0.08)', background: 'var(--bg-elevated)', color: 'var(--text-tertiary)' }}
                      >
                        {m === 'normalized' ? 'Normalized price' : 'Listed price'}
                      </button>
                    ))}
                  </div>
                  <p className="text-[10px] text-zinc-600">
                    Normalized converts all units to a common base (e.g. per audio hour) before ranking.
                  </p>
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
              <div>
                <Label htmlFor="confidence_weighted" className="cursor-pointer">
                  Confidence-weighted ranking
                </Label>
                <p className="text-[10px] text-zinc-500 mt-0.5">
                  Penalizes estimated/low-confidence prices by up to 25% in the ranking score.
                </p>
              </div>
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
                    Exclude providers that don't publish throughput data. Without this,
                    they appear with an "unknown" capacity label.
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
