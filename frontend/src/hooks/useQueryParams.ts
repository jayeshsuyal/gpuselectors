import { useCallback, useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'

type ParamValue = string | number | boolean | null | undefined

/**
 * Typed URL query param hook.
 * Syncs state to URL so links are shareable.
 */
export function useQueryParams<T extends Record<string, ParamValue>>(defaults: T) {
  const [searchParams, setSearchParams] = useSearchParams()

  const params = useMemo(() => {
    const result = { ...defaults } as T
    for (const key of Object.keys(defaults)) {
      const raw = searchParams.get(key)
      if (raw === null) continue

      const defaultVal = defaults[key as keyof T]
      if (typeof defaultVal === 'number') {
        const n = Number(raw)
        if (!isNaN(n)) (result as Record<string, ParamValue>)[key] = n
      } else if (typeof defaultVal === 'boolean') {
        ;(result as Record<string, ParamValue>)[key] = raw === 'true'
      } else {
        ;(result as Record<string, ParamValue>)[key] = raw
      }
    }
    return result
  }, [searchParams, defaults])

  const setParams = useCallback(
    (updates: Partial<T>) => {
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev)
          for (const [key, value] of Object.entries(updates)) {
            if (value === null || value === undefined || value === defaults[key]) {
              next.delete(key)
            } else {
              next.set(key, String(value))
            }
          }
          return next
        },
        { replace: true }
      )
    },
    [setSearchParams, defaults]
  )

  return [params, setParams] as const
}
