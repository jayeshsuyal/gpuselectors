import { createContext, useContext, useState, type ReactNode } from 'react'

interface AIContextState {
  workload_type: string | null
  providers: string[]
}

interface AIContextValue extends AIContextState {
  setAIContext: (ctx: AIContextState) => void
}

const AIContext = createContext<AIContextValue>({
  workload_type: null,
  providers: [],
  setAIContext: () => {},
})

export function AIContextProvider({ children }: { children: ReactNode }) {
  const [ctx, setCtx] = useState<AIContextState>({
    workload_type: null,
    providers: [],
  })

  return (
    <AIContext.Provider value={{ ...ctx, setAIContext: setCtx }}>
      {children}
    </AIContext.Provider>
  )
}

export function useAIContext() {
  return useContext(AIContext)
}
