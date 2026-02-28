import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { AuditResultCard } from '../AuditResultCard'
import type { CostAuditResponse } from '@/services/types'

// ─── Minimal valid response fixture ───────────────────────────────────────────

const BASE_RESPONSE: CostAuditResponse = {
  efficiency_score: 72,
  recommendations: [],
  hardware_recommendation: {
    tier: 'single_gpu',
    gpu_family: 'H100-class',
    deployment_shape: 'dedicated_gpu',
    reasoning: 'Test reasoning.',
  },
  pricing_model_verdict: {
    current_model: 'mixed',
    verdict: 'appropriate',
    reason: 'Mixed pipeline is well-optimised.',
  },
  red_flags: [],
  estimated_monthly_savings: { low_usd: 100, high_usd: 500, basis: 'test' },
  assumptions: [],
  data_gaps: [],
  data_gaps_detailed: [],
}

// ─── Tests ────────────────────────────────────────────────────────────────────

describe('AuditResultCard — per_modality_audits', () => {
  it('renders per-modality table when legs are present', () => {
    render(
      <AuditResultCard
        data={{
          ...BASE_RESPONSE,
          per_modality_audits: [
            { modality: 'llm',       efficiency_score: 68, top_recommendation: 'Enable quantization.', red_flags: [] },
            { modality: 'embeddings', efficiency_score: 81, top_recommendation: null, red_flags: [] },
            { modality: 'asr', efficiency_score: 55, top_recommendation: 'Batch requests.', red_flags: ['Idle GPU hours detected.'] },
          ],
        }}
      />
    )

    // Section heading
    expect(screen.getByText('Per-Modality Breakdown')).toBeInTheDocument()

    // Modality labels
    expect(screen.getByText('LLM')).toBeInTheDocument()
    expect(screen.getByText('Embeddings')).toBeInTheDocument()
    expect(screen.getByText('Speech-to-Text')).toBeInTheDocument()

    // Scores rendered as badges
    expect(screen.getByText('68')).toBeInTheDocument()
    expect(screen.getByText('81')).toBeInTheDocument()
    expect(screen.getByText('55')).toBeInTheDocument()

    // Top recommendation text
    expect(screen.getByText('Enable quantization.')).toBeInTheDocument()
    expect(screen.getByText('Batch requests.')).toBeInTheDocument()

    // Flag count for ASR leg
    expect(screen.getByText('1 flag')).toBeInTheDocument()
  })

  it('does not render per-modality section when legs are absent', () => {
    render(<AuditResultCard data={BASE_RESPONSE} />)
    expect(screen.queryByText('Per-Modality Breakdown')).not.toBeInTheDocument()
  })
})
