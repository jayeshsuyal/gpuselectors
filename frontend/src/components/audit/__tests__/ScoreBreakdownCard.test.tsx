import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ScoreBreakdownCard } from '../ScoreBreakdownCard'
import type { CostAuditScoreBreakdown } from '@/services/types'

const FULL_BREAKDOWN: CostAuditScoreBreakdown = {
  base_score: 100,
  penalty_points: 32,
  bonus_points: 4,
  pre_cap_score: 72,
  post_cap_score: 72,
  major_flags: 2,
  caps_applied: ['major_flags_cap', 'high_switch_savings_cap'],
  combined_savings_pct: 35.5,
}

describe('ScoreBreakdownCard', () => {
  it('shows fallback when breakdown is undefined', () => {
    render(<ScoreBreakdownCard breakdown={undefined} />)
    expect(screen.getByText('Score breakdown unavailable for this run.')).toBeInTheDocument()
    expect(screen.queryByText('Show details')).not.toBeInTheDocument()
  })

  it('renders summary chips when breakdown is present', () => {
    render(<ScoreBreakdownCard breakdown={FULL_BREAKDOWN} />)
    expect(screen.getByText('−32 penalties')).toBeInTheDocument()
    expect(screen.getByText('+4 bonuses')).toBeInTheDocument()
    expect(screen.getByText('2 major flags')).toBeInTheDocument()
  })

  it('renders cap chips with correct labels', () => {
    render(<ScoreBreakdownCard breakdown={FULL_BREAKDOWN} />)
    expect(screen.getByText('Major Flags Cap')).toBeInTheDocument()
    expect(screen.getByText('High Switch Savings Cap')).toBeInTheDocument()
  })

  it('cap chips have aria-label', () => {
    render(<ScoreBreakdownCard breakdown={FULL_BREAKDOWN} />)
    expect(
      screen.getByRole('listitem', { name: 'Cap applied: Major Flags Cap' })
    ).toBeInTheDocument()
  })

  it('hides detail rows by default', () => {
    render(<ScoreBreakdownCard breakdown={FULL_BREAKDOWN} />)
    expect(screen.queryByText('Base score')).not.toBeInTheDocument()
  })

  it('shows detail rows after clicking Show details', async () => {
    const user = userEvent.setup()
    render(<ScoreBreakdownCard breakdown={FULL_BREAKDOWN} />)
    await user.click(screen.getByRole('button', { name: /show details/i }))
    expect(screen.getByText('Base score')).toBeInTheDocument()
    expect(screen.getByText('Post-cap score')).toBeInTheDocument()
    expect(screen.getByText('35.5 %')).toBeInTheDocument()
  })

  it('toggle button has aria-expanded reflecting open state', async () => {
    const user = userEvent.setup()
    render(<ScoreBreakdownCard breakdown={FULL_BREAKDOWN} />)
    const btn = screen.getByRole('button', { name: /show details/i })
    expect(btn).toHaveAttribute('aria-expanded', 'false')
    await user.click(btn)
    expect(btn).toHaveAttribute('aria-expanded', 'true')
  })

  it('does not render bonus chip when bonus_points is 0', () => {
    render(<ScoreBreakdownCard breakdown={{ ...FULL_BREAKDOWN, bonus_points: 0 }} />)
    expect(screen.queryByText(/\+0 bonuses/)).not.toBeInTheDocument()
  })
})
