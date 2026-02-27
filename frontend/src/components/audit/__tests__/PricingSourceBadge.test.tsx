import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { PricingSourceBadge } from '../PricingSourceBadge'

describe('PricingSourceBadge', () => {
  it('renders provider_csv badge correctly', () => {
    render(<PricingSourceBadge source="provider_csv" />)
    expect(screen.getByText('Provider CSV')).toBeInTheDocument()
    expect(screen.getByLabelText('Pricing source: Provider CSV')).toBeInTheDocument()
  })

  it('renders heuristic_prior badge correctly', () => {
    render(<PricingSourceBadge source="heuristic_prior" />)
    expect(screen.getByText('Heuristic Prior')).toBeInTheDocument()
    expect(screen.getByLabelText('Pricing source: Heuristic Prior')).toBeInTheDocument()
  })

  it('renders Unknown badge when source is missing', () => {
    render(<PricingSourceBadge />)
    expect(screen.getByText('Unknown')).toBeInTheDocument()
    expect(screen.getByLabelText('Pricing source: Unknown')).toBeInTheDocument()
  })

  it('renders Unknown badge when source is "unknown"', () => {
    render(<PricingSourceBadge source="unknown" />)
    expect(screen.getByText('Unknown')).toBeInTheDocument()
  })

  it('shows provider detail when provided', () => {
    render(<PricingSourceBadge source="provider_csv" provider="anthropic" />)
    expect(screen.getByText('anthropic')).toBeInTheDocument()
  })

  it('shows GPU detail when provided', () => {
    render(<PricingSourceBadge source="provider_csv" gpu="A100 80GB" />)
    expect(screen.getByText('A100 80GB')).toBeInTheDocument()
  })

  it('handles null provider and gpu safely', () => {
    render(<PricingSourceBadge source="provider_csv" provider={null} gpu={null} />)
    expect(screen.queryByText('Provider:')).not.toBeInTheDocument()
    expect(screen.queryByText('GPU:')).not.toBeInTheDocument()
  })

  it('handles undefined provider and gpu safely', () => {
    render(<PricingSourceBadge source="heuristic_prior" provider={undefined} gpu={undefined} />)
    expect(screen.queryByText('Provider:')).not.toBeInTheDocument()
    expect(screen.queryByText('GPU:')).not.toBeInTheDocument()
  })
})
