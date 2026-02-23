# Frontend Execution Plan (v1 -> v3.1)

## Scope
Deliver a clear, high-trust UX across Optimize, Catalog, Invoice, and AI Assist while keeping backend contracts intact.

## Non-Negotiables
- Do not change backend API contracts from frontend.
- Preserve ranking/exclusion logic semantics in UI.
- Keep interactions accessible (keyboard/focus/contrast).
- Keep context grounding visible (workload/provider scope).

## Success Metrics
### Technical Acceptance
- [ ] `npm run build` passes with 0 type errors.
- [ ] No runtime console errors on key flows.
- [ ] API responses render without schema mismatch.
- [ ] Charts render from real payloads and degrade gracefully when data is missing.

### UX Acceptance
- [ ] Provider support and beta labeling are explicit.
- [ ] Assumptions are visible and understandable.
- [ ] Errors are actionable and specific.
- [ ] Filter interactions are predictable (no stale selections).

### Accessibility
- [ ] Keyboard-only nav through all primary actions.
- [ ] Focus-visible styles on all interactive controls.
- [ ] Hover-only states have focus equivalent.
- [ ] Table headers and form controls remain semantically correct.

## Phased Plan
## Phase 1 (Days 1-2): UX Baseline Stabilization
- Keep dual flow:
  - Ask IA AI
  - Guided Config
- Ensure AI panel receives workload/provider context.
- Remove stale/invalid filter states in catalog browse modes.

Exit Criteria:
- [ ] Optimize + Catalog flows stable.
- [ ] AI prompts grounded to active scope.

## Phase 2 (Days 3-4): Visual System Consistency
- Consolidate repeated inline styles into token-based utilities.
- Remove leftover non-brand colors in interactive states.
- Tighten card/table hierarchy for readability on desktop/mobile.

Exit Criteria:
- [ ] Brand token consistency pass complete.
- [ ] Mobile overflow/truncation pass complete.

## Phase 3 (Days 5-6): Charts + Insights
- Keep current charts and add clearer legends/tooltips.
- Add chart state toggles (cost/risk/composite where available).
- Prepare chart payload adapters for report export.

Exit Criteria:
- [ ] Charts are data-linked and regression-tested.
- [ ] Empty/partial data states are handled cleanly.

## Phase 4 (Days 7-8): Report UX
- Add `Export Report` action.
- Allow toggles:
  - include charts
  - include AI narrative (when available)
- Render export status/errors clearly.

Exit Criteria:
- [ ] Deterministic report export works from UI.
- [ ] User gets clear success/failure feedback.

## Phase 5 (Days 9-10): Polish + QA
- Full responsive QA sweep (mobile/tablet/desktop).
- Accessibility and keyboard QA.
- Final copy pass for clarity and consistency.

Exit Criteria:
- [ ] Build clean.
- [ ] Manual QA checklist complete.

## Commands
```bash
npm --prefix frontend run dev
npm --prefix frontend run build
```

## v2/v3.1 Extension Hooks (Frontend)
- Quality vs Price scatter with Pareto frontier.
- Scaling planner visual panels (GPU count, mode rationale).
- Training/lifecycle cost report sections.
- Invoice Analyzer++ remediation + scenario simulator UI.

