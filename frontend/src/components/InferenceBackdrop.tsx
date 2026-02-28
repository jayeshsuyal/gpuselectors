import { useEffect, useRef } from 'react'

// ─── Types ────────────────────────────────────────────────────────────────────

interface NNode {
  x: number
  y: number
  vx: number
  vy: number
  baseR: number          // radius before breathing offset
  phase: number          // breathing phase offset (0..2π)
  layer: 0 | 1 | 2      // depth: 0=back (slow/dim), 1=mid, 2=front (fast/bright)
  activatedAt: number    // timestamp of last pulse arrival
}

interface Pulse {
  fromIdx: number
  toIdx: number
  t: number              // 0..1 travel progress
  speed: number
  cascade: boolean
}

// ─── Config ───────────────────────────────────────────────────────────────────

const NODE_COUNT     = 44
const CONNECT_DIST   = 168    // px — max edge length
const PULSE_INTERVAL = 650    // ms between regular auto-spawns
const BURST_INTERVAL = 6200   // ms between forward-pass bursts
const BURST_COUNT    = 5      // pulses per burst
const CASCADE_CHANCE = 0.32

// Depth layers: [baseR, speedScale, alphaScale]
const LAYER_CONFIG = [
  { baseR: 1.2, speed: 0.18, alpha: 0.38 },  // 0: background — slow, tiny, dim
  { baseR: 2.0, speed: 0.28, alpha: 0.65 },  // 1: midground
  { baseR: 2.7, speed: 0.42, alpha: 1.00 },  // 2: foreground — fast, large, bright
] as const

// Global opacity ceiling (keeps it a whisper, not a shout)
const EDGE_BASE_ALPHA  = 0.026
const EDGE_GLOW_ALPHA  = 0.11   // active edge (pulse traveling along it)
const NODE_BASE_ALPHA  = 0.060
const PULSE_ALPHA      = 0.58
const RING_ALPHA       = 0.24
const RING_DECAY_MS    = 950
const BREATHE_AMP      = 0.45   // ± px radius oscillation
const BREATHE_SPEED    = 0.0008 // radians/ms

// Inference cyan palette
const C_EDGE  = [22,  163, 192] as const
const C_NODE  = [103, 232, 249] as const
const C_PULSE = [110, 231, 183] as const
const C_RING  = [34,  211, 238] as const

function rgba([r, g, b]: readonly [number, number, number], a: number) {
  return `rgba(${r},${g},${b},${Math.max(0, a).toFixed(4)})`
}

// ─── Component ────────────────────────────────────────────────────────────────

export function InferenceBackdrop() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let animId: number
    let nodes: NNode[] = []
    let pulses: Pulse[] = []
    let lastPulseTime  = 0
    let lastBurstTime  = 0

    // ── Resize (DPR-aware) ───────────────────────────────────────────────────

    function resize() {
      const dpr = Math.min(window.devicePixelRatio ?? 1, 2)
      const w = window.innerWidth
      const h = window.innerHeight
      canvas!.width  = w * dpr
      canvas!.height = h * dpr
      canvas!.style.width  = `${w}px`
      canvas!.style.height = `${h}px`
      ctx!.scale(dpr, dpr)
    }

    // ── Init nodes ───────────────────────────────────────────────────────────

    function initNodes() {
      const w = window.innerWidth
      const h = window.innerHeight
      nodes = Array.from({ length: NODE_COUNT }, () => {
        const layer = (Math.random() < 0.33 ? 0 : Math.random() < 0.5 ? 1 : 2) as 0 | 1 | 2
        const cfg = LAYER_CONFIG[layer]
        return {
          x: Math.random() * w,
          y: Math.random() * h,
          vx: (Math.random() - 0.5) * cfg.speed * 2,
          vy: (Math.random() - 0.5) * cfg.speed * 2,
          baseR: cfg.baseR,
          phase: Math.random() * Math.PI * 2,
          layer,
          activatedAt: -99999,
        }
      })
    }

    // ── Pulse helpers ────────────────────────────────────────────────────────

    function neighborsOf(idx: number): number[] {
      const n = nodes[idx]
      const out: number[] = []
      for (let i = 0; i < nodes.length; i++) {
        if (i === idx) continue
        if (Math.hypot(nodes[i].x - n.x, nodes[i].y - n.y) < CONNECT_DIST) out.push(i)
      }
      return out
    }

    function spawnPulseFrom(fromIdx: number, cascade = false) {
      const nbrs = neighborsOf(fromIdx)
      if (nbrs.length === 0) return
      const toIdx = nbrs[Math.floor(Math.random() * nbrs.length)]
      pulses.push({ fromIdx, toIdx, t: 0, speed: 0.005 + Math.random() * 0.011, cascade })
    }

    /** Forward-pass burst: pick nodes from left 40% and fire pulses rightward. */
    function spawnBurst() {
      const w = window.innerWidth
      const leftNodes = nodes
        .map((n, i) => ({ n, i }))
        .filter(({ n }) => n.x < w * 0.45)
      for (let k = 0; k < BURST_COUNT; k++) {
        if (leftNodes.length === 0) break
        const pick = leftNodes[Math.floor(Math.random() * leftNodes.length)]
        const rightNbrs = neighborsOf(pick.i).filter((j) => nodes[j].x > pick.n.x)
        if (rightNbrs.length > 0) {
          const toIdx = rightNbrs[Math.floor(Math.random() * rightNbrs.length)]
          pulses.push({ fromIdx: pick.i, toIdx, t: 0, speed: 0.008 + Math.random() * 0.008, cascade: false })
        }
      }
    }

    // ── Draw ─────────────────────────────────────────────────────────────────

    function draw(ts: number) {
      const w = window.innerWidth
      const h = window.innerHeight
      ctx!.clearRect(0, 0, w, h)

      // Move nodes — layer-speed-aware, bounce walls
      for (const n of nodes) {
        n.x += n.vx
        n.y += n.vy
        if (n.x <= 0)  { n.x = 0;  n.vx =  Math.abs(n.vx) }
        if (n.x >= w)  { n.x = w;  n.vx = -Math.abs(n.vx) }
        if (n.y <= 0)  { n.y = 0;  n.vy =  Math.abs(n.vy) }
        if (n.y >= h)  { n.y = h;  n.vy = -Math.abs(n.vy) }
      }

      // Timed spawns
      if (ts - lastPulseTime > PULSE_INTERVAL) {
        spawnPulseFrom(
          Math.floor(Math.random() * nodes.length),
          Math.random() < CASCADE_CHANCE,
        )
        lastPulseTime = ts
      }
      if (ts - lastBurstTime > BURST_INTERVAL) {
        spawnBurst()
        lastBurstTime = ts
      }

      // Build active-edge set (edges with a live pulse on them)
      const activeEdgeKeys = new Set<string>()
      for (const p of pulses) {
        const a = Math.min(p.fromIdx, p.toIdx)
        const b = Math.max(p.fromIdx, p.toIdx)
        activeEdgeKeys.add(`${a}:${b}`)
      }

      // ── Edges (draw back-layer first, then front) ──────────────────────────
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const a = nodes[i], b = nodes[j]
          const dist = Math.hypot(b.x - a.x, b.y - a.y)
          if (dist >= CONNECT_DIST) continue

          const proximity = 1 - dist / CONNECT_DIST
          const key = `${i}:${j}`
          const isActive = activeEdgeKeys.has(key)

          ctx!.beginPath()
          ctx!.strokeStyle = isActive
            ? rgba(C_EDGE, EDGE_GLOW_ALPHA * proximity)
            : rgba(C_EDGE, EDGE_BASE_ALPHA * proximity)
          ctx!.lineWidth = isActive ? 1.0 : 0.45
          ctx!.moveTo(a.x, a.y)
          ctx!.lineTo(b.x, b.y)
          ctx!.stroke()
        }
      }

      // ── Nodes (back layers first for correct depth ordering) ───────────────
      for (let layer = 0; layer <= 2; layer++) {
        for (const n of nodes) {
          if (n.layer !== layer) continue

          const layerAlpha = LAYER_CONFIG[n.layer].alpha
          // Breathing oscillation
          const r = n.baseR + Math.sin(ts * BREATHE_SPEED + n.phase) * BREATHE_AMP

          // Activation ring
          const ringAge = ts - n.activatedAt
          if (ringAge < RING_DECAY_MS) {
            const fade  = 1 - ringAge / RING_DECAY_MS
            const ringR = r + 4 + ringAge * 0.005
            ctx!.beginPath()
            ctx!.strokeStyle = rgba(C_RING, RING_ALPHA * fade * layerAlpha)
            ctx!.lineWidth   = 0.9
            ctx!.arc(n.x, n.y, ringR, 0, Math.PI * 2)
            ctx!.stroke()
          }

          // Node dot
          ctx!.beginPath()
          ctx!.fillStyle = rgba(C_NODE, NODE_BASE_ALPHA * layerAlpha)
          ctx!.arc(n.x, n.y, r, 0, Math.PI * 2)
          ctx!.fill()
        }
      }

      // ── Pulses ─────────────────────────────────────────────────────────────
      const alive: Pulse[] = []
      for (const p of pulses) {
        p.t += p.speed

        const from = nodes[p.fromIdx]
        const to   = nodes[p.toIdx]

        if (p.t >= 1) {
          to.activatedAt = ts
          if (p.cascade) {
            const count = 1 + (Math.random() < 0.38 ? 1 : 0)
            for (let c = 0; c < count; c++) spawnPulseFrom(p.toIdx, false)
          }
          continue
        }

        alive.push(p)

        const px = from.x + (to.x - from.x) * p.t
        const py = from.y + (to.y - from.y) * p.t

        // ── Trail: sample 8 points backward along edge ──────────────────────
        const TRAIL_STEPS  = 8
        const TRAIL_EXTENT = 0.14   // how far back (in t-units) the trail reaches
        for (let s = 1; s <= TRAIL_STEPS; s++) {
          const trailT = Math.max(0, p.t - (s / TRAIL_STEPS) * TRAIL_EXTENT)
          const tx = from.x + (to.x - from.x) * trailT
          const ty = from.y + (to.y - from.y) * trailT
          const fade = 1 - s / TRAIL_STEPS
          ctx!.beginPath()
          ctx!.fillStyle = rgba(C_PULSE, PULSE_ALPHA * fade * 0.55)
          ctx!.arc(tx, ty, 1.2 * fade, 0, Math.PI * 2)
          ctx!.fill()
        }

        // ── Pulse glow halo ──────────────────────────────────────────────────
        const grd = ctx!.createRadialGradient(px, py, 0, px, py, 6)
        grd.addColorStop(0, rgba(C_PULSE, PULSE_ALPHA * 0.85))
        grd.addColorStop(1, rgba(C_PULSE, 0))
        ctx!.beginPath()
        ctx!.fillStyle = grd
        ctx!.arc(px, py, 6, 0, Math.PI * 2)
        ctx!.fill()

        // ── Pulse core ───────────────────────────────────────────────────────
        ctx!.beginPath()
        ctx!.fillStyle = rgba(C_PULSE, PULSE_ALPHA)
        ctx!.arc(px, py, 1.6, 0, Math.PI * 2)
        ctx!.fill()
      }
      pulses = alive

      animId = requestAnimationFrame(draw)
    }

    // ── Bootstrap ─────────────────────────────────────────────────────────────

    resize()
    initNodes()
    animId = requestAnimationFrame(draw)

    const ro = new ResizeObserver(() => {
      resize()
      const w = window.innerWidth
      const h = window.innerHeight
      for (const n of nodes) {
        n.x = Math.min(n.x, w)
        n.y = Math.min(n.y, h)
      }
    })
    ro.observe(document.documentElement)

    return () => {
      cancelAnimationFrame(animId)
      ro.disconnect()
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      style={{ position: 'fixed', inset: 0, pointerEvents: 'none', zIndex: 0 }}
    />
  )
}
