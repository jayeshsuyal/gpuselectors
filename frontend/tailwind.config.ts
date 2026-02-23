import type { Config } from 'tailwindcss'

const config: Config = {
  darkMode: ['class'],
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Surface layers — keyed to CSS design tokens
        surface: {
          base: 'var(--bg-base)',
          DEFAULT: 'var(--bg-surface)',
          elevated: 'var(--bg-elevated)',
          showcase: 'var(--bg-showcase)',
        },
        border: {
          DEFAULT: 'rgba(255,255,255,0.07)',
          subtle: 'rgba(255,255,255,0.04)',
          strong: 'rgba(255,255,255,0.13)',
        },
        brand: {
          DEFAULT: '#7c5cfc',
          muted: '#6366f1',
          hover: '#9b7dfe',
          dim: 'rgba(124,92,252,0.10)',
          glow: 'rgba(124,92,252,0.24)',
        },
      },
      fontFamily: {
        // Space Grotesk as primary — geometric, technical-premium feel
        sans: ['Space Grotesk', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'ui-monospace', 'monospace'],
      },
      fontSize: {
        '2xs': ['0.625rem', { lineHeight: '0.875rem' }],
      },
      borderRadius: {
        sm: '6px',
        DEFAULT: '8px',
        md: '10px',
        lg: '14px',
        xl: '18px',
        '2xl': '24px',
      },
      boxShadow: {
        'xs': '0 1px 2px rgba(0,0,0,0.5)',
        'sm': '0 2px 6px rgba(0,0,0,0.5), 0 1px 2px rgba(0,0,0,0.3)',
        'md': '0 4px 16px rgba(0,0,0,0.6), 0 1px 4px rgba(0,0,0,0.35)',
        'lg': '0 8px 32px rgba(0,0,0,0.7), 0 2px 8px rgba(0,0,0,0.4)',
        'glow': '0 0 0 1px rgba(124,92,252,0.4), 0 0 20px rgba(124,92,252,0.25)',
        'glow-sm': '0 0 0 1px rgba(124,92,252,0.3), 0 0 12px rgba(124,92,252,0.18)',
        'inner-highlight': 'inset 0 1px 0 rgba(255,255,255,0.04)',
        'inner-top': 'inset 0 1px 0 rgba(255,255,255,0.06)',
      },
      animation: {
        'fade-in': 'fadeIn 0.2s ease-out both',
        'enter': 'enter 0.22s cubic-bezier(0.16,1,0.3,1) both',
        'enter-fast': 'enter 0.12s cubic-bezier(0.16,1,0.3,1) both',
        'slide-up': 'slideUp 0.25s cubic-bezier(0.16,1,0.3,1) both',
        'scale-in': 'scaleIn 0.15s cubic-bezier(0.34,1.56,0.64,1) both',
        'reveal': 'reveal 0.6s cubic-bezier(0.16,1,0.3,1) both',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4,0,0.6,1) infinite',
        'shimmer': 'shimmer 2s linear infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        enter: {
          '0%': { opacity: '0', transform: 'translateY(6px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(14px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        scaleIn: {
          '0%': { opacity: '0', transform: 'scale(0.94)' },
          '100%': { opacity: '1', transform: 'scale(1)' },
        },
        reveal: {
          '0%': { opacity: '0', transform: 'translateY(24px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
      },
      transitionTimingFunction: {
        'spring': 'cubic-bezier(0.34, 1.56, 0.64, 1)',
        'out-expo': 'cubic-bezier(0.16, 1, 0.3, 1)',
      },
      backdropBlur: {
        xs: '6px',
        sm: '10px',
        DEFAULT: '14px',
        lg: '20px',
        xl: '28px',
      },
    },
  },
  plugins: [],
}

export default config
