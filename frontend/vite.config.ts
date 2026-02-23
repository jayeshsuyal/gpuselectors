import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules/recharts')) return 'vendor-recharts'
          if (id.includes('node_modules/@radix-ui')) return 'vendor-radix'
          if (
            id.includes('node_modules/lucide-react') ||
            id.includes('node_modules/react-hook-form') ||
            id.includes('node_modules/zod') ||
            id.includes('node_modules/@hookform/resolvers')
          ) {
            return 'vendor-forms-icons'
          }
          return undefined
        },
      },
    },
  },
})
