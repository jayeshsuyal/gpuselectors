import { Suspense, lazy } from 'react'
import { createBrowserRouter, RouterProvider, Navigate } from 'react-router-dom'
import { AppShell } from '@/components/AppShell'

const OptimizePage = lazy(() => import('@/pages/Optimize').then((m) => ({ default: m.OptimizePage })))
const AuditPage = lazy(() => import('@/pages/Audit').then((m) => ({ default: m.AuditPage })))

function PageFallback() {
  return <div className="px-4 py-6 text-xs text-zinc-500">Loading page…</div>
}

const router = createBrowserRouter([
  {
    path: '/',
    element: <AppShell />,
    children: [
      { index: true, element: <OptimizePage /> },
      { path: 'audit',   element: <AuditPage /> },
      { path: '*', element: <Navigate to="/" replace /> },
    ],
  },
])

export function App() {
  return (
    <Suspense fallback={<PageFallback />}>
      <RouterProvider router={router} />
    </Suspense>
  )
}
