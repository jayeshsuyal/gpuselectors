import { createBrowserRouter, RouterProvider, Navigate } from 'react-router-dom'
import { AppShell } from '@/components/AppShell'
import { OptimizePage } from '@/pages/Optimize'
import { CatalogPage } from '@/pages/Catalog'
import { InvoicePage } from '@/pages/Invoice'

const router = createBrowserRouter([
  {
    path: '/',
    element: <AppShell />,
    children: [
      { index: true, element: <OptimizePage /> },
      { path: 'catalog', element: <CatalogPage /> },
      { path: 'invoice', element: <InvoicePage /> },
      { path: '*', element: <Navigate to="/" replace /> },
    ],
  },
])

export function App() {
  return <RouterProvider router={router} />
}
