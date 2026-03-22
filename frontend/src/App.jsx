import React, { createContext, useContext, useState, useCallback } from 'react'
import { BrowserRouter, Routes, Route, NavLink, Navigate } from 'react-router-dom'
import { UploadCloud, BarChart3, ClipboardList, Microscope, Layers } from 'lucide-react'
import UploadPage from './pages/UploadPage.jsx'
import ResultsPage from './pages/ResultsPage.jsx'
import HistoryPage from './pages/HistoryPage.jsx'
import StitchPage from './pages/StitchPage.jsx'
import { ToastContainer } from './components/Toast.jsx'

const ToastCtx = createContext(null)
export const useToast = () => useContext(ToastCtx)

// Pipeline mode context — "macro" or "micro"
const PipelineModeCtx = createContext(null)
export const usePipelineMode = () => useContext(PipelineModeCtx)

function Layout({ children, mode, setMode }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', background: 'var(--bg)' }}>
      <Navbar mode={mode} setMode={setMode} />
      <main style={{ flex: 1, padding: '1.5rem 2rem 3rem', maxWidth: 1440, margin: '0 auto', width: '100%' }}>
        {children}
      </main>
    </div>
  )
}

const NAV_TABS = [
  { to: '/detect',  icon: <UploadCloud size={16} />, label: 'Detect' },
  { to: '/stitch',  icon: <Layers size={16} />, label: 'Stitch' },
  { to: '/results', icon: <BarChart3 size={16} />, label: 'Results' },
  { to: '/history', icon: <ClipboardList size={16} />, label: 'History' },
]

function ModeToggle({ mode, setMode }) {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      background: 'var(--surface2)',
      borderRadius: 'var(--radius)',
      border: '1px solid var(--border)',
      padding: '2px',
      gap: 0,
    }}>
      {['macro', 'micro'].map(m => (
        <button
          key={m}
          onClick={() => setMode(m)}
          style={{
            border: 'none',
            cursor: 'pointer',
            padding: '5px 14px',
            borderRadius: 'calc(var(--radius) - 2px)',
            fontSize: '0.75rem',
            fontWeight: mode === m ? 700 : 500,
            letterSpacing: '.03em',
            textTransform: 'uppercase',
            transition: 'all .2s ease',
            background: mode === m
              ? (m === 'macro' ? 'var(--primary)' : '#a855f7')
              : 'transparent',
            color: mode === m ? '#fff' : 'var(--text-muted)',
            boxShadow: mode === m ? '0 1px 4px rgba(0,0,0,.18)' : 'none',
          }}
        >
          {m === 'macro' ? '🔬 Macro' : '🔎 Micro'}
        </button>
      ))}
    </div>
  )
}

function Navbar({ mode, setMode }) {
  return (
    <nav style={{
      background: 'var(--surface)',
      borderBottom: '1px solid var(--border)',
      padding: '0 2rem',
      display: 'flex',
      alignItems: 'center',
      height: 54,
      position: 'sticky',
      top: 0,
      zIndex: 100,
      boxShadow: 'var(--shadow-sm)',
    }}>
      {/* Branding */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginRight: '2.5rem' }}>
        <div style={{
          width: 30, height: 30, borderRadius: 'var(--radius-sm)',
          background: 'var(--primary)', display: 'flex', alignItems: 'center',
          justifyContent: 'center', color: '#fff',
        }}>
          <Microscope size={18} />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', lineHeight: 1.2 }}>
          <span style={{ fontWeight: 700, fontSize: '0.875rem', color: 'var(--text)', letterSpacing: '-.01em' }}>
            MP Detect
          </span>
          <span style={{ fontSize: '0.5625rem', color: 'var(--text-muted)', fontWeight: 600, letterSpacing: '.08em', textTransform: 'uppercase' }}>
            Microplastic Analysis
          </span>
        </div>
      </div>

      {/* Nav tabs */}
      <div style={{ display: 'flex', gap: '0.25rem', height: '100%' }}>
        {NAV_TABS.map(({ to, icon, label }) => (
          <NavLink
            key={to}
            to={to}
            style={({ isActive }) => ({
              textDecoration: 'none',
              color: isActive ? 'var(--primary)' : 'var(--text-muted)',
              fontWeight: isActive ? 600 : 500,
              fontSize: '0.8125rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.375rem',
              padding: '0 1rem',
              borderBottom: isActive ? '2px solid var(--primary)' : '2px solid transparent',
              transition: 'color .15s, border-color .15s',
              marginBottom: -1,
            })}
          >
            <span style={{ display: 'flex', alignItems: 'center' }}>{icon}</span>
            {label}
          </NavLink>
        ))}
      </div>

      {/* Right side: Mode toggle + version */}
      <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
        <ModeToggle mode={mode} setMode={setMode} />
        <span className="badge badge-primary" style={{ fontSize: '0.625rem' }}>v2.0</span>
      </div>
    </nav>
  )
}

function App() {
  const [toasts, setToasts] = useState([])
  const [pipelineMode, setPipelineMode] = useState('macro')

  const addToast = useCallback((msg, type = 'info') => {
    const id = Date.now()
    setToasts(t => [...t, { id, msg, type }])
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 4000)
  }, [])

  return (
    <ToastCtx.Provider value={addToast}>
      <PipelineModeCtx.Provider value={{ mode: pipelineMode, setMode: setPipelineMode }}>
        <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
          <Layout mode={pipelineMode} setMode={setPipelineMode}>
            <Routes>
              <Route path="/" element={<Navigate to="/detect" replace />} />
              <Route path="/detect" element={<UploadPage />} />
              <Route path="/stitch" element={<StitchPage />} />
              <Route path="/results" element={<ResultsPage />} />
              <Route path="/history" element={<HistoryPage />} />
            </Routes>
          </Layout>
          <ToastContainer toasts={toasts} />
        </BrowserRouter>
      </PipelineModeCtx.Provider>
    </ToastCtx.Provider>
  )
}

export default App
