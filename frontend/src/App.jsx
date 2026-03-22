import React, { createContext, useContext, useState, useCallback, useRef } from 'react'
import { BrowserRouter, Routes, Route, NavLink, Navigate, useNavigate, useLocation } from 'react-router-dom'
import { UploadCloud, ClipboardList, Microscope, Layers } from 'lucide-react'
import UploadPage from './pages/UploadPage.jsx'
import HistoryPage from './pages/HistoryPage.jsx'
import StitchPage from './pages/StitchPage.jsx'
import { ToastContainer } from './components/Toast.jsx'
import { runDetection } from './api/detect.js'

const ToastCtx = createContext(null)
export const useToast = () => useContext(ToastCtx)

// Pipeline mode context — "macro" or "micro"
const PipelineModeCtx = createContext(null)
export const usePipelineMode = () => useContext(PipelineModeCtx)

// Background pipeline context — tracks running jobs
const PipelineJobCtx = createContext(null)
export const usePipelineJob = () => useContext(PipelineJobCtx)

// Shared files for stitch — pass uploaded files from Detect → Stitch page
const StitchFilesCtx = createContext(null)
export const useStitchFiles = () => useContext(StitchFilesCtx)

// Persistent upload files context — survives page navigation, stored per-mode
const UploadFilesCtx = createContext(null)
export const useUploadFiles = () => useContext(UploadFilesCtx)

function Layout({ children, mode, setMode }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', background: 'var(--bg)' }}>
      <Navbar mode={mode} setMode={setMode} />
      <main style={{ flex: 1, padding: '1.25rem 2rem 3rem', maxWidth: 1440, margin: '0 auto', width: '100%' }}>
        {children}
      </main>
    </div>
  )
}

function ModeToggle({ mode, setMode }) {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      background: 'var(--surface2)',
      borderRadius: 'var(--radius-sm)',
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
            padding: '4px 14px',
            borderRadius: 'calc(var(--radius-sm) - 2px)',
            fontSize: '0.6875rem',
            fontWeight: mode === m ? 700 : 500,
            letterSpacing: '.05em',
            textTransform: 'uppercase',
            transition: 'all .2s ease',
            background: mode === m ? 'var(--text)' : 'transparent',
            color: mode === m ? '#fff' : 'var(--text-muted)',
            boxShadow: mode === m ? '0 1px 3px rgba(0,0,0,.15)' : 'none',
          }}
        >
          {m}
        </button>
      ))}
    </div>
  )
}

function Navbar({ mode, setMode }) {
  const { running } = usePipelineJob()

  // Build nav tabs — exclude Stitch in micro mode
  const tabs = [
    { to: '/detect', icon: <UploadCloud size={15} strokeWidth={1.8} />, label: 'Detect' },
    ...(mode === 'macro' ? [{ to: '/stitch', icon: <Layers size={15} strokeWidth={1.8} />, label: 'Stitch' }] : []),
    { to: '/history', icon: <ClipboardList size={15} strokeWidth={1.8} />, label: 'History' },
  ]

  return (
    <nav style={{
      background: 'var(--surface)',
      borderBottom: '1px solid var(--border)',
      padding: '0 2rem',
      display: 'flex',
      alignItems: 'center',
      height: 52,
      position: 'sticky',
      top: 0,
      zIndex: 100,
      boxShadow: 'var(--shadow-sm)',
    }}>
      {/* Branding */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginRight: '2.5rem' }}>
        <div style={{
          width: 28, height: 28, borderRadius: 'var(--radius-sm)',
          background: 'var(--text)', display: 'flex', alignItems: 'center',
          justifyContent: 'center', color: '#fff',
        }}>
          <Microscope size={16} strokeWidth={1.8} />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', lineHeight: 1.2 }}>
          <span style={{ fontWeight: 700, fontSize: '0.8125rem', color: 'var(--text)', letterSpacing: '-.01em' }}>
            MP Detect
          </span>
          <span style={{ fontSize: '0.5rem', color: 'var(--text-muted)', fontWeight: 600, letterSpacing: '.1em', textTransform: 'uppercase' }}>
            Microplastic Analysis
          </span>
        </div>
      </div>

      {/* Nav tabs */}
      <div style={{ display: 'flex', gap: '0.125rem', height: '100%' }}>
        {tabs.map(({ to, icon, label }) => (
          <NavLink
            key={to}
            to={to}
            style={({ isActive }) => ({
              textDecoration: 'none',
              color: isActive ? 'var(--text)' : 'var(--text-muted)',
              fontWeight: isActive ? 600 : 500,
              fontSize: '0.75rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.325rem',
              padding: '0 0.875rem',
              borderBottom: isActive ? '2px solid var(--text)' : '2px solid transparent',
              transition: 'color .15s, border-color .15s',
              marginBottom: -1,
              letterSpacing: '.01em',
            })}
          >
            <span style={{ display: 'flex', alignItems: 'center', opacity: 0.7 }}>{icon}</span>
            {label}
          </NavLink>
        ))}
      </div>

      {/* Right side: running indicator + Mode toggle + version */}
      <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
        {running && (
          <div style={{
            display: 'flex', alignItems: 'center', gap: '0.375rem',
            padding: '3px 10px', borderRadius: 'var(--radius-sm)',
            background: 'var(--surface2)', border: '1px solid var(--border)',
            fontSize: '0.625rem', fontWeight: 600, color: 'var(--text-secondary)',
            letterSpacing: '.03em',
          }}>
            <span className="spinner" style={{ width: 10, height: 10, borderWidth: 1.5 }} />
            PROCESSING
          </div>
        )}
        <ModeToggle mode={mode} setMode={setMode} />
        <span className="badge badge-primary" style={{ fontSize: '0.5625rem' }}>v2.0</span>
      </div>
    </nav>
  )
}

function App() {
  const [toasts, setToasts] = useState([])
  const [pipelineMode, setPipelineMode] = useState('macro')

  // Background pipeline state
  const [pipelineRunning, setPipelineRunning] = useState(false)
  const [pipelineResult, setPipelineResult] = useState(null)
  const pipelineResultRef = useRef(null)

  // Shared stitch files state
  const [stitchFiles, setStitchFiles] = useState(null)

  // Upload files state — persists per mode across navigation
  const uploadFilesRef = useRef({ macro: [], micro: [] })
  const [uploadFilesTick, setUploadFilesTick] = useState(0) // force re-render on change
  const uploadFilesValue = {
    files: uploadFilesRef.current[pipelineMode] || [],
    setFiles: (updater) => {
      const prev = uploadFilesRef.current[pipelineMode] || []
      const next = typeof updater === 'function' ? updater(prev) : updater
      uploadFilesRef.current[pipelineMode] = next
      setUploadFilesTick(t => t + 1)
    },
  }

  const addToast = useCallback((msg, type = 'info') => {
    const id = Date.now()
    setToasts(t => [...t, { id, msg, type }])
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 5000)
  }, [])

  // Start pipeline in background — returns immediately, notifies on completion
  const startPipeline = useCallback(async (formData) => {
    setPipelineRunning(true)
    setPipelineResult(null)
    pipelineResultRef.current = null
    addToast('Pipeline started — you can navigate freely.', 'info')

    try {
      const result = await runDetection(formData)
      sessionStorage.setItem('mp_last_result', JSON.stringify(result))
      setPipelineResult(result)
      pipelineResultRef.current = result
      const total = result.images?.reduce((s, im) => s + (im.summary?.total || 0), 0) || 0
      addToast(`Pipeline complete! ${total} detections found.`, 'success')
    } catch (err) {
      addToast(`Pipeline failed: ${err.message}`, 'error')
    } finally {
      setPipelineRunning(false)
    }
  }, [addToast])

  const pipelineJobValue = {
    running: pipelineRunning,
    result: pipelineResult,
    resultRef: pipelineResultRef,
    startPipeline,
    clearResult: () => { setPipelineResult(null); pipelineResultRef.current = null },
  }

  const stitchFilesValue = {
    files: stitchFiles,
    setFiles: setStitchFiles,
    clear: () => setStitchFiles(null),
  }

  return (
    <ToastCtx.Provider value={addToast}>
      <PipelineModeCtx.Provider value={{ mode: pipelineMode, setMode: setPipelineMode }}>
        <PipelineJobCtx.Provider value={pipelineJobValue}>
          <StitchFilesCtx.Provider value={stitchFilesValue}>
            <UploadFilesCtx.Provider value={uploadFilesValue}>
              <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
                <Layout mode={pipelineMode} setMode={setPipelineMode}>
                  <Routes>
                    <Route path="/" element={<Navigate to="/detect" replace />} />
                    <Route path="/detect" element={<UploadPage />} />
                    {pipelineMode === 'macro' && <Route path="/stitch" element={<StitchPage />} />}
                    <Route path="/history" element={<HistoryPage />} />
                    {/* Redirect stitch to detect if user switched to micro */}
                    <Route path="/stitch" element={<Navigate to="/detect" replace />} />
                    {/* Legacy results route — redirect to history */}
                    <Route path="/results" element={<Navigate to="/history" replace />} />
                  </Routes>
                </Layout>
                <ToastContainer toasts={toasts} />
              </BrowserRouter>
            </UploadFilesCtx.Provider>
          </StitchFilesCtx.Provider>
        </PipelineJobCtx.Provider>
      </PipelineModeCtx.Provider>
    </ToastCtx.Provider>
  )
}

export default App
