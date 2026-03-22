import React, { useState, useCallback, useRef, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  FolderOpen, Images, Zap, Sparkles, Send, Trash2, ChevronDown, ChevronUp,
  SlidersHorizontal, RotateCcw, CheckCircle2, AlertTriangle, Loader2,
  ImagePlus, ArrowRight, Info, Eye, Download, Layers
} from 'lucide-react'
import {
  analyzeFolder, thumbnailUrl, runStitch, stitchPreviewUrl,
  enhancePreviewUrl, saveEnhancement, fetchStitchedFile, deleteStitch
} from '../api/stitch.js'
import { runDetection } from '../api/detect.js'
import { useToast } from '../App.jsx'

/* ── tiny helpers ────────────────────────────────── */
const STEPS = ['folder', 'select', 'result', 'enhance']

function StepIndicator({ current }) {
  const labels = ['Input Folder', 'Select Images', 'Stitch Result', 'Enhance']
  const icons  = [<FolderOpen size={13} strokeWidth={1.8}/>, <Images size={13} strokeWidth={1.8}/>, <Layers size={13} strokeWidth={1.8}/>, <Sparkles size={13} strokeWidth={1.8}/>]
  return (
    <div style={{ display:'flex', gap:'0.25rem', marginBottom:'1.25rem' }}>
      {STEPS.map((s, i) => {
        const idx = STEPS.indexOf(current)
        const done = i < idx, active = i === idx
        return (
          <div key={s} style={{
            flex:1, display:'flex', alignItems:'center', gap:'0.5rem',
            padding:'0.5rem 0.75rem', borderRadius:'var(--radius-sm)',
            background: active ? 'var(--surface)' : done ? 'var(--surface)' : 'var(--surface2)',
            border: `1px solid ${active ? 'var(--text)' : done ? 'var(--border-strong)' : 'var(--border)'}`,
            opacity: (i > idx) ? 0.4 : 1, transition:'all .2s',
          }}>
            <span style={{ display:'flex', alignItems:'center', color: done ? 'var(--text-secondary)' : active ? 'var(--text)' : 'var(--text-muted)' }}>
              {done ? <CheckCircle2 size={13} strokeWidth={1.8}/> : icons[i]}
            </span>
            <span style={{ fontSize:'0.6875rem', fontWeight: active ? 600 : 500, color: active ? 'var(--text)' : 'var(--text-muted)', letterSpacing:'.01em' }}>
              {labels[i]}
            </span>
          </div>
        )
      })}
    </div>
  )
}

/* ── STEP 1 : Folder Input ───────────────────────── */
function FolderStep({ folder, setFolder, onAnalyze, loading }) {
  return (
    <div className="card" style={{ display:'flex', flexDirection:'column', gap:'1.25rem' }}>
      <div style={{ fontWeight:600, display:'flex', alignItems:'center', gap:'0.5rem', fontSize:'0.8125rem', color:'var(--text-secondary)' }}>
        <FolderOpen size={15} strokeWidth={1.8} style={{ color:'var(--text-muted)' }} /> Input Folder
      </div>
      <p className="text-sm text-muted" style={{ margin:0 }}>
        Point to a folder containing overlapping partial filter paper images captured with the macro lens.
      </p>
      <div style={{ display:'flex', gap:'0.75rem' }}>
        <input
          type="text" value={folder}
          onChange={e => setFolder(e.target.value)}
          placeholder="e.g.  datasets/raw/Sample1/Macro"
          className="input"
          style={{ flex:1 }}
          onKeyDown={e => e.key === 'Enter' && folder.trim() && onAnalyze()}
        />
        <button
          className="btn btn-primary"
          disabled={!folder.trim() || loading}
          onClick={onAnalyze}
          style={{ whiteSpace:'nowrap' }}
        >
          {loading ? <><Loader2 size={14} className="spin" /> Scanning…</> : <><Eye size={14} strokeWidth={1.8} /> Analyze</>}
        </button>
      </div>
      <div style={{ background:'var(--surface2)', border:'1px solid var(--border)', borderRadius:'var(--radius-sm)', padding:'0.75rem 1rem', fontSize:'0.75rem', color:'var(--text-muted)', display:'flex', gap:'0.5rem', alignItems:'flex-start' }}>
        <Info size={13} strokeWidth={1.8} style={{ marginTop:2, flexShrink:0 }} />
        <span>Images will be grouped by brightness so you can select the best-matching set for stitching.</span>
      </div>
    </div>
  )
}

/* ── STEP 2 : Image Selection ────────────────────── */
function SelectStep({ groups, folder, selected, setSelected, onStitch, loading, advancedMode, setAdvancedMode, outputName, setOutputName, maxDim, setMaxDim, upscale, setUpscale, onBack }) {
  const toggle = (path) => {
    setSelected(prev => prev.includes(path) ? prev.filter(p => p !== path) : [...prev, path])
  }
  const selectGroup = (imgs) => {
    setSelected(imgs.map(i => i.path))
  }
  const minNeeded = advancedMode ? 3 : 2
  const canStitch = selected.length >= minNeeded

  const [settingsOpen, setSettingsOpen] = useState(false)

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:'1.25rem' }}>
      {/* Groups */}
      {Object.entries(groups).map(([level, imgs]) => (
        <div key={level} className="card">
          <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:'1rem' }}>
            <div style={{ fontWeight:600, fontSize:'0.8125rem', display:'flex', alignItems:'center', gap:'0.5rem', color:'var(--text-secondary)' }}>
              <Images size={14} strokeWidth={1.8} style={{ color:'var(--text-muted)' }} />
              Brightness Level {level}
              <span className="badge badge-primary" style={{ fontSize:'0.625rem' }}>{imgs.length} images</span>
            </div>
            <button className="btn btn-sm btn-ghost" onClick={() => selectGroup(imgs)} style={{ fontSize:'0.6875rem' }}>
              Select all
            </button>
          </div>
          <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fill,minmax(150px,1fr))', gap:'0.75rem' }}>
            {imgs.map(img => {
              const active = selected.includes(img.path)
              return (
                <label key={img.path} style={{ cursor:'pointer', position: 'relative' }}>
                  <input type="checkbox" checked={active} onChange={() => toggle(img.path)} style={{ position:'absolute', opacity:0 }} />
                  <div style={{
                    border: `2px solid ${active ? 'var(--text)' : 'var(--border)'}`,
                    borderRadius:'var(--radius-sm)', padding:'0.5rem', textAlign:'center',
                    background: active ? 'var(--surface2)' : 'var(--surface)',
                    transition:'all .15s',
                    boxShadow: active ? '0 0 0 2px rgba(55,65,81,.1)' : 'none',
                  }}>
                    <div style={{ width:'100%', height:100, borderRadius:'var(--radius-sm)', overflow:'hidden', background:'var(--surface2)', marginBottom:'0.5rem' }}>
                      <img src={thumbnailUrl(img.path)} alt={img.filename} loading="lazy"
                        style={{ width:'100%', height:'100%', objectFit:'cover' }} />
                    </div>
                    <div style={{ fontSize:'0.6875rem', fontWeight:500, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{img.filename}</div>
                    <div className="text-xs text-muted">Brightness: {img.brightness}</div>
                  </div>
                </label>
              )
            })}
          </div>
        </div>
      ))}

      {/* Bottom bar */}
      <div className="card" style={{ position:'sticky', bottom:'1rem', zIndex:10, boxShadow:'var(--shadow-lg)' }}>
        {/* Mode toggle */}
        <div style={{ display:'flex', gap:'1rem', alignItems:'center', marginBottom:'1rem', flexWrap:'wrap' }}>
          <label style={{ display:'flex', alignItems:'center', gap:'0.5rem', cursor:'pointer', fontSize:'0.75rem', fontWeight:500 }}>
            <input type="checkbox" checked={advancedMode} onChange={e => setAdvancedMode(e.target.checked)} style={{ accentColor:'var(--text)' }} />
            <Zap size={13} strokeWidth={1.8} style={{ color:'var(--text-muted)' }} />
            Advanced Mode
          </label>
          <span className="text-xs text-muted">
            {advancedMode ? 'Enhanced multi-image stitching (3+ required)' : 'Standard stitching (2+ images)'}
          </span>
          <button className="btn btn-sm btn-ghost" style={{ marginLeft:'auto', fontSize:'0.6875rem' }}
            onClick={() => setSettingsOpen(o => !o)}>
            <SlidersHorizontal size={13} strokeWidth={1.8} /> Settings {settingsOpen ? <ChevronUp size={11}/> : <ChevronDown size={11}/>}
          </button>
        </div>

        {settingsOpen && (
          <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr 1fr', gap:'1rem', marginBottom:'1rem', paddingTop:'0.75rem', borderTop:'1px solid var(--border)' }}>
            <div>
              <label className="field-label" style={{ fontSize:'0.6875rem' }}>Output Filename</label>
              <input className="input" value={outputName} onChange={e => setOutputName(e.target.value)} placeholder="stitched_output.png" style={{ fontSize:'0.75rem' }} />
            </div>
            <div>
              <label className="field-label" style={{ fontSize:'0.6875rem' }}>Max Dimension</label>
              <input className="input" type="number" value={maxDim} onChange={e => setMaxDim(+e.target.value)} step={256} style={{ fontSize:'0.75rem' }} />
            </div>
            <div>
              <label className="field-label" style={{ fontSize:'0.6875rem' }}>Upscale</label>
              <input className="input" type="number" value={upscale} onChange={e => setUpscale(+e.target.value)} min={1} max={4} step={0.5} style={{ fontSize:'0.75rem' }} />
            </div>
          </div>
        )}

        {/* Selection + action row */}
        <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between' }}>
          <div style={{ display:'flex', gap:'0.75rem', alignItems:'center' }}>
            <button className="btn btn-ghost" onClick={onBack}>&larr; Change folder</button>
            <span className="text-sm" style={{ fontWeight:500 }}>
              {selected.length} image{selected.length !== 1 && 's'} selected
              {!canStitch && <span className="text-muted"> (need {minNeeded - selected.length} more)</span>}
            </span>
          </div>
          <button className="btn btn-primary" disabled={!canStitch || loading} onClick={onStitch}>
            {loading ? <><Loader2 size={14} className="spin" /> Stitching…</> : <><ImagePlus size={14} strokeWidth={1.8} /> Stitch {selected.length} Images</>}
          </button>
        </div>
      </div>
    </div>
  )
}

/* ── STEP 3 : Result ─────────────────────────────── */
function ResultStep({ session, meta, numImages, advanced, onEnhance, onSendToPipeline, onDelete, onBack, sendLoading }) {
  return (
    <div className="card" style={{ display:'flex', flexDirection:'column', gap:'1.5rem' }}>
      <div style={{ textAlign:'center' }}>
        <div style={{ display:'inline-flex', alignItems:'center', gap:'0.5rem', background:'var(--surface2)', color:'var(--text)', padding:'0.5rem 1rem', borderRadius:'var(--radius)', fontWeight:600, fontSize:'0.8125rem', marginBottom:'0.75rem', border:'1px solid var(--border)' }}>
          <CheckCircle2 size={15} strokeWidth={1.8} /> Stitching Complete
        </div>
        {advanced && <div className="text-xs text-muted" style={{ marginTop:4 }}>Advanced mode &middot; {numImages} images stitched</div>}
      </div>

      {/* Stats */}
      <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fit,minmax(160px,1fr))', gap:'1rem' }}>
        {[
          { label:'Original Size', value:`${meta.original_size[0]} × ${meta.original_size[1]}` },
          { label:'Output Size', value:`${meta.output_size[0]} × ${meta.output_size[1]}` },
          { label:'Scale', value:`${meta.scale.toFixed(2)}×` },
          { label:'Images', value: numImages },
        ].map(m => (
          <div key={m.label} style={{ background:'var(--surface2)', borderRadius:'var(--radius-sm)', padding:'1rem', textAlign:'center', border:'1px solid var(--surface3)' }}>
            <div className="text-xs text-muted" style={{ textTransform:'uppercase', letterSpacing:'.06em', fontWeight:600, marginBottom:4 }}>{m.label}</div>
            <div style={{ fontSize:'1.125rem', fontWeight:700, color:'var(--text)' }}>{m.value}</div>
          </div>
        ))}
      </div>

      {/* Preview */}
      <div style={{ background:'#111827', borderRadius:'var(--radius)', padding:'0.75rem', textAlign:'center' }}>
        <img src={stitchPreviewUrl(session)} alt="Stitched result" style={{ maxWidth:'100%', maxHeight:500, borderRadius:'var(--radius-sm)', objectFit:'contain' }} />
      </div>

      {/* Output path */}
      <div style={{ background:'var(--surface2)', border:'1px solid var(--border)', borderRadius:'var(--radius-sm)', padding:'0.75rem 1rem', fontFamily:'monospace', fontSize:'0.75rem', textAlign:'center', wordBreak:'break-all', color:'var(--text-secondary)' }}>
        {meta.image_path}
      </div>

      {/* Actions */}
      <div style={{ display:'flex', justifyContent:'center', gap:'0.5rem', flexWrap:'wrap' }}>
        <button className="btn btn-primary" onClick={onSendToPipeline} disabled={sendLoading}>
          {sendLoading ? <><Loader2 size={14} className="spin" /> Preparing…</> : <><Send size={14} strokeWidth={1.8} /> Send to Detection Pipeline</>}
        </button>
        <button className="btn btn-secondary" onClick={onEnhance}><Sparkles size={14} strokeWidth={1.8} /> Enhance Image</button>
        <button className="btn btn-secondary" onClick={onBack}><ImagePlus size={14} strokeWidth={1.8} /> Stitch Another</button>
        <button className="btn btn-danger-ghost" onClick={onDelete}><Trash2 size={14} strokeWidth={1.8} /> Delete</button>
      </div>
    </div>
  )
}

/* ── STEP 4 : Enhance ────────────────────────────── */
const PRESETS = {
  subtle: { sharpen:0.5, denoise:3, contrast:1.0, brightness:5, auto_wb:0 },
  vivid:  { sharpen:1.0, denoise:0, contrast:2.5, brightness:10, auto_wb:1 },
  sharp:  { sharpen:2.0, denoise:5, contrast:1.5, brightness:0, auto_wb:0 },
  clean:  { sharpen:0.3, denoise:12, contrast:1.2, brightness:0, auto_wb:1 },
}

function EnhanceStep({ session, onDone, onSendToPipeline, sendLoading }) {
  const [p, setP] = useState({ sharpen:0, denoise:0, contrast:0, brightness:0, auto_wb:0 })
  const [saving, setSaving] = useState(false)
  const [previewUrl, setPreviewUrl] = useState(stitchPreviewUrl(session))
  const debounceRef = useRef(null)
  const toast = useToast()

  const updatePreview = useCallback((params) => {
    clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => {
      setPreviewUrl(enhancePreviewUrl(session, params))
    }, 350)
  }, [session])

  const set = (key, val) => {
    const next = { ...p, [key]: val }
    setP(next)
    updatePreview(next)
  }

  const applyPreset = (name) => {
    const pr = PRESETS[name]
    setP(pr)
    updatePreview(pr)
  }

  const reset = () => {
    const z = { sharpen:0, denoise:0, contrast:0, brightness:0, auto_wb:0 }
    setP(z)
    updatePreview(z)
  }

  const save = async () => {
    setSaving(true)
    try {
      await saveEnhancement(session, p)
      toast('Enhancement saved', 'success')
    } catch { toast('Save failed', 'error') }
    finally { setSaving(false) }
  }

  const Slider = ({ label, k, min, max, step }) => (
    <div style={{ marginBottom:'0.875rem' }}>
      <div style={{ display:'flex', justifyContent:'space-between', fontSize:'0.75rem', fontWeight:500, marginBottom:4 }}>
        <span>{label}</span>
        <span style={{ color:'var(--text)', fontFamily:"'JetBrains Mono','Fira Code','Consolas',monospace", fontSize:'0.6875rem' }}>{p[k]}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={p[k]} onChange={e => set(k, parseFloat(e.target.value))} style={{ width:'100%' }} />
    </div>
  )

  return (
    <div style={{ display:'grid', gridTemplateColumns:'280px 1fr', gap:'1.25rem', minHeight:500 }}>
      {/* Sidebar */}
      <div className="card" style={{ display:'flex', flexDirection:'column', gap:'0.5rem', overflowY:'auto' }}>
        <div style={{ fontWeight:600, fontSize:'0.8125rem', display:'flex', alignItems:'center', gap:'0.5rem', marginBottom:'0.5rem', color:'var(--text-secondary)' }}>
          <SlidersHorizontal size={14} strokeWidth={1.8} style={{ color:'var(--text-muted)' }} /> Adjustments
        </div>
        <Slider label="Sharpening" k="sharpen" min={0} max={3} step={0.1} />
        <Slider label="Denoising" k="denoise" min={0} max={20} step={1} />
        <Slider label="Contrast" k="contrast" min={0} max={4} step={0.1} />
        <Slider label="Brightness" k="brightness" min={-50} max={50} step={1} />
        <label style={{ display:'flex', alignItems:'center', gap:'0.5rem', cursor:'pointer', fontSize:'0.75rem', fontWeight:500, marginBottom:'0.5rem' }}>
          <input type="checkbox" checked={!!p.auto_wb} onChange={e => set('auto_wb', e.target.checked ? 1 : 0)} style={{ accentColor:'var(--text)' }} />
          Auto White Balance
        </label>
        <button className="btn btn-ghost btn-sm" onClick={reset} style={{ fontSize:'0.6875rem', justifyContent:'center' }}>
          <RotateCcw size={13} strokeWidth={1.8} /> Reset
        </button>

        <div style={{ borderTop:'1px solid var(--border)', paddingTop:'0.75rem', marginTop:'0.25rem' }}>
          <div className="text-xs text-muted" style={{ fontWeight:600, textTransform:'uppercase', letterSpacing:'.05em', marginBottom:'0.5rem' }}>Presets</div>
          <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:'0.375rem' }}>
            {Object.keys(PRESETS).map(name => (
              <button key={name} className="btn btn-sm btn-ghost" onClick={() => applyPreset(name)}
                style={{ textTransform:'capitalize', justifyContent:'center', fontSize:'0.6875rem' }}>
                {name}
              </button>
            ))}
          </div>
        </div>

        <div style={{ marginTop:'auto', display:'flex', flexDirection:'column', gap:'0.375rem', paddingTop:'0.75rem', borderTop:'1px solid var(--border)' }}>
          <button className="btn btn-primary btn-sm" onClick={save} disabled={saving} style={{ justifyContent:'center' }}>
            {saving ? <Loader2 size={13} className="spin" /> : <Download size={13} strokeWidth={1.8} />} Save Enhancement
          </button>
          <button className="btn btn-primary btn-sm" onClick={onSendToPipeline} disabled={sendLoading} style={{ justifyContent:'center' }}>
            {sendLoading ? <Loader2 size={13} className="spin" /> : <Send size={13} strokeWidth={1.8} />} Send to Pipeline
          </button>
          <button className="btn btn-ghost btn-sm" onClick={onDone} style={{ justifyContent:'center', fontSize:'0.6875rem' }}>
            <ArrowRight size={13} strokeWidth={1.8} /> Back to Result
          </button>
        </div>
      </div>

      {/* Preview */}
      <div className="card" style={{ display:'flex', alignItems:'center', justifyContent:'center', background:'var(--surface2)', overflow:'hidden' }}>
        <img src={previewUrl} alt="Enhancement preview" style={{ maxWidth:'100%', maxHeight:'100%', objectFit:'contain', borderRadius:'var(--radius-sm)' }} />
      </div>
    </div>
  )
}

/* ══════════════════════════════════════════════════
   Main StitchPage
   ══════════════════════════════════════════════════ */
export default function StitchPage() {
  const [step, setStep] = useState('folder')
  const [folder, setFolder] = useState('')
  const [groups, setGroups] = useState(null)
  const [selected, setSelected] = useState([])
  const [advancedMode, setAdvancedMode] = useState(false)
  const [outputName, setOutputName] = useState('stitched_output.png')
  const [maxDim, setMaxDim] = useState(8192)
  const [upscale, setUpscale] = useState(1.0)
  const [loading, setLoading] = useState(false)
  const [stitchResult, setStitchResult] = useState(null)
  const [sendLoading, setSendLoading] = useState(false)

  const toast = useToast()
  const navigate = useNavigate()

  /* Analyze folder */
  const handleAnalyze = async () => {
    setLoading(true)
    try {
      const data = await analyzeFolder(folder)
      setGroups(data.groups)
      setSelected([])
      setStep('select')
    } catch (err) {
      toast(err.message, 'error')
    } finally { setLoading(false) }
  }

  /* Run stitch */
  const handleStitch = async () => {
    setLoading(true)
    try {
      const data = await runStitch({ folderPath: folder, selectedImages: selected, advancedMode, outputName, maxDim, upscale })
      setStitchResult(data)
      setStep('result')
      toast('Stitching complete!', 'success')
    } catch (err) {
      toast(err.message, 'error')
    } finally { setLoading(false) }
  }

  /* Send stitched image to detection pipeline */
  const handleSendToPipeline = async () => {
    if (!stitchResult) return
    setSendLoading(true)
    try {
      const file = await fetchStitchedFile(stitchResult.session_id)
      // Put the file into sessionStorage marker so UploadPage knows
      // Instead, navigate to /detect with the file already submitted
      const fd = new FormData()
      fd.append('files', file)
      fd.append('yolo_conf', 0.1)
      fd.append('mask_threshold', 0.5)
      fd.append('pixel_to_micron', 1.0)
      fd.append('crop_padding', 30)
      fd.append('nms_iou', 0.3)
      fd.append('use_maskrcnn', true)
      fd.append('use_effnet', true)
      toast('Running detection pipeline on stitched image…', 'info')
      const result = await runDetection(fd)
      sessionStorage.setItem('mp_last_result', JSON.stringify(result))
      const total = result.images?.reduce((s, im) => s + (im.summary?.total || 0), 0) || 0
      toast(`Done! ${total} detections found.`, 'success')
      navigate('/results')
    } catch (err) {
      toast(err.message, 'error')
    } finally { setSendLoading(false) }
  }

  /* Delete session */
  const handleDelete = async () => {
    if (!stitchResult || !window.confirm('Delete this stitched image? This cannot be undone.')) return
    try {
      await deleteStitch(stitchResult.session_id)
      toast('Deleted', 'info')
      setStitchResult(null)
      setStep('folder')
    } catch (err) { toast(err.message, 'error') }
  }

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:'1.25rem' }}>
      {/* Header */}
      <div>
        <h1 style={{ fontSize:'1.375rem', fontWeight:700, marginBottom:'0.25rem' }}>Image Stitching</h1>
        <p className="text-muted" style={{ fontSize:'0.8125rem' }}>
          Combine overlapping partial filter paper images into a single high-resolution mosaic, then send it through the detection pipeline.
        </p>
      </div>

      <StepIndicator current={step} />

      {step === 'folder' && (
        <FolderStep folder={folder} setFolder={setFolder} onAnalyze={handleAnalyze} loading={loading} />
      )}

      {step === 'select' && groups && (
        <SelectStep
          groups={groups} folder={folder}
          selected={selected} setSelected={setSelected}
          advancedMode={advancedMode} setAdvancedMode={setAdvancedMode}
          outputName={outputName} setOutputName={setOutputName}
          maxDim={maxDim} setMaxDim={setMaxDim}
          upscale={upscale} setUpscale={setUpscale}
          onStitch={handleStitch} loading={loading}
          onBack={() => setStep('folder')}
        />
      )}

      {step === 'result' && stitchResult && (
        <ResultStep
          session={stitchResult.session_id}
          meta={stitchResult.meta}
          numImages={stitchResult.num_images}
          advanced={stitchResult.advanced}
          onEnhance={() => setStep('enhance')}
          onSendToPipeline={handleSendToPipeline}
          onDelete={handleDelete}
          onBack={() => { setStitchResult(null); setStep('folder') }}
          sendLoading={sendLoading}
        />
      )}

      {step === 'enhance' && stitchResult && (
        <EnhanceStep
          session={stitchResult.session_id}
          onDone={() => setStep('result')}
          onSendToPipeline={handleSendToPipeline}
          sendLoading={sendLoading}
        />
      )}
    </div>
  )
}
