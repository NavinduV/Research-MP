import React, { useState, useCallback, useRef, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  FolderOpen, Images, Zap, Sparkles, Send, Trash2, ChevronDown, ChevronUp,
  SlidersHorizontal, RotateCcw, CheckCircle2, AlertTriangle, Loader2,
  ImagePlus, ArrowRight, Info, Eye, Download, Layers, UploadCloud, X,
  MousePointerClick, FolderSearch, ChevronLeft, ChevronRight, ToggleLeft, ToggleRight
} from 'lucide-react'
import {
  analyzeFolder, thumbnailUrl, runStitch, stitchPreviewUrl,
  enhancePreviewUrl, saveEnhancement, fetchStitchedFile, deleteStitch,
  uploadAndAnalyzeForStitch,
} from '../api/stitch.js'
import { runDetection } from '../api/detect.js'
import { useToast, useStitchFiles } from '../App.jsx'

/* ── tiny helpers ────────────────────────────────── */
const STEPS = ['folder', 'select', 'result', 'enhance']
const IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp']

function StepIndicator({ current }) {
  const labels = ['Input', 'Select Images', 'Stitch Result', 'Enhance']
  const icons  = [<UploadCloud size={13} strokeWidth={1.8}/>, <Images size={13} strokeWidth={1.8}/>, <Layers size={13} strokeWidth={1.8}/>, <Sparkles size={13} strokeWidth={1.8}/>]
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

/* ── Image Preview Carousel ───────────────────────── */
function ImagePreviewCarousel({ files, onRemove }) {
  const [scrollIdx, setScrollIdx] = useState(0)
  const VISIBLE = 4

  if (files.length === 0) return null

  const canLeft = scrollIdx > 0
  const canRight = scrollIdx + VISIBLE < files.length
  const visibleFiles = files.slice(scrollIdx, scrollIdx + VISIBLE)

  return (
    <div style={{ marginTop: '0.75rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        {files.length > VISIBLE && (
          <button
            onClick={() => setScrollIdx(i => Math.max(0, i - 1))}
            disabled={!canLeft}
            style={{
              border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
              background: canLeft ? 'var(--surface)' : 'var(--surface2)',
              color: canLeft ? 'var(--text)' : 'var(--text-muted)',
              cursor: canLeft ? 'pointer' : 'default',
              width: 28, height: 28, display: 'flex', alignItems: 'center', justifyContent: 'center',
              opacity: canLeft ? 1 : 0.3, flexShrink: 0,
            }}
          >
            <ChevronLeft size={14} />
          </button>
        )}

        <div style={{ display: 'flex', gap: '0.5rem', flex: 1, overflow: 'hidden' }}>
          {visibleFiles.map((f, vi) => {
            const realIdx = scrollIdx + vi
            const url = URL.createObjectURL(f)
            return (
              <div key={realIdx} style={{
                flex: '1 1 0', minWidth: 0, position: 'relative',
                border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
                overflow: 'hidden', background: 'var(--surface2)',
              }}>
                <img
                  src={url}
                  alt={f.name}
                  style={{ width: '100%', height: 90, objectFit: 'cover', display: 'block' }}
                  onLoad={() => URL.revokeObjectURL(url)}
                />
                <div style={{
                  padding: '0.25rem 0.375rem', display: 'flex', alignItems: 'center',
                  justifyContent: 'space-between', gap: '0.25rem',
                }}>
                  <span className="truncate" style={{ fontSize: '0.625rem', fontWeight: 500, color: 'var(--text-secondary)' }}>
                    {f.name}
                  </span>
                  <button
                    onClick={() => onRemove(realIdx)}
                    style={{
                      background: 'none', border: 'none', cursor: 'pointer',
                      color: 'var(--text-muted)', padding: '2px', display: 'flex',
                      alignItems: 'center', flexShrink: 0,
                    }}
                  >
                    <X size={11} />
                  </button>
                </div>
              </div>
            )
          })}
        </div>

        {files.length > VISIBLE && (
          <button
            onClick={() => setScrollIdx(i => Math.min(files.length - VISIBLE, i + 1))}
            disabled={!canRight}
            style={{
              border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
              background: canRight ? 'var(--surface)' : 'var(--surface2)',
              color: canRight ? 'var(--text)' : 'var(--text-muted)',
              cursor: canRight ? 'pointer' : 'default',
              width: 28, height: 28, display: 'flex', alignItems: 'center', justifyContent: 'center',
              opacity: canRight ? 1 : 0.3, flexShrink: 0,
            }}
          >
            <ChevronRight size={14} />
          </button>
        )}
      </div>

      {files.length > VISIBLE && (
        <div style={{ textAlign: 'center', marginTop: '0.375rem' }}>
          <span className="text-xs text-muted">
            Showing {scrollIdx + 1}–{Math.min(scrollIdx + VISIBLE, files.length)} of {files.length}
          </span>
        </div>
      )}
    </div>
  )
}


/* ── STEP 1 : Folder Input — Revamped ─────────────── */
function FolderStep({ folder, setFolder, onAnalyze, onUploadFiles, loading, autoSelect, setAutoSelect }) {
  const [inputMethod, setInputMethod] = useState('drag')  // 'drag' | 'path' | 'browse'
  const [dragOver, setDragOver] = useState(false)
  const [droppedFiles, setDroppedFiles] = useState([])
  const fileInputRef = useRef()
  const folderInputRef = useRef()

  const isImage = (name) => {
    const ext = '.' + name.split('.').pop().toLowerCase()
    return IMAGE_EXTS.includes(ext)
  }

  /* Handle drag & drop — files or folders */
  const handleDrop = useCallback(async (e) => {
    e.preventDefault()
    setDragOver(false)

    const items = e.dataTransfer.items
    const files = e.dataTransfer.files

    // Check if a folder was dropped (using DataTransferItem API)
    if (items && items.length > 0) {
      const firstItem = items[0]
      if (firstItem.webkitGetAsEntry) {
        const entry = firstItem.webkitGetAsEntry()
        if (entry && entry.isDirectory) {
          // Folder dropped — read all image files from it
          const folderFiles = []
          const readEntries = (dirEntry) => {
            return new Promise((resolve) => {
              const reader = dirEntry.createReader()
              const allEntries = []
              const readBatch = () => {
                reader.readEntries((entries) => {
                  if (entries.length === 0) {
                    resolve(allEntries)
                  } else {
                    allEntries.push(...entries)
                    readBatch()
                  }
                })
              }
              readBatch()
            })
          }
          const entryToFile = (fileEntry) => {
            return new Promise((resolve) => {
              fileEntry.file((f) => resolve(f))
            })
          }
          const entries = await readEntries(entry)
          for (const e of entries) {
            if (e.isFile && isImage(e.name)) {
              const file = await entryToFile(e)
              folderFiles.push(file)
            }
          }
          if (folderFiles.length > 0) {
            setDroppedFiles(folderFiles)
          }
          return
        }
      }
    }

    // Regular file drop
    const imageFiles = Array.from(files).filter(f => f.type.startsWith('image/') || isImage(f.name))
    if (imageFiles.length > 0) {
      setDroppedFiles(prev => [...prev, ...imageFiles])
    }
  }, [])

  const handleFileSelect = (e) => {
    const selected = Array.from(e.target.files).filter(f => f.type.startsWith('image/') || isImage(f.name))
    if (selected.length > 0) {
      setDroppedFiles(prev => [...prev, ...selected])
    }
  }

  const removeFile = (idx) => {
    setDroppedFiles(prev => prev.filter((_, j) => j !== idx))
  }

  const handleUploadDropped = () => {
    if (droppedFiles.length > 0) {
      onUploadFiles(droppedFiles)
    }
  }

  const handleBrowseFolder = () => {
    folderInputRef.current?.click()
  }

  const handleFolderBrowse = (e) => {
    const files = Array.from(e.target.files)
    if (files.length > 0) {
      // Extract the common folder path from the webkitRelativePath
      const first = files[0]
      if (first.webkitRelativePath) {
        const parts = first.webkitRelativePath.split('/')
        if (parts.length > 1) {
          // The first part is the folder name — we'll upload the files
          const imageFiles = files.filter(f => f.type.startsWith('image/') || isImage(f.name))
          if (imageFiles.length > 0) {
            setDroppedFiles(imageFiles)
            setInputMethod('drag')
          }
        }
      }
    }
  }

  const methods = [
    { id: 'drag', icon: <UploadCloud size={13} strokeWidth={1.8} />, label: 'Drag & Drop' },
    { id: 'path', icon: <FolderSearch size={13} strokeWidth={1.8} />, label: 'Folder Path' },
    { id: 'browse', icon: <FolderOpen size={13} strokeWidth={1.8} />, label: 'Browse' },
  ]

  return (
    <div className="card" style={{ display:'flex', flexDirection:'column', gap:'1.25rem' }}>
      {/* Header + Auto/Manual toggle */}
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', flexWrap:'wrap', gap:'0.75rem' }}>
        <div style={{ fontWeight:600, display:'flex', alignItems:'center', gap:'0.5rem', fontSize:'0.8125rem', color:'var(--text-secondary)' }}>
          <UploadCloud size={15} strokeWidth={1.8} style={{ color:'var(--text-muted)' }} /> Input Images
        </div>

        {/* Auto / Manual selection toggle */}
        <div style={{
          display:'flex', alignItems:'center', gap:'0.5rem',
          background:'var(--surface2)', padding:'4px 10px', borderRadius:'var(--radius-sm)',
          border:'1px solid var(--border)',
        }}>
          <span style={{ fontSize:'0.6875rem', fontWeight:500, color:'var(--text-muted)' }}>Selection:</span>
          <button
            onClick={() => setAutoSelect(!autoSelect)}
            style={{
              display:'flex', alignItems:'center', gap:'0.35rem',
              padding:'3px 8px', borderRadius:'calc(var(--radius-sm) - 2px)',
              fontSize:'0.6875rem', fontWeight:600, letterSpacing:'.02em',
              background: autoSelect ? 'var(--text)' : 'transparent',
              color: autoSelect ? '#fff' : 'var(--text-secondary)',
              border:'none', cursor:'pointer', transition:'all .15s',
            }}
          >
            {autoSelect ? <ToggleRight size={12} /> : <ToggleLeft size={12} />}
            {autoSelect ? 'Auto' : 'Manual'}
          </button>
          <span className="text-xs text-muted" style={{ maxWidth:200 }}>
            {autoSelect ? 'All images selected automatically' : 'Manually pick images'}
          </span>
        </div>
      </div>

      {/* Input Method Pill Bar */}
      <div style={{ display:'flex', gap:'0.125rem', background:'var(--surface2)', borderRadius:'var(--radius)', padding:'0.1875rem', border:'1px solid var(--border)' }}>
        {methods.map(m => (
          <button
            key={m.id}
            onClick={() => setInputMethod(m.id)}
            style={{
              flex:1, display:'flex', alignItems:'center', justifyContent:'center', gap:'0.375rem',
              fontSize:'0.75rem', fontWeight: inputMethod === m.id ? 600 : 500,
              padding:'0.3rem 0.75rem', borderRadius:'var(--radius-sm)',
              color: inputMethod === m.id ? 'var(--text)' : 'var(--text-muted)',
              background: inputMethod === m.id ? 'var(--surface)' : 'transparent',
              border:'none', cursor:'pointer', transition:'all .15s',
              boxShadow: inputMethod === m.id ? 'var(--shadow-sm)' : 'none',
            }}
          >
            {m.icon} {m.label}
          </button>
        ))}
      </div>

      {/* --- METHOD: Drag & Drop ---- */}
      {inputMethod === 'drag' && (
        <div style={{ display:'flex', flexDirection:'column', gap:'1rem' }}>
          {/* Hidden inputs */}
          <input ref={fileInputRef} type="file" accept="image/*" multiple hidden onChange={handleFileSelect} />
          <input ref={folderInputRef} type="file" webkitdirectory="" directory="" multiple hidden onChange={handleFolderBrowse} />

          {/* Drop zone */}
          {droppedFiles.length === 0 ? (
            <div
              onDragOver={e => { e.preventDefault(); setDragOver(true) }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              style={{
                border: `2px dashed ${dragOver ? 'var(--text)' : 'var(--border-strong)'}`,
                borderRadius: 'var(--radius)',
                padding: '2.5rem 2rem',
                textAlign: 'center',
                cursor: 'pointer',
                background: dragOver ? 'var(--surface)' : 'var(--surface2)',
                transition: 'all .2s',
                position: 'relative',
              }}
            >
              <div style={{ display:'flex', justifyContent:'center', marginBottom:'0.75rem' }}>
                <div style={{
                  width:48, height:48, borderRadius:'var(--radius)',
                  background: dragOver ? 'var(--text)' : 'var(--surface)',
                  border: `1px solid ${dragOver ? 'var(--text)' : 'var(--border)'}`,
                  display:'flex', alignItems:'center', justifyContent:'center',
                  transition: 'all .2s',
                  color: dragOver ? '#fff' : 'var(--text-muted)',
                }}>
                  <UploadCloud size={22} strokeWidth={1.5} />
                </div>
              </div>
              <p style={{ fontWeight:600, marginBottom:4, fontSize:'0.8125rem', color: dragOver ? 'var(--text)' : 'var(--text-secondary)' }}>
                Drop images or folders here
              </p>
              <p className="text-xs text-muted" style={{ margin:0 }}>
                Drag individual images, multiple files, or an entire folder
              </p>
              <p className="text-xs text-muted" style={{ marginTop:4, opacity:0.7 }}>
                PNG, JPG, TIFF, BMP, WebP supported
              </p>

              {/* Click to browse sub-actions */}
              <div style={{ display:'flex', justifyContent:'center', gap:'0.5rem', marginTop:'1rem' }}>
                <button
                  className="btn btn-sm btn-outline"
                  onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click() }}
                  style={{ fontSize:'0.6875rem' }}
                >
                  <ImagePlus size={12} strokeWidth={1.8} /> Browse Images
                </button>
                <button
                  className="btn btn-sm btn-outline"
                  onClick={(e) => { e.stopPropagation(); folderInputRef.current?.click() }}
                  style={{ fontSize:'0.6875rem' }}
                >
                  <FolderOpen size={12} strokeWidth={1.8} /> Browse Folder
                </button>
              </div>
            </div>
          ) : (
            <div
              onDragOver={e => { e.preventDefault(); setDragOver(true) }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              style={{
                outline: dragOver ? '2px dashed var(--text-secondary)' : 'none',
                borderRadius: 'var(--radius)',
                outlineOffset: -2,
              }}
            >
              <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:'0.5rem' }}>
                <span style={{ fontSize:'0.75rem', fontWeight:600, color:'var(--text-secondary)', display:'flex', alignItems:'center', gap:'0.375rem' }}>
                  <Images size={13} strokeWidth={1.8} style={{ color:'var(--text-muted)' }} />
                  {droppedFiles.length} image{droppedFiles.length !== 1 ? 's' : ''} ready
                </span>
                <div style={{ display:'flex', gap:'0.375rem' }}>
                  <button
                    className="btn btn-sm btn-ghost"
                    onClick={() => fileInputRef.current?.click()}
                    style={{ fontSize:'0.6875rem' }}
                  >
                    <UploadCloud size={12} strokeWidth={1.8} /> Add More
                  </button>
                  <button
                    className="btn btn-sm btn-ghost"
                    onClick={() => setDroppedFiles([])}
                    style={{ fontSize:'0.6875rem', color:'#991b1b' }}
                  >
                    <Trash2 size={12} strokeWidth={1.8} /> Clear All
                  </button>
                </div>
              </div>
              <ImagePreviewCarousel files={droppedFiles} onRemove={removeFile} />
            </div>
          )}

          {/* Analyze button for dropped files */}
          {droppedFiles.length > 0 && (
            <button
              className="btn btn-primary"
              disabled={loading}
              onClick={handleUploadDropped}
              style={{ alignSelf:'stretch' }}
            >
              {loading ? <><Loader2 size={14} className="spin" /> Uploading & Analyzing…</> : <><Eye size={14} strokeWidth={1.8} /> Upload & Analyze {droppedFiles.length} Images</>}
            </button>
          )}
        </div>
      )}

      {/* --- METHOD: Folder Path ---- */}
      {inputMethod === 'path' && (
        <div style={{ display:'flex', flexDirection:'column', gap:'0.75rem' }}>
          <p className="text-sm text-muted" style={{ margin:0 }}>
            Enter the full path to a folder containing overlapping partial filter paper images.
          </p>
          <div style={{ display:'flex', gap:'0.75rem' }}>
            <input
              type="text" value={folder}
              onChange={e => setFolder(e.target.value)}
              placeholder="e.g.  D:\datasets\raw\Sample1\Macro"
              className="input"
              style={{ flex:1, fontFamily:"'JetBrains Mono','Fira Code','Consolas',monospace", fontSize:'0.75rem' }}
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
        </div>
      )}

      {/* --- METHOD: Browse ---- */}
      {inputMethod === 'browse' && (
        <div style={{ display:'flex', flexDirection:'column', gap:'0.75rem' }}>
          <input ref={folderInputRef} type="file" webkitdirectory="" directory="" multiple hidden onChange={handleFolderBrowse} />

          <p className="text-sm text-muted" style={{ margin:0 }}>
            Use your system's file browser to select a folder containing the images to stitch.
          </p>

          <div style={{ display:'flex', gap:'0.75rem', alignItems:'center' }}>
            <button
              className="btn btn-secondary"
              onClick={handleBrowseFolder}
              style={{ whiteSpace:'nowrap' }}
            >
              <FolderOpen size={14} strokeWidth={1.8} /> Select Folder
            </button>
            {droppedFiles.length > 0 && (
              <span style={{ fontSize:'0.75rem', fontWeight:500, color:'var(--text-secondary)', display:'flex', alignItems:'center', gap:'0.375rem' }}>
                <CheckCircle2 size={13} strokeWidth={1.8} style={{ color:'var(--success)' }} />
                {droppedFiles.length} images found
              </span>
            )}
          </div>

          {droppedFiles.length > 0 && (
            <>
              <ImagePreviewCarousel files={droppedFiles} onRemove={removeFile} />
              <button
                className="btn btn-primary"
                disabled={loading}
                onClick={handleUploadDropped}
                style={{ alignSelf:'stretch' }}
              >
                {loading ? <><Loader2 size={14} className="spin" /> Uploading & Analyzing…</> : <><Eye size={14} strokeWidth={1.8} /> Upload & Analyze {droppedFiles.length} Images</>}
              </button>
            </>
          )}
        </div>
      )}

      {/* Info tip */}
      <div style={{ background:'var(--surface2)', border:'1px solid var(--border)', borderRadius:'var(--radius-sm)', padding:'0.75rem 1rem', fontSize:'0.75rem', color:'var(--text-muted)', display:'flex', gap:'0.5rem', alignItems:'flex-start' }}>
        <Info size={13} strokeWidth={1.8} style={{ marginTop:2, flexShrink:0 }} />
        <span>
          Images will be grouped by brightness so you can select the best-matching set for stitching.
          {autoSelect && <strong> Auto mode will pre-select all images for you.</strong>}
        </span>
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
            <button className="btn btn-ghost" onClick={onBack}>&larr; Change input</button>
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
  const [autoSelect, setAutoSelect] = useState(true)

  const toast = useToast()
  const navigate = useNavigate()
  const stitchCtx = useStitchFiles()

  /* ── Auto-process files passed from Detect page ── */
  useEffect(() => {
    if (!stitchCtx.files || stitchCtx.files.length < 2) return
    const filesToUpload = stitchCtx.files
    stitchCtx.clear() // consume so it doesn't re-trigger
    const process = async () => {
      setLoading(true)
      toast(`Uploading ${filesToUpload.length} images for stitching…`, 'info')
      try {
        const data = await uploadAndAnalyzeForStitch(filesToUpload)
        setFolder(data.folder)
        setGroups(data.groups)
        const allPaths = Object.values(data.groups).flat().map(img => img.path)
        setSelected(allPaths)
        setStep('select')
        toast(`${data.uploaded_count} images uploaded and analyzed.`, 'success')
      } catch (err) {
        toast(err.message, 'error')
      } finally { setLoading(false) }
    }
    process()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  /* Analyze folder (path mode) */
  const handleAnalyze = async () => {
    setLoading(true)
    try {
      const data = await analyzeFolder(folder)
      setGroups(data.groups)
      // Auto-select all if auto mode is on
      if (autoSelect) {
        const allPaths = Object.values(data.groups).flat().map(img => img.path)
        setSelected(allPaths)
      } else {
        setSelected([])
      }
      setStep('select')
    } catch (err) {
      toast(err.message, 'error')
    } finally { setLoading(false) }
  }

  /* Upload dropped/browsed files */
  const handleUploadFiles = async (files) => {
    setLoading(true)
    toast(`Uploading ${files.length} images for stitching…`, 'info')
    try {
      const data = await uploadAndAnalyzeForStitch(files)
      setFolder(data.folder)
      setGroups(data.groups)
      // Auto-select all if auto mode is on
      if (autoSelect) {
        const allPaths = Object.values(data.groups).flat().map(img => img.path)
        setSelected(allPaths)
      } else {
        setSelected([])
      }
      setStep('select')
      toast(`${data.uploaded_count} images uploaded and analyzed.`, 'success')
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
        <FolderStep
          folder={folder} setFolder={setFolder}
          onAnalyze={handleAnalyze}
          onUploadFiles={handleUploadFiles}
          loading={loading}
          autoSelect={autoSelect}
          setAutoSelect={setAutoSelect}
        />
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
