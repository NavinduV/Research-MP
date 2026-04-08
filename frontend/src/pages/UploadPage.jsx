import React, { useState, useRef, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { UploadCloud, Image as ImageIcon, X, Settings, SlidersHorizontal, FileImage, Play, Info, ChevronLeft, ChevronRight, Layers, FlaskConical } from 'lucide-react'
import { useToast, usePipelineMode, usePipelineJob, useStitchFiles, useUploadFiles, useSoilWeight } from '../App.jsx'

const DEFAULTS = {
  yolo_conf: 0.1,
  mask_threshold: 0.5,
  crop_padding: 30,
  nms_iou: 0.3,
  use_maskrcnn: true,
  use_effnet: true,
  yolo_path: '',
  maskrcnn_path: '',
  effnet_path: '',
}

function Toggle({ checked, onChange, label, description }) {
  return (
    <label className="toggle-wrap" style={{ display: 'flex', alignItems: 'flex-start', gap: '0.75rem', cursor: 'pointer' }}>
      <span className={`toggle-switch ${checked ? 'active' : ''}`} style={{ marginTop: 2 }}>
        <input type="checkbox" className="sr-only" checked={checked} onChange={e => onChange(e.target.checked)} />
      </span>
      <span>
        <div style={{ fontWeight: 500, fontSize: '0.8125rem', color: 'var(--text)' }}>{label}</div>
        {description && <div className="text-xs text-muted" style={{ marginTop: 2 }}>{description}</div>}
      </span>
    </label>
  )
}

function InfoTip({ tip }) {
  return (
    <span className="info-tip-wrap">
      <Info size={13} className="info-tip-icon" />
      <span className="info-tip-bubble">{tip}</span>
    </span>
  )
}

function SliderField({ label, value, onChange, min, max, step, unit, tip }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.375rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
          <label className="field-label" style={{ margin: 0 }}>{label}</label>
          {tip && <InfoTip tip={tip} />}
        </span>
        <span style={{
          fontWeight: 700, color: 'var(--text)', minWidth: 48, textAlign: 'right',
          fontFamily: "'JetBrains Mono','Fira Code','Consolas',monospace",
          fontSize: '0.75rem',
        }}>
          {value}{unit}
        </span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(parseFloat(e.target.value))} />
    </div>
  )
}

/* ── Image Preview Carousel ──────────────────────── */
function ImagePreviewCarousel({ files, onRemove }) {
  const [scrollIdx, setScrollIdx] = useState(0)
  const VISIBLE = 3

  if (files.length === 0) return null

  const canLeft = scrollIdx > 0
  const canRight = scrollIdx + VISIBLE < files.length
  const visibleFiles = files.slice(scrollIdx, scrollIdx + VISIBLE)

  return (
    <div style={{ marginTop: '0.75rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        {/* Left arrow */}
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

        {/* Image previews */}
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
                  style={{ width: '100%', height: 120, objectFit: 'cover', display: 'block' }}
                  onLoad={() => URL.revokeObjectURL(url)}
                />
                <div style={{
                  padding: '0.375rem 0.5rem', display: 'flex', alignItems: 'center',
                  justifyContent: 'space-between', gap: '0.25rem',
                }}>
                  <span className="truncate" style={{ fontSize: '0.6875rem', fontWeight: 500, color: 'var(--text-secondary)' }}>
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
                    <X size={12} />
                  </button>
                </div>
              </div>
            )
          })}
        </div>

        {/* Right arrow */}
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

      {/* Pagination indicator */}
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

function DropZone({ files, onChange, addBtnRef }) {
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef()

  // Expose the file picker trigger so the parent "Add More" button can use it
  React.useImperativeHandle(addBtnRef, () => ({
    openPicker: () => inputRef.current?.click(),
  }))

  const handleDrop = useCallback(e => {
    e.preventDefault()
    setDragOver(false)
    const dropped = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'))
    onChange(prev => [...prev, ...dropped])
  }, [onChange])

  const handleFiles = e => {
    const selected = Array.from(e.target.files).filter(f => f.type.startsWith('image/'))
    onChange(prev => [...prev, ...selected])
  }

  const removeFile = (idx) => {
    onChange(prev => prev.filter((_, j) => j !== idx))
  }

  const hasFiles = files.length > 0

  return (
    <div
      onDragOver={hasFiles ? (e => { e.preventDefault(); setDragOver(true) }) : undefined}
      onDragLeave={hasFiles ? (() => setDragOver(false)) : undefined}
      onDrop={hasFiles ? handleDrop : undefined}
      style={hasFiles && dragOver ? { outline: '2px dashed var(--text-secondary)', borderRadius: 'var(--radius)', outlineOffset: -2 } : {}}
    >
      {/* Hidden file input — always present */}
      <input ref={inputRef} type="file" accept="image/*" multiple hidden onChange={handleFiles} />

      {/* Show full drop zone only when no files added yet */}
      {!hasFiles && (
        <div
          onClick={() => inputRef.current?.click()}
          onDragOver={e => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          style={{
            border: `2px dashed ${dragOver ? 'var(--text-secondary)' : 'var(--border)'}`,
            borderRadius: 'var(--radius)',
            padding: '2.5rem',
            textAlign: 'center',
            cursor: 'pointer',
            background: 'var(--surface2)',
            transition: 'all .15s',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '0.5rem', color: 'var(--text-muted)' }}>
            <UploadCloud size={28} strokeWidth={1.5} />
          </div>
          <p style={{ fontWeight: 500, marginBottom: 4, fontSize: '0.8125rem' }}>Drop images here or click to browse</p>
          <p className="text-xs text-muted">PNG, JPG, TIFF, BMP supported · Multiple files OK</p>
        </div>
      )}

      {/* Image preview carousel — shown when files exist */}
      {hasFiles && <ImagePreviewCarousel files={files} onRemove={removeFile} />}
    </div>
  )
}

export default function UploadPage() {
  const { files, setFiles } = useUploadFiles()
  const [cfg, setCfg] = useState(DEFAULTS)
  const toast = useToast()
  const { mode: pipelineMode } = usePipelineMode()
  const { running, startPipeline } = usePipelineJob()
  const stitchCtx = useStitchFiles()
  const navigate = useNavigate()
  const dropZoneRef = useRef()
  const soilWeightCtx = useSoilWeight()

  const set = (key, val) => setCfg(c => ({ ...c, [key]: val }))

  const handleRun = async () => {
    if (files.length === 0) { toast('Please select at least one image.', 'error'); return }
    if (running) { toast('A pipeline is already running.', 'error'); return }

    const fd = new FormData()
    files.forEach(f => fd.append('files', f))
    fd.append('yolo_conf',       cfg.yolo_conf)
    fd.append('mask_threshold',  cfg.mask_threshold)
    fd.append('crop_padding',    cfg.crop_padding)
    fd.append('nms_iou',         cfg.nms_iou)
    fd.append('use_maskrcnn',    cfg.use_maskrcnn)
    fd.append('use_effnet',      cfg.use_effnet)
    fd.append('pipeline_mode',   pipelineMode)
    if (cfg.yolo_path)     fd.append('yolo_path',     cfg.yolo_path)
    if (cfg.maskrcnn_path) fd.append('maskrcnn_path', cfg.maskrcnn_path)
    if (cfg.effnet_path)   fd.append('effnet_path',   cfg.effnet_path)

    // Fire and forget — pipeline runs in background via App context
    // Save soil weight to sessionStorage for results page
    if (pipelineMode === 'macro' && soilWeightCtx.weight) {
      sessionStorage.setItem('mp_soil_weight', soilWeightCtx.weight)
      fd.append('soil_weight_g', soilWeightCtx.weight)
    } else {
      sessionStorage.removeItem('mp_soil_weight')
    }
    startPipeline(fd)
    setFiles([])
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Header */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.25rem' }}>
          <h1 style={{ fontSize: '1.375rem', fontWeight: 700, margin: 0 }}>
            Microplastic Detection
          </h1>
          <span style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.25rem',
            padding: '3px 10px',
            borderRadius: 'var(--radius-sm)',
            fontSize: '0.625rem',
            fontWeight: 700,
            letterSpacing: '.06em',
            textTransform: 'uppercase',
            background: 'var(--surface2)',
            color: 'var(--text-secondary)',
            border: '1px solid var(--border)',
          }}>
            {pipelineMode} mode
          </span>
        </div>
        <p className="text-muted" style={{ fontSize: '0.8125rem' }}>
          Upload {pipelineMode === 'macro' ? 'macro' : 'micro'} microscopy images. The pipeline runs YOLO → EfficientNet classification → Mask&nbsp;R-CNN segmentation → size analysis.
        </p>
      </div>

      {/* Running indicator */}
      {running && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: '0.625rem',
          padding: '0.75rem 1rem', borderRadius: 'var(--radius)',
          background: 'var(--surface2)', border: '1px solid var(--border)',
        }}>
          <span className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} />
          <span style={{ fontSize: '0.8125rem', fontWeight: 500, color: 'var(--text-secondary)' }}>
            Pipeline is running in the background. You can navigate freely — you'll be notified when it completes.
          </span>
        </div>
      )}

      {/* Top row: Input Images + Parameters side by side */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.25rem', alignItems: 'stretch' }}>
        {/* Input Images */}
        <div className="card">
          <div style={{ fontWeight: 600, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>
            <ImageIcon size={15} strokeWidth={1.8} style={{ color: 'var(--text-muted)' }} /> Input Images
            {files.length > 0 && (
              <span className="badge badge-neutral" style={{ marginLeft: '0.25rem' }}>{files.length} file{files.length !== 1 ? 's' : ''}</span>
            )}
            {/* Add More button — appears at top-right when files exist */}
            {files.length > 0 && (
              <button
                className="btn btn-ghost btn-sm"
                onClick={() => dropZoneRef.current?.openPicker()}
                style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '0.3rem', fontSize: '0.6875rem' }}
              >
                <UploadCloud size={13} strokeWidth={1.8} /> Add More
              </button>
            )}
          </div>
          <DropZone files={files} onChange={setFiles} addBtnRef={dropZoneRef} />
        </div>

        {/* Parameters */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          <div
            style={{
              fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem',
              color: 'var(--text-secondary)', fontSize: '0.8125rem', width: '100%',
            }}
          >
            <SlidersHorizontal size={15} strokeWidth={1.8} style={{ color: 'var(--text-muted)' }} />
            Parameters
          </div>

          <>
            <SliderField label="YOLO Confidence" value={cfg.yolo_conf} onChange={v => set('yolo_conf', v)}
              min={0.01} max={0.95} step={0.01} unit=""
              tip="Minimum score a detection must have to be kept." />
            <SliderField label="Mask Threshold" value={cfg.mask_threshold} onChange={v => set('mask_threshold', v)}
              min={0.1} max={0.9} step={0.05} unit=""
              tip="Cutoff for converting mask probabilities into a binary shape." />
            <SliderField label="Crop Padding" value={cfg.crop_padding} onChange={v => set('crop_padding', v)}
              min={0} max={80} step={1} unit=" px"
              tip="Extra pixels added around each detected bounding box." />
            <SliderField label="NMS IoU" value={cfg.nms_iou} onChange={v => set('nms_iou', v)}
              min={0.05} max={0.9} step={0.05} unit=""
              tip="Overlap threshold for removing duplicate detections." />
          </>
        </div>
      </div>

      {/* Pipeline toggles + Soil Weight — side by side */}
      <div style={{ display: 'grid', gridTemplateColumns: pipelineMode === 'macro' ? '1fr 1fr' : '1fr', gap: '1.25rem', alignItems: 'stretch' }}>
        {/* Pipeline toggles */}
        <div className="card">
          <div style={{ fontWeight: 600, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>
            <Settings size={15} strokeWidth={1.8} style={{ color: 'var(--text-muted)' }} /> Pipeline Modules
            {/* Pipeline mode indicator */}
            <span className="badge badge-primary" style={{ marginLeft: 'auto', fontSize: '0.5625rem' }}>
              {!cfg.use_effnet ? 'YOLO Only' : !cfg.use_maskrcnn ? 'YOLO + EfficientNet' : 'Full Pipeline'}
            </span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            <Toggle
              checked={cfg.use_effnet}
              onChange={v => {
                setCfg(c => ({
                  ...c,
                  use_effnet: v,
                  // Auto-disable Mask R-CNN when EfficientNet is turned off
                  use_maskrcnn: v ? c.use_maskrcnn : false,
                }))
              }}
              label="EfficientNet Classification"
              description="Refines YOLO class predictions (fiber / film / fragment)"
            />
            <div style={{ opacity: cfg.use_effnet ? 1 : 0.45, transition: 'opacity .2s' }}>
              <Toggle
                checked={cfg.use_maskrcnn}
                onChange={v => {
                  if (cfg.use_effnet) set('use_maskrcnn', v)
                }}
                label="Mask R-CNN Segmentation"
                description={
                  cfg.use_effnet
                    ? 'Precise pixel-level masks for size measurement. Falls back to ellipse if disabled.'
                    : '⚠ Requires EfficientNet Classification to be enabled first.'
                }
              />
            </div>
          </div>
        </div>

        {/* Soil Weight Input — macro mode only */}
        {pipelineMode === 'macro' && (
          <div className="card">
            <div style={{ fontWeight: 600, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8125rem', color: 'var(--text-secondary)' }}>
              <FlaskConical size={15} strokeWidth={1.8} style={{ color: 'var(--text-muted)' }} /> Soil Sample Weight
              <InfoTip tip="Enter the weight of the soil sample in grams. Used to estimate particle concentration (particles per kg)." />
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              <div style={{ position: 'relative', maxWidth: 280 }}>
                <input
                  type="number"
                  className="input"
                  placeholder="e.g. 250"
                  min="0.1"
                  step="any"
                  value={soilWeightCtx.weight}
                  onChange={e => soilWeightCtx.setWeight(e.target.value)}
                  style={{
                    paddingRight: '2.5rem',
                    fontFamily: "'JetBrains Mono','Fira Code','Consolas',monospace",
                    fontSize: '0.875rem',
                    fontWeight: 600,
                  }}
                />
                <span style={{
                  position: 'absolute',
                  right: '0.75rem',
                  top: '50%',
                  transform: 'translateY(-50%)',
                  fontSize: '0.75rem',
                  fontWeight: 600,
                  color: 'var(--text-muted)',
                  pointerEvents: 'none',
                }}>
                  g
                </span>
              </div>
              <span className="text-xs text-muted">
                Concentration will be calculated as <strong>particles per kg</strong> in the results.
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Action buttons */}
      <div style={{ display: 'flex', gap: '0.625rem' }}>
        {/* Stitch button — macro mode only, 2+ images */}
        {pipelineMode === 'macro' && files.length > 1 && (
          <button
            className="btn btn-secondary btn-lg"
            onClick={() => {
              stitchCtx.setFiles([...files])
              navigate('/stitch')
            }}
            disabled={running}
            style={{ flex: '0 0 auto', justifyContent: 'center', fontSize: '0.8125rem', letterSpacing: '.02em' }}
          >
            <Layers size={15} strokeWidth={1.8} /> Stitch {files.length} Images
          </button>
        )}

        {/* Run detection button */}
        <button
          className="btn btn-primary btn-lg"
          onClick={handleRun}
          disabled={running || files.length === 0}
          style={{ flex: 1, justifyContent: 'center', fontSize: '0.8125rem', letterSpacing: '.02em' }}
        >
          {running ? (
            <>
              <span className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} />
              Pipeline running…
            </>
          ) : (
            <><Play size={15} fill="currentColor" /> Run {pipelineMode.charAt(0).toUpperCase() + pipelineMode.slice(1)} Detection Pipeline</>
          )}
        </button>
      </div>
    </div>
  )
}
