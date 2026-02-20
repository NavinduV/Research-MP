import React, { useState, useRef, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { runDetection } from '../api/detect.js'
import { useToast } from '../App.jsx'

const DEFAULTS = {
  yolo_conf: 0.1,
  mask_threshold: 0.5,
  pixel_to_micron: 1.0,
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
    <label className="toggle-wrap" style={{ alignItems: 'flex-start', gap: '0.75rem' }}>
      <span className="toggle-switch" style={{ marginTop: 2 }}>
        <input type="checkbox" checked={checked} onChange={e => onChange(e.target.checked)} />
        <span className="toggle-track">
          <span className="toggle-thumb" />
        </span>
      </span>
      <span>
        <div style={{ fontWeight: 500, fontSize: '0.875rem', color: 'var(--text)' }}>{label}</div>
        {description && <div className="text-xs text-muted" style={{ marginTop: 2 }}>{description}</div>}
      </span>
    </label>
  )
}

function SliderField({ label, value, onChange, min, max, step, unit, helperText }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.375rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
        <label className="field-label" style={{ margin: 0 }}>{label}</label>
        <span style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--primary)', minWidth: 48, textAlign: 'right' }}>
          {value}{unit}
        </span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(parseFloat(e.target.value))} />
      {helperText && <span className="text-xs text-muted">{helperText}</span>}
    </div>
  )
}

function DropZone({ files, onChange }) {
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef()

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

  return (
    <div>
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={e => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        style={{
          border: `2px dashed ${dragOver ? 'var(--primary)' : 'var(--border)'}`,
          borderRadius: 'var(--radius)',
          padding: '2.5rem',
          textAlign: 'center',
          cursor: 'pointer',
          background: dragOver ? 'rgba(13,148,136,0.05)' : 'var(--surface2)',
          transition: 'all .15s',
        }}
      >
        <div style={{ fontSize: 36, marginBottom: '0.75rem', opacity: 0.6 }}>📂</div>
        <p style={{ fontWeight: 500, marginBottom: 4 }}>Drop images here or click to browse</p>
        <p className="text-sm text-muted">PNG, JPG, TIFF, BMP supported • Multiple files OK</p>
        <input ref={inputRef} type="file" accept="image/*" multiple hidden onChange={handleFiles} />
      </div>

      {files.length > 0 && (
        <div style={{ marginTop: '1rem', display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
          {files.map((f, i) => (
            <div key={i} style={{
              display: 'flex', alignItems: 'center', gap: '0.5rem',
              background: 'var(--surface2)', border: '1px solid var(--border)',
              borderRadius: 'var(--radius-sm)', padding: '0.375rem 0.75rem',
              fontSize: '0.8125rem',
            }}>
              <span>🖼️</span>
              <span className="truncate" style={{ maxWidth: 180 }}>{f.name}</span>
              <button
                onClick={() => onChange(prev => prev.filter((_, j) => j !== i))}
                style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', fontSize: 14, padding: '0 2px' }}
              >✕</button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default function UploadPage() {
  const [files, setFiles] = useState([])
  const [cfg, setCfg] = useState(DEFAULTS)
  const [loading, setLoading] = useState(false)
  const [advanced, setAdvanced] = useState(false)
  const navigate = useNavigate()
  const toast = useToast()

  const set = (key, val) => setCfg(c => ({ ...c, [key]: val }))

  const handleRun = async () => {
    if (files.length === 0) { toast('Please select at least one image.', 'error'); return }
    setLoading(true)
    try {
      const fd = new FormData()
      files.forEach(f => fd.append('files', f))
      fd.append('yolo_conf',       cfg.yolo_conf)
      fd.append('mask_threshold',  cfg.mask_threshold)
      fd.append('pixel_to_micron', cfg.pixel_to_micron)
      fd.append('crop_padding',    cfg.crop_padding)
      fd.append('nms_iou',         cfg.nms_iou)
      fd.append('use_maskrcnn',    cfg.use_maskrcnn)
      fd.append('use_effnet',      cfg.use_effnet)
      if (cfg.yolo_path)     fd.append('yolo_path',     cfg.yolo_path)
      if (cfg.maskrcnn_path) fd.append('maskrcnn_path', cfg.maskrcnn_path)
      if (cfg.effnet_path)   fd.append('effnet_path',   cfg.effnet_path)

      const result = await runDetection(fd)
      // Persist to sessionStorage for the results page
      sessionStorage.setItem('mp_last_result', JSON.stringify(result))
      toast(`Done! ${result.images?.reduce((s, im) => s + (im.summary?.total || 0), 0)} detections.`, 'success')
      navigate('/results')
    } catch (err) {
      toast(err.message, 'error')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: '1.5rem', alignItems: 'start' }}>
      {/* Left: Upload + Run */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        {/* Header */}
        <div>
          <h1 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: '0.25rem' }}>
            Microplastic Detection
          </h1>
          <p className="text-muted" style={{ fontSize: '0.875rem' }}>
            Upload microscopy images. The pipeline runs YOLO → EfficientNet classification → Mask&nbsp;R-CNN segmentation → size analysis.
          </p>
        </div>

        {/* Drop zone */}
        <div className="card">
          <div style={{ fontWeight: 600, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span>📤</span> Input Images
          </div>
          <DropZone files={files} onChange={setFiles} />
        </div>

        {/* Pipeline toggles */}
        <div className="card">
          <div style={{ fontWeight: 600, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span>⚙️</span> Pipeline Modules
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            <Toggle
              checked={cfg.use_effnet}
              onChange={v => set('use_effnet', v)}
              label="EfficientNet Classification"
              description="Refines YOLO class predictions (fiber / film / fragment)"
            />
            <Toggle
              checked={cfg.use_maskrcnn}
              onChange={v => set('use_maskrcnn', v)}
              label="Mask R-CNN Segmentation"
              description="Precise pixel-level masks for size measurement. Falls back to ellipse if disabled."
            />
          </div>
        </div>

        {/* Run button */}
        <button
          className="btn btn-primary btn-lg"
          onClick={handleRun}
          disabled={loading || files.length === 0}
          style={{ width: '100%', justifyContent: 'center' }}
        >
          {loading ? (
            <>
              <span className="spinner" style={{ width: 18, height: 18, borderWidth: 2 }} />
              Running pipeline…
            </>
          ) : (
            <>▶ Run Detection Pipeline</>
          )}
        </button>
      </div>

      {/* Right: Config */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', position: 'sticky', top: 72 }}>
        <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          <div style={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span>🎛️</span> Parameters
          </div>

          <SliderField
            label="YOLO Confidence"
            value={cfg.yolo_conf}
            onChange={v => set('yolo_conf', v)}
            min={0.01} max={0.95} step={0.01}
            unit=""
            helperText="Minimum detection confidence (lower = more detections)"
          />

          <SliderField
            label="Mask Threshold"
            value={cfg.mask_threshold}
            onChange={v => set('mask_threshold', v)}
            min={0.1} max={0.9} step={0.05}
            unit=""
            helperText="Mask binarisation threshold"
          />

          <SliderField
            label="Pixel → Micron"
            value={cfg.pixel_to_micron}
            onChange={v => set('pixel_to_micron', v)}
            min={0.1} max={10.0} step={0.1}
            unit=" µm/px"
            helperText="Calibrate with a stage micrometer"
          />

          <SliderField
            label="Crop Padding"
            value={cfg.crop_padding}
            onChange={v => set('crop_padding', v)}
            min={0} max={80} step={1}
            unit=" px"
            helperText="Padding around each YOLO detection box"
          />

          <SliderField
            label="NMS IoU"
            value={cfg.nms_iou}
            onChange={v => set('nms_iou', v)}
            min={0.05} max={0.9} step={0.05}
            unit=""
            helperText="IoU threshold for duplicate removal (lower = more aggressive)"
          />
        </div>

        {/* Advanced: custom model paths */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <button
            className="btn btn-ghost btn-sm"
            onClick={() => setAdvanced(a => !a)}
            style={{ alignSelf: 'flex-start' }}
          >
            {advanced ? '▲' : '▼'} Advanced: Custom Models
          </button>

          {advanced && (
            <>
              {[
                { key: 'yolo_path', label: 'YOLO weights (.pt)', ph: 'experiments/yolo/best.pt' },
                { key: 'maskrcnn_path', label: 'Mask R-CNN checkpoint (.pth)', ph: 'experiments/maskrcnn/maskrcnn_crops_best.pth' },
                { key: 'effnet_path', label: 'EfficientNet checkpoint (.pth)', ph: 'experiments/efficientnet/efficientnet_best.pth' },
              ].map(({ key, label, ph }) => (
                <div key={key}>
                  <label className="field-label">{label}</label>
                  <input
                    className="input"
                    type="text"
                    placeholder={ph}
                    value={cfg[key]}
                    onChange={e => set(key, e.target.value)}
                  />
                </div>
              ))}
            </>
          )}
        </div>

        {/* Quick tips */}
        <div className="card" style={{ background: 'rgba(13,148,136,0.04)', borderColor: 'rgba(13,148,136,0.18)' }}>
          <div style={{ fontWeight: 600, marginBottom: '0.75rem', fontSize: '0.8125rem', color: 'var(--primary)' }}>
            💡 Tips
          </div>
          <ul style={{ paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.8125rem', color: 'var(--text-muted)' }}>
            <li>Lower YOLO confidence to catch faint particles</li>
            <li>Set pixel-to-micron ratio for real-world sizes</li>
            <li>Disable Mask R-CNN for faster processing</li>
            <li>Multiple images are processed as a batch</li>
          </ul>
        </div>
      </div>
    </div>
  )
}
