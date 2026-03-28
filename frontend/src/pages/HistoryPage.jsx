import React, { useState, useEffect, useMemo, useCallback } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { getJobs, getResult, imageUrl, maskUrl, originalUrl } from '../api/detect.js'
import { usePipelineMode, usePipelineJob } from '../App.jsx'
import {
  AlertTriangle, ClipboardList, Loader2, ArrowRight, ChevronDown, ChevronUp,
  Download, Image as ImageIcon, Layers, Maximize, BarChart2, Activity, FileText,
  Microscope, FlaskConical, Hash, Ruler, Circle, Ratio, ShieldCheck, ArrowRightLeft,
  Printer, X,
} from 'lucide-react'
import {
  PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  ScatterChart, Scatter, ZAxis,
} from 'recharts'

/* ═══════════════════════════════════════════════════════════════
   CONSTANTS & HELPERS
   ═══════════════════════════════════════════════════════════════ */
const CLASS_COLORS = { fiber: '#dc2626', film: '#d97706', fragment: '#059669' }
const CLASS_ICONS  = { fiber: '─', film: '▢', fragment: '△' }
const CLASS_ORDER  = ['fiber', 'film', 'fragment']

function fmt(v, dp = 2) { return v == null ? '—' : Number(v).toFixed(dp) }
function pct(n, total) { return total ? ((n / total) * 100).toFixed(1) : '0.0' }
function round2(v) { return Math.round(v * 100) / 100 }
function timeAgo(ts) {
  const diff = Math.floor(Date.now() / 1000 - ts)
  if (diff < 60) return `${diff}s ago`
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  return `${Math.floor(diff / 86400)}d ago`
}

function computeStats(arr) {
  const n = arr.length
  if (!n) return {}
  const sorted = [...arr].sort((a, b) => a - b)
  const mean = arr.reduce((s, x) => s + x, 0) / n
  const median = sorted[Math.floor(n / 2)]
  const min = sorted[0], max = sorted[n - 1]
  const std = Math.sqrt(arr.reduce((s, x) => s + (x - mean) ** 2, 0) / n)
  return { mean: round2(mean), median: round2(median), min: round2(min), max: round2(max), std: round2(std), n }
}

function buildHistogram(values, bins) {
  if (!values.length) return null
  const min = Math.min(...values), max = Math.max(...values) + 1e-9
  const step = (max - min) / bins
  const counts = new Array(bins).fill(0)
  const edges = Array.from({ length: bins + 1 }, (_, i) => min + i * step)
  for (const v of values) { const i = Math.min(Math.floor((v - min) / step), bins - 1); counts[i]++ }
  return { counts, bin_edges: edges.map(e => Math.round(e * 10) / 10), unit: 'px' }
}

/* ═══════════════════════════════════════════════════════════════
   CHART THEME
   ═══════════════════════════════════════════════════════════════ */
const TIP_STYLE = {
  background: '#fff', border: '1px solid #e3e6eb', color: '#111827',
  borderRadius: 6, fontSize: 11, boxShadow: '0 4px 12px rgba(0,0,0,.06)', padding: '6px 10px',
}
const AXIS_TICK = { fill: '#6b7280', fontSize: 10 }
const GRID = { strokeDasharray: '3 3', stroke: '#e3e6eb' }

/* ═══════════════════════════════════════════════════════════════
   SECTION WRAPPER
   ═══════════════════════════════════════════════════════════════ */
function Section({ title, icon: Icon, open, onToggle, children }) {
  return (
    <section className="lab-section">
      <div className="lab-section__header" onClick={onToggle} role="button" tabIndex={0}>
        <div className="lab-section__title-group">
          {Icon && <Icon size={14} strokeWidth={1.8} className="lab-section__icon" />}
          <h2 className="lab-section__title">{title}</h2>
        </div>
        <div className="lab-section__toggle">
          {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </div>
      </div>
      {open && <div className="lab-section__body">{children}</div>}
    </section>
  )
}

/* ═══════════════════════════════════════════════════════════════
   KPI DASHBOARD
   ═══════════════════════════════════════════════════════════════ */
function KpiDashboard({ totalN, counts, avgYoloConf, avgEffConf, reclassified, avgLength, unit }) {
  return (
    <div className="kpi-grid">
      <div className="kpi-card kpi-card--primary">
        <div className="kpi-card__icon-wrap kpi-card__icon-wrap--primary"><Hash size={16} strokeWidth={1.8} /></div>
        <div className="kpi-card__body">
          <span className="kpi-card__label">Total Particles</span>
          <span className="kpi-card__value">{totalN}</span>
        </div>
      </div>
      {CLASS_ORDER.map(c => (
        <div key={c} className="kpi-card" style={{ borderLeftColor: CLASS_COLORS[c] }}>
          <div className="kpi-card__icon-wrap">
            <span style={{ fontSize: 14, fontWeight: 700, lineHeight: 1, color: 'var(--text-muted)' }}>{CLASS_ICONS[c]}</span>
          </div>
          <div className="kpi-card__body">
            <span className="kpi-card__label">{c.charAt(0).toUpperCase() + c.slice(1)}</span>
            <span className="kpi-card__value">{counts[c]}</span>
            <span className="kpi-card__sub">{pct(counts[c], totalN)}%</span>
          </div>
        </div>
      ))}
      <div className="kpi-card">
        <div className="kpi-card__icon-wrap"><ShieldCheck size={14} strokeWidth={1.8} /></div>
        <div className="kpi-card__body">
          <span className="kpi-card__label">Avg YOLO Conf</span>
          <span className="kpi-card__value">{fmt(avgYoloConf)}</span>
        </div>
      </div>
      <div className="kpi-card">
        <div className="kpi-card__icon-wrap"><ShieldCheck size={14} strokeWidth={1.8} /></div>
        <div className="kpi-card__body">
          <span className="kpi-card__label">Avg ENet Conf</span>
          <span className="kpi-card__value">{fmt(avgEffConf)}</span>
        </div>
      </div>
      <div className="kpi-card">
        <div className="kpi-card__icon-wrap"><ArrowRightLeft size={14} strokeWidth={1.8} /></div>
        <div className="kpi-card__body">
          <span className="kpi-card__label">Reclassified</span>
          <span className="kpi-card__value">{reclassified}</span>
          <span className="kpi-card__sub">{pct(reclassified, totalN)}%</span>
        </div>
      </div>
      <div className="kpi-card">
        <div className="kpi-card__icon-wrap"><Ruler size={14} strokeWidth={1.8} /></div>
        <div className="kpi-card__body">
          <span className="kpi-card__label">Avg Length</span>
          <span className="kpi-card__value">{fmt(avgLength)}</span>
          <span className="kpi-card__sub">{unit}</span>
        </div>
      </div>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════
   COMPOSITION BAR
   ═══════════════════════════════════════════════════════════════ */
function CompositionBar({ counts, total }) {
  if (!total) return null
  return (
    <div className="composition-panel">
      <div className="composition-bar">
        {CLASS_ORDER.filter(c => counts[c] > 0).map(c => (
          <div key={c} className="composition-bar__segment"
            style={{ width: `${(counts[c] / total) * 100}%`, background: CLASS_COLORS[c] }}
            title={`${c}: ${counts[c]} (${pct(counts[c], total)}%)`} />
        ))}
      </div>
      <div className="composition-legend">
        {CLASS_ORDER.map(c => (
          <div key={c} className="composition-legend__item">
            <span className="composition-legend__swatch" style={{ background: CLASS_COLORS[c] }} />
            <span className="composition-legend__label">{c}</span>
            <span className="composition-legend__count">{counts[c]}</span>
            <span className="composition-legend__pct">{pct(counts[c], total)}%</span>
          </div>
        ))}
      </div>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════
   DONUT
   ═══════════════════════════════════════════════════════════════ */
function CountDonut({ counts }) {
  const data = CLASS_ORDER.filter(c => counts[c] > 0).map(c => ({ name: c, value: counts[c] }))
  return (
    <div className="chart-panel">
      <h3 className="chart-panel__title">Class Distribution</h3>
      <ResponsiveContainer width="100%" height={210}>
        <PieChart>
          <Pie data={data} cx="50%" cy="50%" innerRadius={52} outerRadius={85}
            dataKey="value" paddingAngle={2} label={({ name, value }) => `${name}: ${value}`}
            labelLine={false} stroke="none">
            {data.map(d => <Cell key={d.name} fill={CLASS_COLORS[d.name]} />)}
          </Pie>
          <Tooltip contentStyle={TIP_STYLE} />
        </PieChart>
      </ResponsiveContainer>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════
   LENGTH HISTOGRAM
   ═══════════════════════════════════════════════════════════════ */
function LengthHistogram({ histogram }) {
  if (!histogram) return null
  const { counts, bin_edges, unit } = histogram
  const data = counts.map((cnt, i) => ({ range: `${fmt(bin_edges[i], 1)}–${fmt(bin_edges[i + 1], 1)}`, count: cnt }))
  return (
    <div className="chart-panel">
      <h3 className="chart-panel__title">Length Distribution ({unit})</h3>
      <ResponsiveContainer width="100%" height={210}>
        <BarChart data={data} margin={{ top: 4, right: 12, bottom: 0, left: -4 }}>
          <CartesianGrid {...GRID} />
          <XAxis dataKey="range" tick={{ ...AXIS_TICK, fontSize: 9 }} interval={1} />
          <YAxis tick={AXIS_TICK} allowDecimals={false} />
          <Tooltip contentStyle={TIP_STYLE} />
          <Bar dataKey="count" fill="#374151" radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════
   CONFIDENCE BARS
   ═══════════════════════════════════════════════════════════════ */
function ConfidenceBars({ detections }) {
  const buckets = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
  const data = buckets.slice(0, -1).map((lo, i) => {
    const hi = buckets[i + 1]
    const inB = detections.filter(d => d.yolo_confidence >= lo && d.yolo_confidence < hi)
    const row = { range: `${(lo * 100).toFixed(0)}–${(hi > 1 ? 100 : hi * 100).toFixed(0)}%` }
    for (const c of CLASS_ORDER) row[c] = inB.filter(d => d.final_class === c).length
    return row
  })
  return (
    <div className="chart-panel">
      <h3 className="chart-panel__title">YOLO Confidence Distribution</h3>
      <ResponsiveContainer width="100%" height={210}>
        <BarChart data={data} margin={{ top: 4, right: 12, bottom: 0, left: -4 }}>
          <CartesianGrid {...GRID} />
          <XAxis dataKey="range" tick={AXIS_TICK} />
          <YAxis tick={AXIS_TICK} allowDecimals={false} />
          <Tooltip contentStyle={TIP_STYLE} />
          {CLASS_ORDER.map(c => (
            <Bar key={c} dataKey={c} stackId="a" fill={CLASS_COLORS[c]}
              radius={c === 'fragment' ? [3, 3, 0, 0] : [0, 0, 0, 0]} />
          ))}
          <Legend wrapperStyle={{ fontSize: 10, paddingTop: 4 }} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════
   PER-CLASS BARS
   ═══════════════════════════════════════════════════════════════ */
function PerClassBars({ perClass, metric, unit }) {
  const data = CLASS_ORDER.filter(c => perClass[c]).map(c => ({
    name: c, mean: perClass[c][metric]?.mean ?? 0,
  }))
  return (
    <ResponsiveContainer width="100%" height={170}>
      <BarChart data={data} margin={{ top: 4, right: 12, bottom: 0, left: -4 }}>
        <CartesianGrid {...GRID} />
        <XAxis dataKey="name" tick={{ ...AXIS_TICK, fontSize: 11 }} />
        <YAxis tick={AXIS_TICK} unit={unit ? ` ${unit}` : ''} />
        <Tooltip contentStyle={TIP_STYLE} formatter={v => [fmt(v, 2), '']} />
        <Bar dataKey="mean" radius={[3, 3, 0, 0]}>
          {data.map(d => <Cell key={d.name} fill={CLASS_COLORS[d.name]} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

/* ═══════════════════════════════════════════════════════════════
   SCATTER
   ═══════════════════════════════════════════════════════════════ */
function LengthCircScatter({ detections, pixelToMicron }) {
  const byClass = useMemo(() => {
    const m = {}
    for (const d of detections) {
      const cls = d.final_class
      if (!m[cls]) m[cls] = []
      m[cls].push({ x: round2(d.size.length_px * pixelToMicron), y: d.size.circularity, z: d.size.area_px })
    }
    return m
  }, [detections, pixelToMicron])
  return (
    <div className="chart-panel">
      <h3 className="chart-panel__title">Length vs Circularity</h3>
      <ResponsiveContainer width="100%" height={240}>
        <ScatterChart margin={{ top: 4, right: 16, bottom: 4, left: 0 }}>
          <CartesianGrid {...GRID} />
          <XAxis dataKey="x" name="Length" type="number" tick={AXIS_TICK} />
          <YAxis dataKey="y" name="Circularity" domain={[0, 1]} tick={AXIS_TICK} />
          <ZAxis dataKey="z" range={[30, 280]} name="Area" />
          <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={TIP_STYLE} />
          {CLASS_ORDER.filter(c => byClass[c]).map(c => (
            <Scatter key={c} name={c} data={byClass[c]} fill={CLASS_COLORS[c]} opacity={0.7} />
          ))}
          <Legend wrapperStyle={{ fontSize: 10 }} />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════
   MORPHOMETRIC TABLE
   ═══════════════════════════════════════════════════════════════ */
function MorphometricTable({ perClass, unit }) {
  const metrics = [
    { key: 'length', label: `Length (${unit})` },
    { key: 'width', label: `Width (${unit})` },
    { key: 'area', label: `Area (${unit}²)` },
    { key: 'circularity', label: 'Circularity' },
    { key: 'aspect_ratio', label: 'Aspect Ratio' },
    { key: 'yolo_confidence', label: 'YOLO Conf' },
    { key: 'effnet_confidence', label: 'ENet Conf' },
  ]
  const cols = CLASS_ORDER.filter(c => perClass[c])
  return (
    <div className="morpho-table-wrap">
      <table className="morpho-table">
        <thead>
          <tr>
            <th className="morpho-table__metric-header" rowSpan={2}>Metric</th>
            {cols.map(c => (
              <th key={c} colSpan={5} className="morpho-table__class-header" style={{ color: CLASS_COLORS[c] }}>
                <span className="morpho-table__class-swatch" style={{ background: CLASS_COLORS[c] }} />
                {c.charAt(0).toUpperCase() + c.slice(1)}
              </th>
            ))}
          </tr>
          <tr>{cols.map(c => (
            <React.Fragment key={c}>
              <th className="morpho-table__stat-header">Mean</th>
              <th className="morpho-table__stat-header">Med</th>
              <th className="morpho-table__stat-header">SD</th>
              <th className="morpho-table__stat-header">Min</th>
              <th className="morpho-table__stat-header">Max</th>
            </React.Fragment>
          ))}</tr>
        </thead>
        <tbody>
          {metrics.map(({ key, label }, ri) => (
            <tr key={key} className={ri % 2 === 0 ? 'morpho-table__row--even' : ''}>
              <td className="morpho-table__metric-cell">{label}</td>
              {cols.map(c => {
                const s = perClass[c]?.[key]
                return (
                  <React.Fragment key={c}>
                    <td className="morpho-table__num">{fmt(s?.mean)}</td>
                    <td className="morpho-table__num">{fmt(s?.median)}</td>
                    <td className="morpho-table__num morpho-table__num--muted">{fmt(s?.std)}</td>
                    <td className="morpho-table__num morpho-table__num--muted">{fmt(s?.min)}</td>
                    <td className="morpho-table__num morpho-table__num--muted">{fmt(s?.max)}</td>
                  </React.Fragment>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════
   DETECTION TABLE
   ═══════════════════════════════════════════════════════════════ */
function DetectionTable({ detections, pixelToMicron }) {
  const [filter, setFilter] = useState('all')
  const [sortKey, setSortKey] = useState('id')
  const [sortDir, setSortDir] = useState(1)
  const toggleSort = key => { if (sortKey === key) setSortDir(d => -d); else { setSortKey(key); setSortDir(1) } }
  const shown = useMemo(() => {
    return detections
      .filter(d => filter === 'all' || d.final_class === filter)
      .slice().sort((a, b) => {
        let av = sortKey === 'length' ? a.size.length_px : sortKey === 'area' ? a.size.area_px : a[sortKey]
        let bv = sortKey === 'length' ? b.size.length_px : sortKey === 'area' ? b.size.area_px : b[sortKey]
        if (typeof av === 'string') { av = av.toLowerCase(); bv = bv.toLowerCase() }
        return sortDir * (av < bv ? -1 : av > bv ? 1 : 0)
      })
  }, [detections, filter, sortKey, sortDir])
  const scale = pixelToMicron
  const SortHd = ({ k, children }) => (
    <th onClick={() => toggleSort(k)} className="detection-table__sortable">
      {children} <span className="detection-table__sort-indicator">{sortKey === k ? (sortDir > 0 ? '↑' : '↓') : '↕'}</span>
    </th>
  )
  return (
    <div>
      <div className="detection-table__toolbar">
        <div className="pill-bar">
          {['all', ...CLASS_ORDER].map(c => (
            <button key={c} className={`pill ${filter === c ? 'active' : ''}`} onClick={() => setFilter(c)}>
              {c === 'all' ? `All (${detections.length})` : `${c} (${detections.filter(d => d.final_class === c).length})`}
            </button>
          ))}
        </div>
        <span className="text-xs text-muted">{shown.length} rows</span>
      </div>
      <div className="detection-table__scroll">
        <table className="data-table">
          <thead><tr>
            <SortHd k="id">#</SortHd><SortHd k="final_class">Class</SortHd>
            <SortHd k="yolo_confidence">YOLO</SortHd><SortHd k="effnet_confidence">ENet</SortHd>
            <SortHd k="mask_confidence">Mask</SortHd><SortHd k="length">Length</SortHd>
            <th>Width</th><th>Area</th><th>Circ.</th><th>AR</th><th>Seg</th>
          </tr></thead>
          <tbody>
            {shown.map(d => (
              <tr key={d.id}>
                <td className="text-muted mono">{d.id}</td>
                <td>
                  <span className={`badge badge-${d.final_class}`}>{d.final_class}</span>
                  {d.yolo_class !== d.final_class && <span className="text-xs text-muted" style={{ marginLeft: 4 }}>← {d.yolo_class}</span>}
                </td>
                <td className="mono">{fmt(d.yolo_confidence)}</td>
                <td className="mono">{fmt(d.effnet_confidence)}</td>
                <td className="mono">{fmt(d.mask_confidence)}</td>
                <td className="mono" style={{ fontWeight: 600 }}>{fmt(d.size.length_px * scale, 1)}</td>
                <td className="mono">{fmt(d.size.width_px * scale, 1)}</td>
                <td className="mono">{fmt(d.size.area_px * scale * scale, 0)}</td>
                <td className="mono">{fmt(d.size.circularity, 3)}</td>
                <td className="mono">{fmt(d.size.aspect_ratio, 2)}</td>
                <td className="text-muted text-xs">{d.segmentation_source}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════
   IMAGE VIEWER
   ═══════════════════════════════════════════════════════════════ */
function ImageViewer({ jobId, images }) {
  const [imgIdx, setImgIdx] = useState(0)
  const [mode, setMode] = useState('vis')
  const im = images[imgIdx]
  const src = mode === 'vis' ? imageUrl(jobId, imgIdx) : mode === 'mask' ? maskUrl(jobId, imgIdx) : originalUrl(jobId, imgIdx)
  return (
    <div>
      <div className="image-viewer__controls">
        {images.length > 1 && (
          <div className="pill-bar" style={{ marginRight: '0.75rem' }}>
            {images.map((img, i) => (
              <button key={i} onClick={() => setImgIdx(i)} className={`pill ${imgIdx === i ? 'active' : ''}`} style={{ fontSize: '0.6875rem' }}>
                {img.filename || `Image ${i + 1}`}
              </button>
            ))}
          </div>
        )}
        <div className="pill-bar">
          {[['vis', <><ImageIcon size={12} strokeWidth={1.8} /> Annotated</>],
            ['mask', <><Layers size={12} strokeWidth={1.8} /> Masks</>],
            ['original', <><Maximize size={12} strokeWidth={1.8} /> Original</>],
          ].map(([k, lbl]) => (
            <button key={k} className={`pill ${mode === k ? 'active' : ''}`} onClick={() => setMode(k)}
              style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>{lbl}</button>
          ))}
        </div>
      </div>
      <div className="image-viewer__frame">
        <img key={src} src={src} alt={mode} className="image-viewer__img" onError={e => { e.target.style.opacity = 0.3 }} />
      </div>
      {im && <p className="text-xs text-muted" style={{ marginTop: '0.5rem' }}>
        {im.filename} · {im.image_size?.width} × {im.image_size?.height} px · {im.summary?.total} detections
      </p>}
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════
   CONFIG PANEL
   ═══════════════════════════════════════════════════════════════ */
function ConfigPanel({ config }) {
  return (
    <div className="config-grid">
      {Object.entries(config).map(([k, v]) => (
        <div key={k} className="config-grid__item">
          <span className="config-grid__key">{k.replace(/_/g, ' ')}</span>
          <span className="config-grid__val">{String(v)}</span>
        </div>
      ))}
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════
   INLINE RESULTS VIEWER  (expanded from job row)
   ═══════════════════════════════════════════════════════════════ */
function InlineResults({ result, onClose }) {
  const [sections, setSections] = useState({
    overview: true, composition: true, charts: true, shape: false,
    perClass: false, images: true, morpho: false, detections: false, config: false,
  })
  const toggle = key => setSections(s => ({ ...s, [key]: !s[key] }))

  const { job_id, images = [], config = {}, pipeline_mode: pm } = result
  const pixelToMicron = config.pixel_to_micron || 1.0
  const unit = pixelToMicron !== 1.0 ? 'µm' : 'px'
  const allDetections = images.flatMap(im => im.detections || [])
  const totalCounts = CLASS_ORDER.reduce((acc, c) => { acc[c] = allDetections.filter(d => d.final_class === c).length; return acc }, {})
  const totalN = allDetections.length

  const combinedSummary = (() => {
    const perClass = {}
    for (const c of CLASS_ORDER) {
      const grp = allDetections.filter(d => d.final_class === c)
      if (!grp.length) continue
      perClass[c] = {
        count: grp.length,
        length: computeStats(grp.map(d => d.size.length_px * pixelToMicron)),
        width: computeStats(grp.map(d => d.size.width_px * pixelToMicron)),
        area: computeStats(grp.map(d => d.size.area_px * pixelToMicron * pixelToMicron)),
        circularity: computeStats(grp.map(d => d.size.circularity)),
        aspect_ratio: computeStats(grp.map(d => d.size.aspect_ratio)),
        yolo_confidence: computeStats(grp.map(d => d.yolo_confidence)),
        effnet_confidence: computeStats(grp.map(d => d.effnet_confidence)),
      }
    }
    const lengths = allDetections.map(d => d.size.length_px * pixelToMicron)
    return { per_class: perClass, length_histogram: buildHistogram(lengths, 10), overall: { length: computeStats(lengths) } }
  })()

  const { per_class: perClass = {}, length_histogram: histogram } = combinedSummary
  const reclassified = allDetections.filter(d => d.yolo_class !== d.final_class).length
  const avgYoloConf = totalN ? allDetections.reduce((s, d) => s + d.yolo_confidence, 0) / totalN : 0
  const avgEffConf = totalN ? allDetections.reduce((s, d) => s + d.effnet_confidence, 0) / totalN : 0
  const avgLength = combinedSummary.overall?.length?.mean ?? null

  const handleExport = () => {
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' })
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob)
    a.download = `mp_report_${job_id.slice(0, 8)}.json`; a.click()
  }

  return (
    <div className="results-page" style={{ marginTop: '1rem' }}>
      {/* Report header */}
      <div className="lab-report-header">
        <div className="lab-report-header__top">
          <div className="lab-report-header__title-group">
            <div className="lab-report-header__icon"><Microscope size={20} strokeWidth={1.8} /></div>
            <div>
              <h2 className="lab-report-header__title" style={{ fontSize: '1.125rem' }}>Analysis Report</h2>
              <p className="lab-report-header__subtitle">
                Job {job_id.slice(0, 8)}…
                {pm && <span style={{ display: 'inline-flex', alignItems: 'center', marginLeft: '0.5rem', padding: '2px 8px', borderRadius: 'var(--radius-sm)', fontSize: '0.5625rem', fontWeight: 700, letterSpacing: '.06em', textTransform: 'uppercase', background: 'var(--surface2)', color: 'var(--text-secondary)', border: '1px solid var(--border)' }}>{pm}</span>}
              </p>
            </div>
          </div>
          <div className="lab-report-header__actions">
            <button className="btn btn-outline btn-sm" onClick={() => window.print()}>
              <Printer size={13} strokeWidth={1.8} /> Print
            </button>
            <button className="btn btn-primary btn-sm" onClick={handleExport}>
              <Download size={13} strokeWidth={1.8} /> Export JSON
            </button>
            <button className="btn btn-ghost btn-sm" onClick={onClose}>
              <X size={13} strokeWidth={1.8} /> Close
            </button>
          </div>
        </div>
        <div className="lab-report-header__meta">
          <div className="lab-meta-field"><span className="lab-meta-field__label">Images</span><span className="lab-meta-field__value">{images.length}</span></div>
          <div className="lab-meta-field"><span className="lab-meta-field__label">Total Particles</span><span className="lab-meta-field__value">{totalN}</span></div>
          <div className="lab-meta-field"><span className="lab-meta-field__label">Scale</span><span className="lab-meta-field__value">{pixelToMicron === 1 ? '1 px' : `1 px = ${pixelToMicron} ${unit}`}</span></div>
        </div>
      </div>

      <Section title="Sample Overview" icon={Activity} open={sections.overview} onToggle={() => toggle('overview')}>
        <KpiDashboard totalN={totalN} counts={totalCounts} avgYoloConf={avgYoloConf} avgEffConf={avgEffConf} reclassified={reclassified} avgLength={avgLength} unit={unit} />
      </Section>

      <Section title="Particle Composition" icon={BarChart2} open={sections.composition} onToggle={() => toggle('composition')}>
        <div className="grid-2">
          <CompositionBar counts={totalCounts} total={totalN} />
          <CountDonut counts={totalCounts} />
        </div>
      </Section>

      <Section title="Size Distribution" icon={BarChart2} open={sections.charts} onToggle={() => toggle('charts')}>
        <div className="grid-2">
          <LengthHistogram histogram={histogram} />
          <ConfidenceBars detections={allDetections} />
        </div>
      </Section>

      <Section title="Per-Class Morphometry" icon={BarChart2} open={sections.perClass} onToggle={() => toggle('perClass')}>
        <div className="grid-3">
          {[{ key: 'length', label: `Mean Length (${unit})` }, { key: 'area', label: `Mean Area (${unit}²)` }, { key: 'circularity', label: 'Mean Circularity' }].map(({ key, label }) => (
            <div key={key} className="chart-panel">
              <h3 className="chart-panel__title">{label}</h3>
              <PerClassBars perClass={perClass} metric={key} unit={key === 'circularity' ? '' : unit} />
            </div>
          ))}
        </div>
      </Section>

      <Section title="Shape Analysis" icon={Activity} open={sections.shape} onToggle={() => toggle('shape')}>
        <LengthCircScatter detections={allDetections} pixelToMicron={pixelToMicron} />
      </Section>

      <Section title="Microscopy Visualisations" icon={ImageIcon} open={sections.images} onToggle={() => toggle('images')}>
        {job_id && images.length > 0 && <ImageViewer jobId={job_id} images={images} />}
      </Section>

      <Section title="Detailed Morphometric Statistics" icon={FileText} open={sections.morpho} onToggle={() => toggle('morpho')}>
        <MorphometricTable perClass={perClass} unit={unit} />
      </Section>

      <Section title="Individual Detections" icon={FileText} open={sections.detections} onToggle={() => toggle('detections')}>
        {allDetections.length > 0 && <DetectionTable detections={allDetections} pixelToMicron={pixelToMicron} />}
      </Section>

      <Section title="Pipeline Configuration" icon={FlaskConical} open={sections.config} onToggle={() => toggle('config')}>
        <ConfigPanel config={config} />
      </Section>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════
   MAIN HISTORY PAGE
   ═══════════════════════════════════════════════════════════════ */
export default function HistoryPage() {
  const [jobs, setJobs] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [expandedJob, setExpandedJob] = useState(null) // job_id of expanded row
  const [expandedResult, setExpandedResult] = useState(null) // full result data
  const [loadingJob, setLoadingJob] = useState(null)

  const { mode: pipelineMode } = usePipelineMode()
  const { running } = usePipelineJob()
  const location = useLocation()
  const navigate = useNavigate()

  const fetchJobs = useCallback(() => {
    getJobs()
      .then(data => { setJobs(data.jobs || []); setLoading(false) })
      .catch(err => { setError(err.message); setLoading(false) })
  }, [])

  useEffect(() => { fetchJobs() }, [fetchJobs])

  // Auto-open job if passed via route state (e.g. immediately after pipeline completion)
  useEffect(() => {
    if (location.state?.autoOpenJob && location.state?.autoOpenResult) {
      if (expandedJob !== location.state.autoOpenJob) {
        setExpandedJob(location.state.autoOpenJob)
        setExpandedResult(location.state.autoOpenResult)
        
        // Temporarily append to jobs list if it's not there yet to ensure it's rendered inline
        setJobs(prevJobs => {
          if (!prevJobs.find(j => j.job_id === location.state.autoOpenJob)) {
            const resultData = location.state.autoOpenResult
            const detections = resultData.images?.reduce((sum, im) => sum + (im.summary?.total || 0), 0) || 0
            const newJob = {
              job_id: location.state.autoOpenJob,
              status: 'done',
              created_at: Date.now() / 1000,
              total_detections: detections,
              image_count: resultData.images?.length || 0,
              pipeline_mode: resultData.pipeline_mode || pipelineMode
            }
            return [newJob, ...prevJobs]
          }
          return prevJobs
        })
        
        // Remove it from history state so it doesn't reopen if the user navigates away and back
        navigate(location.pathname, { replace: true, state: {} })
      }
    }
  }, [location.state, location.pathname, expandedJob, pipelineMode, navigate])

  // Auto-refresh when pipeline finishes
  useEffect(() => {
    if (!running) {
      // Small delay to let backend finish writing
      const t = setTimeout(fetchJobs, 1000)
      return () => clearTimeout(t)
    }
  }, [running, fetchJobs])

  const loadJob = async (jobId) => {
    if (expandedJob === jobId) {
      setExpandedJob(null)
      setExpandedResult(null)
      return
    }
    setLoadingJob(jobId)
    try {
      const result = await getResult(jobId)
      setExpandedJob(jobId)
      setExpandedResult(result)
    } catch (err) {
      alert('Could not load job: ' + err.message)
    } finally {
      setLoadingJob(null)
    }
  }

  // Filter jobs by current pipeline mode
  const filteredJobs = useMemo(() => {
    return [...jobs]
      .filter(j => {
        const jMode = j.pipeline_mode || 'macro'
        return jMode === pipelineMode
      })
      .sort((a, b) => (b.created_at || 0) - (a.created_at || 0))
  }, [jobs, pipelineMode])

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', padding: '4rem' }}>
        <Loader2 className="spinner" size={28} style={{ color: 'var(--text-muted)' }} />
      </div>
    )
  }

  if (error) {
    return (
      <div className="empty-state">
        <AlertTriangle size={44} strokeWidth={1.5} style={{ color: 'var(--text-muted)' }} />
        <p style={{ marginTop: '1rem' }}>Could not reach backend: <strong>{error}</strong></p>
        <p className="text-sm text-muted" style={{ marginTop: '0.5rem' }}>Make sure the FastAPI server is running on port 8000.</p>
      </div>
    )
  }

  if (filteredJobs.length === 0) {
    return (
      <div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
          <h1 style={{ fontSize: '1.375rem', fontWeight: 700, margin: 0 }}>Job History</h1>
          <span style={{ fontSize: '0.6875rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '.05em' }}>
            {pipelineMode} mode
          </span>
        </div>
        <div className="empty-state" style={{ marginTop: '3rem' }}>
          <ClipboardList size={44} strokeWidth={1.5} style={{ color: 'var(--text-muted)' }} />
          <h2 style={{ fontWeight: 600, marginBottom: '0.5rem', marginTop: '1rem' }}>No {pipelineMode} jobs yet</h2>
          <p className="text-muted text-sm">Run a detection pipeline in {pipelineMode} mode to see results here.</p>
        </div>
      </div>
    )
  }

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
        <h1 style={{ fontSize: '1.375rem', fontWeight: 700, margin: 0 }}>Job History</h1>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          {running && (
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.375rem', fontSize: '0.6875rem', fontWeight: 600, color: 'var(--text-muted)' }}>
              <span className="spinner" style={{ width: 10, height: 10, borderWidth: 1.5 }} /> Running…
            </span>
          )}
          <span style={{ fontSize: '0.6875rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '.05em' }}>
            {pipelineMode} · {filteredJobs.length} job{filteredJobs.length !== 1 ? 's' : ''}
          </span>
        </div>
      </div>

      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>Job ID</th>
              <th>Status</th>
              <th>Detections</th>
              <th>Images</th>
              <th>Created</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {filteredJobs.map(j => (
              <React.Fragment key={j.job_id}>
                <tr style={{ cursor: j.status === 'done' ? 'pointer' : 'default', background: expandedJob === j.job_id ? 'var(--surface2)' : undefined }}
                  onClick={() => j.status === 'done' && loadJob(j.job_id)}>
                  <td>
                    <code style={{ fontSize: '0.6875rem', background: 'var(--surface2)', padding: '2px 6px', borderRadius: 4 }}>
                      {j.job_id.slice(0, 12)}…
                    </code>
                  </td>
                  <td>
                    <span className={`badge ${j.status === 'done' ? 'badge-primary' : j.status === 'running' ? 'badge-neutral' : ''}`}
                      style={j.status === 'error' ? { background: 'rgba(153,27,27,.08)', color: '#991b1b', border: '1px solid rgba(153,27,27,.15)' } : j.status === 'running' ? { display: 'inline-flex', alignItems: 'center', gap: '0.25rem' } : {}}>
                      {j.status === 'running' && <span className="spinner" style={{ width: 8, height: 8, borderWidth: 1.5 }} />}
                      {j.status}
                    </span>
                  </td>
                  <td className="mono" style={{ fontWeight: 600 }}>{j.total_detections ?? '—'}</td>
                  <td className="mono">{j.image_count ?? '—'}</td>
                  <td className="text-muted text-sm">{j.created_at ? timeAgo(j.created_at) : '—'}</td>
                  <td>
                    {j.status === 'done' && (
                      <button className="btn btn-ghost btn-sm" style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}
                        onClick={e => { e.stopPropagation(); loadJob(j.job_id) }}>
                        {loadingJob === j.job_id ? <Loader2 size={13} className="spin" /> :
                          expandedJob === j.job_id ? <ChevronUp size={13} /> : <ArrowRight size={13} strokeWidth={1.8} />}
                        {expandedJob === j.job_id ? 'Collapse' : 'View'}
                      </button>
                    )}
                  </td>
                </tr>
                {/* Inline results panel */}
                {expandedJob === j.job_id && expandedResult && (
                  <tr>
                    <td colSpan={6} style={{ padding: 0 }}>
                      <div style={{ padding: '0.5rem 1rem 1.5rem', background: 'var(--bg)' }}>
                        <InlineResults result={expandedResult} onClose={() => { setExpandedJob(null); setExpandedResult(null) }} />
                      </div>
                    </td>
                  </tr>
                )}
              </React.Fragment>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
