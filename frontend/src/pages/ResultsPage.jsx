import React, { useState, useEffect, useMemo } from 'react'
import {
  PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  ScatterChart, Scatter, ZAxis,
} from 'recharts'
import {
  ChevronDown, ChevronUp, Download, Image as ImageIcon,
  Layers, Maximize, BarChart2, Activity, FileText,
  Microscope, FlaskConical, Hash, Ruler, Circle, Ratio,
  ShieldCheck, ArrowRightLeft, Printer,
} from 'lucide-react'
import { imageUrl, maskUrl, originalUrl } from '../api/detect.js'

/* ═══════════════════════════════════════════════════════════════
   CONSTANTS
   ═══════════════════════════════════════════════════════════════ */
const CLASS_COLORS = { fiber: '#dc2626', film: '#d97706', fragment: '#059669' }
const CLASS_BG     = { fiber: 'rgba(220,38,38,.06)', film: 'rgba(217,119,6,.06)', fragment: 'rgba(5,150,105,.06)' }
const CLASS_ICONS  = { fiber: '─', film: '▢', fragment: '△' }
const CLASS_ORDER  = ['fiber', 'film', 'fragment']

/* ═══════════════════════════════════════════════════════════════
   HELPERS
   ═══════════════════════════════════════════════════════════════ */
function fmt(v, dp = 2)  { return v == null ? '—' : Number(v).toFixed(dp) }
function pct(n, total)   { return total ? ((n / total) * 100).toFixed(1) : '0.0' }
function round2(v)       { return Math.round(v * 100) / 100 }

function computeStats(arr) {
  const n = arr.length
  if (!n) return {}
  const sorted = [...arr].sort((a, b) => a - b)
  const mean   = arr.reduce((s, x) => s + x, 0) / n
  const median = sorted[Math.floor(n / 2)]
  const min    = sorted[0]
  const max    = sorted[n - 1]
  const std    = Math.sqrt(arr.reduce((s, x) => s + (x - mean) ** 2, 0) / n)
  return { mean: round2(mean), median: round2(median), min: round2(min), max: round2(max), std: round2(std), n }
}

function buildHistogram(values, bins) {
  if (!values.length) return null
  const min = Math.min(...values)
  const max = Math.max(...values) + 1e-9
  const step = (max - min) / bins
  const counts = new Array(bins).fill(0)
  const edges  = Array.from({ length: bins + 1 }, (_, i) => min + i * step)
  for (const v of values) {
    const i = Math.min(Math.floor((v - min) / step), bins - 1)
    counts[i]++
  }
  return { counts, bin_edges: edges.map(e => Math.round(e * 10) / 10), unit: 'px' }
}

/* ═══════════════════════════════════════════════════════════════
   CHART THEME
   ═══════════════════════════════════════════════════════════════ */
const TIP_STYLE = {
  background: '#fff', border: '1px solid #e3e6eb', color: '#111827',
  borderRadius: 6, fontSize: 11, boxShadow: '0 4px 12px rgba(0,0,0,.06)',
  padding: '6px 10px',
}
const AXIS_TICK = { fill: '#6b7280', fontSize: 10 }
const GRID      = { strokeDasharray: '3 3', stroke: '#e3e6eb' }

/* ═══════════════════════════════════════════════════════════════
   LAB REPORT HEADER
   ═══════════════════════════════════════════════════════════════ */
function ReportHeader({ jobId, imageCount, unit, pixelToMicron, totalN, onExport, onPrint, pipelineMode }) {
  const now = new Date()
  return (
    <div className="lab-report-header">
      <div className="lab-report-header__top">
        <div className="lab-report-header__title-group">
          <div className="lab-report-header__icon">
            <Microscope size={20} strokeWidth={1.8} />
          </div>
          <div>
            <h1 className="lab-report-header__title">Microplastic Analysis Report</h1>
            <p className="lab-report-header__subtitle">
              Automated detection and morphometric characterisation
              {pipelineMode && (
                <span style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  marginLeft: '0.5rem',
                  padding: '2px 8px',
                  borderRadius: 'var(--radius-sm)',
                  fontSize: '0.5625rem',
                  fontWeight: 700,
                  letterSpacing: '.06em',
                  textTransform: 'uppercase',
                  background: 'var(--surface2)',
                  color: 'var(--text-secondary)',
                  border: '1px solid var(--border)',
                  verticalAlign: 'middle',
                }}>
                  {pipelineMode}
                </span>
              )}
            </p>
          </div>
        </div>
        <div className="lab-report-header__actions">
          <button className="btn btn-outline btn-sm" onClick={onPrint}>
            <Printer size={13} strokeWidth={1.8} /> Print
          </button>
          <button className="btn btn-primary btn-sm" onClick={onExport}>
            <Download size={13} strokeWidth={1.8} /> Export JSON
          </button>
        </div>
      </div>
      <div className="lab-report-header__meta">
        <MetaField label="Job ID" value={jobId.slice(0, 8) + '...'} mono />
        <MetaField label="Date" value={now.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' })} />
        <MetaField label="Images" value={imageCount} />
        <MetaField label="Total Particles" value={totalN} />
        <MetaField label="Scale" value={pixelToMicron === 1 ? '1 px' : `1 px = ${pixelToMicron} ${unit}`} />
      </div>
    </div>
  )
}

function MetaField({ label, value, mono }) {
  return (
    <div className="lab-meta-field">
      <span className="lab-meta-field__label">{label}</span>
      <span className={`lab-meta-field__value ${mono ? 'mono' : ''}`}>{value}</span>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════
   SECTION WRAPPER
   ═══════════════════════════════════════════════════════════════ */
function Section({ id, title, icon: Icon, open, onToggle, children, className = '' }) {
  return (
    <section className={`lab-section ${className}`}>
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
   KPI DASHBOARD (top-level numbers)
   ═══════════════════════════════════════════════════════════════ */
function KpiDashboard({ totalN, counts, avgYoloConf, avgEffConf, reclassified, avgLength, unit }) {
  return (
    <div className="kpi-grid">
      {/* Total */}
      <div className="kpi-card kpi-card--primary">
        <div className="kpi-card__icon-wrap kpi-card__icon-wrap--primary"><Hash size={16} strokeWidth={1.8} /></div>
        <div className="kpi-card__body">
          <span className="kpi-card__label">Total Particles</span>
          <span className="kpi-card__value">{totalN}</span>
        </div>
      </div>

      {/* Per class */}
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

      {/* Confidence */}
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
   COMPOSITION BAR  (horizontal stacked bar showing proportions)
   ═══════════════════════════════════════════════════════════════ */
function CompositionBar({ counts, total }) {
  if (!total) return null
  return (
    <div className="composition-panel">
      <div className="composition-bar">
        {CLASS_ORDER.filter(c => counts[c] > 0).map(c => (
          <div
            key={c}
            className="composition-bar__segment"
            style={{ width: `${(counts[c] / total) * 100}%`, background: CLASS_COLORS[c] }}
            title={`${c}: ${counts[c]} (${pct(counts[c], total)}%)`}
          />
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
   DONUT CHART
   ═══════════════════════════════════════════════════════════════ */
function CountDonut({ counts, total }) {
  const data = CLASS_ORDER.filter(c => counts[c] > 0).map(c => ({ name: c, value: counts[c] }))
  return (
    <div className="chart-panel">
      <h3 className="chart-panel__title">Class Distribution</h3>
      <ResponsiveContainer width="100%" height={210}>
        <PieChart>
          <Pie
            data={data} cx="50%" cy="50%"
            innerRadius={52} outerRadius={85}
            dataKey="value" paddingAngle={2}
            label={({ name, value }) => `${name}: ${value}`}
            labelLine={false}
            stroke="none"
          >
            {data.map(d => <Cell key={d.name} fill={CLASS_COLORS[d.name]} />)}
          </Pie>
          <Tooltip contentStyle={TIP_STYLE} formatter={(v, n) => [v, n]} />
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
  const data = counts.map((cnt, i) => ({
    range: `${fmt(bin_edges[i], 1)}–${fmt(bin_edges[i + 1], 1)}`,
    count: cnt,
  }))
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
   CONFIDENCE DISTRIBUTION (stacked bars)
   ═══════════════════════════════════════════════════════════════ */
function ConfidenceBars({ detections }) {
  const buckets = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
  const data = buckets.slice(0, -1).map((lo, i) => {
    const hi = buckets[i + 1]
    const inBucket = detections.filter(d => d.yolo_confidence >= lo && d.yolo_confidence < hi)
    const row = { range: `${(lo * 100).toFixed(0)}–${(hi > 1 ? 100 : hi * 100).toFixed(0)}%` }
    for (const c of CLASS_ORDER) row[c] = inBucket.filter(d => d.final_class === c).length
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
   PER-CLASS BAR CHARTS
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
   SCATTER: LENGTH vs CIRCULARITY
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
      <p className="chart-panel__sub">Bubble size proportional to area</p>
      <ResponsiveContainer width="100%" height={240}>
        <ScatterChart margin={{ top: 4, right: 16, bottom: 4, left: 0 }}>
          <CartesianGrid {...GRID} />
          <XAxis dataKey="x" name="Length" type="number" tick={AXIS_TICK}
            label={{ value: 'Length', fill: '#6b7280', position: 'insideBottomRight', offset: -6, fontSize: 10 }} />
          <YAxis dataKey="y" name="Circularity" domain={[0, 1]} tick={AXIS_TICK} />
          <ZAxis dataKey="z" range={[30, 280]} name="Area" />
          <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={TIP_STYLE}
            formatter={(v, n) => [fmt(v, 3), n]} />
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
   MORPHOMETRIC STATISTICS TABLE
   ═══════════════════════════════════════════════════════════════ */
function MorphometricTable({ perClass, unit }) {
  const metrics = [
    { key: 'length',            label: `Length (${unit})`,     icon: Ruler },
    { key: 'width',             label: `Width (${unit})`,      icon: Ruler },
    { key: 'area',              label: `Area (${unit}²)`,      icon: null },
    { key: 'circularity',       label: 'Circularity',          icon: Circle },
    { key: 'aspect_ratio',      label: 'Aspect Ratio',         icon: Ratio },
    { key: 'yolo_confidence',   label: 'YOLO Conf',            icon: ShieldCheck },
    { key: 'effnet_confidence', label: 'EfficientNet Conf',    icon: ShieldCheck },
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
          <tr>
            {cols.map(c => (
              <React.Fragment key={c}>
                <th className="morpho-table__stat-header">Mean</th>
                <th className="morpho-table__stat-header">Med</th>
                <th className="morpho-table__stat-header">SD</th>
                <th className="morpho-table__stat-header">Min</th>
                <th className="morpho-table__stat-header">Max</th>
              </React.Fragment>
            ))}
          </tr>
        </thead>
        <tbody>
          {metrics.map(({ key, label, icon: MIcon }, ri) => (
            <tr key={key} className={ri % 2 === 0 ? 'morpho-table__row--even' : ''}>
              <td className="morpho-table__metric-cell">
                {MIcon && <MIcon size={11} strokeWidth={1.8} style={{ color: 'var(--text-muted)' }} />}
                {label}
              </td>
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

  const toggleSort = key => {
    if (sortKey === key) setSortDir(d => -d)
    else { setSortKey(key); setSortDir(1) }
  }

  const shown = useMemo(() => {
    return detections
      .filter(d => filter === 'all' || d.final_class === filter)
      .slice()
      .sort((a, b) => {
        let av = sortKey === 'length' ? a.size.length_px : sortKey === 'area' ? a.size.area_px : a[sortKey]
        let bv = sortKey === 'length' ? b.size.length_px : sortKey === 'area' ? b.size.area_px : b[sortKey]
        if (typeof av === 'string') { av = av.toLowerCase(); bv = bv.toLowerCase() }
        return sortDir * (av < bv ? -1 : av > bv ? 1 : 0)
      })
  }, [detections, filter, sortKey, sortDir])

  const scale = pixelToMicron
  const unit  = pixelToMicron !== 1.0 ? 'µm' : 'px'

  const SortHd = ({ k, children }) => (
    <th onClick={() => toggleSort(k)} className="detection-table__sortable">
      {children}
      <span className="detection-table__sort-indicator">
        {sortKey === k ? (sortDir > 0 ? '↑' : '↓') : '↕'}
      </span>
    </th>
  )

  return (
    <div>
      {/* Filter pills */}
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
          <thead>
            <tr>
              <SortHd k="id">#</SortHd>
              <SortHd k="final_class">Class</SortHd>
              <SortHd k="yolo_confidence">YOLO</SortHd>
              <SortHd k="effnet_confidence">ENet</SortHd>
              <SortHd k="mask_confidence">Mask</SortHd>
              <SortHd k="length">Length</SortHd>
              <th>Width</th>
              <th>Area</th>
              <th>Circ.</th>
              <th>AR</th>
              <th>Seg</th>
            </tr>
          </thead>
          <tbody>
            {shown.map(d => (
              <tr key={d.id}>
                <td className="text-muted mono">{d.id}</td>
                <td>
                  <span className={`badge badge-${d.final_class}`}>{d.final_class}</span>
                  {d.yolo_class !== d.final_class && (
                    <span className="text-xs text-muted" style={{ marginLeft: 4 }}>← {d.yolo_class}</span>
                  )}
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
  const total = images.length
  const im  = images[imgIdx]
  const src = mode === 'vis' ? imageUrl(jobId, imgIdx)
            : mode === 'mask' ? maskUrl(jobId, imgIdx)
            : originalUrl(jobId, imgIdx)

  return (
    <div>
      <div className="image-viewer__controls">
        {total > 1 && (
          <div className="pill-bar" style={{ marginRight: '0.75rem' }}>
            {images.map((img, i) => (
              <button key={i} onClick={() => setImgIdx(i)} className={`pill ${imgIdx === i ? 'active' : ''}`} style={{ fontSize: '0.6875rem' }}>
                {img.filename || `Image ${i + 1}`}
              </button>
            ))}
          </div>
        )}
        <div className="pill-bar">
          {[
            ['vis', <><ImageIcon size={12} strokeWidth={1.8} /> Annotated</>],
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
      {im && (
        <p className="text-xs text-muted" style={{ marginTop: '0.5rem' }}>
          {im.filename} &middot; {im.image_size?.width} × {im.image_size?.height} px
          &middot; {im.summary?.total} detections
        </p>
      )}
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════
   PIPELINE CONFIG PANEL
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
   MAIN PAGE
   ═══════════════════════════════════════════════════════════════ */
export default function ResultsPage() {
  const [result, setResult] = useState(null)
  const [sections, setSections] = useState({
    overview: true, composition: true, charts: true,
    shape: true, confidence: true, perClass: true,
    images: true, morpho: true, detections: true, config: false,
  })

  useEffect(() => {
    const raw = sessionStorage.getItem('mp_last_result')
    if (raw) { try { setResult(JSON.parse(raw)) } catch {} }
  }, [])

  const toggle = key => setSections(s => ({ ...s, [key]: !s[key] }))

  /* ── Empty state ── */
  if (!result) {
    return (
      <div className="empty-state" style={{ marginTop: '5rem' }}>
        <FlaskConical size={44} strokeWidth={1.3} style={{ color: 'var(--text-muted)', marginBottom: '1rem', opacity: 0.4 }} />
        <h2 style={{ fontWeight: 600, marginBottom: '0.5rem' }}>No analysis results</h2>
        <p className="text-muted" style={{ fontSize: '0.8125rem' }}>
          Run the detection pipeline on the <a href="/detect" style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>Detect</a> page to generate a report.
        </p>
      </div>
    )
  }

  /* ── Data extraction ── */
  const { job_id, images = [], config = {}, pipeline_mode: pipelineMode } = result
  const pixelToMicron = config.pixel_to_micron || 1.0
  const unit = pixelToMicron !== 1.0 ? 'µm' : 'px'

  const allDetections = images.flatMap(im => im.detections || [])
  const totalCounts = CLASS_ORDER.reduce((acc, c) => {
    acc[c] = allDetections.filter(d => d.final_class === c).length
    return acc
  }, {})
  const totalN = allDetections.length

  /* ── Aggregate per-class stats ── */
  const combinedSummary = images.length === 1
    ? images[0].summary
    : (() => {
        const perClass = {}
        for (const c of CLASS_ORDER) {
          const grp = allDetections.filter(d => d.final_class === c)
          if (!grp.length) continue
          perClass[c] = {
            count: grp.length,
            length: computeStats(grp.map(d => d.size.length_px * pixelToMicron)),
            width:  computeStats(grp.map(d => d.size.width_px  * pixelToMicron)),
            area:   computeStats(grp.map(d => d.size.area_px   * pixelToMicron * pixelToMicron)),
            circularity:      computeStats(grp.map(d => d.size.circularity)),
            aspect_ratio:     computeStats(grp.map(d => d.size.aspect_ratio)),
            yolo_confidence:  computeStats(grp.map(d => d.yolo_confidence)),
            effnet_confidence:computeStats(grp.map(d => d.effnet_confidence)),
          }
        }
        const lengths = allDetections.map(d => d.size.length_px * pixelToMicron)
        const overall = { length: computeStats(lengths) }
        const hist = buildHistogram(lengths, 10)
        return { total: totalN, counts: totalCounts, per_class: perClass, length_histogram: hist, unit, pixel_to_micron: pixelToMicron, overall }
      })()

  const { per_class: perClass = {}, length_histogram: histogram } = combinedSummary

  const reclassified = allDetections.filter(d => d.yolo_class !== d.final_class).length
  const avgYoloConf  = totalN ? allDetections.reduce((s, d) => s + d.yolo_confidence, 0) / totalN : 0
  const avgEffConf   = totalN ? allDetections.reduce((s, d) => s + d.effnet_confidence, 0) / totalN : 0
  const avgLength    = combinedSummary.overall?.length?.mean ?? null

  const handleExport = () => {
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `mp_report_${job_id.slice(0, 8)}.json`
    a.click()
  }
  const handlePrint = () => window.print()

  /* ── Render ── */
  return (
    <div className="results-page">

      <ReportHeader
        jobId={job_id} imageCount={images.length} unit={unit}
        pixelToMicron={pixelToMicron} totalN={totalN}
        onExport={handleExport} onPrint={handlePrint}
        pipelineMode={pipelineMode || config.pipeline_mode || 'macro'}
      />

      {/* 1 — KPI Overview */}
      <Section id="overview" title="Sample Overview" icon={Activity}
        open={sections.overview} onToggle={() => toggle('overview')}>
        <KpiDashboard
          totalN={totalN} counts={totalCounts}
          avgYoloConf={avgYoloConf} avgEffConf={avgEffConf}
          reclassified={reclassified} avgLength={avgLength} unit={unit}
        />
      </Section>

      {/* 2 — Composition */}
      <Section id="composition" title="Particle Composition" icon={BarChart2}
        open={sections.composition} onToggle={() => toggle('composition')}>
        <div className="grid-2">
          <CompositionBar counts={totalCounts} total={totalN} />
          <CountDonut counts={totalCounts} total={totalN} />
        </div>
      </Section>

      {/* 3 — Distribution Charts */}
      <Section id="charts" title="Size Distribution" icon={BarChart2}
        open={sections.charts} onToggle={() => toggle('charts')}>
        <div className="grid-2">
          <LengthHistogram histogram={histogram} />
          <ConfidenceBars detections={allDetections} />
        </div>
      </Section>

      {/* 4 — Per-class bars */}
      <Section id="perClass" title="Per-Class Morphometry" icon={BarChart2}
        open={sections.perClass} onToggle={() => toggle('perClass')}>
        <div className="grid-3">
          {[
            { key: 'length',      label: `Mean Length (${unit})` },
            { key: 'area',        label: `Mean Area (${unit}²)` },
            { key: 'circularity', label: 'Mean Circularity' },
          ].map(({ key, label }) => (
            <div key={key} className="chart-panel">
              <h3 className="chart-panel__title">{label}</h3>
              <PerClassBars perClass={perClass} metric={key} unit={key === 'circularity' ? '' : unit} />
            </div>
          ))}
        </div>
      </Section>

      {/* 5 — Shape scatter */}
      <Section id="shape" title="Shape Analysis" icon={Activity}
        open={sections.shape} onToggle={() => toggle('shape')}>
        <LengthCircScatter detections={allDetections} pixelToMicron={pixelToMicron} />
      </Section>

      {/* 6 — Image viewer */}
      <Section id="images" title="Microscopy Visualisations" icon={ImageIcon}
        open={sections.images} onToggle={() => toggle('images')}>
        {job_id && images.length > 0 && <ImageViewer jobId={job_id} images={images} />}
      </Section>

      {/* 7 — Full statistics table */}
      <Section id="morpho" title="Detailed Morphometric Statistics" icon={FileText}
        open={sections.morpho} onToggle={() => toggle('morpho')}>
        <MorphometricTable perClass={perClass} unit={unit} />
      </Section>

      {/* 8 — Individual detections */}
      <Section id="detections" title="Individual Detections" icon={FileText}
        open={sections.detections} onToggle={() => toggle('detections')}>
        {allDetections.length > 0 && (
          <DetectionTable detections={allDetections} pixelToMicron={pixelToMicron} />
        )}
      </Section>

      {/* 9 — Pipeline config */}
      <Section id="config" title="Pipeline Configuration" icon={FlaskConical}
        open={sections.config} onToggle={() => toggle('config')}>
        <ConfigPanel config={config} />
      </Section>

    </div>
  )
}
