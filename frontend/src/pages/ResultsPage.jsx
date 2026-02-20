import React, { useState, useEffect } from 'react'
import {
  PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  ScatterChart, Scatter, ZAxis,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from 'recharts'
import { ChevronDown, ChevronUp, Download, Image as ImageIcon, Layers, Maximize, BarChart2, Activity, FileText } from 'lucide-react'
import { imageUrl, maskUrl, originalUrl } from '../api/detect.js'

const CLASS_COLORS = { fiber: '#ef4444', film: '#f59e0b', fragment: '#10b981' }
const CLASS_ORDER   = ['fiber', 'film', 'fragment']

/* ──────────────────────────────────────────── helpers ─── */
function fmt(v, dp = 2) { return v == null ? '—' : Number(v).toFixed(dp) }
function pct(n, total) { return total ? ((n / total) * 100).toFixed(1) + '%' : '—' }

function StatTile({ label, value, sub, color, icon: Icon }) {
  return (
    <div className="stat-tile" style={{ display: 'flex', alignItems: 'flex-start', gap: '1rem' }}>
      {Icon && <div style={{ color: color || 'var(--primary)', background: 'var(--surface2)', padding: '0.5rem', borderRadius: 'var(--radius-sm)' }}><Icon size={20} /></div>}
      <div>
        <span className="stat-label">{label}</span>
        <span className="stat-value" style={color ? { color } : {}}>{value}</span>
        {sub && <span className="stat-sub">{sub}</span>}
      </div>
    </div>
  )
}

/* ──────────────────────────────────────────── Chart wrappers ─── */
const CHART_STYLE = { background: 'transparent', fontSize: 12 }
const TIP_STYLE   = { background: '#ffffff', border: '1px solid #dce0e8', color: '#1a1e2c', borderRadius: 8, fontSize: 12, boxShadow: '0 4px 12px rgba(0,0,0,.08)' }

function SectionTitle({ children, toggle, onToggle, icon: Icon }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
      <h2 style={{ fontSize: '1rem', fontWeight: 700, margin: 0, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        {Icon && <Icon size={18} className="text-primary" />}
        {children}
      </h2>
      {onToggle && (
        <button className="btn btn-ghost btn-sm" onClick={onToggle} style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
          {toggle ? <><ChevronUp size={14} /> collapse</> : <><ChevronDown size={14} /> expand</>}
        </button>
      )}
    </div>
  )
}

/* ──────────────────────────────────────────── Detection counts donut ─── */
function CountDonut({ counts, total }) {
  const data = CLASS_ORDER.filter(c => counts[c] > 0).map(c => ({
    name: c, value: counts[c],
  }))
  return (
    <div className="card" style={{ minHeight: 280 }}>
      <SectionTitle>Detection Counts</SectionTitle>
      <ResponsiveContainer width="100%" height={220}>
        <PieChart style={CHART_STYLE}>
          <Pie data={data} cx="50%" cy="50%" innerRadius={55} outerRadius={90}
            dataKey="value" label={({ name, value }) => `${name}: ${value}`}
            labelLine={false} paddingAngle={3}
          >
            {data.map(d => <Cell key={d.name} fill={CLASS_COLORS[d.name]} />)}
          </Pie>
          <Tooltip contentStyle={TIP_STYLE} formatter={(v, n) => [v, n]} />
        </PieChart>
      </ResponsiveContainer>
      <div style={{ display: 'flex', justifyContent: 'center', gap: '1.5rem', flexWrap: 'wrap', marginTop: '0.5rem' }}>
        {CLASS_ORDER.filter(c => counts[c] > 0).map(c => (
          <div key={c} style={{ textAlign: 'center' }}>
            <span style={{ width: 10, height: 10, borderRadius: '50%', background: CLASS_COLORS[c], display: 'inline-block', marginRight: 5 }} />
            <span style={{ fontSize: 13, fontWeight: 600, color: CLASS_COLORS[c] }}>{counts[c]}</span>
            <span className="text-xs text-muted" style={{ marginLeft: 4 }}>{c}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

/* ──────────────────────────────────────────── Length histogram ─── */
function LengthHistogram({ histogram }) {
  if (!histogram) return null
  const { counts, bin_edges, unit } = histogram
  const data = counts.map((cnt, i) => ({
    range: `${fmt(bin_edges[i], 1)}–${fmt(bin_edges[i + 1], 1)}`,
    count: cnt,
  }))
  return (
    <div className="card" style={{ minHeight: 280 }}>
      <SectionTitle>Length Distribution ({unit})</SectionTitle>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} style={CHART_STYLE} margin={{ top: 0, right: 16, bottom: 0, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis dataKey="range" tick={{ fill: '#64748b', fontSize: 10 }} interval={1} />
          <YAxis tick={{ fill: '#64748b', fontSize: 11 }} allowDecimals={false} />
          <Tooltip contentStyle={TIP_STYLE} />
          <Bar dataKey="count" fill="#0ea5e9" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

/* ──────────────────────────────────────────── Per-class bar charts ─── */
function PerClassBars({ perClass, metric, unit }) {
  const data = CLASS_ORDER.filter(c => perClass[c]).map(c => {
    const s = perClass[c][metric]
    return { name: c, mean: s?.mean, min: s?.min, max: s?.max }
  })
  return (
    <ResponsiveContainer width="100%" height={180}>
      <BarChart data={data} style={CHART_STYLE} margin={{ top: 0, right: 16, bottom: 0, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 12 }} />
        <YAxis tick={{ fill: '#64748b', fontSize: 11 }} unit={unit ? ` ${unit}` : ''} />
        <Tooltip contentStyle={TIP_STYLE} formatter={v => [fmt(v, 2), '']} />
        <Bar dataKey="mean" radius={[4, 4, 0, 0]}>
          {data.map(d => <Cell key={d.name} fill={CLASS_COLORS[d.name]} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

/* ──────────────────────────────────────────── Scatter: length vs circularity ─── */
function LengthCircScatter({ detections, pixelToMicron }) {
  const byClass = {}
  for (const d of detections) {
    const cls = d.final_class
    if (!byClass[cls]) byClass[cls] = []
    const scale = pixelToMicron
    byClass[cls].push({ x: round2(d.size.length_px * scale), y: d.size.circularity, z: d.size.area_px })
  }
  return (
    <div className="card">
      <SectionTitle>Length vs Circularity (bubble = area)</SectionTitle>
      <ResponsiveContainer width="100%" height={240}>
        <ScatterChart style={CHART_STYLE} margin={{ top: 0, right: 20, bottom: 0, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis dataKey="x" name="Length" type="number" tick={{ fill: '#64748b', fontSize: 11 }} label={{ value: 'Length', fill: '#64748b', position: 'insideBottomRight', offset: -10, fontSize: 11 }} />
          <YAxis dataKey="y" name="Circularity" domain={[0, 1]} tick={{ fill: '#64748b', fontSize: 11 }} />
          <ZAxis dataKey="z" range={[30, 300]} name="Area" />
          <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={TIP_STYLE}
            formatter={(v, n) => [fmt(v, 3), n]} />
          {CLASS_ORDER.filter(c => byClass[c]).map(c => (
            <Scatter key={c} name={c} data={byClass[c]} fill={CLASS_COLORS[c]} opacity={0.75} />
          ))}
          <Legend wrapperStyle={{ fontSize: 12, color: '#64748b' }} />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  )
}
function round2(v) { return Math.round(v * 100) / 100 }

/* ──────────────────────────────────────────── Radar: per-class shape ─── */
function ShapeRadar({ perClass }) {
  const metrics = ['aspect_ratio', 'circularity']
  const rows = metrics.map(m => {
    const row = { metric: m }
    for (const c of CLASS_ORDER) {
      if (perClass[c]?.[m]) row[c] = perClass[c][m].mean
    }
    return row
  })

  // Normalise all to 0-1 for radar readability
  const maxVals = {}
  for (const m of metrics) {
    maxVals[m] = Math.max(...CLASS_ORDER.map(c => perClass[c]?.[m]?.mean || 0))
  }
  const norm = rows.map(r => {
    const nr = { metric: r.metric }
    for (const c of CLASS_ORDER) nr[c] = maxVals[r.metric] ? (r[c] / maxVals[r.metric]) : 0
    return nr
  })

  return (
    <div className="card">
      <SectionTitle>Shape Profile (normalised)</SectionTitle>
      <ResponsiveContainer width="100%" height={240}>
        <RadarChart data={norm} style={CHART_STYLE}>
          <PolarGrid stroke="#e2e8f0" />
          <PolarAngleAxis dataKey="metric" tick={{ fill: '#64748b', fontSize: 12 }} />
          <PolarRadiusAxis tick={{ fill: '#64748b', fontSize: 10 }} domain={[0, 1]} />
          {CLASS_ORDER.filter(c => perClass[c]).map(c => (
            <Radar key={c} name={c} dataKey={c} stroke={CLASS_COLORS[c]} fill={CLASS_COLORS[c]} fillOpacity={0.18} />
          ))}
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Tooltip contentStyle={TIP_STYLE} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  )
}

/* ──────────────────────────────────────────── Confidence dist bar ─── */
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
    <div className="card">
      <SectionTitle>YOLO Confidence Distribution</SectionTitle>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} style={CHART_STYLE} margin={{ top: 0, right: 16, bottom: 0, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis dataKey="range" tick={{ fill: '#64748b', fontSize: 11 }} />
          <YAxis tick={{ fill: '#64748b', fontSize: 11 }} allowDecimals={false} />
          <Tooltip contentStyle={TIP_STYLE} />
          {CLASS_ORDER.map(c => (
            <Bar key={c} dataKey={c} stackId="a" fill={CLASS_COLORS[c]} radius={c === 'fragment' ? [4, 4, 0, 0] : [0, 0, 0, 0]} />
          ))}
          <Legend wrapperStyle={{ fontSize: 12 }} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

/* ──────────────────────────────────────────── Metric table ─── */
function MetricTable({ perClass, unit }) {
  const metrics = [
    { key: 'length', label: `Length (${unit})` },
    { key: 'width',  label: `Width (${unit})` },
    { key: 'area',   label: `Area (${unit}²)` },
    { key: 'circularity', label: 'Circularity' },
    { key: 'aspect_ratio', label: 'Aspect Ratio' },
    { key: 'yolo_confidence', label: 'YOLO Conf.' },
    { key: 'effnet_confidence', label: 'EfficientNet Conf.' },
  ]
  const cols = CLASS_ORDER.filter(c => perClass[c])

  return (
    <div style={{ overflowX: 'auto' }}>
      <table className="data-table">
        <thead>
          <tr>
            <th>Metric</th>
            {cols.map(c => (
              <th key={c} colSpan={4} style={{ color: CLASS_COLORS[c], textAlign: 'center' }}>
                {c.toUpperCase()}
              </th>
            ))}
          </tr>
          <tr>
            <th />
            {cols.map(c => (
              <React.Fragment key={c}>
                <th>Mean</th><th>Median</th><th>Min</th><th>Max</th>
              </React.Fragment>
            ))}
          </tr>
        </thead>
        <tbody>
          {metrics.map(({ key, label }) => (
            <tr key={key}>
              <td style={{ color: 'var(--text-muted)', fontWeight: 500, whiteSpace: 'nowrap' }}>{label}</td>
              {cols.map(c => {
                const s = perClass[c]?.[key]
                return (
                  <React.Fragment key={c}>
                    <td>{fmt(s?.mean)}</td>
                    <td>{fmt(s?.median)}</td>
                    <td>{fmt(s?.min)}</td>
                    <td>{fmt(s?.max)}</td>
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

/* ──────────────────────────────────────────── Detection table ─── */
function DetectionTable({ detections, pixelToMicron }) {
  const [filter, setFilter] = useState('all')
  const [sortKey, setSortKey] = useState('id')
  const [sortDir, setSortDir] = useState(1)

  const toggleSort = key => {
    if (sortKey === key) setSortDir(d => -d)
    else { setSortKey(key); setSortDir(1) }
  }

  const shown = detections
    .filter(d => filter === 'all' || d.final_class === filter)
    .slice()
    .sort((a, b) => {
      let av = sortKey === 'length' ? a.size.length_px : sortKey === 'area' ? a.size.area_px : a[sortKey]
      let bv = sortKey === 'length' ? b.size.length_px : sortKey === 'area' ? b.size.area_px : b[sortKey]
      if (typeof av === 'string') av = av.toLowerCase(), bv = bv.toLowerCase()
      return sortDir * (av < bv ? -1 : av > bv ? 1 : 0)
    })

  const scale = pixelToMicron
  const unit  = pixelToMicron !== 1.0 ? 'µm' : 'px'

  const SortHd = ({ k, children }) => (
    <th onClick={() => toggleSort(k)} style={{ cursor: 'pointer', userSelect: 'none', whiteSpace: 'nowrap' }}>
      {children} {sortKey === k ? (sortDir > 0 ? '↑' : '↓') : <span style={{ opacity: .3 }}>↕</span>}
    </th>
  )

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem', flexWrap: 'wrap', gap: '0.5rem' }}>
        <div className="pill-bar">
          {['all', ...CLASS_ORDER].map(c => (
            <button key={c} className={`pill ${filter === c ? 'active' : ''}`} onClick={() => setFilter(c)}>
              {c === 'all' ? `All (${detections.length})` : `${c} (${detections.filter(d => d.final_class === c).length})`}
            </button>
          ))}
        </div>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table className="data-table">
          <thead>
            <tr>
              <SortHd k="id">#</SortHd>
              <SortHd k="final_class">Class</SortHd>
              <SortHd k="yolo_confidence">YOLO conf</SortHd>
              <SortHd k="effnet_confidence">ENet conf</SortHd>
              <SortHd k="mask_confidence">Mask conf</SortHd>
              <SortHd k="length">Length ({unit})</SortHd>
              <th>Width ({unit})</th>
              <th>Area ({unit}²)</th>
              <th>Circ.</th>
              <th>AR</th>
              <th>Seg source</th>
            </tr>
          </thead>
          <tbody>
            {shown.map(d => (
              <tr key={d.id}>
                <td className="text-muted">{d.id}</td>
                <td>
                  <span className={`badge badge-${d.final_class}`}>{d.final_class}</span>
                  {d.yolo_class !== d.final_class && (
                    <span className="text-xs text-muted" style={{ marginLeft: 4 }}>
                      (was {d.yolo_class})
                    </span>
                  )}
                </td>
                <td>{fmt(d.yolo_confidence)}</td>
                <td>{fmt(d.effnet_confidence)}</td>
                <td>{fmt(d.mask_confidence)}</td>
                <td style={{ fontWeight: 500 }}>{fmt(d.size.length_px * scale, 1)}</td>
                <td>{fmt(d.size.width_px * scale, 1)}</td>
                <td>{fmt(d.size.area_px * scale * scale, 0)}</td>
                <td>{fmt(d.size.circularity, 3)}</td>
                <td>{fmt(d.size.aspect_ratio, 2)}</td>
                <td className="text-muted text-xs">{d.segmentation_source}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

/* ──────────────────────────────────────────── Image viewer ─── */
function ImageViewer({ jobId, images }) {
  const [imgIdx, setImgIdx] = useState(0)
  const [mode, setMode] = useState('vis')   // vis | mask | original
  const total = images.length

  const im = images[imgIdx]
  const src = mode === 'vis' ? imageUrl(jobId, imgIdx)
            : mode === 'mask' ? maskUrl(jobId, imgIdx)
            : originalUrl(jobId, imgIdx)

  return (
    <div className="card">
      <SectionTitle>Image Preview</SectionTitle>
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.75rem', flexWrap: 'wrap', alignItems: 'center' }}>
        {/* Image selector */}
        {total > 1 && (
          <div style={{ display: 'flex', gap: '0.25rem', marginRight: '0.5rem' }}>
            {images.map((img, i) => (
              <button key={i} onClick={() => setImgIdx(i)}
                className={`pill ${imgIdx === i ? 'active' : ''}`}
                style={{ fontSize: '0.75rem' }}>
                {img.filename || `Image ${i + 1}`}
              </button>
            ))}
          </div>
        )}
        {/* Mode toggles */}
        <div className="pill-bar">
          {[['vis', <><ImageIcon size={14} /> Annotated</>], ['mask', <><Layers size={14} /> Masks</>], ['original', <><Maximize size={14} /> Original</>]].map(([k, lbl]) => (
            <button key={k} className={`pill ${mode === k ? 'active' : ''}`} onClick={() => setMode(k)} style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>{lbl}</button>
          ))}
        </div>
      </div>
      <img
        key={src}
        src={src}
        alt={mode}
        style={{
          width: '100%',
          maxHeight: 520,
          objectFit: 'contain',
          borderRadius: 'var(--radius-sm)',
          background: '#f8fafc',
          border: '1px solid var(--border)',
        }}
        onError={e => { e.target.style.opacity = 0.3 }}
      />
      {im && (
        <p className="text-xs text-muted" style={{ marginTop: '0.5rem' }}>
          {im.filename}  ·  {im.image_size?.width} × {im.image_size?.height} px
          ·  {im.summary?.total} detections
        </p>
      )}
    </div>
  )
}

/* ──────────────────────────────────────────── Main page ─── */
export default function ResultsPage() {
  const [result, setResult] = useState(null)
  const [activeSections, setActiveSections] = useState({
    overview: true,
    charts: true,
    scatter: true,
    confidence: true,
    perClassBars: true,
    table: true,
    radar: true,
    metricTable: true,
    images: true,
  })

  useEffect(() => {
    const raw = sessionStorage.getItem('mp_last_result')
    if (raw) {
      try { setResult(JSON.parse(raw)) } catch {}
    }
  }, [])

  const toggle = key => setActiveSections(s => ({ ...s, [key]: !s[key] }))

  if (!result) {
    return (
      <div className="empty-state" style={{ marginTop: '4rem' }}>
        <div className="icon text-muted"><BarChart2 size={48} strokeWidth={1.5} /></div>
        <h2 style={{ fontWeight: 600, marginBottom: '0.5rem' }}>No results yet</h2>
        <p className="text-muted">Run the detection pipeline on the <a href="/detect" style={{ color: 'var(--primary)' }}>Detect</a> page first.</p>
      </div>
    )
  }

  const { job_id, images = [], config = {} } = result
  const pixelToMicron = config.pixel_to_micron || 1.0
  const unit = pixelToMicron !== 1.0 ? 'µm' : 'px'

  // Aggregate across all processed images
  const allDetections = images.flatMap(im => im.detections || [])
  const totalCounts   = CLASS_ORDER.reduce((acc, c) => {
    acc[c] = allDetections.filter(d => d.final_class === c).length
    return acc
  }, {})
  const totalN = allDetections.length

  // Merge per-class summaries (use first image's if multi — or re-aggregate)
  const combinedSummary = images.length === 1
    ? images[0].summary
    : (() => {
        // Re-build from allDetections
        const counts = { ...totalCounts }
        const perClass = {}
        for (const c of CLASS_ORDER) {
          const grp = allDetections.filter(d => d.final_class === c)
          if (!grp.length) continue
          const stats = (arr) => {
            const n = arr.length
            if (!n) return {}
            const sorted = [...arr].sort((a, b) => a - b)
            const mean   = arr.reduce((s, x) => s + x, 0) / n
            const median = sorted[Math.floor(n / 2)]
            const min    = sorted[0]
            const max    = sorted[n - 1]
            const std    = Math.sqrt(arr.reduce((s, x) => s + (x - mean) ** 2, 0) / n)
            return { mean: round2(mean), median: round2(median), min: round2(min), max: round2(max), std: round2(std) }
          }
          perClass[c] = {
            count: grp.length,
            length: stats(grp.map(d => d.size.length_px * pixelToMicron)),
            width:  stats(grp.map(d => d.size.width_px  * pixelToMicron)),
            area:   stats(grp.map(d => d.size.area_px   * pixelToMicron * pixelToMicron)),
            circularity:  stats(grp.map(d => d.size.circularity)),
            aspect_ratio: stats(grp.map(d => d.size.aspect_ratio)),
            yolo_confidence:   stats(grp.map(d => d.yolo_confidence)),
            effnet_confidence: stats(grp.map(d => d.effnet_confidence)),
          }
        }
        // histogram
        const lengths = allDetections.map(d => d.size.length_px * pixelToMicron)
        const hist = buildHistogram(lengths, 10)
        return { total: totalN, counts, per_class: perClass, length_histogram: hist, unit, pixel_to_micron: pixelToMicron }
      })()

  const { per_class: perClass = {}, length_histogram: histogram } = combinedSummary

  // YOLO reclassification rate
  const reclassified = allDetections.filter(d => d.yolo_class !== d.final_class).length
  const avgYoloConf  = totalN ? (allDetections.reduce((s, d) => s + d.yolo_confidence, 0) / totalN) : 0
  const avgEffConf   = totalN ? (allDetections.reduce((s, d) => s + d.effnet_confidence, 0) / totalN) : 0

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', paddingBottom: '2rem' }}>

      {/* ── Page header ── */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h1 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: '0.25rem' }}>Pipeline Results</h1>
          <p className="text-muted text-sm">
            Job <code style={{ background: 'var(--surface2)', padding: '0 6px', borderRadius: 4 }}>{job_id}</code>
            &nbsp;·&nbsp;{images.length} image{images.length !== 1 ? 's' : ''}
            &nbsp;·&nbsp;pixel→{unit}: {pixelToMicron}
          </p>
        </div>
        <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
          <button className="btn btn-ghost btn-sm" onClick={() => {
            const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' })
            const a = document.createElement('a'); a.href = URL.createObjectURL(blob)
            a.download = `mp_report_${job_id.slice(0, 8)}.json`; a.click()
          }} style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}><Download size={14} /> Export JSON</button>
        </div>
      </div>

      {/* ── 1: KPI overview tiles ── */}
      <section>
        <SectionTitle toggle={activeSections.overview} onToggle={() => toggle('overview')} icon={Activity}>
          Overview
        </SectionTitle>
        {activeSections.overview && (
          <div className="grid-4" style={{ marginTop: 0 }}>
            <StatTile label="Total Detections" value={totalN} sub={`${images.length} image(s)`} icon={BarChart2} />
            <StatTile label="Fiber" value={totalCounts.fiber} sub={pct(totalCounts.fiber, totalN)} color="var(--fiber)" />
            <StatTile label="Film" value={totalCounts.film} sub={pct(totalCounts.film, totalN)} color="var(--film)" />
            <StatTile label="Fragment" value={totalCounts.fragment} sub={pct(totalCounts.fragment, totalN)} color="var(--fragment)" />
            <StatTile label="Avg YOLO Conf." value={fmt(avgYoloConf)} />
            <StatTile label="Avg EfficientNet Conf." value={fmt(avgEffConf)} />
            <StatTile label="ENet Reclassified" value={reclassified} sub={pct(reclassified, totalN)} />
            <StatTile label="Avg Length" value={fmt(combinedSummary.overall?.length?.mean)} sub={unit} />
          </div>
        )}
      </section>

      {/* ── 2: Charts row ── */}
      <section>
        <SectionTitle toggle={activeSections.charts} onToggle={() => toggle('charts')} icon={BarChart2}>
          Distribution Charts
        </SectionTitle>
        {activeSections.charts && (
          <div className="grid-2">
            <CountDonut counts={totalCounts} total={totalN} />
            <LengthHistogram histogram={histogram} />
          </div>
        )}
      </section>

      {/* ── 3: Per-class bar charts ── */}
      <section>
        <SectionTitle toggle={activeSections.perClassBars} onToggle={() => toggle('perClassBars')} icon={BarChart2}>
          Per-Class Size Comparison
        </SectionTitle>
        {activeSections.perClassBars && (
          <div className="grid-3">
            {[
              { key: 'length', label: `Mean Length (${unit})` },
              { key: 'area',   label: `Mean Area (${unit}²)` },
              { key: 'circularity', label: 'Mean Circularity' },
            ].map(({ key, label }) => (
              <div key={key} className="card">
                <div style={{ fontWeight: 600, marginBottom: '0.75rem', fontSize: '0.875rem' }}>{label}</div>
                <PerClassBars perClass={perClass} metric={key} unit={key === 'circularity' ? '' : unit} />
              </div>
            ))}
          </div>
        )}
      </section>

      {/* ── 4: Scatter & Radar ── */}
      <section>
        <SectionTitle toggle={activeSections.scatter} onToggle={() => toggle('scatter')} icon={Activity}>
          Shape Analysis
        </SectionTitle>
        {activeSections.scatter && (
          <div className="grid-2">
            <LengthCircScatter detections={allDetections} pixelToMicron={pixelToMicron} />
            <ShapeRadar perClass={perClass} />
          </div>
        )}
      </section>

      {/* ── 5: Confidence distribution ── */}
      <section>
        <SectionTitle toggle={activeSections.confidence} onToggle={() => toggle('confidence')} icon={BarChart2}>
          Confidence Analysis
        </SectionTitle>
        {activeSections.confidence && <ConfidenceBars detections={allDetections} />}
      </section>

      {/* ── 6: Image viewer ── */}
      <section>
        <SectionTitle toggle={activeSections.images} onToggle={() => toggle('images')} icon={ImageIcon}>
          Visualizations
        </SectionTitle>
        {activeSections.images && job_id && images.length > 0 && (
          <ImageViewer jobId={job_id} images={images} />
        )}
      </section>

      {/* ── 7: Stats metric table ── */}
      <section>
        <SectionTitle toggle={activeSections.metricTable} onToggle={() => toggle('metricTable')} icon={FileText}>
          Detailed Statistics Table
        </SectionTitle>
        {activeSections.metricTable && (
          <div className="card">
            <MetricTable perClass={perClass} unit={unit} />
          </div>
        )}
      </section>

      {/* ── 8: Detection table ── */}
      <section>
        <SectionTitle toggle={activeSections.table} onToggle={() => toggle('table')} icon={FileText}>
          All Detections
        </SectionTitle>
        {activeSections.table && allDetections.length > 0 && (
          <div className="card">
            <DetectionTable detections={allDetections} pixelToMicron={pixelToMicron} />
          </div>
        )}
      </section>

      {/* ── 9: Raw config ── */}
      <section>
        <div className="card">
          <div style={{ fontWeight: 600, marginBottom: '0.75rem', fontSize: '0.875rem' }}>Pipeline Config</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem' }}>
            {Object.entries(config).map(([k, v]) => (
              <div key={k} style={{
                background: 'var(--surface2)', borderRadius: 6,
                padding: '0.375rem 0.75rem', fontSize: '0.75rem',
              }}>
                <span style={{ color: 'var(--text-muted)' }}>{k}: </span>
                <span style={{ fontWeight: 600 }}>{String(v)}</span>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  )
}

/* ── tiny histogram builder ── */
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
