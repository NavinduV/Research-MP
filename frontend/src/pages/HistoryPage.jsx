import React, { useState, useEffect } from 'react'
import { getJobs, getResult } from '../api/detect.js'
import { useNavigate } from 'react-router-dom'
import { AlertTriangle, ClipboardList, Loader2, ArrowRight } from 'lucide-react'

function timeAgo(ts) {
  const diff = Math.floor(Date.now() / 1000 - ts)
  if (diff < 60) return `${diff}s ago`
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  return `${Math.floor(diff / 86400)}d ago`
}

export default function HistoryPage() {
  const [jobs, setJobs] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const navigate = useNavigate()

  useEffect(() => {
    getJobs()
      .then(data => { setJobs(data.jobs || []); setLoading(false) })
      .catch(err => { setError(err.message); setLoading(false) })
  }, [])

  const loadJob = async (jobId) => {
    try {
      const result = await getResult(jobId)
      sessionStorage.setItem('mp_last_result', JSON.stringify(result))
      navigate('/results')
    } catch (err) {
      alert('Could not load job: ' + err.message)
    }
  }

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', padding: '4rem' }}>
        <Loader2 className="spinner text-primary" size={32} />
      </div>
    )
  }

  if (error) {
    return (
      <div className="empty-state">
        <div className="icon text-danger"><AlertTriangle size={48} strokeWidth={1.5} /></div>
        <p>Could not reach backend: <strong>{error}</strong></p>
        <p className="text-sm text-muted" style={{ marginTop: '0.5rem' }}>Make sure the FastAPI server is running on port 8000.</p>
      </div>
    )
  }

  if (jobs.length === 0) {
    return (
      <div className="empty-state" style={{ marginTop: '4rem' }}>
        <div className="icon text-muted"><ClipboardList size={48} strokeWidth={1.5} /></div>
        <h2 style={{ fontWeight: 600, marginBottom: '0.5rem' }}>No past jobs</h2>
        <p className="text-muted text-sm">Past detection jobs will appear here (current session only).</p>
      </div>
    )
  }

  return (
    <div>
      <h1 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: '1.5rem' }}>Job History</h1>
      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>Job ID</th>
              <th>Status</th>
              <th>Created</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {[...jobs].reverse().map(j => (
              <tr key={j.job_id}>
                <td>
                  <code style={{ fontSize: '0.75rem', background: 'var(--surface2)', padding: '2px 6px', borderRadius: 4 }}>
                    {j.job_id.slice(0, 12)}…
                  </code>
                </td>
                <td>
                  <span className={`badge ${j.status === 'done' ? 'badge-primary' : j.status === 'error' ? '' : 'badge-neutral'}`}
                    style={j.status === 'error' ? { background: 'rgba(248,113,113,.15)', color: 'var(--danger)' } : {}}>
                    {j.status}
                  </span>
                </td>
                <td className="text-muted text-sm">{j.created_at ? timeAgo(j.created_at) : '—'}</td>
                <td>
                  {j.status === 'done' && (
                    <button className="btn btn-ghost btn-sm" onClick={() => loadJob(j.job_id)} style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                      Load Results <ArrowRight size={14} />
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
