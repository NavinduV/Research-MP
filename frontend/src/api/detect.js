const BASE = '/api'

/**
 * POST /api/detect
 * @param {FormData} formData
 */
export async function runDetection(formData) {
  const res = await fetch(`${BASE}/detect`, { method: 'POST', body: formData })
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(detail.detail || 'Detection failed')
  }
  return res.json()
}

/** GET /api/results/:jobId */
export async function getResult(jobId) {
  const res = await fetch(`${BASE}/results/${jobId}`)
  if (!res.ok) throw new Error('Result not found')
  return res.json()
}

/** GET /api/health */
export async function getHealth() {
  const res = await fetch(`${BASE}/health`)
  if (!res.ok) throw new Error('Backend offline')
  return res.json()
}

/** GET /api/jobs */
export async function getJobs() {
  const res = await fetch(`${BASE}/jobs`)
  if (!res.ok) throw new Error('Could not fetch jobs')
  return res.json()
}

export const imageUrl         = (jobId, idx) => `${BASE}/image/${jobId}/${idx}`
export const maskUrl          = (jobId, idx) => `${BASE}/mask/${jobId}/${idx}`
export const originalUrl      = (jobId, idx) => `${BASE}/original/${jobId}/${idx}`
