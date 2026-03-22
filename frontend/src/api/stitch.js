const BASE = '/api'

/**
 * POST /api/stitch/analyze  — scan a folder and return brightness groups
 * @param {string} folderPath
 */
export async function analyzeFolder(folderPath) {
  const fd = new FormData()
  fd.append('folder_path', folderPath)
  const res = await fetch(`${BASE}/stitch/analyze`, { method: 'POST', body: fd })
  if (!res.ok) {
    const d = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(d.detail || 'Analyze failed')
  }
  return res.json()
}

/** Thumbnail URL for an on-disk image */
export const thumbnailUrl = (path) => `${BASE}/stitch/thumbnail?path=${encodeURIComponent(path)}`

/**
 * POST /api/stitch/run  — run stitching on selected images
 */
export async function runStitch({ folderPath, selectedImages, advancedMode, outputName, maxDim, upscale }) {
  const fd = new FormData()
  fd.append('folder_path', folderPath)
  fd.append('selected_images', selectedImages.join('|||'))
  fd.append('advanced_mode', advancedMode)
  fd.append('output_name', outputName || 'stitched_output.png')
  fd.append('max_dim', maxDim || 8192)
  fd.append('upscale', upscale || 1.0)
  const res = await fetch(`${BASE}/stitch/run`, { method: 'POST', body: fd })
  if (!res.ok) {
    const d = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(d.detail || 'Stitch failed')
  }
  return res.json()
}

/** Preview image URL for a stitch session */
export const stitchPreviewUrl = (sessionId) => `${BASE}/stitch/preview/${sessionId}`

/** Live enhancement preview URL */
export const enhancePreviewUrl = (sessionId, params) => {
  const q = new URLSearchParams({ ...params, t: Date.now() })
  return `${BASE}/stitch/enhance-preview/${sessionId}?${q}`
}

/**
 * POST /api/stitch/enhance/:sessionId  — save enhanced image
 */
export async function saveEnhancement(sessionId, { sharpen, denoise, contrast, brightness, auto_wb }) {
  const fd = new FormData()
  fd.append('sharpen', sharpen)
  fd.append('denoise', denoise)
  fd.append('contrast', contrast)
  fd.append('brightness', brightness)
  fd.append('auto_wb', auto_wb)
  const res = await fetch(`${BASE}/stitch/enhance/${sessionId}`, { method: 'POST', body: fd })
  if (!res.ok) throw new Error('Enhancement save failed')
  return res.json()
}

/**
 * Fetch the stitched file as a File object so it can be appended to FormData
 * for the detection pipeline.
 */
export async function fetchStitchedFile(sessionId) {
  const res = await fetch(`${BASE}/stitch/send-to-pipeline/${sessionId}`, { method: 'POST' })
  if (!res.ok) throw new Error('Could not fetch stitched image')
  const blob = await res.blob()
  // Extract filename from Content-Disposition header
  const cd = res.headers.get('content-disposition') || ''
  const match = cd.match(/filename="?(.+?)"?$/i)
  const name = match ? match[1] : 'stitched.png'
  return new File([blob], name, { type: blob.type })
}

/** DELETE /api/stitch/:sessionId */
export async function deleteStitch(sessionId) {
  const res = await fetch(`${BASE}/stitch/${sessionId}`, { method: 'DELETE' })
  if (!res.ok) throw new Error('Delete failed')
  return res.json()
}

/**
 * POST /api/stitch/upload-and-analyze
 * Upload browser File objects for stitching — saves them to a temp
 * folder on the server and runs brightness analysis.
 */
export async function uploadAndAnalyzeForStitch(files) {
  const fd = new FormData()
  files.forEach(f => fd.append('files', f))
  const res = await fetch(`${BASE}/stitch/upload-and-analyze`, { method: 'POST', body: fd })
  if (!res.ok) {
    const d = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(d.detail || 'Upload failed')
  }
  return res.json()
}
