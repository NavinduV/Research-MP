from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
import io
import cv2
import numpy as np
from ..inference import analyze_image
from ..macro_stitch_pipeline import process_folder, enhance_image

app = FastAPI(title="Filter Paper Stitching Pipeline")

# Store for stitched images awaiting enhancement (in production, use proper storage)
_stitched_cache = {}

# Serve stitched images
@app.get("/images/{filepath:path}")
async def serve_image(filepath: str):
    """Serve stitched images for preview."""
    full_path = filepath
    if os.path.exists(full_path):
        return FileResponse(full_path)
    return HTMLResponse("<p>Image not found</p>", status_code=404)

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    file_path = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = analyze_image(file_path)
    return result


@app.get("/", response_class=HTMLResponse)
async def home():
    """Home page with links to all features."""
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Filter Paper Analysis</title>
            <style>
                body { font-family: system-ui, -apple-system, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; background: #f5f5f5; }
                .card { background: white; border-radius: 12px; padding: 2rem; margin: 1rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
                h1 { color: #333; }
                a.btn { display: inline-block; background: #4CAF50; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; margin: 0.5rem 0.5rem 0.5rem 0; }
                a.btn:hover { background: #45a049; }
                a.btn.secondary { background: #2196F3; }
                a.btn.secondary:hover { background: #1976D2; }
            </style>
        </head>
        <body>
            <div class="card">
                <h1>🔬 Filter Paper Analysis Pipeline</h1>
                <p>Tools for stitching partial filter paper images and detecting microplastics.</p>
                <a href="/stitch" class="btn">📷 Image Stitching</a>
                <a href="/enhance/open" class="btn secondary">✨ Enhance Image</a>
                <a href="/docs" class="btn secondary">📚 API Docs</a>
            </div>
        </body>
    </html>
    """


@app.get("/stitch", response_class=HTMLResponse)
async def stitch_form():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Filter Paper Stitching</title>
            <style>
                body { font-family: system-ui, -apple-system, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; background: #f5f5f5; }
                .card { background: white; border-radius: 12px; padding: 2rem; margin: 1rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
                h1 { color: #333; margin-top: 0; }
                label { display: block; margin: 1rem 0 0.5rem; font-weight: 500; color: #555; }
                input[type="text"], input[type="number"], select { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px; box-sizing: border-box; }
                input:focus, select:focus { outline: none; border-color: #4CAF50; }
                .row { display: flex; gap: 1rem; }
                .row > div { flex: 1; }
                button { background: #4CAF50; color: white; padding: 14px 28px; border: none; border-radius: 6px; font-size: 16px; cursor: pointer; margin-top: 1.5rem; }
                button:hover { background: #45a049; }
                .info { background: #e3f2fd; padding: 1rem; border-radius: 6px; margin: 1rem 0; font-size: 14px; }
                a { color: #2196F3; }
                .section-title { font-size: 14px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin: 2rem 0 1rem; border-bottom: 1px solid #eee; padding-bottom: 0.5rem; }
            </style>
        </head>
        <body>
            <div class="card">
                <h1>📷 Filter Paper Image Stitching</h1>
                <p>Stitch partial filter paper images into a complete mosaic for microplastic detection.</p>
                
                <div class="info">
                    <strong>How it works:</strong> Place your overlapping partial images in a folder. 
                    The algorithm uses feature matching to align and blend them into a seamless output.
                </div>
                
                <form method="post" action="/stitch/run">
                    <div class="section-title">Input / Output</div>
                    
                    <label>Input Folder (containing partial images):</label>
                    <input type="text" name="input_folder" placeholder="dev-test/raw/s6" required>
                    
                    <label>Output Image Path:</label>
                    <input type="text" name="output_image" placeholder="dev-test/stitched/output.png" required>
                    
                    <div class="section-title">Quality Options</div>
                    
                    <div class="row">
                        <div>
                            <label>Max Output Dimension (px):</label>
                            <input type="number" name="max_dim" value="8192" min="512" step="256">
                        </div>
                        <div>
                            <label>Max Stitch Dimension (px):</label>
                            <input type="number" name="max_stitch_dim" value="8192" min="512" step="256">
                        </div>
                    </div>
                    
                    <div class="section-title">Upscale Options</div>
                    
                    <div class="row">
                        <div>
                            <label>Upscale Factor (1.0 = no upscale):</label>
                            <input type="number" name="upscale" value="1.0" min="1.0" max="4.0" step="0.5">
                        </div>
                        <div>
                            <label>Upscale Method:</label>
                            <select name="upscale_method">
                                <option value="lanczos" selected>Lanczos (Best Quality)</option>
                                <option value="cubic">Cubic</option>
                                <option value="linear">Linear (Fastest)</option>
                            </select>
                        </div>
                    </div>
                    
                    <button type="submit">🚀 Run Stitching Pipeline</button>
                </form>
                
                <p style="margin-top: 2rem;"><a href="/">← Back to Home</a></p>
            </div>
        </body>
    </html>
    """


@app.post("/stitch/run")
async def stitch_run(
    input_folder: str = Form(...),
    output_image: str = Form(...),
    max_dim: int = Form(8192),
    max_stitch_dim: int = Form(8192),
    upscale: float = Form(1.0),
    upscale_method: str = Form("lanczos"),
):
    import traceback
    import uuid
    
    try:
        meta = process_folder(
            input_folder=input_folder,
            output_image_path=output_image,
            max_dim=int(max_dim),
            max_stitch_dim=int(max_stitch_dim),
            upscale=float(upscale),
            upscale_method=upscale_method,
        )
    except Exception as e:
        tb = traceback.format_exc()
        return HTMLResponse(
            f"""
            <!DOCTYPE html>
            <html>
                <head>
                    <title>Stitching Error</title>
                    <style>
                        body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; background: #f5f5f5; }}
                        .card {{ background: white; border-radius: 12px; padding: 2rem; margin: 1rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
                        h2 {{ color: #c62828; margin-top: 0; }}
                        pre {{ white-space: pre-wrap; background: #ffebee; padding: 1rem; border-radius: 6px; overflow-x: auto; font-size: 13px; }}
                        a {{ color: #2196F3; }}
                    </style>
                </head>
                <body>
                    <div class="card">
                        <h2>❌ Error while running pipeline</h2>
                        <pre>{tb}</pre>
                        <p><a href="/stitch">← Try Again</a></p>
                    </div>
                </body>
            </html>
            """
        )

    # Get absolute path for image serving
    abs_image_path = os.path.abspath(meta['image_path'])
    
    # Generate a session ID and cache the image path for enhancement
    session_id = str(uuid.uuid4())[:8]
    _stitched_cache[session_id] = {
        'image_path': abs_image_path,
        'label_path': meta['label_path'],
        'meta': meta
    }
    
    return HTMLResponse(
        f"""
        <!DOCTYPE html>
        <html>
            <head>
                <title>Stitching Complete</title>
                <style>
                    body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 1100px; margin: 2rem auto; padding: 0 1rem; background: #f5f5f5; }}
                    .card {{ background: white; border-radius: 12px; padding: 2rem; margin: 1rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
                    h2 {{ color: #2e7d32; margin-top: 0; }}
                    .meta {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1rem 0; }}
                    .meta-item {{ background: #f5f5f5; padding: 1rem; border-radius: 6px; }}
                    .meta-item label {{ font-size: 12px; color: #888; text-transform: uppercase; }}
                    .meta-item value {{ font-size: 18px; font-weight: 500; display: block; margin-top: 4px; }}
                    .preview {{ margin: 1.5rem 0; text-align: center; }}
                    .preview img {{ max-width: 100%; max-height: 400px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); }}
                    a {{ color: #2196F3; }}
                    .btn {{ display: inline-block; background: #4CAF50; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; margin-right: 0.5rem; font-size: 16px; }}
                    .btn:hover {{ background: #45a049; }}
                    .btn.secondary {{ background: #2196F3; }}
                    .btn.secondary:hover {{ background: #1976D2; }}
                    .btn.primary {{ background: #ff9800; }}
                    .btn.primary:hover {{ background: #f57c00; }}
                </style>
            </head>
            <body>
                <div class="card">
                    <h2>✅ Stitching Complete!</h2>
                    
                    <div class="meta">
                        <div class="meta-item">
                            <label>Original Size</label>
                            <value>{meta['original_size'][0]} × {meta['original_size'][1]} px</value>
                        </div>
                        <div class="meta-item">
                            <label>Output Size</label>
                            <value>{meta['output_size'][0]} × {meta['output_size'][1]} px</value>
                        </div>
                        <div class="meta-item">
                            <label>Scale Factor</label>
                            <value>{meta['scale']:.3f}</value>
                        </div>
                    </div>
                    
                    <div class="preview">
                        <h3>Preview</h3>
                        <img src="/images/{abs_image_path}" alt="Stitched Result">
                    </div>
                    
                    <p><strong>Image saved to:</strong> <code>{meta['image_path']}</code></p>
                    
                    <p style="margin-top: 1.5rem;">
                        <a href="/enhance/{session_id}" class="btn primary">✨ Enhance Image</a>
                        <a href="/stitch" class="btn secondary">🔄 Stitch Another</a>
                        <a href="/" class="btn secondary">🏠 Home</a>
                    </p>
                </div>
            </body>
        </html>
        """
    )


@app.get("/enhance/{session_id}", response_class=HTMLResponse)
async def enhance_editor(session_id: str):
    """Interactive enhancement editor with live preview."""
    if session_id not in _stitched_cache:
        return HTMLResponse("<p>Session expired. Please stitch again.</p>", status_code=404)
    
    cache = _stitched_cache[session_id]
    image_path = cache['image_path']
    
    return f"""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Enhance Image</title>
            <style>
                * {{ box-sizing: border-box; }}
                body {{ font-family: system-ui, -apple-system, sans-serif; margin: 0; padding: 1rem; background: #1a1a2e; color: #eee; min-height: 100vh; }}
                .container {{ display: grid; grid-template-columns: 300px 1fr; gap: 1rem; max-width: 1600px; margin: 0 auto; height: calc(100vh - 2rem); }}
                .controls {{ background: #16213e; border-radius: 12px; padding: 1.5rem; overflow-y: auto; }}
                .preview-area {{ background: #0f0f23; border-radius: 12px; display: flex; flex-direction: column; overflow: hidden; }}
                h1 {{ margin: 0 0 1.5rem; font-size: 1.5rem; color: #fff; }}
                h3 {{ margin: 1.5rem 0 1rem; font-size: 0.85rem; color: #888; text-transform: uppercase; letter-spacing: 1px; border-bottom: 1px solid #333; padding-bottom: 0.5rem; }}
                
                .slider-group {{ margin: 1rem 0; }}
                .slider-group label {{ display: flex; justify-content: space-between; margin-bottom: 0.5rem; font-size: 0.9rem; }}
                .slider-group label span {{ color: #4CAF50; font-weight: 600; min-width: 50px; text-align: right; }}
                input[type="range"] {{ width: 100%; height: 6px; border-radius: 3px; background: #333; outline: none; -webkit-appearance: none; }}
                input[type="range"]::-webkit-slider-thumb {{ -webkit-appearance: none; width: 18px; height: 18px; border-radius: 50%; background: #4CAF50; cursor: pointer; }}
                input[type="range"]::-webkit-slider-thumb:hover {{ background: #66BB6A; }}
                
                .checkbox-group {{ margin: 1rem 0; }}
                .checkbox-group label {{ display: flex; align-items: center; gap: 0.75rem; cursor: pointer; padding: 0.5rem; border-radius: 6px; transition: background 0.2s; }}
                .checkbox-group label:hover {{ background: rgba(255,255,255,0.05); }}
                .checkbox-group input {{ width: 18px; height: 18px; accent-color: #4CAF50; }}
                
                .btn {{ display: inline-block; padding: 12px 24px; border: none; border-radius: 6px; font-size: 14px; cursor: pointer; text-decoration: none; text-align: center; transition: all 0.2s; }}
                .btn-primary {{ background: #4CAF50; color: white; width: 100%; margin-top: 1rem; }}
                .btn-primary:hover {{ background: #66BB6A; }}
                .btn-secondary {{ background: #333; color: #fff; }}
                .btn-secondary:hover {{ background: #444; }}
                .btn-reset {{ background: #ff5722; color: white; width: 100%; margin-top: 0.5rem; }}
                .btn-reset:hover {{ background: #ff7043; }}
                
                .preview-header {{ padding: 1rem; background: rgba(255,255,255,0.05); display: flex; justify-content: space-between; align-items: center; }}
                .preview-header h2 {{ margin: 0; font-size: 1rem; }}
                .status {{ font-size: 0.85rem; color: #888; }}
                .status.loading {{ color: #ff9800; }}
                
                .image-container {{ flex: 1; display: flex; align-items: center; justify-content: center; padding: 1rem; overflow: auto; }}
                .image-container img {{ max-width: 100%; max-height: 100%; object-fit: contain; border-radius: 8px; transition: opacity 0.3s; }}
                .image-container img.loading {{ opacity: 0.5; }}
                
                .presets {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-top: 1rem; }}
                .preset-btn {{ padding: 8px 12px; font-size: 12px; }}
                
                .back-link {{ display: block; margin-top: 1rem; text-align: center; color: #888; text-decoration: none; }}
                .back-link:hover {{ color: #fff; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="controls">
                    <h1>✨ Enhance Image</h1>
                    
                    <h3>Adjustments</h3>
                    
                    <div class="slider-group">
                        <label>Sharpening <span id="sharpen-val">0</span></label>
                        <input type="range" id="sharpen" min="0" max="3" step="0.1" value="0">
                    </div>
                    
                    <div class="slider-group">
                        <label>Denoising <span id="denoise-val">0</span></label>
                        <input type="range" id="denoise" min="0" max="20" step="1" value="0">
                    </div>
                    
                    <div class="slider-group">
                        <label>Contrast <span id="contrast-val">0</span></label>
                        <input type="range" id="contrast" min="0" max="4" step="0.1" value="0">
                    </div>
                    
                    <div class="slider-group">
                        <label>Brightness <span id="brightness-val">0</span></label>
                        <input type="range" id="brightness" min="-50" max="50" step="1" value="0">
                    </div>
                    
                    <div class="checkbox-group">
                        <label>
                            <input type="checkbox" id="auto_wb">
                            Auto White Balance
                        </label>
                    </div>
                    
                    <h3>Presets</h3>
                    <div class="presets">
                        <button class="btn btn-secondary preset-btn" onclick="applyPreset('subtle')">Subtle</button>
                        <button class="btn btn-secondary preset-btn" onclick="applyPreset('vivid')">Vivid</button>
                        <button class="btn btn-secondary preset-btn" onclick="applyPreset('sharp')">Sharp</button>
                        <button class="btn btn-secondary preset-btn" onclick="applyPreset('clean')">Clean</button>
                    </div>
                    
                    <button class="btn btn-reset" onclick="resetAll()">Reset All</button>
                    
                    <h3>Save</h3>
                    <button class="btn btn-primary" onclick="saveEnhanced()">💾 Save Enhanced Image</button>
                    
                    <a href="/stitch" class="back-link">← Back to Stitching</a>
                </div>
                
                <div class="preview-area">
                    <div class="preview-header">
                        <h2>Live Preview</h2>
                        <span class="status" id="status">Ready</span>
                    </div>
                    <div class="image-container">
                        <img id="preview" src="/enhance/preview/{session_id}?sharpen=0&denoise=0&contrast=0&brightness=0&auto_wb=0&t=0" alt="Preview">
                    </div>
                </div>
            </div>
            
            <script>
                const sessionId = "{session_id}";
                let debounceTimer = null;
                
                const sliders = ['sharpen', 'denoise', 'contrast', 'brightness'];
                
                sliders.forEach(id => {{
                    const slider = document.getElementById(id);
                    const display = document.getElementById(id + '-val');
                    slider.addEventListener('input', () => {{
                        display.textContent = slider.value;
                        debounceUpdate();
                    }});
                }});
                
                document.getElementById('auto_wb').addEventListener('change', debounceUpdate);
                
                function debounceUpdate() {{
                    clearTimeout(debounceTimer);
                    document.getElementById('status').textContent = 'Updating...';
                    document.getElementById('status').className = 'status loading';
                    document.getElementById('preview').classList.add('loading');
                    
                    debounceTimer = setTimeout(updatePreview, 300);
                }}
                
                function updatePreview() {{
                    const params = new URLSearchParams({{
                        sharpen: document.getElementById('sharpen').value,
                        denoise: document.getElementById('denoise').value,
                        contrast: document.getElementById('contrast').value,
                        brightness: document.getElementById('brightness').value,
                        auto_wb: document.getElementById('auto_wb').checked ? '1' : '0',
                        t: Date.now()
                    }});
                    
                    const img = document.getElementById('preview');
                    const newImg = new Image();
                    newImg.onload = () => {{
                        img.src = newImg.src;
                        img.classList.remove('loading');
                        document.getElementById('status').textContent = 'Ready';
                        document.getElementById('status').className = 'status';
                    }};
                    newImg.onerror = () => {{
                        document.getElementById('status').textContent = 'Error';
                        img.classList.remove('loading');
                    }};
                    newImg.src = `/enhance/preview/${{sessionId}}?${{params}}`;
                }}
                
                function applyPreset(name) {{
                    const presets = {{
                        subtle: {{ sharpen: 0.5, denoise: 3, contrast: 1.0, brightness: 5, auto_wb: false }},
                        vivid: {{ sharpen: 1.0, denoise: 0, contrast: 2.5, brightness: 10, auto_wb: true }},
                        sharp: {{ sharpen: 2.0, denoise: 5, contrast: 1.5, brightness: 0, auto_wb: false }},
                        clean: {{ sharpen: 0.3, denoise: 12, contrast: 1.2, brightness: 0, auto_wb: true }}
                    }};
                    const p = presets[name];
                    if (!p) return;
                    
                    document.getElementById('sharpen').value = p.sharpen;
                    document.getElementById('sharpen-val').textContent = p.sharpen;
                    document.getElementById('denoise').value = p.denoise;
                    document.getElementById('denoise-val').textContent = p.denoise;
                    document.getElementById('contrast').value = p.contrast;
                    document.getElementById('contrast-val').textContent = p.contrast;
                    document.getElementById('brightness').value = p.brightness;
                    document.getElementById('brightness-val').textContent = p.brightness;
                    document.getElementById('auto_wb').checked = p.auto_wb;
                    
                    debounceUpdate();
                }}
                
                function resetAll() {{
                    document.getElementById('sharpen').value = 0;
                    document.getElementById('sharpen-val').textContent = '0';
                    document.getElementById('denoise').value = 0;
                    document.getElementById('denoise-val').textContent = '0';
                    document.getElementById('contrast').value = 0;
                    document.getElementById('contrast-val').textContent = '0';
                    document.getElementById('brightness').value = 0;
                    document.getElementById('brightness-val').textContent = '0';
                    document.getElementById('auto_wb').checked = false;
                    debounceUpdate();
                }}
                
                async function saveEnhanced() {{
                    const params = new URLSearchParams({{
                        sharpen: document.getElementById('sharpen').value,
                        denoise: document.getElementById('denoise').value,
                        contrast: document.getElementById('contrast').value,
                        brightness: document.getElementById('brightness').value,
                        auto_wb: document.getElementById('auto_wb').checked ? '1' : '0'
                    }});
                    
                    document.getElementById('status').textContent = 'Saving...';
                    document.getElementById('status').className = 'status loading';
                    
                    try {{
                        const response = await fetch(`/enhance/save/${{sessionId}}?${{params}}`, {{ method: 'POST' }});
                        const result = await response.json();
                        if (result.success) {{
                            alert('Image saved successfully to: ' + result.path);
                            document.getElementById('status').textContent = 'Saved!';
                            document.getElementById('status').className = 'status';
                        }} else {{
                            alert('Error: ' + result.error);
                            document.getElementById('status').textContent = 'Error';
                        }}
                    }} catch (e) {{
                        alert('Error saving: ' + e.message);
                        document.getElementById('status').textContent = 'Error';
                    }}
                }}
            </script>
        </body>
    </html>
    """


@app.get("/enhance/preview/{session_id}")
async def enhance_preview(
    session_id: str,
    sharpen: float = Query(0),
    denoise: int = Query(0),
    contrast: float = Query(0),
    brightness: int = Query(0),
    auto_wb: str = Query("0"),
    t: str = Query("")
):
    """Generate enhanced preview image."""
    if session_id not in _stitched_cache:
        return HTMLResponse("Session expired", status_code=404)
    
    image_path = _stitched_cache[session_id]['image_path']
    
    img = cv2.imread(image_path)
    if img is None:
        return HTMLResponse("Image not found", status_code=404)
    
    max_preview = 1200
    h, w = img.shape[:2]
    if max(h, w) > max_preview:
        scale = max_preview / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    if sharpen > 0 or denoise > 0 or contrast > 0 or brightness != 0 or auto_wb == "1":
        img = enhance_image(img, sharpen=sharpen, denoise=denoise, 
                           contrast=contrast, brightness=brightness, 
                           auto_wb=(auto_wb == "1"))
    
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    
    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache"}
    )


@app.post("/enhance/save/{session_id}")
async def enhance_save(
    session_id: str,
    sharpen: float = Query(0),
    denoise: int = Query(0),
    contrast: float = Query(0),
    brightness: int = Query(0),
    auto_wb: str = Query("0")
):
    """Save the enhanced image."""
    if session_id not in _stitched_cache:
        return {"success": False, "error": "Session expired"}
    
    cache = _stitched_cache[session_id]
    image_path = cache['image_path']
    
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"success": False, "error": "Image not found"}
        
        if sharpen > 0 or denoise > 0 or contrast > 0 or brightness != 0 or auto_wb == "1":
            img = enhance_image(img, sharpen=sharpen, denoise=denoise,
                               contrast=contrast, brightness=brightness,
                               auto_wb=(auto_wb == "1"))
        
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_enhanced{ext}"
        
        if ext.lower() == '.png':
            cv2.imwrite(output_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        else:
            cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 98])
        
        return {"success": True, "path": output_path}
    
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/enhance/open", response_class=HTMLResponse)
async def enhance_open_form():
    """Form to open an existing image for enhancement."""
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Open Image for Enhancement</title>
            <style>
                body { font-family: system-ui, -apple-system, sans-serif; max-width: 600px; margin: 2rem auto; padding: 0 1rem; background: #f5f5f5; }
                .card { background: white; border-radius: 12px; padding: 2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
                h1 { color: #333; margin-top: 0; }
                label { display: block; margin: 1rem 0 0.5rem; font-weight: 500; }
                input[type="text"] { width: 100%; padding: 12px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px; box-sizing: border-box; }
                button { background: #ff9800; color: white; padding: 14px 28px; border: none; border-radius: 6px; font-size: 16px; cursor: pointer; margin-top: 1.5rem; }
                button:hover { background: #f57c00; }
                a { color: #2196F3; }
            </style>
        </head>
        <body>
            <div class="card">
                <h1>✨ Enhance Existing Image</h1>
                <p>Open any image file for enhancement with live preview.</p>
                
                <form method="post" action="/enhance/open">
                    <label>Image Path:</label>
                    <input type="text" name="image_path" placeholder="D:/path/to/image.png" required>
                    <button type="submit">Open in Editor</button>
                </form>
                
                <p style="margin-top: 2rem;"><a href="/">← Back to Home</a></p>
            </div>
        </body>
    </html>
    """


@app.post("/enhance/open")
async def enhance_open_image(image_path: str = Form(...)):
    """Open an existing image for enhancement."""
    import uuid
    
    abs_path = os.path.abspath(image_path)
    if not os.path.exists(abs_path):
        return HTMLResponse(f"<p>Image not found: {abs_path}</p><a href='/enhance/open'>Try again</a>", status_code=404)
    
    session_id = str(uuid.uuid4())[:8]
    _stitched_cache[session_id] = {
        'image_path': abs_path,
        'label_path': None,
        'meta': None
    }
    
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=f"/enhance/{session_id}", status_code=303)