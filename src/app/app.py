from fastapi import FastAPI, UploadFile, File, Form, Query, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
import io
import cv2
import numpy as np
from ..inference import analyze_image
from ..macro_stitch_pipeline import process_folder, enhance_image, group_images_by_brightness

app = FastAPI(title="Filter Paper Stitching Pipeline")

# Store for stitched images awaiting enhancement (in production, use proper storage)
_stitched_cache = {}

# Global processing state
_processing_state = {
    'logs': [],
    'done': False,
    'error': None,
    'session_id': None,
    'meta': None
}

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
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Filter Paper Stitching</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary: #2563eb;
                --primary-hover: #1d4ed8;
                --bg: #f8fafc;
                --card-bg: #ffffff;
                --text: #1e293b;
                --text-muted: #64748b;
                --border: #e2e8f0;
            }
            body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 2rem; line-height: 1.5; }
            .container { max-width: 800px; margin: 0 auto; }
            .card { background: var(--card-bg); border-radius: 16px; padding: 2.5rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); }
            h1 { font-size: 1.875rem; font-weight: 600; margin-bottom: 0.5rem; color: var(--text); }
            p { color: var(--text-muted); margin-bottom: 2rem; }
            
            .form-group { margin-bottom: 1.5rem; }
            label { display: block; font-weight: 500; margin-bottom: 0.5rem; color: var(--text); }
            input[type="text"] { 
                width: 100%; 
                padding: 0.75rem 1rem; 
                border: 1px solid var(--border); 
                border-radius: 8px; 
                font-size: 1rem; 
                transition: border-color 0.15s ease;
                box-sizing: border-box;
            }
            input[type="text"]:focus { outline: none; border-color: var(--primary); box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1); }
            
            .btn { 
                display: inline-flex; 
                align-items: center; 
                justify-content: center; 
                background: var(--primary); 
                color: white; 
                padding: 0.75rem 1.5rem; 
                border-radius: 8px; 
                font-weight: 500; 
                text-decoration: none; 
                border: none; 
                cursor: pointer; 
                transition: background-color 0.15s ease; 
                font-size: 1rem;
            }
            .btn:hover { background: var(--primary-hover); }
            .btn-icon { margin-right: 0.5rem; }
            
            .back-link { display: inline-block; margin-top: 2rem; color: var(--text-muted); text-decoration: none; font-size: 0.875rem; }
            .back-link:hover { color: var(--primary); }

            .info-box {
                background: #eff6ff;
                border-left: 4px solid var(--primary);
                padding: 1rem;
                border-radius: 4px;
                margin-bottom: 2rem;
                font-size: 0.875rem;
                color: #1e40af;
            }
            
            .checkbox-container {
                display: flex;
                align-items: center;
                cursor: pointer;
                user-select: none;
                margin-top: 1rem;
            }
            .checkbox-container input {
                position: absolute;
                opacity: 0;
                cursor: pointer;
                height: 0;
                width: 0;
            }
            .checkmark {
                height: 20px;
                width: 20px;
                background-color: #fff;
                border: 1px solid var(--border);
                border-radius: 4px;
                margin-right: 10px;
                position: relative;
                transition: all 0.2s;
            }
            .checkbox-container:hover .checkmark {
                border-color: var(--primary);
            }
            .checkbox-container input:checked ~ .checkmark {
                background-color: var(--primary);
                border-color: var(--primary);
            }
            .checkmark:after {
                content: "";
                position: absolute;
                display: none;
            }
            .checkbox-container input:checked ~ .checkmark:after {
                display: block;
            }
            .checkbox-container .checkmark:after {
                left: 7px;
                top: 3px;
                width: 4px;
                height: 8px;
                border: solid white;
                border-width: 0 2px 2px 0;
                transform: rotate(45deg);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card">
                <h1>📷 New Stitching Job</h1>
                <p>Create a high-resolution mosaic from partial filter paper images.</p>
                
                <div class="info-box">
                    <strong>Tip:</strong> Ensure your input folder contains overlapping images. The system will help you select the best ones based on brightness.
                </div>
                
                <form method="post" action="/stitch/select-images">
                    <div class="form-group">
                        <label for="input_folder">Input Folder Path</label>
                        <input type="text" id="input_folder" name="input_folder" placeholder="e.g., dev-test/raw/Sample1/Micro" required>
                    </div>
                    
                    <div class="form-group">
                        <label class="checkbox-container">
                            <input type="checkbox" name="auto_select" value="true">
                            <span class="checkmark"></span>
                            Auto-select best images (matching brightness)
                        </label>
                    </div>
                    
                    <button type="submit" class="btn">
                        <span class="btn-icon">📊</span> Analyze & Select Images
                    </button>
                </form>
                
                <a href="/" class="back-link">← Back to Dashboard</a>
            </div>
        </div>
    </body>
    </html>
    """


@app.post("/stitch/select-images", response_class=HTMLResponse)
async def select_images_form(input_folder: str = Form(...), auto_select: bool = Form(False)):
    """Analyze images in folder by brightness and show selection interface."""
    import traceback
    
    try:
        # Verify folder exists
        if not os.path.isdir(input_folder):
            return HTMLResponse(
                f"""
                <!DOCTYPE html>
                <html lang="en">
                    <head>
                        <meta charset="UTF-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                        <title>Error</title>
                        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
                        <style>
                            body {{ font-family: 'Inter', sans-serif; background: #f8fafc; color: #1e293b; margin: 0; padding: 2rem; }}
                            .container {{ max-width: 600px; margin: 0 auto; }}
                            .card {{ background: white; border-radius: 16px; padding: 2.5rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); text-align: center; }}
                            h2 {{ color: #ef4444; margin-top: 0; }}
                            .btn {{ display: inline-block; background: #2563eb; color: white; padding: 0.75rem 1.5rem; border-radius: 8px; text-decoration: none; margin-top: 1rem; }}
                        </style>
                    </head>
                    <body>
                        <div class="container">
                            <div class="card">
                                <h2>❌ Folder Not Found</h2>
                                <p>The folder "<strong>{input_folder}</strong>" does not exist.</p>
                                <a href="/stitch" class="btn">Try Again</a>
                            </div>
                        </div>
                    </body>
                </html>
                """,
                status_code=404
            )
        
        # Group images by brightness
        brightness_groups = group_images_by_brightness(input_folder)
        
        if not brightness_groups:
            return HTMLResponse(
                f"""
                <!DOCTYPE html>
                <html lang="en">
                    <head>
                        <meta charset="UTF-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                        <title>Error</title>
                        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
                        <style>
                            body {{ font-family: 'Inter', sans-serif; background: #f8fafc; color: #1e293b; margin: 0; padding: 2rem; }}
                            .container {{ max-width: 600px; margin: 0 auto; }}
                            .card {{ background: white; border-radius: 16px; padding: 2.5rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); text-align: center; }}
                            h2 {{ color: #ef4444; margin-top: 0; }}
                            .btn {{ display: inline-block; background: #2563eb; color: white; padding: 0.75rem 1.5rem; border-radius: 8px; text-decoration: none; margin-top: 1rem; }}
                        </style>
                    </head>
                    <body>
                        <div class="container">
                            <div class="card">
                                <h2>❌ No Images Found</h2>
                                <p>No image files (PNG, JPG, JPEG) found in "<strong>{input_folder}</strong>".</p>
                                <a href="/stitch" class="btn">Try Again</a>
                            </div>
                        </div>
                    </body>
                </html>
                """
            )
        
        # Auto-selection logic
        pre_selected_paths = []
        if auto_select:
            # Find the group with the most images
            best_group_key = max(brightness_groups, key=lambda k: len(brightness_groups[k]))
            best_images = brightness_groups[best_group_key]
            
            # Sort by filename to try and get sequence (assuming sequence implies overlap)
            best_images.sort(key=lambda x: x['filename'])
            
            # Pick up to 3 images
            # If we have many images, picking 3 evenly spaced might be better, but for now let's pick first 3
            # or if we have exactly 3, pick them.
            # Let's pick the first 3 for now as a simple heuristic for "best" in a sequence
            count = min(len(best_images), 3)
            for i in range(count):
                pre_selected_paths.append(best_images[i]['path'])

        # Build HTML for brightness groups
        brightness_html = ""
        for brightness_level, images in sorted(brightness_groups.items()):
            brightness_html += f"""
            <div class="brightness-group">
                <div class="group-header">
                    <h3>Brightness Level: {brightness_level}</h3>
                    <span class="badge">{len(images)} images</span>
                </div>
                <div class="image-grid">
            """
            
            for img_info in images:
                brightness_val = round(img_info['brightness'], 1)
                is_checked = "checked" if img_info['path'] in pre_selected_paths else ""
                
                # Ensure path is web-accessible. 
                # If path is absolute, we need to make it relative to the app or serve it via /images/ endpoint
                # The /images/ endpoint takes the full path, so we can use that.
                # We need to handle backslashes for URL
                img_url = f"/images/{img_info['path'].replace(os.sep, '/')}"
                
                brightness_html += f"""
                    <label class="image-card">
                        <input type="checkbox" name="img_selection" value="{img_info['path']}" onchange="updateLeftMiddleRight()" {is_checked}>
                        <div class="image-content">
                            <div class="image-preview-container">
                                <img src="{img_url}" class="image-preview" loading="lazy" alt="{img_info['filename']}">
                                <button type="button" class="preview-btn" onclick="event.preventDefault(); window.open('{img_url}', '_blank');" title="View Full Resolution">🔍</button>
                            </div>
                            <div class="image-name">{img_info['filename']}</div>
                            <div class="image-meta">Brightness: {brightness_val}</div>
                        </div>
                    </label>
                """
            
            brightness_html += """
                </div>
            </div>
            """
        
        return f"""
        <!DOCTYPE html>
        <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Select Images</title>
                <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
                <style>
                    :root {{
                        --primary: #2563eb;
                        --primary-hover: #1d4ed8;
                        --bg: #f8fafc;
                        --card-bg: #ffffff;
                        --text: #1e293b;
                        --text-muted: #64748b;
                        --border: #e2e8f0;
                        --success: #10b981;
                        --warning: #f59e0b;
                    }}
                    body {{ font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 2rem; line-height: 1.5; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    .header {{ margin-bottom: 2rem; }}
                    h1 {{ font-size: 1.875rem; font-weight: 600; margin: 0 0 0.5rem 0; }}
                    p {{ color: var(--text-muted); margin: 0; }}
                    
                    .card {{ background: var(--card-bg); border-radius: 16px; padding: 2rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); margin-bottom: 2rem; }}
                    
                    .brightness-group {{ margin-bottom: 2rem; }}
                    .group-header {{ display: flex; align-items: center; margin-bottom: 1rem; }}
                    .group-header h3 {{ margin: 0; font-size: 1.1rem; font-weight: 600; }}
                    .badge {{ background: #e0e7ff; color: var(--primary); padding: 0.25rem 0.75rem; border-radius: 999px; font-size: 0.75rem; font-weight: 600; margin-left: 1rem; }}
                    
                    .image-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 1rem; }}
                    
                    .image-card {{ cursor: pointer; position: relative; }}
                    .image-card input {{ position: absolute; opacity: 0; }}
                    .image-content {{ 
                        border: 2px solid var(--border); 
                        border-radius: 12px; 
                        padding: 0.75rem; 
                        text-align: center; 
                        transition: all 0.2s ease;
                        background: white;
                        height: 100%;
                        display: flex;
                        flex-direction: column;
                    }}
                    .image-card:hover .image-content {{ border-color: var(--primary); transform: translateY(-2px); }}
                    .image-card input:checked + .image-content {{ 
                        border-color: var(--primary); 
                        background: #eff6ff; 
                        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2);
                    }}
                    
                    .image-preview-container {{
                        width: 100%;
                        height: 120px;
                        background: #f1f5f9;
                        border-radius: 8px;
                        margin-bottom: 0.75rem;
                        overflow: hidden;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        position: relative;
                    }}
                    .image-preview {{
                        width: 100%;
                        height: 100%;
                        object-fit: cover;
                    }}
                    .preview-btn {{
                        position: absolute;
                        top: 5px;
                        right: 5px;
                        background: rgba(255, 255, 255, 0.9);
                        border: 1px solid #e2e8f0;
                        border-radius: 50%;
                        width: 30px;
                        height: 30px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        cursor: pointer;
                        font-size: 14px;
                        transition: all 0.2s;
                        z-index: 10;
                        color: #1e293b;
                    }}
                    .preview-btn:hover {{
                        background: #fff;
                        transform: scale(1.1);
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        color: var(--primary);
                    }}
                    
                    .image-name {{ font-weight: 500; font-size: 0.85rem; margin-bottom: 0.25rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
                    .image-meta {{ font-size: 0.75rem; color: var(--text-muted); }}
                    
                    .selection-panel {{ 
                        position: sticky; 
                        bottom: 2rem; 
                        background: white; 
                        border-radius: 16px; 
                        padding: 1.5rem; 
                        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
                        border: 1px solid var(--border);
                        z-index: 100;
                    }}
                    
                    .selection-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; margin-bottom: 1.5rem; }}
                    .selection-slot {{ background: #f8fafc; border: 2px dashed var(--border); border-radius: 8px; padding: 1rem; text-align: center; }}
                    .selection-slot.filled {{ border-style: solid; border-color: var(--success); background: #ecfdf5; }}
                    .slot-label {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); margin-bottom: 0.5rem; font-weight: 600; }}
                    .slot-value {{ font-weight: 500; font-size: 0.9rem; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
                    
                    .options-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.5rem; margin-bottom: 1.5rem; padding-top: 1.5rem; border-top: 1px solid var(--border); }}
                    .form-group label {{ display: block; font-size: 0.875rem; font-weight: 500; margin-bottom: 0.5rem; }}
                    .form-group input {{ width: 100%; padding: 0.5rem; border: 1px solid var(--border); border-radius: 6px; box-sizing: border-box; }}
                    
                    .actions {{ display: flex; align-items: center; justify-content: space-between; }}
                    .btn {{ 
                        background: var(--primary); 
                        color: white; 
                        padding: 0.75rem 1.5rem; 
                        border-radius: 8px; 
                        font-weight: 500; 
                        border: none; 
                        cursor: pointer; 
                        font-size: 1rem;
                        transition: all 0.2s;
                    }}
                    .btn:disabled {{ background: var(--text-muted); cursor: not-allowed; opacity: 0.7; }}
                    .btn:hover:not(:disabled) {{ background: var(--primary-hover); }}
                    .back-link {{ color: var(--text-muted); text-decoration: none; }}
                    .back-link:hover {{ color: var(--text); }}
                    
                    /* Loading Overlay */
                    .loading-overlay {{
                        position: fixed;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 100%;
                        background: rgba(255, 255, 255, 0.9);
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: center;
                        z-index: 1000;
                        opacity: 0;
                        pointer-events: none;
                        transition: opacity 0.3s ease;
                    }}
                    .loading-overlay.active {{
                        opacity: 1;
                        pointer-events: all;
                    }}
                    .spinner {{
                        width: 50px;
                        height: 50px;
                        border: 4px solid #e2e8f0;
                        border-top-color: var(--primary);
                        border-radius: 50%;
                        animation: spin 1s linear infinite;
                        margin-bottom: 1rem;
                    }}
                    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
                    .loading-text {{
                        font-weight: 500;
                        color: var(--text);
                        font-size: 1.1rem;
                    }}
                    .loading-subtext {{
                        color: var(--text-muted);
                        font-size: 0.9rem;
                        margin-top: 0.5rem;
                    }}
                </style>
                <script>
                    function updateLeftMiddleRight() {{
                        const leftInput = document.getElementById('left_image');
                        const middleInput = document.getElementById('middle_image');
                        const rightInput = document.getElementById('right_image');
                        
                        const leftDisplay = document.getElementById('left_display');
                        const middleDisplay = document.getElementById('middle_display');
                        const rightDisplay = document.getElementById('right_display');
                        
                        const leftSlot = document.getElementById('left_slot');
                        const middleSlot = document.getElementById('middle_slot');
                        const rightSlot = document.getElementById('right_slot');
                        
                        // Reset
                        leftInput.value = ""; middleInput.value = ""; rightInput.value = "";
                        leftDisplay.textContent = "Select an image"; middleDisplay.textContent = "Select an image"; rightDisplay.textContent = "Select an image";
                        leftSlot.classList.remove('filled'); middleSlot.classList.remove('filled'); rightSlot.classList.remove('filled');
                        
                        // Get all checked inputs
                        const allChecked = Array.from(document.querySelectorAll('input[name="img_selection"]:checked'));
                        
                        if (allChecked.length > 0) {{
                            const imgPath = allChecked[0].value;
                            const filename = imgPath.split(/[\\\\/]/).pop();
                            leftInput.value = imgPath;
                            leftDisplay.textContent = filename;
                            leftSlot.classList.add('filled');
                        }}
                        
                        if (allChecked.length > 1) {{
                            const imgPath = allChecked[1].value;
                            const filename = imgPath.split(/[\\\\/]/).pop();
                            middleInput.value = imgPath;
                            middleDisplay.textContent = filename;
                            middleSlot.classList.add('filled');
                        }}
                        
                        if (allChecked.length > 2) {{
                            const imgPath = allChecked[2].value;
                            const filename = imgPath.split(/[\\\\/]/).pop();
                            rightInput.value = imgPath;
                            rightDisplay.textContent = filename;
                            rightSlot.classList.add('filled');
                        }}
                        
                        const canSubmit = leftInput.value && middleInput.value && rightInput.value;
                        document.getElementById('submit_btn').disabled = !canSubmit;
                        
                        if (canSubmit) {{
                            document.getElementById('submit_btn').innerHTML = "🚀 Run Stitching Pipeline";
                        }} else {{
                            document.getElementById('submit_btn').innerHTML = `Select ${{3 - allChecked.length}} more image(s)`;
                        }}
                    }}
                    
                    async function submitStitching(event) {{
                        event.preventDefault();
                        const form = document.getElementById('imageSelectionForm');
                        const overlay = document.getElementById('loadingOverlay');
                        const statusText = document.getElementById('loadingStatus');
                        
                        overlay.classList.add('active');
                        statusText.textContent = "Processing... This may take a while.";
                        
                        try {{
                            const formData = new FormData(form);
                            const response = await fetch('/stitch/run', {{
                                method: 'POST',
                                body: formData
                            }});
                            
                            if (response.ok) {{
                                // Fetch follows redirects automatically. 
                                // The response.url will be the final destination (result page).
                                window.location.href = response.url;
                            }} else {{
                                const text = await response.text();
                                throw new Error('Stitching failed: ' + text);
                            }}
                            
                        }} catch (e) {{
                            alert('Error: ' + e.message);
                            overlay.classList.remove('active');
                        }}
                    }}
                    
                    // Run on load to handle auto-selection
                    window.onload = function() {{
                        updateLeftMiddleRight();
                        document.getElementById('imageSelectionForm').addEventListener('submit', submitStitching);
                    }};
                </script>
            </head>
            <body>
                <div id="loadingOverlay" class="loading-overlay">
                    <div class="spinner"></div>
                    <div class="loading-text">Stitching Images...</div>
                    <div id="loadingStatus" class="loading-subtext">Initializing pipeline...</div>
                </div>

                <div class="container">
                    <div class="header">
                        <h1>Select Images</h1>
                        <p>Choose 3 images with similar brightness for the best result.</p>
                    </div>
                    
                    <form id="imageSelectionForm" method="post" action="/stitch/run">
                        <div class="card">
                            {brightness_html}
                        </div>
                        
                        <div class="selection-panel">
                            <div class="selection-grid">
                                <div class="selection-slot" id="left_slot">
                                    <div class="slot-label">Left Image</div>
                                    <div class="slot-value" id="left_display">Select an image</div>
                                    <input type="hidden" id="left_image" name="left_image">
                                </div>
                                <div class="selection-slot" id="middle_slot">
                                    <div class="slot-label">Middle Image</div>
                                    <div class="slot-value" id="middle_display">Select an image</div>
                                    <input type="hidden" id="middle_image" name="middle_image">
                                </div>
                                <div class="selection-slot" id="right_slot">
                                    <div class="slot-label">Right Image</div>
                                    <div class="slot-value" id="right_display">Select an image</div>
                                    <input type="hidden" id="right_image" name="right_image">
                                </div>
                            </div>
                            
                            <div class="options-grid">
                                <div class="form-group">
                                    <label>Output Path</label>
                                    <input type="text" name="output_image" placeholder="dev-test/stitched/output.png" required>
                                </div>
                                <div class="form-group">
                                    <label>Max Dimension</label>
                                    <input type="number" name="max_dim" value="8192" step="256">
                                </div>
                                <div class="form-group">
                                    <label>Upscale</label>
                                    <input type="number" name="upscale" value="1.0" step="0.5" max="4.0">
                                </div>
                            </div>
                            
                            <input type="hidden" name="input_folder" value="{input_folder}">
                            
                            <div class="actions">
                                <a href="/stitch" class="back-link">Cancel</a>
                                <button type="submit" id="submit_btn" class="btn" disabled>Select 3 images</button>
                            </div>
                        </div>
                    </form>
                </div>
            </body>
        </html>
        """
    
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return HTMLResponse(f"Error: {tb}", status_code=500)


@app.post("/stitch/run")
def stitch_run(
    input_folder: str = Form(...),
    output_image: str = Form(...),
    left_image: str = Form(None),
    middle_image: str = Form(None),
    right_image: str = Form(None),
    max_dim: int = Form(8192),
    max_stitch_dim: int = Form(8192),
    upscale: float = Form(1.0),
    upscale_method: str = Form("lanczos"),
):
    import traceback
    import uuid
    import sys
    
    try:
        # If specific images are selected, use only those
        if left_image and middle_image and right_image:
            # Load only the selected images in left-middle-right order
            selected_images = [left_image, middle_image, right_image]
            images_to_stitch = []
            for img_path in selected_images:
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        images_to_stitch.append(img)
            
            if len(images_to_stitch) != 3:
                return HTMLResponse(f"Error: Could not load all 3 selected images. Loaded {len(images_to_stitch)}", status_code=400)
            
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_image) or ".", exist_ok=True)
            
            # Stitch the selected images
            from ..macro_stitch_pipeline import stitch_images, prepare_for_yolo
            pano = stitch_images(images_to_stitch, max_stitch_dim=max_stitch_dim)
            
            # Upscale if requested
            if upscale > 1.0:
                from ..macro_stitch_pipeline import upscale_image
                pano = upscale_image(pano, scale=upscale, method=upscale_method)
            
            # Save and prepare for YOLO
            meta = prepare_for_yolo(pano, output_image, max_dim=max_dim)
        else:
            # Fall back to stitching all images in the folder
            from ..macro_stitch_pipeline import process_folder
            meta = process_folder(
                input_folder=input_folder,
                output_image_path=output_image,
                max_dim=int(max_dim),
                max_stitch_dim=int(max_stitch_dim),
                upscale=float(upscale),
                upscale_method=upscale_method,
            )
        
        # Cache the result
        session_id = str(uuid.uuid4())[:8]
        abs_image_path = os.path.abspath(meta['image_path'])
        _stitched_cache[session_id] = {
            'image_path': abs_image_path,
            'label_path': meta['label_path'],
            'meta': meta
        }
        
        return RedirectResponse(url=f"/stitch/result?session={session_id}", status_code=303)
        
    except Exception as e:
        tb = traceback.format_exc()
        return HTMLResponse(f"<h1>Error</h1><pre>{tb}</pre>", status_code=500)


@app.post("/stitch/delete/{session_id}")
async def delete_stitch(session_id: str):
    """Delete the stitched image and clear from cache."""
    if session_id in _stitched_cache:
        cache = _stitched_cache[session_id]
        image_path = cache['image_path']
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
        except Exception as e:
            print(f"Error deleting file: {e}")
        
        del _stitched_cache[session_id]
        
    return RedirectResponse(url="/stitch", status_code=303)


@app.get("/stitch/logs")
async def get_logs():
    """Get current processing logs and status"""
    return JSONResponse({
        'logs': _processing_state['logs'],
        'done': _processing_state['done'],
        'error': _processing_state['error'],
        'session_id': _processing_state['session_id']
    })


@app.get("/stitch/result")
async def stitch_result(session: str = Query(...)):
    """Show stitching result after completion"""
    if session not in _stitched_cache:
        return HTMLResponse("<p>Session expired. Please stitch again.</p>", status_code=404)
    
    cache = _stitched_cache[session]
    meta = cache['meta']
    abs_image_path = cache['image_path']
    
    return HTMLResponse(
        f"""
        <!DOCTYPE html>
        <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Stitching Complete</title>
                <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
                <style>
                    :root {{
                        --primary: #2563eb;
                        --primary-hover: #1d4ed8;
                        --bg: #f8fafc;
                        --card-bg: #ffffff;
                        --text: #1e293b;
                        --text-muted: #64748b;
                        --border: #e2e8f0;
                        --success: #10b981;
                    }}
                    body {{ font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 2rem; line-height: 1.5; }}
                    .container {{ max-width: 1000px; margin: 0 auto; }}
                    .card {{ background: var(--card-bg); border-radius: 16px; padding: 2.5rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }}
                    
                    .header {{ text-align: center; margin-bottom: 2rem; }}
                    h1 {{ font-size: 2rem; font-weight: 600; margin: 0 0 0.5rem 0; color: var(--success); }}
                    p {{ color: var(--text-muted); }}
                    
                    .meta-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }}
                    .meta-item {{ background: #f1f5f9; padding: 1.5rem; border-radius: 12px; text-align: center; }}
                    .meta-label {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); font-weight: 600; margin-bottom: 0.5rem; }}
                    .meta-value {{ font-size: 1.5rem; font-weight: 600; color: var(--text); }}
                    
                    .preview-container {{ 
                        background: #0f172a; 
                        border-radius: 12px; 
                        padding: 1rem; 
                        margin-bottom: 2rem; 
                        text-align: center;
                        overflow: hidden;
                    }}
                    .preview-img {{ max-width: 100%; max-height: 600px; border-radius: 8px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5); }}
                    
                    .path-box {{ 
                        background: #eff6ff; 
                        border: 1px solid #bfdbfe; 
                        color: #1e40af; 
                        padding: 1rem; 
                        border-radius: 8px; 
                        font-family: monospace; 
                        margin-bottom: 2rem;
                        word-break: break-all;
                        text-align: center;
                    }}
                    
                    .actions {{ display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap; }}
                    .btn {{ 
                        display: inline-flex; 
                        align-items: center; 
                        justify-content: center; 
                        padding: 0.75rem 1.5rem; 
                        border-radius: 8px; 
                        font-weight: 500; 
                        text-decoration: none; 
                        transition: all 0.2s;
                        font-size: 1rem;
                    }}
                    .btn-primary {{ background: var(--primary); color: white; }}
                    .btn-primary:hover {{ background: var(--primary-hover); }}
                    .btn-secondary {{ background: white; color: var(--text); border: 1px solid var(--border); }}
                    .btn-secondary:hover {{ background: #f1f5f9; border-color: #cbd5e1; }}
                    .btn-danger {{ background: #ef4444; color: white; border: none; }}
                    .btn-danger:hover {{ background: #dc2626; }}
                    
                    .btn-icon {{ margin-right: 0.5rem; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="card">
                        <div class="header">
                            <h1>✨ Stitching Complete!</h1>
                            <p>Your mosaic has been successfully generated.</p>
                        </div>
                        
                        <div class="meta-grid">
                            <div class="meta-item">
                                <div class="meta-label">Original Size</div>
                                <div class="meta-value">{meta['original_size'][0]} × {meta['original_size'][1]}</div>
                            </div>
                            <div class="meta-item">
                                <div class="meta-label">Output Size</div>
                                <div class="meta-value">{meta['output_size'][0]} × {meta['output_size'][1]}</div>
                            </div>
                            <div class="meta-item">
                                <div class="meta-label">Scale Factor</div>
                                <div class="meta-value">{meta['scale']:.2f}x</div>
                            </div>
                        </div>
                        
                        <div class="preview-container">
                            <img src="/images/{abs_image_path}" alt="Stitched Result" class="preview-img">
                        </div>
                        
                        <div class="path-box">
                            📂 {meta['image_path']}
                        </div>
                        
                        <div class="actions">
                            <a href="/enhance/{session}" class="btn btn-primary">
                                <span class="btn-icon">✨</span> Enhance Image
                            </a>
                            <a href="/stitch" class="btn btn-secondary">
                                <span class="btn-icon">🔄</span> Stitch Another
                            </a>
                            <a href="/" class="btn btn-secondary">
                                <span class="btn-icon">🏠</span> Home
                            </a>
                            <form action="/stitch/delete/{session}" method="post" onsubmit="return confirm('Are you sure you want to delete this image? This cannot be undone.');" style="display:inline;">
                                <button type="submit" class="btn btn-danger">
                                    <span class="btn-icon">🗑️</span> Delete
                                </button>
                            </form>
                        </div>
                    </div>
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
    <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Enhance Image</title>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
            <style>
                :root {{
                    --primary: #2563eb;
                    --primary-hover: #1d4ed8;
                    --bg: #f8fafc;
                    --card-bg: #ffffff;
                    --text: #1e293b;
                    --text-muted: #64748b;
                    --border: #e2e8f0;
                    --success: #10b981;
                }}
                * {{ box-sizing: border-box; }}
                body {{ font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); margin: 0; height: 100vh; display: flex; flex-direction: column; }}
                
                .header {{ 
                    background: var(--card-bg); 
                    border-bottom: 1px solid var(--border); 
                    padding: 1rem 2rem; 
                    display: flex; 
                    align-items: center; 
                    justify-content: space-between;
                    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
                    z-index: 10;
                }}
                .header h1 {{ font-size: 1.25rem; font-weight: 600; margin: 0; display: flex; align-items: center; gap: 0.5rem; }}
                .header-actions {{ display: flex; gap: 1rem; }}
                
                .main-container {{ display: flex; flex: 1; overflow: hidden; }}
                
                .controls-sidebar {{ 
                    width: 320px; 
                    background: var(--card-bg); 
                    border-right: 1px solid var(--border); 
                    padding: 1.5rem; 
                    overflow-y: auto; 
                    display: flex;
                    flex-direction: column;
                    gap: 2rem;
                }}
                
                .control-section h3 {{ 
                    font-size: 0.75rem; 
                    text-transform: uppercase; 
                    letter-spacing: 0.05em; 
                    color: var(--text-muted); 
                    margin: 0 0 1rem 0; 
                    font-weight: 600;
                }}
                
                .slider-group {{ margin-bottom: 1.5rem; }}
                .slider-header {{ display: flex; justify-content: space-between; margin-bottom: 0.5rem; font-size: 0.875rem; font-weight: 500; }}
                .slider-value {{ color: var(--primary); font-feature-settings: "tnum"; }}
                
                input[type="range"] {{ 
                    width: 100%; 
                    height: 6px; 
                    border-radius: 3px; 
                    background: #e2e8f0; 
                    outline: none; 
                    -webkit-appearance: none; 
                    cursor: pointer;
                }}
                input[type="range"]::-webkit-slider-thumb {{ 
                    -webkit-appearance: none; 
                    width: 18px; 
                    height: 18px; 
                    border-radius: 50%; 
                    background: var(--primary); 
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    transition: transform 0.1s;
                }}
                input[type="range"]::-webkit-slider-thumb:hover {{ transform: scale(1.1); }}
                
                .checkbox-group label {{ 
                    display: flex; 
                    align-items: center; 
                    gap: 0.75rem; 
                    cursor: pointer; 
                    font-size: 0.875rem; 
                    font-weight: 500;
                    user-select: none;
                }}
                .checkbox-group input {{ width: 16px; height: 16px; accent-color: var(--primary); }}
                
                .presets-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; }}
                .preset-btn {{ 
                    background: #f1f5f9; 
                    border: 1px solid transparent; 
                    padding: 0.5rem; 
                    border-radius: 6px; 
                    font-size: 0.875rem; 
                    color: var(--text); 
                    cursor: pointer; 
                    transition: all 0.2s;
                    font-weight: 500;
                }}
                .preset-btn:hover {{ background: #e2e8f0; }}
                
                .preview-area {{ 
                    flex: 1; 
                    background: #f1f5f9; 
                    display: flex; 
                    flex-direction: column; 
                    position: relative;
                    overflow: hidden;
                }}
                
                .image-wrapper {{ 
                    flex: 1; 
                    display: flex; 
                    align-items: center; 
                    justify-content: center; 
                    padding: 2rem; 
                    overflow: auto; 
                }}
                .image-wrapper img {{ 
                    max-width: 100%; 
                    max-height: 100%; 
                    object-fit: contain; 
                    border-radius: 8px; 
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                    transition: opacity 0.2s;
                }}
                .image-wrapper img.loading {{ opacity: 0.6; filter: blur(2px); }}
                
                .status-bar {{ 
                    position: absolute; 
                    top: 1rem; 
                    right: 1rem; 
                    background: rgba(255, 255, 255, 0.9); 
                    backdrop-filter: blur(4px);
                    padding: 0.5rem 1rem; 
                    border-radius: 999px; 
                    font-size: 0.875rem; 
                    font-weight: 500;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                    opacity: 0;
                    transition: opacity 0.2s;
                }}
                .status-bar.visible {{ opacity: 1; }}
                .status-dot {{ width: 8px; height: 8px; border-radius: 50%; background: var(--text-muted); }}
                .status-bar.loading .status-dot {{ background: #f59e0b; animation: pulse 1s infinite; }}
                .status-bar.saved .status-dot {{ background: var(--success); }}
                
                @keyframes pulse {{ 0% {{ opacity: 1; }} 50% {{ opacity: 0.5; }} 100% {{ opacity: 1; }} }}
                
                .btn {{ 
                    display: inline-flex; 
                    align-items: center; 
                    justify-content: center; 
                    padding: 0.6rem 1.2rem; 
                    border-radius: 6px; 
                    font-weight: 500; 
                    text-decoration: none; 
                    border: none; 
                    cursor: pointer; 
                    font-size: 0.875rem;
                    transition: all 0.2s;
                    gap: 0.5rem;
                }}
                .btn-primary {{ background: var(--primary); color: white; }}
                .btn-primary:hover {{ background: var(--primary-hover); }}
                .btn-secondary {{ background: white; color: var(--text); border: 1px solid var(--border); }}
                .btn-secondary:hover {{ background: #f8fafc; border-color: #cbd5e1; }}
                .btn-ghost {{ color: var(--text-muted); background: transparent; }}
                .btn-ghost:hover {{ color: var(--text); background: #f1f5f9; }}
                
                .reset-link {{ 
                    display: block; 
                    text-align: center; 
                    margin-top: 1rem; 
                    color: var(--text-muted); 
                    font-size: 0.875rem; 
                    text-decoration: underline; 
                    cursor: pointer; 
                }}
                .reset-link:hover {{ color: var(--text); }}

            </style>
        </head>
        <body>
            <div class="header">
                <h1>✨ Enhance Image</h1>
                <div class="header-actions">
                    <a href="/stitch" class="btn btn-secondary">← Back</a>
                    <button class="btn btn-primary" onclick="saveEnhanced()">
                        <span>💾</span> Save Changes
                    </button>
                </div>
            </div>
            
            <div class="main-container">
                <div class="controls-sidebar">
                    <div class="control-section">
                        <h3>Adjustments</h3>
                        
                        <div class="slider-group">
                            <div class="slider-header">
                                <label>Sharpening</label>
                                <span class="slider-value" id="sharpen-val">0</span>
                            </div>
                            <input type="range" id="sharpen" min="0" max="3" step="0.1" value="0">
                        </div>
                        
                        <div class="slider-group">
                            <div class="slider-header">
                                <label>Denoising</label>
                                <span class="slider-value" id="denoise-val">0</span>
                            </div>
                            <input type="range" id="denoise" min="0" max="20" step="1" value="0">
                        </div>
                        
                        <div class="slider-group">
                            <div class="slider-header">
                                <label>Contrast</label>
                                <span class="slider-value" id="contrast-val">0</span>
                            </div>
                            <input type="range" id="contrast" min="0" max="4" step="0.1" value="0">
                        </div>
                        
                        <div class="slider-group">
                            <div class="slider-header">
                                <label>Brightness</label>
                                <span class="slider-value" id="brightness-val">0</span>
                            </div>
                            <input type="range" id="brightness" min="-50" max="50" step="1" value="0">
                        </div>
                        
                        <div class="checkbox-group">
                            <label>
                                <input type="checkbox" id="auto_wb">
                                Auto White Balance
                            </label>
                        </div>
                        
                        <div class="reset-link" onclick="resetAll()">Reset to Original</div>
                    </div>
                    
                    <div class="control-section">
                        <h3>Quick Presets</h3>
                        <div class="presets-grid">
                            <button class="preset-btn" onclick="applyPreset('subtle')">Subtle</button>
                            <button class="preset-btn" onclick="applyPreset('vivid')">Vivid</button>
                            <button class="preset-btn" onclick="applyPreset('sharp')">Sharp</button>
                            <button class="preset-btn" onclick="applyPreset('clean')">Clean</button>
                        </div>
                    </div>
                </div>
                
                <div class="preview-area">
                    <div class="status-bar" id="status-bar">
                        <div class="status-dot"></div>
                        <span id="status-text">Ready</span>
                    </div>
                    
                    <div class="image-wrapper">
                        <img id="preview" src="/enhance/preview/{session_id}?sharpen=0&denoise=0&contrast=0&brightness=0&auto_wb=0&t=0" alt="Preview">
                    </div>
                </div>
            </div>
            
            <script>
                const sessionId = "{session_id}";
                let debounceTimer = null;
                const statusBar = document.getElementById('status-bar');
                const statusText = document.getElementById('status-text');
                
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
                
                function showStatus(msg, type) {{
                    statusText.textContent = msg;
                    statusBar.className = 'status-bar visible ' + type;
                }}
                
                function debounceUpdate() {{
                    clearTimeout(debounceTimer);
                    showStatus('Updating preview...', 'loading');
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
                        showStatus('Preview updated', '');
                        setTimeout(() => statusBar.classList.remove('visible'), 2000);
                    }};
                    newImg.onerror = () => {{
                        showStatus('Error loading preview', 'error');
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
                    
                    showStatus('Saving image...', 'loading');
                    
                    try {{
                        const response = await fetch(`/enhance/save/${{sessionId}}?${{params}}`, {{ method: 'POST' }});
                        const result = await response.json();
                        if (result.success) {{
                            console.log('Image saved to: ' + result.path);
                            showStatus('Saved successfully!', 'saved');
                            setTimeout(() => statusBar.classList.remove('visible'), 3000);
                        }} else {{
                            console.error('Error: ' + result.error);
                            showStatus('Save failed: ' + result.error, 'error');
                        }}
                    }} catch (e) {{
                        console.error('Error saving: ' + e.message);
                        showStatus('Save failed: ' + e.message, 'error');
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
        output_path = f"{base}_enhanced{ext}";
        
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