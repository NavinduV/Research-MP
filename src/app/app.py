from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
import shutil
from ..inference import analyze_image
from ..macro_stitch_pipeline import process_folder

app = FastAPI()

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = analyze_image(file_path)
    return result


@app.get("/stitch", response_class=HTMLResponse)
async def stitch_form():
        return """
        <html>
            <head><title>Macro Stitching Pipeline</title></head>
            <body style="font-family: system-ui; max-width: 800px; margin: 2rem auto;">
                <h1>Macro Image Stitching (OpenCV)</h1>
                <p>Run classical illumination correction + CLAHE + stitching, and produce YOLO-ready output.</p>
                <form method="post" action="/stitch/run">
                    <label>Input folder (tiles):<br/><input type="text" name="input_folder" style="width:100%" placeholder="data/raw/sample_tiles" required></label><br/><br/>
                    <label>Output image path:<br/><input type="text" name="output_image" style="width:100%" placeholder="data/stitched/stitched_filter.png" required></label><br/><br/>
                    <label>Illumination blur (odd int):<br/><input type="number" name="illum_blur" value="101" min="3" step="2"></label><br/><br/>
                    <label>CLAHE clip limit:<br/><input type="number" name="clahe_clip" value="2.0" step="0.1"></label><br/><br/>
                        <label>Max output dimension:<br/><input type="number" name="max_dim" value="2048" step="1"></label><br/><br/>
                        <label>Max stitch tile dimension (controls memory, lower = safer):<br/>
                            <input type="number" name="max_stitch_dim" value="2048" step="1"></label><br/><br/>
                    <button type="submit">Run Pipeline</button>
                </form>
            </body>
        </html>
        """


@app.post("/stitch/run")
async def stitch_run(
        input_folder: str = Form(...),
        output_image: str = Form(...),
        illum_blur: int = Form(101),
        clahe_clip: float = Form(2.0),
    max_dim: int = Form(2048),
    max_stitch_dim: int = Form(2048),
):
        import traceback
        try:
            meta = process_folder(
                input_folder=input_folder,
                output_image_path=output_image,
                illum_blur=int(illum_blur),
                clahe_clip=float(clahe_clip),
                max_dim=int(max_dim),
                max_stitch_dim=int(max_stitch_dim) if max_stitch_dim is not None else 2048,
            )
        except Exception as e:
                tb = traceback.format_exc()
                return HTMLResponse(
                        f"""
                        <html>
                            <head><title>Stitching Error</title></head>
                            <body style='font-family: system-ui; max-width: 900px; margin: 2rem auto;'>
                                <h2>Error while running pipeline</h2>
                                <pre style='white-space:pre-wrap;background:#f8f8f8;padding:1rem;border-radius:6px;'>{tb}</pre>
                                <p><a href="/stitch">Back</a></p>
                            </body>
                        </html>
                        """
                )

        # Return a simple HTML summary with links/paths
        return HTMLResponse(
                f"""
                <html>
                    <head><title>Stitching Result</title></head>
                    <body style='font-family: system-ui; max-width: 800px; margin: 2rem auto;'>
                        <h2>Pipeline completed</h2>
                        <p><b>Image:</b> {meta['image_path']}</p>
                        <p><b>Label file:</b> {meta['label_path']}</p>
                        <p><b>Original size (w,h):</b> {meta['original_size']}</p>
                        <p><b>Output size (w,h):</b> {meta['output_size']}</p>
                        <p><b>Scale:</b> {meta['scale']}</p>
                        <p><a href="/stitch">Run another</a></p>
                    </body>
                </html>
                """
        )
