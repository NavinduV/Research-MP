# Start backend
Write-Host "Starting FastAPI backend on http://localhost:8000 ..." -ForegroundColor Cyan
Set-Location $PSScriptRoot
if (Test-Path "venv/Scripts/activate") {
    . venv/Scripts/activate
}
$env:PYTHONPATH = $PSScriptRoot
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot'; if (Test-Path 'venv/Scripts/activate') { . venv/Scripts/activate }; `$env:PYTHONPATH='$PSScriptRoot'; uvicorn src.api.pipeline_api:app --reload --host 0.0.0.0 --port 8000"

# Start frontend
Write-Host "Starting React frontend on http://localhost:5173 ..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot/frontend'; npm run dev"

Write-Host ""
Write-Host "Dashboard: http://localhost:5173" -ForegroundColor Yellow
Write-Host "API docs:  http://localhost:8000/docs" -ForegroundColor Yellow
