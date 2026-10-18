param([int]$Port = 8000, [switch]$Preview)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonExe = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw 'Install the project dependencies first. See README.md.'
}
$env:HF_HOME = Join-Path $PSScriptRoot '.model-cache/huggingface'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:HF_HUB_DISABLE_XET = '1'
if ($Preview) { $env:KOSHUR_SKIP_MODEL_LOAD = '1' }
else { Remove-Item Env:KOSHUR_SKIP_MODEL_LOAD -ErrorAction SilentlyContinue }
Write-Host "Open http://127.0.0.1:$Port in your browser. Press Ctrl+C to stop."
& $pythonExe -m uvicorn app.main:app --host 127.0.0.1 --port $Port --no-access-log
exit $LASTEXITCODE
