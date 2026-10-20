$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonExe = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Run the README installation steps first.' }
& $pythonExe -c 'import httpx, uvicorn, starlette'
if ($LASTEXITCODE -ne 0) { throw 'Install sharing dependencies: .\.venv\Scripts\python.exe -m pip install -r requirements-sharing.txt' }
& $pythonExe scripts/share.py
exit $LASTEXITCODE
