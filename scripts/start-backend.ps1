$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $projectRoot
& "$PSScriptRoot/start-database.ps1"
$dbPassword = (Get-Content '.local/db-password' -Raw).Trim()
$env:JIGSAW_DATABASE_URL = "postgresql://jigsaw:${dbPassword}@127.0.0.1:55432/jigsaw_ai"
& '.venv/Scripts/python.exe' -m uvicorn jigsaw_ai.api:app --host 127.0.0.1 --port 8000
