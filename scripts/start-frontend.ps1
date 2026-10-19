$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$env:PATH = "$projectRoot/.tools/node-v24.22.0-win-x64;$env:PATH"
Set-Location "$projectRoot/frontend"
if (!(Test-Path 'node_modules')) {
    & npm.cmd ci
    if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed.' }
}
& npm.cmd run dev
