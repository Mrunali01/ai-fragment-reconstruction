$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
& "$projectRoot/.tools/pgsql/bin/pg_ctl.exe" -D "$projectRoot/.local/postgres" -m fast -w stop
if ($LASTEXITCODE -ne 0) { throw 'Database did not stop successfully.' }
