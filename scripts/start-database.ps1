$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$postgresBin = Join-Path $projectRoot '.tools/pgsql/bin'
$cluster = Join-Path $projectRoot '.local/postgres'
$secret = Join-Path $projectRoot '.local/db-password'
if (!(Test-Path "$postgresBin/pg_ctl.exe")) { throw 'Run .venv/Scripts/python.exe scripts/setup_tools.py first.' }
New-Item -ItemType Directory -Force (Join-Path $projectRoot '.local') | Out-Null
if (!(Test-Path "$cluster/PG_VERSION")) {
    if (!(Test-Path $secret)) {
        ([guid]::NewGuid().ToString('N') + [guid]::NewGuid().ToString('N')) | Set-Content -Encoding ascii $secret
    }
    & "$postgresBin/initdb.exe" -D $cluster -U jigsaw --pwfile=$secret --auth=scram-sha-256 --encoding=UTF8 --locale=C
    if ($LASTEXITCODE -ne 0) { throw 'Database initialization failed.' }
}
& "$postgresBin/pg_ctl.exe" -D $cluster status *> $null
if ($LASTEXITCODE -ne 0) {
    & "$postgresBin/pg_ctl.exe" -D $cluster -l (Join-Path $projectRoot '.local/postgres.log') -o '-h 127.0.0.1 -p 55432' -w start
    if ($LASTEXITCODE -ne 0) { throw 'Database startup failed. See .local/postgres.log.' }
}
$previousPassword = $env:PGPASSWORD
try {
    $env:PGPASSWORD = (Get-Content $secret -Raw).Trim()
    $exists = & "$postgresBin/psql.exe" -h 127.0.0.1 -p 55432 -U jigsaw -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='jigsaw_ai'"
    if ($LASTEXITCODE -ne 0) { throw 'Database connection failed.' }
    if ($exists -ne '1') {
        & "$postgresBin/createdb.exe" -h 127.0.0.1 -p 55432 -U jigsaw jigsaw_ai
        if ($LASTEXITCODE -ne 0) { throw 'Database creation failed.' }
    }
} finally { $env:PGPASSWORD = $previousPassword }
Write-Host 'PostgreSQL ready on 127.0.0.1:55432 (database jigsaw_ai).'
