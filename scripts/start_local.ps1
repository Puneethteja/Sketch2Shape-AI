$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$port = if ($env:PORT) { [int]$env:PORT } else { 8000 }
$envFile = Join-Path $projectRoot ".env"
$savedProvider = if ($env:SKETCH2SHAPE_PROVIDER) { $env:SKETCH2SHAPE_PROVIDER.Trim().ToLowerInvariant() } else { "tokenhub" }
$savedTokenHubKey = -not [string]::IsNullOrWhiteSpace($env:TENCENT_TOKENHUB_API_KEY) -or -not [string]::IsNullOrWhiteSpace($env:HUNYUAN3D_API_KEY)
$savedMeshyKey = -not [string]::IsNullOrWhiteSpace($env:MESHY_API_KEY)
if (Test-Path -LiteralPath $envFile) {
    foreach ($line in (Get-Content -LiteralPath $envFile)) {
        if ($line -match '^\s*(TENCENT_TOKENHUB_API_KEY|HUNYUAN3D_API_KEY|MESHY_API_KEY|SKETCH2SHAPE_PROVIDER)\s*=\s*(.*?)\s*$') {
            $name = $matches[1]
            $value = $matches[2].Trim()
            if ($value.Length -ge 2 -and (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'")))) {
                $value = $value.Substring(1, $value.Length - 2)
            }
            switch ($name) {
                "SKETCH2SHAPE_PROVIDER" { if ($value) { $savedProvider = $value.ToLowerInvariant() } }
                "MESHY_API_KEY" { if ($value) { $savedMeshyKey = $true } }
                { $_ -in @("TENCENT_TOKENHUB_API_KEY", "HUNYUAN3D_API_KEY") } { if ($value) { $savedTokenHubKey = $true } }
            }
        }
    }
}
if ($env:SKETCH2SHAPE_PROVIDER) { $savedProvider = $env:SKETCH2SHAPE_PROVIDER.Trim().ToLowerInvariant() }
$savedSelectedProviderKey = if ($savedProvider -eq "meshy") { $savedMeshyKey } elseif ($savedProvider -eq "tokenhub") { $savedTokenHubKey } else { $true }

try {
    $existingBackend = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 2
    if ($existingBackend.status -eq "ok" -and $existingBackend.version -eq "2.0.0") {
        $backendReportsKey = if ($existingBackend.provider_setting -eq "meshy") { [bool]$existingBackend.meshy_configured } elseif ($existingBackend.provider_setting -eq "tokenhub") { [bool]$existingBackend.tokenhub_configured } else { $true }
        if (($existingBackend.PSObject.Properties.Name -contains "provider_setting") -and ($existingBackend.provider_setting -eq $savedProvider) -and ($backendReportsKey -or -not $savedSelectedProviderKey)) {
            if (-not $backendReportsKey -and -not $savedSelectedProviderKey -and $savedProvider -eq "meshy") {
                Write-Host "Sketch2Shape is running at http://127.0.0.1:$port, but Meshy is not configured. Run .\scripts\configure_meshy.ps1, then run this script again."
            } elseif (-not $backendReportsKey -and -not $savedSelectedProviderKey -and $savedProvider -eq "tokenhub") {
                Write-Host "Sketch2Shape is running at http://127.0.0.1:$port, but Tencent TokenHub is not configured. Run .\scripts\configure_tencent.ps1, then run this script again."
            } else {
                Write-Host "Sketch2Shape is already running with the current provider configuration at http://127.0.0.1:$port"
            }
            return
        }

        # Restart only when the port listener is demonstrably this application's Uvicorn.
        $legacyCommand = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '(?i)-m uvicorn api\.main:app' }
        $portOwners = (Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue).OwningProcess
        $legacyPids = @($legacyCommand | Where-Object { $portOwners -contains $_.ProcessId } | ForEach-Object { $_.ProcessId })
        if ($legacyPids.Count -gt 0) {
            $parentIds = @($legacyCommand | Where-Object { $legacyPids -contains $_.ProcessId } | ForEach-Object { $_.ParentProcessId })
            $legacyPids += @($legacyCommand | Where-Object { $parentIds -contains $_.ProcessId } | ForEach-Object { $_.ProcessId })
            foreach ($processId in ($legacyPids | Select-Object -Unique)) {
                Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
            }
            Start-Sleep -Seconds 1
            Write-Host "Stopped the Sketch2Shape backend so the saved provider configuration can load."
        } else {
            Write-Host "An older backend is using port $port, but it could not be safely identified as Sketch2Shape. Stop that server manually, then rerun this script."
            exit 1
        }
    }
} catch {
    # No Sketch2Shape backend is listening on this port; start one below.
}

$candidates = [System.Collections.Generic.List[string]]::new()
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $venvPython) { $candidates.Add($venvPython) }

$pythonOnPath = Get-Command python -ErrorAction SilentlyContinue
if ($pythonOnPath) { $candidates.Add($pythonOnPath.Source) }

if ($env:LOCALAPPDATA) {
    foreach ($version in @("Python311", "Python312", "Python310", "Python313", "Python314")) {
        $candidate = Join-Path $env:LOCALAPPDATA "Programs\Python\$version\python.exe"
        if (Test-Path -LiteralPath $candidate) { $candidates.Add($candidate) }
    }
}

$pythonExe = $null
foreach ($candidate in ($candidates | Select-Object -Unique)) {
    # TokenHub generation does not require the heavyweight local Diffusers stack.
    & $candidate -c "import fastapi, uvicorn, multipart, PIL, numpy" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $pythonExe = $candidate
        break
    }
}

if (-not $pythonExe) {
    Write-Error "The backend Python environment is incomplete. Run .\scripts\setup_backend.ps1 first."
}

Write-Host "Starting Sketch2Shape API and frontend at http://127.0.0.1:$port"
if (-not $savedSelectedProviderKey -and $savedProvider -eq "meshy") {
    Write-Host "Meshy API key not configured. Run .\scripts\configure_meshy.ps1, then restart this server."
} elseif (-not $savedSelectedProviderKey -and $savedProvider -eq "tokenhub") {
    Write-Host "Tencent HY-3D API key not configured. Run .\scripts\configure_tencent.ps1, then restart this server."
}
Write-Host "The first generation may take time while model weights are initialized. Press Ctrl+C to stop."
& $pythonExe -m uvicorn api.main:app --host 127.0.0.1 --port $port --log-level info
