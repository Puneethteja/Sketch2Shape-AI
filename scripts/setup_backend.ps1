$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$python = Get-Command python -ErrorAction SilentlyContinue
$pythonExe = if ($python) { $python.Source } else { $null }
if (-not $pythonExe -and $env:LOCALAPPDATA) {
    foreach ($version in @("Python311", "Python312", "Python310", "Python313", "Python314")) {
        $candidate = Join-Path $env:LOCALAPPDATA "Programs\Python\$version\python.exe"
        if (Test-Path -LiteralPath $candidate) { $pythonExe = $candidate; break }
    }
}
if (-not $pythonExe) {
    Write-Error "Python 3.11 or newer is required. Install Python, then rerun this setup script."
}

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host "Creating project environment with $pythonExe"
    & $pythonExe -m venv .venv
}

Write-Host "Installing PyTorch with CUDA 13.0 support"
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Could not upgrade pip." }
& $venvPython -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
if ($LASTEXITCODE -ne 0) { throw "Could not install CUDA-enabled PyTorch." }

Write-Host "Installing Sketch2Shape API and inference dependencies"
& $venvPython -m pip install -r requirements-backend.txt
if ($LASTEXITCODE -ne 0) { throw "Could not install backend dependencies." }

Write-Host "Backend environment is ready. Start it with: .\scripts\start_local.ps1"
