param(
    [switch]$WithLocalGenerator,
    [switch]$WithGradio,
    [switch]$ResetEnvironment,
    [ValidateSet("cpu", "cu130")]
    [string]$TorchVariant = "cpu"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

if ($WithGradio -and -not $WithLocalGenerator) {
    throw "The legacy Gradio interface uses the local generator. Add -WithLocalGenerator when requesting -WithGradio."
}

$python = Get-Command python -ErrorAction SilentlyContinue
$pythonExe = if ($python) { $python.Source } else { $null }
if (-not $pythonExe -and $env:LOCALAPPDATA) {
    foreach ($version in @("Python312", "Python311", "Python313", "Python310", "Python314")) {
        $candidate = Join-Path $env:LOCALAPPDATA "Programs\Python\$version\python.exe"
        if (Test-Path -LiteralPath $candidate) { $pythonExe = $candidate; break }
    }
}
if (-not $pythonExe) {
    throw "Python 3.10 or newer is required. Install Python, then rerun this setup script."
}

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if ($ResetEnvironment -and (Test-Path -LiteralPath (Join-Path $projectRoot ".venv"))) {
    $expectedVenv = [IO.Path]::GetFullPath((Join-Path $projectRoot ".venv"))
    $resolvedVenv = [IO.Path]::GetFullPath((Resolve-Path -LiteralPath (Join-Path $projectRoot ".venv")).Path)
    if ($resolvedVenv -ne $expectedVenv -or (Split-Path -Leaf $resolvedVenv) -ne ".venv") {
        throw "Refusing to reset an unexpected environment path: $resolvedVenv"
    }
    Write-Host "Removing the existing project virtual environment to clear previously installed optional packages."
    Remove-Item -LiteralPath $resolvedVenv -Recurse -Force
}
if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host "Creating project environment with $pythonExe"
    & $pythonExe -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Could not create the project virtual environment." }
}

& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Could not upgrade pip." }

Write-Host "Installing the hosted-provider backend dependencies (target: under 500 MB installed)."
& $venvPython -m pip install -r requirements-backend.txt
if ($LASTEXITCODE -ne 0) { throw "Could not install backend dependencies." }

if ($WithLocalGenerator) {
    $torchIndex = if ($TorchVariant -eq "cu130") { "https://download.pytorch.org/whl/cu130" } else { "https://download.pytorch.org/whl/cpu" }
    Write-Host "Installing optional $TorchVariant PyTorch and local Diffusers dependencies. This adds more than the hosted-runtime size budget."
    & $venvPython -m pip install torch torchvision --index-url $torchIndex
    if ($LASTEXITCODE -ne 0) { throw "Could not install the selected PyTorch build." }
    & $venvPython -m pip install -r requirements-local.txt
    if ($LASTEXITCODE -ne 0) { throw "Could not install local generator dependencies." }
}

if ($WithGradio) {
    & $venvPython -m pip install -r requirements-gradio.txt
    if ($LASTEXITCODE -ne 0) { throw "Could not install Gradio dependencies." }
}

if (-not $WithLocalGenerator) {
    $sitePackages = Join-Path $projectRoot ".venv\Lib\site-packages"
    $installedBytes = (Get-ChildItem -LiteralPath $sitePackages -Recurse -File | Measure-Object -Property Length -Sum).Sum
    $installedMB = [math]::Round($installedBytes / 1MB)
    Write-Host "Installed backend Python packages occupy approximately $installedMB MB."
    if ($installedMB -ge 500) {
        throw "The environment is still at least 500 MB. Run .\scripts\setup_backend.ps1 -ResetEnvironment to remove previously installed optional packages."
    }
}

Write-Host "Backend environment is ready. Start it with: .\scripts\start_local.ps1"
if (-not $WithLocalGenerator) {
    Write-Host "This setup uses hosted TokenHub/Meshy generation and does not install PyTorch, Diffusers, Gradio, or local model weights."
    Write-Host "Optional local generator: .\scripts\setup_backend.ps1 -WithLocalGenerator -TorchVariant cu130"
}
