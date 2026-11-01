$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$envPath = Join-Path $projectRoot ".env"
$keySecure = Read-Host "Meshy API key (input hidden)" -AsSecureString
$pointer = [IntPtr]::Zero
try {
    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($keySecure)
    $apiKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    if ([string]::IsNullOrWhiteSpace($apiKey)) { throw "An API key is required." }

    $existing = @()
    if (Test-Path -LiteralPath $envPath) { $existing = Get-Content -LiteralPath $envPath }
    $lines = @($existing | Where-Object { $_ -notmatch '^\s*(MESHY_API_KEY|MESHY_AI_MODEL|SKETCH2SHAPE_PROVIDER)\s*=' })
    $lines += "MESHY_API_KEY=$apiKey"
    $lines += "MESHY_AI_MODEL=meshy-7.1"
    $lines += "SKETCH2SHAPE_PROVIDER=meshy"
    [IO.File]::WriteAllLines($envPath, $lines, [Text.UTF8Encoding]::new($false))
    Write-Host "Meshy Image-to-3D configured. Restart with .\scripts\start_local.ps1."
}
finally {
    if ($pointer -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer) }
    if ($keySecure) { $keySecure.Dispose() }
    $apiKey = $null
}
