$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$envPath = Join-Path $projectRoot ".env"
$keySecure = Read-Host "Tencent TokenHub API key (input hidden)" -AsSecureString
$pointer = [IntPtr]::Zero
try {
    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($keySecure)
    $apiKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    if ([string]::IsNullOrWhiteSpace($apiKey)) { throw "An API key is required." }
    $existing = @()
    if (Test-Path -LiteralPath $envPath) { $existing = Get-Content -LiteralPath $envPath }
    $lines = @($existing | Where-Object { $_ -notmatch '^\s*(TENCENT_TOKENHUB_API_KEY|HUNYUAN3D_MODEL|SKETCH2SHAPE_PROVIDER)\s*=' })
    $lines += "TENCENT_TOKENHUB_API_KEY=$apiKey"
    $lines += "HUNYUAN3D_MODEL=hi3d-3.0"
    $lines += "SKETCH2SHAPE_PROVIDER=tokenhub"
    [IO.File]::WriteAllLines($envPath, $lines, [Text.UTF8Encoding]::new($false))
    Write-Host "Tencent TokenHub configured for Hi3D 3.0 image-to-3D. Restart the backend with .\scripts\start_local.ps1."
}
finally {
    if ($pointer -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer) }
    if ($keySecure) { $keySecure.Dispose() }
    $apiKey = $null
}
