param(
    [string]$LogDir = "stress_test_logs"
)

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$logPath = Join-Path $root $LogDir

if (-not (Test-Path $logPath)) {
    Write-Error "Log klasoru bulunamadi: $logPath"
    exit 1
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$zipPath = Join-Path $root "stress_test_logs_$timestamp.zip"

if (Test-Path $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}

Compress-Archive -Path (Join-Path $logPath "*") -DestinationPath $zipPath -Force
Write-Host "Stress test log paketi hazir: $zipPath"
