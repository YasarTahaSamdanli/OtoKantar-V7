param(
    [string]$LogDir = "stress_test_logs",
    [string]$LaravelLogDir = "storage/logs/stress_test"
)

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$logPath = Join-Path $root $LogDir
$laravelLogPath = Join-Path $root $LaravelLogDir

if (-not (Test-Path $logPath) -and -not (Test-Path $laravelLogPath)) {
    Write-Error "Log klasoru bulunamadi: $logPath veya $laravelLogPath"
    exit 1
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$zipPath = Join-Path $root "stress_test_logs_$timestamp.zip"

if (Test-Path $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}

$paths = @()
if (Test-Path $logPath) {
    $paths += Join-Path $logPath "*"
}
if (Test-Path $laravelLogPath) {
    $paths += Join-Path $laravelLogPath "*"
}

Compress-Archive -Path $paths -DestinationPath $zipPath -Force
Write-Host "Stress test log paketi hazir: $zipPath"
