<?php

namespace App\Services;

use App\Models\VehiclePass;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\File;
use Throwable;

class SystemHealthService
{
    public function __construct(
        private readonly QueueOperationsService $queueOperations,
    ) {}

    public function report(): array
    {
        $checks = [
            'app' => $this->appCheck(),
            'database' => $this->databaseCheck(),
            'runtime' => $this->runtimeCheck(),
            'queue' => $this->queueCheck(),
            'disk' => $this->diskCheck(),
            'backup' => $this->backupCheck(),
            'ingest' => $this->ingestCheck(),
        ];
        $score = $this->healthScore($checks);
        $overallStatus = $this->overallStatusLabel($checks, $score);

        return [
            'status' => $this->overallStatus($checks),
            'overall_status' => $overallStatus,
            'health_score' => $score,
            'generated_at' => now()->toIso8601String(),
            'checks' => $checks,
        ];
    }

    private function appCheck(): array
    {
        return [
            'status' => 'ok',
            'env' => app()->environment(),
            'debug' => (bool) config('app.debug'),
            'url' => config('app.url'),
        ];
    }

    private function databaseCheck(): array
    {
        try {
            DB::connection()->select('select 1');

            return [
                'status' => 'ok',
                'connection' => config('database.default'),
            ];
        } catch (Throwable $e) {
            return [
                'status' => 'critical',
                'connection' => config('database.default'),
                'error' => $e->getMessage(),
            ];
        }
    }

    private function runtimeCheck(): array
    {
        $root = $this->runtimeRoot();
        $files = [];
        $staleAfter = (int) config('system_health.stale_after_seconds', 120);
        $status = is_dir($root) ? 'ok' : 'warning';

        foreach ((array) config('system_health.runtime_files', []) as $name) {
            $path = $root.DIRECTORY_SEPARATOR.$name;
            $exists = is_file($path);
            $mtime = $exists ? (int) filemtime($path) : null;
            $age = $mtime ? max(0, time() - $mtime) : null;

            if ($name === 'canli_durum.json' && (!$exists || ($age !== null && $age > $staleAfter))) {
                $status = 'warning';
            }

            $files[$name] = [
                'exists' => $exists,
                'modified_at' => $mtime ? date('c', $mtime) : null,
                'age_seconds' => $age,
                'bytes' => $exists ? (int) filesize($path) : null,
            ];
        }

        return [
            'status' => $status,
            'path' => $root,
            'exists' => is_dir($root),
            'files' => $files,
        ];
    }

    private function queueCheck(): array
    {
        $queue = $this->queueOperations->metrics();
        $level = strtoupper((string) ($queue['status'] ?? 'WARNING'));

        return [
            'status' => match ($level) {
                'CRITICAL' => 'critical',
                'WARNING' => 'warning',
                default => 'ok',
            },
            'level' => $level,
            'connection' => config('queue.default'),
            'queue' => $queue['queue'] ?? config('services.legacy_runtime.queue', 'live-ingest'),
            'metrics' => $queue['metrics'] ?? [],
            'levels' => $queue['levels'] ?? [],
            'worker' => $queue['worker'] ?? [],
            'failed_job_sample' => $queue['failed_job_sample'] ?? [],
            'error' => $queue['error'] ?? null,
        ];
    }

    private function diskCheck(): array
    {
        $path = storage_path();
        $freeBytes = @disk_free_space($path);
        $totalBytes = @disk_total_space($path);
        $freeMb = $freeBytes === false ? null : round($freeBytes / 1024 / 1024, 2);
        $totalMb = $totalBytes === false ? null : round($totalBytes / 1024 / 1024, 2);
        $warningMb = (int) config('system_health.disk_warning_free_mb', 1024);

        return [
            'status' => $freeMb !== null && $freeMb < $warningMb ? 'warning' : 'ok',
            'path' => $path,
            'free_mb' => $freeMb,
            'total_mb' => $totalMb,
            'warning_below_mb' => $warningMb,
        ];
    }

    private function backupCheck(): array
    {
        $root = $this->absolutePath((string) config('backup.path', storage_path('app/backups')));
        if (! is_dir($root)) {
            return [
                'status' => 'warning',
                'path' => $root,
                'latest_backup_at' => null,
                'message' => 'Backup dizini bulunamadi.',
            ];
        }

        $directories = File::directories($root);
        rsort($directories);
        $latest = $directories[0] ?? null;

        return [
            'status' => $latest ? 'ok' : 'warning',
            'path' => $root,
            'latest_backup_at' => $latest ? date('c', (int) filemtime($latest)) : null,
            'latest_backup' => $latest ? basename($latest) : null,
        ];
    }

    private function ingestCheck(): array
    {
        try {
            $latest = VehiclePass::query()->latest('passed_at')->first(['id', 'plate', 'direction', 'passed_at']);

            return [
                'status' => $latest ? 'ok' : 'warning',
                'latest_vehicle_pass' => $latest ? [
                    'id' => $latest->id,
                    'plate' => $latest->plate,
                    'direction' => $latest->direction,
                    'passed_at' => optional($latest->passed_at)->toIso8601String(),
                ] : null,
            ];
        } catch (Throwable $e) {
            return [
                'status' => 'warning',
                'error' => $e->getMessage(),
            ];
        }
    }

    private function tableCount(string $table): int
    {
        if (! DB::getSchemaBuilder()->hasTable($table)) {
            return 0;
        }

        return (int) DB::table($table)->count();
    }

    private function runtimeRoot(): string
    {
        return $this->absolutePath(rtrim((string) config('services.legacy_runtime.path'), '\\/'));
    }

    private function absolutePath(string $path): string
    {
        if ($path === '') {
            return storage_path();
        }

        if (str_starts_with($path, '/') || preg_match('/^[A-Za-z]:[\/\\\\]/', $path) === 1) {
            return $path;
        }

        return base_path($path);
    }

    private function overallStatus(array $checks): string
    {
        $statuses = array_map(fn (array $check): string => (string) ($check['status'] ?? 'unknown'), $checks);

        if (in_array('critical', $statuses, true)) {
            return 'critical';
        }

        if (in_array('warning', $statuses, true)) {
            return 'warning';
        }

        return 'ok';
    }

    private function healthScore(array $checks): int
    {
        $score = 100;

        foreach ($checks as $check) {
            $status = (string) ($check['status'] ?? 'warning');
            if ($status === 'critical') {
                $score -= 30;
            } elseif ($status === 'warning') {
                $score -= 12;
            }
        }

        $queueLevels = $checks['queue']['levels'] ?? [];
        foreach ($queueLevels as $level) {
            if ($level === 'CRITICAL') {
                $score -= 10;
            } elseif ($level === 'WARNING') {
                $score -= 4;
            }
        }

        return max(0, min(100, $score));
    }

    private function overallStatusLabel(array $checks, int $score): string
    {
        $status = $this->overallStatus($checks);

        if ($status === 'critical' || $score < 60) {
            return 'Critical';
        }

        if ($status === 'warning' || $score < 85) {
            return 'Warning';
        }

        return 'Healthy';
    }
}
