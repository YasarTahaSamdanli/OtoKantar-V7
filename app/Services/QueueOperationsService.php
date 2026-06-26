<?php

namespace App\Services;

use App\Models\LiveIngestJobRun;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\File;
use Throwable;

class QueueOperationsService
{
    public function metrics(): array
    {
        $queue = $this->queueName();
        $historyHours = (int) config('system_health.queue.history_hours', 24);
        $since = now()->subHours($historyHours);

        try {
            $pending = $this->pendingMetrics($queue);
            $failedJobs = $this->failedJobMetrics($queue);
            $runs = $this->runMetrics($queue, $since);
            $tempImages = $this->tempImageMetrics();
            $metrics = array_merge($pending, $failedJobs, $runs, $tempImages);
            $levels = $this->levels($metrics);

            return [
                'status' => $this->worstLevel($levels),
                'queue' => $queue,
                'history_hours' => $historyHours,
                'metrics' => $metrics,
                'levels' => $levels,
                'worker' => $this->workerStatus($metrics, $levels),
                'failed_job_sample' => $this->failedJobSample($queue),
            ];
        } catch (Throwable $e) {
            return [
                'status' => 'CRITICAL',
                'queue' => $queue,
                'history_hours' => $historyHours,
                'metrics' => [],
                'levels' => ['collector' => 'CRITICAL'],
                'worker' => ['status' => 'unknown', 'message' => 'Queue metrics could not be collected.'],
                'error' => $e->getMessage(),
                'failed_job_sample' => [],
            ];
        }
    }

    public function supportSummary(): array
    {
        $metrics = $this->metrics();

        return [
            'queue' => $metrics['queue'] ?? $this->queueName(),
            'status' => $metrics['status'] ?? 'CRITICAL',
            'metrics' => $metrics['metrics'] ?? [],
            'levels' => $metrics['levels'] ?? [],
            'worker' => $metrics['worker'] ?? [],
            'failed_job_sample' => $metrics['failed_job_sample'] ?? [],
        ];
    }

    private function pendingMetrics(string $queue): array
    {
        $table = (string) config('queue.connections.database.table', 'jobs');
        if (! DB::getSchemaBuilder()->hasTable($table)) {
            return [
                'pending_jobs' => 0,
                'oldest_pending_job_age_seconds' => null,
                'oldest_pending_job_created_at' => null,
            ];
        }

        $query = DB::table($table)->where('queue', $queue);
        $oldest = (clone $query)->min('created_at');

        return [
            'pending_jobs' => (int) $query->count(),
            'oldest_pending_job_age_seconds' => $oldest ? max(0, time() - (int) $oldest) : null,
            'oldest_pending_job_created_at' => $oldest ? date('c', (int) $oldest) : null,
        ];
    }

    private function failedJobMetrics(string $queue): array
    {
        $table = (string) config('queue.failed.table', 'failed_jobs');
        if (! DB::getSchemaBuilder()->hasTable($table)) {
            return [
                'failed_jobs' => 0,
                'last_failed_job_at' => null,
            ];
        }

        $query = DB::table($table)->where('queue', $queue);
        $last = (clone $query)->max('failed_at');

        return [
            'failed_jobs' => (int) $query->count(),
            'last_failed_job_at' => $last ? date('c', strtotime((string) $last)) : null,
        ];
    }

    private function runMetrics(string $queue, \Illuminate\Support\Carbon $since): array
    {
        if (! DB::getSchemaBuilder()->hasTable('live_ingest_job_runs')) {
            return [
                'successful_jobs_24h' => 0,
                'failed_runs_24h' => 0,
                'last_successful_ingest_at' => null,
                'last_failed_ingest_at' => null,
                'avg_job_duration_ms_24h' => null,
            ];
        }

        $base = LiveIngestJobRun::query()->where('queue', $queue);
        $window = (clone $base)->where('finished_at', '>=', $since);

        $lastSuccess = (clone $base)->where('status', 'success')->latest('finished_at')->value('finished_at');
        $lastFailure = (clone $base)->where('status', 'failed')->latest('finished_at')->value('finished_at');

        return [
            'successful_jobs_24h' => (int) (clone $window)->where('status', 'success')->count(),
            'failed_runs_24h' => (int) (clone $window)->where('status', 'failed')->count(),
            'last_successful_ingest_at' => $lastSuccess ? $this->isoDate($lastSuccess) : null,
            'last_failed_ingest_at' => $lastFailure ? $this->isoDate($lastFailure) : null,
            'avg_job_duration_ms_24h' => ($avg = (clone $window)->avg('duration_ms')) !== null ? round((float) $avg, 2) : null,
        ];
    }

    private function tempImageMetrics(): array
    {
        $path = storage_path('app/live-ingest-pending');

        return [
            'pending_temp_images' => is_dir($path) ? count(File::files($path)) : 0,
            'pending_temp_image_path' => $path,
        ];
    }

    private function levels(array $metrics): array
    {
        return [
            'pending_jobs' => $this->thresholdLevel((int) ($metrics['pending_jobs'] ?? 0), 'pending'),
            'failed_jobs' => $this->thresholdLevel((int) ($metrics['failed_jobs'] ?? 0), 'failed'),
            'oldest_pending_job_age_seconds' => $this->thresholdLevel($metrics['oldest_pending_job_age_seconds'], 'oldest_pending'),
            'pending_temp_images' => $this->thresholdLevel((int) ($metrics['pending_temp_images'] ?? 0), 'temp_images'),
            'avg_job_duration_ms_24h' => $this->thresholdLevel($metrics['avg_job_duration_ms_24h'], 'avg_duration'),
            'worker' => $this->workerLevel($metrics),
        ];
    }

    private function thresholdLevel(mixed $value, string $key): string
    {
        if ($value === null) {
            return 'OK';
        }

        $value = (float) $value;
        $configMap = [
            'pending' => ['queue.pending_warning', 'queue.pending_critical'],
            'failed' => ['queue.failed_warning', 'queue.failed_critical'],
            'oldest_pending' => ['queue.oldest_pending_warning_seconds', 'queue.oldest_pending_critical_seconds'],
            'temp_images' => ['queue.temp_images_warning', 'queue.temp_images_critical'],
            'avg_duration' => ['queue.avg_duration_warning_ms', 'queue.avg_duration_critical_ms'],
        ];
        [$warningKey, $criticalKey] = $configMap[$key];
        $warning = (float) config('system_health.'.$warningKey);
        $critical = (float) config('system_health.'.$criticalKey);

        if ($critical > 0 && $value >= $critical) {
            return 'CRITICAL';
        }

        if ($warning > 0 && $value >= $warning) {
            return 'WARNING';
        }

        return 'OK';
    }

    private function workerLevel(array $metrics): string
    {
        $pending = (int) ($metrics['pending_jobs'] ?? 0);
        $oldestAge = $metrics['oldest_pending_job_age_seconds'] ?? null;
        $downAfter = (int) config('system_health.queue.worker_down_after_seconds', 300);

        if ($pending > 0 && $oldestAge !== null && $oldestAge >= $downAfter) {
            return 'CRITICAL';
        }

        if ($pending > 0 && $oldestAge !== null && $oldestAge >= (int) config('system_health.queue.oldest_pending_warning_seconds', 60)) {
            return 'WARNING';
        }

        return 'OK';
    }

    private function workerStatus(array $metrics, array $levels): array
    {
        $level = (string) ($levels['worker'] ?? 'OK');
        $pending = (int) ($metrics['pending_jobs'] ?? 0);
        $age = $metrics['oldest_pending_job_age_seconds'] ?? null;

        return [
            'status' => $level,
            'message' => $level === 'CRITICAL'
                ? 'Queue worker calismiyor olabilir veya backlog eritilemiyor.'
                : ($level === 'WARNING' ? 'Queue backlog yavas eriyor.' : 'Queue backlog normal.'),
            'pending_jobs' => $pending,
            'oldest_pending_job_age_seconds' => $age,
        ];
    }

    private function failedJobSample(string $queue): array
    {
        $table = (string) config('queue.failed.table', 'failed_jobs');
        if (! DB::getSchemaBuilder()->hasTable($table)) {
            return [];
        }

        return DB::table($table)
            ->where('queue', $queue)
            ->latest('failed_at')
            ->limit((int) config('system_health.queue.failed_job_sample', 10))
            ->get(['id', 'uuid', 'connection', 'queue', 'exception', 'failed_at'])
            ->map(fn ($row): array => [
                'id' => $row->id,
                'uuid' => $row->uuid,
                'connection' => $row->connection,
                'queue' => $row->queue,
                'failed_at' => $this->isoDate($row->failed_at),
                'exception_summary' => mb_substr((string) $row->exception, 0, 500),
            ])
            ->all();
    }

    private function worstLevel(array $levels): string
    {
        if (in_array('CRITICAL', $levels, true)) {
            return 'CRITICAL';
        }

        if (in_array('WARNING', $levels, true)) {
            return 'WARNING';
        }

        return 'OK';
    }

    private function queueName(): string
    {
        return (string) config('system_health.queue.name', config('services.legacy_runtime.queue', 'live-ingest'));
    }

    private function isoDate(mixed $value): ?string
    {
        if ($value === null || $value === '') {
            return null;
        }

        $timestamp = strtotime((string) $value);

        return $timestamp === false ? null : date('c', $timestamp);
    }
}
