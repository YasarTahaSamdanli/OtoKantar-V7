<?php

namespace App\Services;

use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\Http;
use Throwable;

class HealthAlertService
{
    public function __construct(
        private readonly SystemHealthService $health,
    ) {}

    public function checkAndNotify(bool $force = false, bool $dryRun = false): array
    {
        $report = $this->health->report();
        $payload = $this->payload($report);
        $shouldAlert = $force || $this->shouldAlert($report);
        $signature = $this->signature($report);
        $cacheKey = 'system_health_alert:'.$signature;
        $cooldownMinutes = max(0, (int) config('system_health.alerts.cooldown_minutes', 15));

        if (! $shouldAlert) {
            return [
                'status' => 'skipped',
                'reason' => 'health_below_alert_threshold',
                'report' => $report,
                'payload' => $payload,
                'sent' => [],
            ];
        }

        if (! $force && $cooldownMinutes > 0 && Cache::has($cacheKey)) {
            return [
                'status' => 'skipped',
                'reason' => 'cooldown',
                'report' => $report,
                'payload' => $payload,
                'sent' => [],
            ];
        }

        if ($dryRun) {
            return [
                'status' => 'dry-run',
                'reason' => null,
                'report' => $report,
                'payload' => $payload,
                'sent' => [],
            ];
        }

        $sent = $this->send($payload);
        $ok = collect($sent)->contains(fn (array $result): bool => ($result['status'] ?? null) === 'sent');

        if ($ok && $cooldownMinutes > 0) {
            Cache::put($cacheKey, true, now()->addMinutes($cooldownMinutes));
        }

        return [
            'status' => $ok ? 'sent' : 'not_configured',
            'reason' => $ok ? null : 'no_channel_sent',
            'report' => $report,
            'payload' => $payload,
            'sent' => $sent,
        ];
    }

    private function shouldAlert(array $report): bool
    {
        if (! (bool) config('system_health.alerts.enabled', false)) {
            return false;
        }

        return $this->severity($report) >= $this->configuredSeverity();
    }

    private function send(array $payload): array
    {
        return [
            'telegram' => $this->sendTelegram($payload),
            'n8n' => $this->sendN8n($payload),
        ];
    }

    private function sendTelegram(array $payload): array
    {
        $token = (string) config('system_health.alerts.telegram.bot_token', '');
        $chatId = (string) config('system_health.alerts.telegram.chat_id', '');

        if ($token === '' || $chatId === '') {
            return ['status' => 'skipped', 'reason' => 'not_configured'];
        }

        try {
            $response = Http::timeout($this->timeoutSeconds())
                ->asForm()
                ->post('https://api.telegram.org/bot'.$token.'/sendMessage', [
                    'chat_id' => $chatId,
                    'text' => $this->telegramText($payload),
                    'disable_web_page_preview' => true,
                ]);

            return [
                'status' => $response->successful() ? 'sent' : 'failed',
                'http_status' => $response->status(),
            ];
        } catch (Throwable $e) {
            return ['status' => 'failed', 'error' => $e->getMessage()];
        }
    }

    private function sendN8n(array $payload): array
    {
        $url = (string) config('system_health.alerts.n8n.webhook_url', '');

        if ($url === '') {
            return ['status' => 'skipped', 'reason' => 'not_configured'];
        }

        try {
            $response = Http::timeout($this->timeoutSeconds())->post($url, $payload);

            return [
                'status' => $response->successful() ? 'sent' : 'failed',
                'http_status' => $response->status(),
            ];
        } catch (Throwable $e) {
            return ['status' => 'failed', 'error' => $e->getMessage()];
        }
    }

    private function payload(array $report): array
    {
        $queue = $report['checks']['queue'] ?? [];
        $metrics = $queue['metrics'] ?? [];
        $worker = $queue['worker'] ?? [];

        return [
            'customer' => (string) config('system_health.alerts.customer_name', config('app.name', 'OtoKantar')),
            'app_url' => (string) config('app.url'),
            'generated_at' => $report['generated_at'] ?? now()->toIso8601String(),
            'status' => $report['status'] ?? 'unknown',
            'overall_status' => $report['overall_status'] ?? 'Unknown',
            'health_score' => $report['health_score'] ?? null,
            'queue' => [
                'name' => $queue['queue'] ?? config('services.legacy_runtime.queue', 'live-ingest'),
                'level' => $queue['level'] ?? 'UNKNOWN',
                'pending_jobs' => $metrics['pending_jobs'] ?? 0,
                'failed_jobs' => $metrics['failed_jobs'] ?? 0,
                'last_successful_ingest_at' => $metrics['last_successful_ingest_at'] ?? null,
                'last_failed_ingest_at' => $metrics['last_failed_ingest_at'] ?? null,
                'avg_job_duration_ms_24h' => $metrics['avg_job_duration_ms_24h'] ?? null,
                'oldest_pending_job_age_seconds' => $metrics['oldest_pending_job_age_seconds'] ?? null,
                'pending_temp_images' => $metrics['pending_temp_images'] ?? 0,
                'worker_status' => $worker['status'] ?? 'UNKNOWN',
                'worker_message' => $worker['message'] ?? null,
            ],
            'failed_job_sample' => $queue['failed_job_sample'] ?? [],
        ];
    }

    private function telegramText(array $payload): string
    {
        $queue = $payload['queue'] ?? [];

        return implode(PHP_EOL, [
            'OtoKantar alarm: '.($payload['overall_status'] ?? 'Unknown'),
            'Musteri: '.($payload['customer'] ?? '-'),
            'Health score: '.($payload['health_score'] ?? '-'),
            'Queue: '.($queue['name'] ?? '-').' / '.($queue['level'] ?? '-'),
            'Pending: '.($queue['pending_jobs'] ?? 0).' | Failed: '.($queue['failed_jobs'] ?? 0).' | Temp: '.($queue['pending_temp_images'] ?? 0),
            'Worker: '.($queue['worker_status'] ?? '-').' - '.($queue['worker_message'] ?? '-'),
            'Son basarili ingest: '.($queue['last_successful_ingest_at'] ?? '-'),
            'Son hatali ingest: '.($queue['last_failed_ingest_at'] ?? '-'),
            'Zaman: '.($payload['generated_at'] ?? '-'),
            'URL: '.($payload['app_url'] ?? '-'),
        ]);
    }

    private function signature(array $report): string
    {
        $queue = $report['checks']['queue'] ?? [];

        return sha1(json_encode([
            'status' => $report['status'] ?? null,
            'overall_status' => $report['overall_status'] ?? null,
            'queue_level' => $queue['level'] ?? null,
            'queue_levels' => $queue['levels'] ?? [],
            'worker_status' => $queue['worker']['status'] ?? null,
        ]) ?: '');
    }

    private function severity(array $report): int
    {
        return match (strtolower((string) ($report['status'] ?? 'warning'))) {
            'critical' => 2,
            'warning' => 1,
            default => 0,
        };
    }

    private function configuredSeverity(): int
    {
        return match (strtolower((string) config('system_health.alerts.min_level', 'warning'))) {
            'critical' => 2,
            'ok', 'healthy' => 0,
            default => 1,
        };
    }

    private function timeoutSeconds(): int
    {
        return max(1, (int) config('system_health.alerts.timeout_seconds', 5));
    }
}
