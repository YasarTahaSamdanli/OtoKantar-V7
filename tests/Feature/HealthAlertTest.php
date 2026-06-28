<?php

namespace Tests\Feature;

use Illuminate\Foundation\Testing\RefreshDatabase;
use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Http;
use Tests\TestCase;

class HealthAlertTest extends TestCase
{
    use RefreshDatabase;

    protected function setUp(): void
    {
        parent::setUp();

        Cache::flush();
        config([
            'cache.default' => 'array',
            'system_health.alerts.enabled' => true,
            'system_health.alerts.customer_name' => 'Test Kantar',
            'system_health.alerts.min_level' => 'warning',
            'system_health.alerts.cooldown_minutes' => 15,
            'system_health.alerts.n8n.webhook_url' => 'https://n8n.example.test/webhook/health',
            'system_health.alerts.telegram.bot_token' => null,
            'system_health.alerts.telegram.chat_id' => null,
            'system_health.queue.name' => 'live-ingest',
            'services.legacy_runtime.queue' => 'live-ingest',
            'system_health.queue.pending_warning' => 1,
            'system_health.queue.pending_critical' => 2,
            'system_health.queue.oldest_pending_warning_seconds' => 60,
            'system_health.queue.oldest_pending_critical_seconds' => 120,
            'system_health.queue.worker_down_after_seconds' => 120,
        ]);
    }

    public function test_health_alert_command_sends_n8n_payload_when_queue_is_critical(): void
    {
        Http::fake([
            'n8n.example.test/*' => Http::response(['ok' => true], 200),
        ]);

        $this->insertPendingJobs(2, 180);

        $this->artisan('otokantar:health-alerts')
            ->expectsOutputToContain('"status": "sent"')
            ->assertExitCode(0);

        Http::assertSent(function ($request): bool {
            $payload = $request->data();

            return $request->url() === 'https://n8n.example.test/webhook/health'
                && $payload['customer'] === 'Test Kantar'
                && collect($payload['problems'])->contains(fn (array $problem): bool => $problem['name'] === 'queue')
                && $payload['queue']['level'] === 'CRITICAL'
                && $payload['queue']['pending_jobs'] === 2
                && $payload['queue']['worker_status'] === 'CRITICAL';
        });
    }

    public function test_health_alert_command_suppresses_duplicate_alerts_during_cooldown(): void
    {
        Http::fake([
            'n8n.example.test/*' => Http::response(['ok' => true], 200),
        ]);

        $this->insertPendingJobs(2, 180);

        $this->artisan('otokantar:health-alerts')->assertExitCode(0);
        $this->artisan('otokantar:health-alerts')
            ->expectsOutputToContain('"reason": "cooldown"')
            ->assertExitCode(0);

        Http::assertSentCount(1);
    }

    public function test_health_alert_command_dry_run_does_not_send_requests(): void
    {
        Http::fake();

        $this->insertPendingJobs(2, 180);

        $this->artisan('otokantar:health-alerts --dry-run')
            ->expectsOutputToContain('"status": "dry-run"')
            ->assertExitCode(0);

        Http::assertNothingSent();
    }

    private function insertPendingJobs(int $count, int $ageSeconds): void
    {
        foreach (range(1, $count) as $index) {
            DB::table('jobs')->insert([
                'queue' => 'live-ingest',
                'payload' => json_encode(['displayName' => 'ProcessLiveIngest '.$index]),
                'attempts' => 0,
                'reserved_at' => null,
                'available_at' => time() - $ageSeconds,
                'created_at' => time() - $ageSeconds,
            ]);
        }
    }
}
