<?php

namespace Tests\Feature;

use App\Models\User;
use Illuminate\Foundation\Testing\RefreshDatabase;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\File;
use Tests\TestCase;

class SystemHealthTest extends TestCase
{
    use RefreshDatabase;

    protected function tearDown(): void
    {
        File::deleteDirectory(storage_path('framework/testing/system-health'));
        File::deleteDirectory(storage_path('framework/testing/support-bundles'));

        parent::tearDown();
    }

    public function test_admin_can_read_system_health_report(): void
    {
        $runtimePath = storage_path('framework/testing/system-health/runtime');
        File::ensureDirectoryExists($runtimePath);
        File::put($runtimePath.DIRECTORY_SEPARATOR.'canli_durum.json', json_encode([
            'son_guncelleme' => now()->toIso8601String(),
            'kantar_kg' => 1234,
        ]));

        config(['services.legacy_runtime.path' => $runtimePath]);

        $admin = User::factory()->create(['role' => 'admin']);

        $this->actingAs($admin)
            ->getJson('/admin/system-health')
            ->assertOk()
            ->assertJsonStructure([
                'status',
                'overall_status',
                'health_score',
                'generated_at',
                'checks' => [
                    'app',
                    'database',
                    'runtime',
                    'queue',
                    'disk',
                    'backup',
                    'ingest',
                    'devices',
                ],
            ])
            ->assertJsonPath('checks.runtime.exists', true);
    }

    public function test_device_status_sqlite_is_reported_by_system_health(): void
    {
        $runtimePath = storage_path('framework/testing/system-health/runtime');
        File::ensureDirectoryExists($runtimePath);

        $pdo = new \PDO('sqlite:'.$runtimePath.DIRECTORY_SEPARATOR.'device_status.sqlite');
        $pdo->exec(
            "CREATE TABLE device_status (
                device_key TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                last_error TEXT,
                last_seen TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )"
        );
        $now = now()->toIso8601String();
        $stmt = $pdo->prepare(
            'INSERT INTO device_status (
                device_key, status, level, message, last_error, last_seen, updated_at, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)'
        );
        $stmt->execute([
            'printer',
            'FAILED',
            'error',
            'Yazıcı çevrimdışı veya hata verdi.',
            'Printer Offline',
            $now,
            $now,
            json_encode(['plate' => '34ABC123', 'status' => 'FAILED'], JSON_UNESCAPED_UNICODE),
        ]);

        config(['services.legacy_runtime.path' => $runtimePath]);

        $admin = User::factory()->create(['role' => 'admin']);

        $this->actingAs($admin)
            ->getJson('/admin/system-health')
            ->assertOk()
            ->assertJsonPath('checks.devices.status', 'warning')
            ->assertJsonPath('checks.devices.devices.printer.status', 'FAILED')
            ->assertJsonPath('checks.devices.devices.printer.last_error', 'Printer Offline')
            ->assertJsonPath('checks.devices.devices.printer.payload.plate', '34ABC123');
    }

    public function test_queue_backlog_changes_health_status_and_score(): void
    {
        config([
            'system_health.queue.pending_warning' => 1,
            'system_health.queue.pending_critical' => 2,
            'system_health.queue.oldest_pending_critical_seconds' => 60,
            'system_health.queue.worker_down_after_seconds' => 60,
        ]);

        DB::table('jobs')->insert([
            [
                'queue' => 'live-ingest',
                'payload' => '{}',
                'attempts' => 0,
                'reserved_at' => null,
                'available_at' => time() - 120,
                'created_at' => time() - 120,
            ],
            [
                'queue' => 'live-ingest',
                'payload' => '{}',
                'attempts' => 0,
                'reserved_at' => null,
                'available_at' => time() - 120,
                'created_at' => time() - 120,
            ],
        ]);

        $admin = User::factory()->create(['role' => 'admin']);

        $response = $this->actingAs($admin)->getJson('/admin/system-health')->assertOk();

        $response->assertJsonPath('checks.queue.levels.pending_jobs', 'CRITICAL')
            ->assertJsonPath('checks.queue.levels.worker', 'CRITICAL')
            ->assertJsonPath('checks.queue.metrics.pending_jobs', 2);

        $this->assertLessThan(100, $response->json('health_score'));
    }

    public function test_employee_cannot_read_system_health_report(): void
    {
        $employee = User::factory()->create(['role' => 'employee']);

        $this->actingAs($employee)
            ->getJson('/admin/system-health')
            ->assertForbidden();
    }

    public function test_health_check_command_outputs_json(): void
    {
        $this->artisan('otokantar:health-check --json')
            ->expectsOutputToContain('"checks"')
            ->assertExitCode(0);
    }

    public function test_support_bundle_contains_health_queue_and_failed_job_files(): void
    {
        $root = storage_path('framework/testing/support-bundles');

        $this->artisan('otokantar:support-bundle', ['--path' => $root])
            ->expectsOutputToContain('Support bundle hazir')
            ->assertExitCode(0);

        $bundles = File::directories($root);
        $this->assertCount(1, $bundles);
        $this->assertFileExists($bundles[0].DIRECTORY_SEPARATOR.'health.json');
        $this->assertFileExists($bundles[0].DIRECTORY_SEPARATOR.'queue.json');
        $this->assertFileExists($bundles[0].DIRECTORY_SEPARATOR.'failed_jobs.json');
        $this->assertFileExists($bundles[0].DIRECTORY_SEPARATOR.'summary.txt');
    }
}
