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
                ],
            ])
            ->assertJsonPath('checks.runtime.exists', true);
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
