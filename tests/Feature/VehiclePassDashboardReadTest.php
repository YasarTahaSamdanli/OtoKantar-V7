<?php

namespace Tests\Feature;

use App\Models\User;
use App\Models\VehiclePass;
use Illuminate\Foundation\Testing\RefreshDatabase;
use Illuminate\Support\Facades\Cache;
use Tests\TestCase;

class VehiclePassDashboardReadTest extends TestCase
{
    use RefreshDatabase;

    protected function setUp(): void
    {
        parent::setUp();

        Cache::flush();
    }

    public function test_panel_prefers_vehicle_passes_when_records_exist(): void
    {
        $user = User::factory()->create(['role' => 'employee']);
        $this->createVehiclePass([
            'event_id' => '34NEW001-GIRIS-2026-06-23-18-00-00',
            'plate' => '34NEW001',
            'direction' => 'GIRIS',
            'passed_at' => '2026-06-23 18:00:00',
            'entry_weight_kg' => 12450,
            'confidence' => 0.91,
        ]);

        $this->actingAs($user)
            ->getJson('/canli/api?action=panel&limit=10')
            ->assertOk()
            ->assertJsonPath('_source', 'vehicle_passes')
            ->assertJsonPath('toplam', 1)
            ->assertJsonPath('kayitlar.0.plaka', '34NEW001')
            ->assertJsonPath('kayitlar.0.tip', 'GIRIS')
            ->assertJsonPath('kayitlar.0.giris_agirlik', 12450);
    }

    public function test_archive_reads_vehicle_passes_with_filters(): void
    {
        $user = User::factory()->create(['role' => 'employee']);
        $this->createVehiclePass([
            'event_id' => '34NEW001-CIKIS-2026-06-23-18-15-00',
            'plate' => '34NEW001',
            'direction' => 'CIKIS',
            'passed_at' => '2026-06-23 18:15:00',
            'exit_weight_kg' => 8400,
            'net_weight_kg' => 4050,
        ]);
        $this->createVehiclePass([
            'event_id' => '06OLD001-GIRIS-2026-06-22-10-00-00',
            'plate' => '06OLD001',
            'direction' => 'GIRIS',
            'passed_at' => '2026-06-22 10:00:00',
        ]);

        $this->actingAs($user)
            ->getJson('/canli/archive?period=day&date=2026-06-23&plate=34NEW')
            ->assertOk()
            ->assertJsonPath('_source', 'vehicle_passes')
            ->assertJsonPath('toplam', 1)
            ->assertJsonPath('kayitlar.0.plaka', '34NEW001')
            ->assertJsonPath('kayitlar.0.tip', 'CIKIS')
            ->assertJsonPath('kayitlar.0.net_agirlik', 4050);
    }

    public function test_exit_record_exposes_vehicle_and_material_weight_difference(): void
    {
        $user = User::factory()->create(['role' => 'employee']);
        $this->createVehiclePass([
            'event_id' => '34NET001-CIKIS-2026-06-23-18-15-00',
            'plate' => '34NET001',
            'direction' => 'CIKIS',
            'passed_at' => '2026-06-23 18:15:00',
            'entry_weight_kg' => 42000,
            'exit_weight_kg' => 12000,
            'net_weight_kg' => null,
        ]);

        $this->actingAs($user)
            ->getJson('/canli/archive?period=day&date=2026-06-23&plate=34NET')
            ->assertOk()
            ->assertJsonPath('kayitlar.0.arac_agirlik', 12000)
            ->assertJsonPath('kayitlar.0.malzeme_agirlik', -30000)
            ->assertJsonPath('kayitlar.0.net_agirlik', -30000);
    }

    public function test_admin_csv_export_prefers_vehicle_passes_when_records_exist(): void
    {
        $admin = User::factory()->create(['role' => 'admin']);
        $this->createVehiclePass([
            'event_id' => '34CSV001-GIRIS-2026-06-23-19-00-00',
            'plate' => '34CSV001',
            'direction' => 'GIRIS',
            'passed_at' => '2026-06-23 19:00:00',
            'entry_weight_kg' => 10100,
            'confidence' => 0.88,
        ]);

        $response = $this->actingAs($admin)->get('/canli/csv');

        $response->assertOk();
        $content = $response->getContent();
        $this->assertStringContainsString('34CSV001', $content);
        $this->assertStringContainsString('GirisKg', $content);
    }

    private function createVehiclePass(array $attributes): VehiclePass
    {
        return VehiclePass::create(array_merge([
            'plate' => '34TEST34',
            'direction' => 'GIRIS',
            'status' => 'transition',
            'passed_at' => '2026-06-23 12:00:00',
            'source' => 'remote_ingest',
            'legacy_pass_key' => sha1(($attributes['plate'] ?? '34TEST34').'|'.($attributes['direction'] ?? 'GIRIS').'|'.($attributes['passed_at'] ?? '2026-06-23 12:00:00')),
        ], $attributes));
    }
}
