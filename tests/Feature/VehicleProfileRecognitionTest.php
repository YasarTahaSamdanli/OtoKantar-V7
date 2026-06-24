<?php

namespace Tests\Feature;

use App\Models\VehiclePass;
use App\Models\VehicleProfile;
use App\Services\VehicleProfileService;
use Illuminate\Foundation\Testing\RefreshDatabase;
use Tests\TestCase;

class VehicleProfileRecognitionTest extends TestCase
{
    use RefreshDatabase;

    public function test_first_plate_creates_new_vehicle_profile(): void
    {
        $pass = $this->createVehiclePass([
            'plate' => '34ABC123',
            'passed_at' => '2026-06-24 10:00:00',
            'company_name' => 'Test Firma',
            'driver_name' => 'Test Sofor',
            'net_weight_kg' => 12000,
        ]);

        $this->app->make(VehicleProfileService::class)->syncForPass($pass, true);

        $profile = VehicleProfile::where('plate', '34ABC123')->firstOrFail();

        $this->assertSame('YENI_ARAC', $pass->refresh()->vehicle_recognition_status);
        $this->assertSame($profile->id, $pass->vehicle_profile_id);
        $this->assertSame(1, $profile->total_entry_count);
        $this->assertSame('12000.000', $profile->total_net_weight_kg);
        $this->assertSame('Test Firma', $profile->company_name);
    }

    public function test_known_plate_is_marked_recognized_and_counted_once_per_new_pass(): void
    {
        $firstPass = $this->createVehiclePass([
            'event_id' => 'first-pass',
            'plate' => '06OLD001',
            'passed_at' => '2026-06-24 09:00:00',
        ]);
        $service = $this->app->make(VehicleProfileService::class);
        $service->syncForPass($firstPass, true);

        $secondPass = $this->createVehiclePass([
            'event_id' => 'second-pass',
            'plate' => '06OLD001',
            'passed_at' => '2026-06-24 11:00:00',
            'net_weight_kg' => 5000,
        ]);
        $service->syncForPass($secondPass, true);
        $service->syncForPass($secondPass->refresh(), false);

        $profile = VehicleProfile::where('plate', '06OLD001')->firstOrFail();

        $this->assertSame('YENI_ARAC', $firstPass->refresh()->vehicle_recognition_status);
        $this->assertSame('TANINAN_ARAC', $secondPass->refresh()->vehicle_recognition_status);
        $this->assertSame(2, $profile->total_entry_count);
        $this->assertSame('5000.000', $profile->total_net_weight_kg);
    }

    private function createVehiclePass(array $attributes): VehiclePass
    {
        return VehiclePass::create(array_merge([
            'plate' => '34TEST34',
            'direction' => 'GIRIS',
            'status' => 'transition',
            'passed_at' => '2026-06-24 12:00:00',
            'source' => 'test',
            'legacy_pass_key' => sha1(($attributes['plate'] ?? '34TEST34').'|'.($attributes['passed_at'] ?? '2026-06-24 12:00:00')),
        ], $attributes));
    }
}
