<?php

namespace Tests\Feature;

use App\Models\User;
use App\Models\VehicleProfile;
use Carbon\Carbon;
use Illuminate\Foundation\Testing\RefreshDatabase;
use Tests\TestCase;

class VehicleProfileDashboardTest extends TestCase
{
    use RefreshDatabase;

    protected function tearDown(): void
    {
        Carbon::setTestNow();

        parent::tearDown();
    }

    public function test_dashboard_profile_lists_are_scoped_to_the_expected_vehicles(): void
    {
        Carbon::setTestNow('2026-06-29 15:00:00');

        $user = User::factory()->create(['role' => 'admin']);

        $this->createProfile('34TODAY01', '2026-06-29 10:00:00', 1);
        $this->createProfile('34TODAY02', '2026-06-29 11:00:00', 2);
        $this->createProfile('34TODAY03', '2026-06-29 12:00:00', 3);
        $this->createProfile('34TODAY04', '2026-06-29 13:00:00', 4);

        $this->createProfile('34FREQ01', '2026-06-20 09:00:00', 40);
        $this->createProfile('34FREQ02', '2026-06-20 08:00:00', 30);
        $this->createProfile('34FREQ03', '2026-06-20 07:00:00', 20);
        $this->createProfile('34FREQ04', '2026-06-20 06:00:00', 10);

        $response = $this->actingAs($user)->get('/dashboard');

        $response->assertOk();
        $response->assertSee('34TODAY04');
        $response->assertSee('34TODAY03');
        $response->assertSee('34TODAY02');
        $response->assertSee('34FREQ01');
        $response->assertSee('34FREQ02');
        $response->assertSee('34FREQ03');
        $response->assertDontSee('34TODAY01');
        $response->assertDontSee('34FREQ04');
    }

    public function test_new_vehicle_tab_only_lists_profiles_first_seen_today(): void
    {
        Carbon::setTestNow('2026-06-29 15:00:00');

        $user = User::factory()->create(['role' => 'admin']);

        $this->createProfile('34TODAY01', '2026-06-29 10:00:00', 1);
        $this->createProfile('34OLD001', '2026-06-28 10:00:00', 1);

        $this->actingAs($user)
            ->get(route('vehicle-profiles.index', ['tab' => 'new'], absolute: false))
            ->assertOk()
            ->assertSee('34TODAY01')
            ->assertDontSee('34OLD001');
    }

    private function createProfile(string $plate, string $seenAt, int $entryCount): VehicleProfile
    {
        return VehicleProfile::create([
            'plate' => $plate,
            'first_seen_at' => $seenAt,
            'last_seen_at' => $seenAt,
            'total_entry_count' => $entryCount,
        ]);
    }
}
