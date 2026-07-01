<?php

namespace Tests\Feature;

use App\Models\User;
use App\Models\VehiclePass;
use Illuminate\Foundation\Testing\RefreshDatabase;
use Tests\TestCase;

class AdminUserManagementTest extends TestCase
{
    use RefreshDatabase;

    public function test_admin_can_create_an_admin_user(): void
    {
        $admin = User::factory()->create(['role' => 'admin']);

        $this->actingAs($admin)
            ->post(route('admin.users.store'), [
                'name' => 'Second Admin',
                'email' => 'second-admin@example.com',
                'password' => 'Password123!',
                'password_confirmation' => 'Password123!',
                'role' => 'admin',
            ])
            ->assertRedirect(route('admin.users.index', absolute: false));

        $this->assertDatabaseHas('users', [
            'email' => 'second-admin@example.com',
            'role' => 'admin',
        ]);
    }

    public function test_user_role_must_be_valid_when_created_by_admin(): void
    {
        $admin = User::factory()->create(['role' => 'admin']);

        $this->actingAs($admin)
            ->post(route('admin.users.store'), [
                'name' => 'Bad Role',
                'email' => 'bad-role@example.com',
                'password' => 'Password123!',
                'password_confirmation' => 'Password123!',
                'role' => 'owner',
            ])
            ->assertSessionHasErrors('role');
    }

    public function test_admin_can_reset_live_data_from_operations_page(): void
    {
        $admin = User::factory()->create(['role' => 'admin']);
        VehiclePass::create([
            'plate' => '34RESET',
            'direction' => 'GIRIS',
            'status' => 'transition',
            'passed_at' => '2026-06-23 12:00:00',
            'source' => 'test',
            'legacy_pass_key' => 'reset-test-key',
        ]);

        $this->actingAs($admin)
            ->post(route('admin.operations.reset-live-data'), ['confirm' => 'SIFIRLA'])
            ->assertRedirect(route('admin.operations.index', absolute: false))
            ->assertSessionHas('status');

        $this->assertDatabaseCount('vehicle_passes', 0);
    }

    public function test_employee_cannot_reset_live_data(): void
    {
        $employee = User::factory()->create(['role' => 'employee']);

        $this->actingAs($employee)
            ->post(route('admin.operations.reset-live-data'), ['confirm' => 'SIFIRLA'])
            ->assertForbidden();
    }
}
