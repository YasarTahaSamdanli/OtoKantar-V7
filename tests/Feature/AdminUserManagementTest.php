<?php

namespace Tests\Feature;

use App\Models\User;
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
}
