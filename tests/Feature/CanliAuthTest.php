<?php

namespace Tests\Feature;

use App\Models\User;
use Illuminate\Foundation\Testing\RefreshDatabase;
use Tests\TestCase;

class CanliAuthTest extends TestCase
{
    use RefreshDatabase;

    public function test_guest_cannot_open_canli_panel(): void
    {
        $this->get('/')
            ->assertRedirect(route('login', absolute: false));

        $this->get('/canli')
            ->assertRedirect(route('login', absolute: false));
    }

    public function test_guest_cannot_open_legacy_canli_entrypoints(): void
    {
        $this->get('/index.php')
            ->assertRedirect(route('login', absolute: false));

        $this->get('/api_canli.php?action=panel')
            ->assertRedirect(route('login', absolute: false));

        $this->get('/canli_kare.jpg')
            ->assertRedirect(route('login', absolute: false));
    }

    public function test_employee_user_can_open_canli_panel_and_summary_but_not_admin_pages(): void
    {
        $user = User::factory()->create(['role' => 'employee']);

        $this->actingAs($user)
            ->get('/')
            ->assertRedirect(route('canli.view', absolute: false));

        $this->actingAs($user)
            ->get('/canli')
            ->assertOk();

        $this->actingAs($user)
            ->get('/dashboard')
            ->assertOk();

        $this->actingAs($user)
            ->get('/admin/users')
            ->assertForbidden();
    }

    public function test_unknown_role_cannot_open_canli_panel(): void
    {
        $user = User::factory()->create(['role' => 'user']);

        $this->actingAs($user)
            ->get('/canli')
            ->assertForbidden();
    }

    public function test_admin_user_can_open_canli_panel(): void
    {
        $admin = User::factory()->create(['role' => 'admin']);

        $this->actingAs($admin)
            ->get('/')
            ->assertRedirect(route('dashboard', absolute: false));

        $this->actingAs($admin)
            ->get('/canli')
            ->assertOk();
    }
}
