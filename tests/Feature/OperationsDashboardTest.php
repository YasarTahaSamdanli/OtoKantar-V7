<?php

namespace Tests\Feature;

use App\Models\User;
use Illuminate\Foundation\Testing\RefreshDatabase;
use Tests\TestCase;

class OperationsDashboardTest extends TestCase
{
    use RefreshDatabase;

    public function test_admin_can_view_operations_dashboard(): void
    {
        $admin = User::factory()->create(['role' => 'admin']);

        $this->actingAs($admin)
            ->get('/admin/operations')
            ->assertOk()
            ->assertSee('Operations Dashboard')
            ->assertSee('Live ingest pending')
            ->assertSee('Failed job gecmisi');
    }

    public function test_employee_cannot_view_operations_dashboard(): void
    {
        $employee = User::factory()->create(['role' => 'employee']);

        $this->actingAs($employee)
            ->get('/admin/operations')
            ->assertForbidden();
    }
}
