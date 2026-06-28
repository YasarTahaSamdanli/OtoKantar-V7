<?php

namespace Tests\Feature;

use App\Models\AuditLog;
use App\Models\User;
use Illuminate\Foundation\Testing\RefreshDatabase;
use Illuminate\Support\Facades\Queue;
use Tests\TestCase;

class AuditLogTest extends TestCase
{
    use RefreshDatabase;

    public function test_successful_login_is_audited(): void
    {
        $user = User::factory()->create(['role' => 'admin']);

        $this->post('/login', [
            'email' => $user->email,
            'password' => 'password',
        ])->assertRedirect(route('dashboard', absolute: false));

        $this->assertDatabaseHas('audit_logs', [
            'user_id' => $user->id,
            'action' => 'auth.login',
        ]);
    }

    public function test_logout_is_audited(): void
    {
        $user = User::factory()->create(['role' => 'employee']);

        $this->actingAs($user)
            ->post('/logout')
            ->assertRedirect('/');

        $this->assertDatabaseHas('audit_logs', [
            'user_id' => $user->id,
            'action' => 'auth.logout',
        ]);
    }

    public function test_admin_user_creation_is_audited(): void
    {
        $admin = User::factory()->create(['role' => 'admin']);

        $this->actingAs($admin)
            ->post(route('admin.users.store'), [
                'name' => 'Audit Employee',
                'email' => 'audit-employee@example.com',
                'password' => 'Password123!',
                'password_confirmation' => 'Password123!',
                'role' => 'employee',
            ])
            ->assertRedirect(route('admin.users.index', absolute: false));

        $createdUser = User::query()->where('email', 'audit-employee@example.com')->firstOrFail();

        $this->assertDatabaseHas('audit_logs', [
            'user_id' => $admin->id,
            'action' => 'admin.user.created',
            'subject_type' => User::class,
            'subject_id' => (string) $createdUser->id,
        ]);
    }

    public function test_only_admin_can_view_audit_logs(): void
    {
        $employee = User::factory()->create(['role' => 'employee']);
        $admin = User::factory()->create(['role' => 'admin']);

        AuditLog::create([
            'user_id' => $admin->id,
            'action' => 'auth.login',
            'created_at' => now(),
        ]);

        $this->actingAs($employee)
            ->get(route('admin.audit-logs.index'))
            ->assertForbidden();

        $this->actingAs($admin)
            ->get(route('admin.audit-logs.index'))
            ->assertOk()
            ->assertSee('Audit Log')
            ->assertSee('auth.login');
    }

    public function test_successful_live_ingest_is_not_written_to_audit_log(): void
    {
        Queue::fake();
        config(['services.legacy_runtime.api_token' => 'test-token']);

        $this->withHeader('Authorization', 'Bearer test-token')
            ->postJson('/api/live-ingest', [
                'son_guncelleme' => now()->toIso8601String(),
                'kantar_kg' => 12000,
                'kantar_sabit' => true,
            ])
            ->assertOk();

        $this->assertDatabaseMissing('audit_logs', [
            'action' => 'live_ingest.accepted',
        ]);
    }

    public function test_admin_can_delete_live_ingest_accepted_noise_only(): void
    {
        $admin = User::factory()->create(['role' => 'admin']);

        AuditLog::create(['action' => 'live_ingest.accepted', 'created_at' => now()]);
        AuditLog::create(['action' => 'live_ingest.accepted', 'created_at' => now()]);
        AuditLog::create(['action' => 'auth.login', 'user_id' => $admin->id, 'created_at' => now()]);
        AuditLog::create(['action' => 'live_ingest.rejected', 'created_at' => now()]);

        $this->actingAs($admin)
            ->delete(route('admin.audit-logs.destroy-live-ingest-accepted'))
            ->assertRedirect(route('admin.audit-logs.index', absolute: false));

        $this->assertDatabaseMissing('audit_logs', ['action' => 'live_ingest.accepted']);
        $this->assertDatabaseHas('audit_logs', ['action' => 'auth.login']);
        $this->assertDatabaseHas('audit_logs', ['action' => 'live_ingest.rejected']);
        $this->assertDatabaseHas('audit_logs', [
            'user_id' => $admin->id,
            'action' => 'admin.audit_logs.cleaned',
        ]);
    }

    public function test_employee_cannot_delete_audit_log_noise(): void
    {
        $employee = User::factory()->create(['role' => 'employee']);

        AuditLog::create(['action' => 'live_ingest.accepted', 'created_at' => now()]);

        $this->actingAs($employee)
            ->delete(route('admin.audit-logs.destroy-live-ingest-accepted'))
            ->assertForbidden();

        $this->assertDatabaseHas('audit_logs', ['action' => 'live_ingest.accepted']);
    }
}
