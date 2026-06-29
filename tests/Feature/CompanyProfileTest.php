<?php

namespace Tests\Feature;

use App\Models\Company;
use App\Models\User;
use App\Models\VehiclePass;
use Illuminate\Foundation\Testing\RefreshDatabase;
use Tests\TestCase;

class CompanyProfileTest extends TestCase
{
    use RefreshDatabase;

    public function test_employee_can_create_company_card(): void
    {
        $user = User::factory()->create(['role' => 'employee']);

        $this->actingAs($user)
            ->post(route('companies.store'), [
                'name' => 'Acme Metal',
                'type' => 'supplier',
                'tax_number' => '1234567890',
                'contact_name' => 'Ali Veli',
                'phone' => '05550000000',
                'email' => 'operasyon@acme.test',
                'is_active' => '1',
            ])
            ->assertRedirect();

        $this->assertDatabaseHas('companies', [
            'name' => 'Acme Metal',
            'type' => 'supplier',
            'is_active' => true,
        ]);
    }

    public function test_company_detail_lists_linked_and_legacy_named_passes(): void
    {
        $user = User::factory()->create(['role' => 'employee']);
        $company = Company::query()->create(['name' => 'Beta Kum', 'type' => 'supplier']);

        $this->createPass([
            'event_id' => 'beta-linked',
            'company_id' => $company->id,
            'company_name' => 'Beta Kum',
            'plate' => '34BET123',
            'passed_at' => '2026-06-25 10:00:00',
            'material_type' => 'Kum',
            'net_weight_kg' => 12000,
        ]);
        $this->createPass([
            'event_id' => 'beta-legacy',
            'company_name' => 'Beta Kum',
            'plate' => '34LEG123',
            'passed_at' => '2026-06-26 10:00:00',
            'material_type' => 'Cakil',
            'net_weight_kg' => 8000,
        ]);
        $this->createPass([
            'event_id' => 'other-company',
            'company_name' => 'Baska Firma',
            'plate' => '06OTH123',
            'passed_at' => '2026-06-26 10:00:00',
            'net_weight_kg' => 5000,
        ]);

        $this->actingAs($user)
            ->get(route('companies.show', $company))
            ->assertOk()
            ->assertSee('34BET123')
            ->assertSee('34LEG123')
            ->assertDontSee('06OTH123')
            ->assertSee('20,00');
    }

    public function test_company_csv_respects_filters(): void
    {
        $user = User::factory()->create(['role' => 'employee']);
        $company = Company::query()->create(['name' => 'Gamma Lojistik', 'type' => 'carrier']);

        $this->createPass([
            'event_id' => 'gamma-in-filter',
            'company_id' => $company->id,
            'company_name' => 'Gamma Lojistik',
            'plate' => '35GAM001',
            'passed_at' => '2026-06-20 09:00:00',
            'material_type' => 'Demir',
        ]);
        $this->createPass([
            'event_id' => 'gamma-out-filter',
            'company_id' => $company->id,
            'company_name' => 'Gamma Lojistik',
            'plate' => '35GAM002',
            'passed_at' => '2026-06-21 09:00:00',
            'material_type' => 'Kum',
        ]);

        $response = $this->actingAs($user)
            ->get(route('companies.csv', [
                'company' => $company,
                'date_from' => '2026-06-20',
                'date_to' => '2026-06-20',
                'material' => 'Demir',
            ]));

        $response->assertOk();
        $content = $response->streamedContent();

        $this->assertStringContainsString('35GAM001', $content);
        $this->assertStringContainsString('Demir', $content);
        $this->assertStringNotContainsString('35GAM002', $content);
    }

    private function createPass(array $overrides = []): VehiclePass
    {
        return VehiclePass::query()->create(array_merge([
            'event_id' => fake()->uuid(),
            'plate' => '34TST123',
            'direction' => 'GIRIS',
            'passed_at' => '2026-06-20 08:00:00',
            'source' => 'test',
        ], $overrides));
    }
}
