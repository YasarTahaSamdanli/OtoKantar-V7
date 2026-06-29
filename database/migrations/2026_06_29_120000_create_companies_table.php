<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Schema;
use Illuminate\Support\Str;

return new class extends Migration
{
    public function up(): void
    {
        Schema::create('companies', function (Blueprint $table) {
            $table->id();
            $table->string('name')->unique();
            $table->string('type', 30)->default('supplier');
            $table->string('tax_number')->nullable();
            $table->string('contact_name')->nullable();
            $table->string('phone')->nullable();
            $table->string('email')->nullable();
            $table->boolean('is_active')->default(true);
            $table->text('notes')->nullable();
            $table->timestamps();

            $table->index('type');
            $table->index('is_active');
        });

        Schema::table('vehicle_profiles', function (Blueprint $table) {
            $table->foreignId('company_id')->nullable()->after('plate')->constrained('companies')->nullOnDelete();
        });

        Schema::table('vehicle_passes', function (Blueprint $table) {
            $table->foreignId('company_id')->nullable()->after('vehicle_profile_id')->constrained('companies')->nullOnDelete();
        });

        $names = DB::table('vehicle_passes')
            ->whereNotNull('company_name')
            ->pluck('company_name')
            ->merge(DB::table('vehicle_profiles')->whereNotNull('company_name')->pluck('company_name'))
            ->map(fn ($name) => trim((string) $name))
            ->filter()
            ->unique(fn ($name) => Str::lower($name))
            ->values();

        foreach ($names as $name) {
            $companyId = DB::table('companies')->insertGetId([
                'name' => $name,
                'type' => 'supplier',
                'created_at' => now(),
                'updated_at' => now(),
            ]);

            DB::table('vehicle_profiles')
                ->whereNull('company_id')
                ->where('company_name', $name)
                ->update(['company_id' => $companyId]);

            DB::table('vehicle_passes')
                ->whereNull('company_id')
                ->where('company_name', $name)
                ->update(['company_id' => $companyId]);
        }
    }

    public function down(): void
    {
        Schema::table('vehicle_passes', function (Blueprint $table) {
            $table->dropConstrainedForeignId('company_id');
        });

        Schema::table('vehicle_profiles', function (Blueprint $table) {
            $table->dropConstrainedForeignId('company_id');
        });

        Schema::dropIfExists('companies');
    }
};
