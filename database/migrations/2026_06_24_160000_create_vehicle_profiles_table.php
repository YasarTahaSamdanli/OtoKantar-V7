<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::create('vehicle_profiles', function (Blueprint $table) {
            $table->id();
            $table->string('plate', 20)->unique();
            $table->string('company_name')->nullable();
            $table->string('driver_name')->nullable();
            $table->timestampTz('first_seen_at')->nullable();
            $table->timestampTz('last_seen_at')->nullable();
            $table->unsignedInteger('total_entry_count')->default(0);
            $table->decimal('total_net_weight_kg', 14, 3)->nullable();
            $table->text('notes')->nullable();
            $table->timestamps();

            $table->index('first_seen_at');
            $table->index('last_seen_at');
            $table->index('total_entry_count');
        });

        Schema::table('vehicle_passes', function (Blueprint $table) {
            $table->foreignId('vehicle_profile_id')->nullable()->after('id')->constrained('vehicle_profiles')->nullOnDelete();
            $table->string('vehicle_recognition_status', 20)->nullable()->after('vehicle_profile_id');

            $table->index('vehicle_recognition_status');
        });

        $passes = DB::table('vehicle_passes')
            ->selectRaw('
                plate,
                MIN(passed_at) as first_seen_at,
                MAX(passed_at) as last_seen_at,
                COUNT(*) as total_entry_count,
                SUM(net_weight_kg) as total_net_weight_kg,
                MAX(company_name) as company_name,
                MAX(driver_name) as driver_name
            ')
            ->whereNotNull('plate')
            ->groupBy('plate')
            ->get();

        foreach ($passes as $pass) {
            $profileId = DB::table('vehicle_profiles')->insertGetId([
                'plate' => strtoupper(trim((string) $pass->plate)),
                'company_name' => $pass->company_name,
                'driver_name' => $pass->driver_name,
                'first_seen_at' => $pass->first_seen_at,
                'last_seen_at' => $pass->last_seen_at,
                'total_entry_count' => (int) $pass->total_entry_count,
                'total_net_weight_kg' => $pass->total_net_weight_kg,
                'created_at' => now(),
                'updated_at' => now(),
            ]);

            DB::table('vehicle_passes')
                ->where('plate', $pass->plate)
                ->update([
                    'vehicle_profile_id' => $profileId,
                    'vehicle_recognition_status' => 'TANINAN_ARAC',
                ]);

            $firstPassId = DB::table('vehicle_passes')
                ->where('plate', $pass->plate)
                ->orderBy('passed_at')
                ->orderBy('id')
                ->value('id');

            if ($firstPassId !== null) {
                DB::table('vehicle_passes')
                    ->where('id', $firstPassId)
                    ->update(['vehicle_recognition_status' => 'YENI_ARAC']);
            }
        }
    }

    public function down(): void
    {
        Schema::table('vehicle_passes', function (Blueprint $table) {
            $table->dropConstrainedForeignId('vehicle_profile_id');
            $table->dropColumn('vehicle_recognition_status');
        });

        Schema::dropIfExists('vehicle_profiles');
    }
};
