<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        DB::table('vehicle_passes')
            ->select('legacy_pass_key')
            ->whereNotNull('legacy_pass_key')
            ->groupBy('legacy_pass_key')
            ->havingRaw('COUNT(*) > 1')
            ->orderBy('legacy_pass_key')
            ->chunk(100, function ($duplicates): void {
                foreach ($duplicates as $duplicate) {
                    $ids = DB::table('vehicle_passes')
                        ->where('legacy_pass_key', $duplicate->legacy_pass_key)
                        ->orderBy('id')
                        ->pluck('id')
                        ->all();

                    $idsToClear = array_slice($ids, 1);
                    if ($idsToClear !== []) {
                        DB::table('vehicle_passes')
                            ->whereIn('id', $idsToClear)
                            ->update(['legacy_pass_key' => null]);
                    }
                }
            });

        Schema::table('vehicle_passes', function (Blueprint $table) {
            $table->unique('legacy_pass_key', 'vehicle_passes_legacy_pass_key_unique');
        });
    }

    public function down(): void
    {
        Schema::table('vehicle_passes', function (Blueprint $table) {
            $table->dropUnique('vehicle_passes_legacy_pass_key_unique');
        });
    }
};
