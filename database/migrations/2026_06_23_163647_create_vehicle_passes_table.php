<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::create('vehicle_passes', function (Blueprint $table) {
            $table->id();
            $table->string('event_id')->nullable()->unique();
            $table->string('plate', 20);
            $table->string('direction', 10);
            $table->string('status', 20)->nullable();
            $table->timestampTz('passed_at');
            $table->timestampTz('entry_at')->nullable();
            $table->timestampTz('exit_at')->nullable();
            $table->decimal('entry_weight_kg', 12, 3)->nullable();
            $table->decimal('exit_weight_kg', 12, 3)->nullable();
            $table->decimal('net_weight_kg', 12, 3)->nullable();
            $table->decimal('scale_weight_kg', 12, 3)->nullable();
            $table->decimal('confidence', 6, 4)->nullable();
            $table->string('snapshot_disk')->nullable();
            $table->string('snapshot_path')->nullable();
            $table->string('snapshot_url')->nullable();
            $table->string('source', 40)->default('unknown');
            $table->json('source_payload')->nullable();
            $table->unsignedBigInteger('legacy_vehicle_id')->nullable();
            $table->string('legacy_pass_key')->nullable();
            $table->boolean('is_blacklisted')->default(false);
            $table->string('operator')->nullable();
            $table->string('company_name')->nullable();
            $table->string('driver_name')->nullable();
            $table->string('driver_phone')->nullable();
            $table->string('material_type')->nullable();
            $table->string('dispatch_no')->nullable();
            $table->timestamps();

            $table->index('plate');
            $table->index('direction');
            $table->index('passed_at');
            $table->index('source');
            $table->index('legacy_pass_key');
            $table->index(['plate', 'passed_at']);
            $table->index(['direction', 'passed_at']);
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('vehicle_passes');
    }
};
