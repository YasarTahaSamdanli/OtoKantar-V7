<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::create('live_ingest_job_runs', function (Blueprint $table) {
            $table->id();
            $table->string('event_id')->nullable();
            $table->string('queue', 80)->default('live-ingest');
            $table->string('status', 20);
            $table->unsignedInteger('duration_ms')->default(0);
            $table->string('exception_class')->nullable();
            $table->string('exception_message', 500)->nullable();
            $table->timestamp('started_at')->nullable();
            $table->timestamp('finished_at')->nullable();
            $table->timestamps();

            $table->index(['queue', 'status', 'finished_at']);
            $table->index('event_id');
            $table->index('finished_at');
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('live_ingest_job_runs');
    }
};
