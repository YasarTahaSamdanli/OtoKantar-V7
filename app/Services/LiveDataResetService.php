<?php

namespace App\Services;

use Illuminate\Support\Facades\Artisan;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\File;

class LiveDataResetService
{
    public function reset(bool $withCompanies = true): array
    {
        $tables = [
            'vehicle_passes',
            'vehicle_profiles',
            'live_ingest_job_runs',
            'jobs',
            'failed_jobs',
            'job_batches',
            'cache',
            'cache_locks',
        ];

        if ($withCompanies) {
            $tables[] = 'companies';
        }

        DB::transaction(function () use ($tables): void {
            foreach ($tables as $table) {
                if (! DB::getSchemaBuilder()->hasTable($table)) {
                    continue;
                }

                DB::table($table)->delete();
            }
        });

        $runtimeRoot = $this->runtimeRoot();
        File::ensureDirectoryExists($runtimeRoot, 0775);

        foreach (['canli_durum.json', 'canli_kare.jpg', 'kantar_raporu.csv', 'kantar_fisi.txt'] as $name) {
            File::delete($runtimeRoot.DIRECTORY_SEPARATOR.$name);
        }

        foreach (['gecis_gecmisi.jsonl', 'sync_queue.jsonl'] as $name) {
            File::put($runtimeRoot.DIRECTORY_SEPARATOR.$name, '');
        }

        $captures = $runtimeRoot.DIRECTORY_SEPARATOR.'captures';
        if (is_dir($captures)) {
            File::cleanDirectory($captures);
        }

        Artisan::call('cache:clear');

        return [
            'runtime' => $runtimeRoot,
            'tables' => $tables,
        ];
    }

    private function runtimeRoot(): string
    {
        $root = rtrim((string) config('services.legacy_runtime.path', storage_path('app/legacy_python')), '\\/');
        if ($root === '' || ! (str_starts_with($root, '/') || preg_match('/^[A-Za-z]:[\/\\\\]/', $root) === 1)) {
            $root = base_path($root);
        }

        return $root;
    }
}
