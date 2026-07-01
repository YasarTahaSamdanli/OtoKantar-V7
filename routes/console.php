<?php

use App\Models\User;
use App\Models\VehiclePass;
use App\Services\HealthAlertService;
use App\Services\LiveDataResetService;
use App\Services\ProjectBackupService;
use App\Services\QueueOperationsService;
use App\Services\SystemHealthService;
use Illuminate\Foundation\Inspiring;
use Illuminate\Support\Facades\Artisan;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Hash;

Artisan::command('inspire', function () {
    $this->comment(Inspiring::quote());
})->purpose('Display an inspiring quote');

Artisan::command('otokantar:ensure-admin', function () {
    $email = env('ADMIN_EMAIL');
    $password = env('ADMIN_PASSWORD');
    $name = env('ADMIN_NAME', 'Admin');

    if (! $email || ! $password) {
        $this->warn('ADMIN_EMAIL veya ADMIN_PASSWORD tanimli degil; admin olusturulmadi.');

        return 0;
    }

    User::updateOrCreate(
        ['email' => $email],
        [
            'name' => $name,
            'password' => Hash::make($password),
            'role' => 'admin',
        ]
    );

    $this->info('Admin kullanici hazir: '.$email);

    return 0;
})->purpose('Create or update the production admin user from environment variables');

Artisan::command('otokantar:reset-live-data {--force : Required confirmation flag}', function (LiveDataResetService $reset) {
    if (! (bool) $this->option('force')) {
        $this->error('Bu komut canli gecis ve arac verilerini siler. Calistirmak icin --force ekle.');

        return 1;
    }

    $result = $reset->reset();

    $this->info('Canli test verileri sifirlandi.');
    $this->line('Runtime: '.$result['runtime']);
    $this->line('Silinen tablolar: '.implode(', ', $result['tables']));

    return 0;
})->purpose('Delete live vehicle records and runtime files for a clean test run');

Artisan::command('vehicle-passes:latest {--limit=10 : Number of latest records to show}', function () {
    $limit = max(1, min(100, (int) $this->option('limit')));
    $total = VehiclePass::count();
    $latest = VehiclePass::query()
        ->latest('passed_at')
        ->limit($limit)
        ->get([
            'id',
            'event_id',
            'plate',
            'direction',
            'passed_at',
            'entry_weight_kg',
            'exit_weight_kg',
            'net_weight_kg',
            'confidence',
            'source',
            'legacy_pass_key',
        ]);

    $this->info('VehiclePass toplam kayit: '.$total);

    if ($latest->isEmpty()) {
        $this->warn('VehiclePass kaydi bulunamadi.');

        return 0;
    }

    $this->table(
        ['ID', 'Plaka', 'Yon', 'Gecis Zamani', 'Giris Kg', 'Cikis Kg', 'Net Kg', 'Guven', 'Kaynak', 'Event/Legacy Key'],
        $latest->map(fn (VehiclePass $pass): array => [
            $pass->id,
            $pass->plate,
            $pass->direction,
            optional($pass->passed_at)->toDateTimeString(),
            $pass->entry_weight_kg,
            $pass->exit_weight_kg,
            $pass->net_weight_kg,
            $pass->confidence,
            $pass->source,
            $pass->event_id ?: $pass->legacy_pass_key,
        ])->all()
    );

    return 0;
})->purpose('Show latest central vehicle pass records');

Artisan::command('vehicle-passes:verify {--sample=10 : Number of missing/extra records to show}', function () {
    $sample = max(1, min(100, (int) $this->option('sample')));
    $now = now();
    $last24h = $now->copy()->subDay();
    $last7d = $now->copy()->subDays(7);

    $normalizeDirection = function (mixed $value): string {
        $raw = strtoupper(trim((string) $value));

        return str_contains($raw, 'CIKIS') || str_contains($raw, 'TAMAMLANDI') ? 'CIKIS' : 'GIRIS';
    };

    $normalizeDateTime = function (mixed $value): ?string {
        if ($value === null || trim((string) $value) === '') {
            return null;
        }

        $timestamp = strtotime((string) $value);

        return $timestamp === false ? null : date('Y-m-d H:i:s', $timestamp);
    };

    $recordKey = function (string $plate, string $direction, string $passedAt): string {
        return strtoupper(trim($plate)).'|'.$direction.'|'.$passedAt;
    };

    $runtimePath = function (string $name): string {
        $root = rtrim((string) config('services.legacy_runtime.path', base_path('legacy')), '\\/');
        if ($root === '' || ! (str_starts_with($root, '/') || preg_match('/^[A-Za-z]:[\/\\\\]/', $root) === 1)) {
            $root = base_path($root);
        }

        return $root.DIRECTORY_SEPARATOR.$name;
    };

    $vehiclePasses = VehiclePass::query()
        ->where('passed_at', '>=', $last7d)
        ->orderByDesc('passed_at')
        ->get(['id', 'plate', 'direction', 'passed_at', 'source', 'event_id', 'legacy_pass_key'])
        ->map(function (VehiclePass $pass) use ($recordKey): array {
            $passedAt = optional($pass->passed_at)->toDateTimeString();

            return [
                'key' => $passedAt ? $recordKey((string) $pass->plate, (string) $pass->direction, $passedAt) : '',
                'plate' => (string) $pass->plate,
                'direction' => (string) $pass->direction,
                'passed_at' => $passedAt,
                'source' => (string) $pass->source,
                'ref' => $pass->event_id ?: $pass->legacy_pass_key ?: ('id:'.$pass->id),
            ];
        })
        ->filter(fn (array $row): bool => $row['key'] !== '')
        ->values();

    $dashboardSource = 'legacy_db';
    $dashboardRecords = collect();

    try {
        $pdo = DB::connection('legacy')->getPdo();
        $stmt = $pdo->prepare(
            'SELECT a.plaka, g.yon, g.gecis_zamani
             FROM gecisler g
             INNER JOIN araclar a ON a.id = g.id
             WHERE g.gecis_zamani >= :cutoff
             ORDER BY g.gecis_zamani DESC'
        );
        $stmt->execute(['cutoff' => $last7d->toDateTimeString()]);
        $dashboardRecords = collect($stmt->fetchAll(PDO::FETCH_ASSOC))
            ->map(function (array $row) use ($normalizeDirection, $normalizeDateTime, $recordKey): array {
                $plate = strtoupper(trim((string) ($row['plaka'] ?? '')));
                $direction = $normalizeDirection($row['yon'] ?? 'GIRIS');
                $passedAt = $normalizeDateTime($row['gecis_zamani'] ?? null);

                return [
                    'key' => ($plate !== '' && $passedAt !== null) ? $recordKey($plate, $direction, $passedAt) : '',
                    'plate' => $plate,
                    'direction' => $direction,
                    'passed_at' => $passedAt,
                    'source' => 'legacy_db',
                    'ref' => 'legacy_db',
                ];
            })
            ->filter(fn (array $row): bool => $row['key'] !== '')
            ->values();
    } catch (Throwable $e) {
        $dashboardSource = 'jsonl+csv';
        $rows = [];

        $jsonl = $runtimePath('gecis_gecmisi.jsonl');
        if (is_file($jsonl)) {
            foreach (file($jsonl, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES) ?: [] as $line) {
                $decoded = json_decode((string) $line, true);
                if (! is_array($decoded)) {
                    continue;
                }

                $plate = strtoupper(trim((string) ($decoded['plaka'] ?? '')));
                $direction = $normalizeDirection($decoded['tip'] ?? $decoded['durum'] ?? $decoded['yon'] ?? 'GIRIS');
                $passedAt = $normalizeDateTime($decoded['gecis_zamani'] ?? null);
                if ($plate === '' || $passedAt === null || strtotime($passedAt) < $last7d->timestamp) {
                    continue;
                }

                $rows[] = [
                    'key' => $recordKey($plate, $direction, $passedAt),
                    'plate' => $plate,
                    'direction' => $direction,
                    'passed_at' => $passedAt,
                    'source' => 'jsonl',
                    'ref' => (string) ($decoded['_event_id'] ?? 'jsonl'),
                ];
            }
        }

        $csv = $runtimePath('kantar_raporu.csv');
        if (is_file($csv) && ($handle = fopen($csv, 'r')) !== false) {
            try {
                $header = fgetcsv($handle, 0, ';') ?: [];
                $header = array_map(fn ($value): string => preg_replace('/^\xEF\xBB\xBF/', '', trim((string) $value)) ?? '', $header);

                while (($line = fgetcsv($handle, 0, ';')) !== false) {
                    $row = [];
                    foreach ($header as $index => $key) {
                        $row[$key] = $line[$index] ?? '';
                    }

                    $plate = strtoupper(trim((string) ($row['Plaka'] ?? '')));
                    $direction = $normalizeDirection($row['Durum'] ?? 'GIRIS');
                    $date = $direction === 'CIKIS'
                        ? (string) ($row['CikisTarih'] ?? $row['GirisTarih'] ?? '')
                        : (string) ($row['GirisTarih'] ?? $row['CikisTarih'] ?? '');
                    $time = $direction === 'CIKIS'
                        ? (string) ($row['CikisSaat'] ?? $row['GirisSaat'] ?? '')
                        : (string) ($row['GirisSaat'] ?? $row['CikisSaat'] ?? '');
                    $passedAt = $normalizeDateTime(trim($date.' '.$time));
                    if ($plate === '' || $passedAt === null || strtotime($passedAt) < $last7d->timestamp) {
                        continue;
                    }

                    $rows[] = [
                        'key' => $recordKey($plate, $direction, $passedAt),
                        'plate' => $plate,
                        'direction' => $direction,
                        'passed_at' => $passedAt,
                        'source' => 'csv',
                        'ref' => 'csv',
                    ];
                }
            } finally {
                fclose($handle);
            }
        }

        $seen = [];
        $dashboardRecords = collect($rows)
            ->filter(function (array $row) use (&$seen): bool {
                if (isset($seen[$row['key']])) {
                    return false;
                }
                $seen[$row['key']] = true;

                return true;
            })
            ->sortByDesc('passed_at')
            ->values();
    }

    $vpKeys = $vehiclePasses->keyBy('key');
    $dashboardKeys = $dashboardRecords->keyBy('key');
    $missing = $dashboardRecords->reject(fn (array $row): bool => $vpKeys->has($row['key']))->values();
    $extra = $vehiclePasses->reject(fn (array $row): bool => $dashboardKeys->has($row['key']))->values();
    $vpLast = $vehiclePasses->first();
    $dashboardLast = $dashboardRecords->first();

    $this->info('VehiclePass shadow verification');
    $this->line('Dashboard source: '.$dashboardSource);
    $this->line('Window: '.$last7d->toDateTimeString().' -> '.$now->toDateTimeString());

    $this->table(
        ['Metrik', 'VehiclePass', 'Dashboard Source', 'Fark'],
        [
            [
                'Son 24 saat',
                $vehiclePasses->filter(fn (array $row): bool => strtotime((string) $row['passed_at']) >= $last24h->timestamp)->count(),
                $dashboardRecords->filter(fn (array $row): bool => strtotime((string) $row['passed_at']) >= $last24h->timestamp)->count(),
                $vehiclePasses->filter(fn (array $row): bool => strtotime((string) $row['passed_at']) >= $last24h->timestamp)->count()
                    - $dashboardRecords->filter(fn (array $row): bool => strtotime((string) $row['passed_at']) >= $last24h->timestamp)->count(),
            ],
            [
                'Son 7 gun',
                $vehiclePasses->count(),
                $dashboardRecords->count(),
                $vehiclePasses->count() - $dashboardRecords->count(),
            ],
            [
                'Eksik kayit',
                '-',
                $missing->count(),
                'Dashboard var / VehiclePass yok',
            ],
            [
                'Fazla kayit',
                $extra->count(),
                '-',
                'VehiclePass var / Dashboard yok',
            ],
        ]
    );

    $this->table(
        ['Kaynak', 'Plaka', 'Yon', 'Gecis Zamani', 'Ref'],
        [
            ['VehiclePass son', $vpLast['plate'] ?? '-', $vpLast['direction'] ?? '-', $vpLast['passed_at'] ?? '-', $vpLast['ref'] ?? '-'],
            ['Dashboard son', $dashboardLast['plate'] ?? '-', $dashboardLast['direction'] ?? '-', $dashboardLast['passed_at'] ?? '-', $dashboardLast['ref'] ?? '-'],
        ]
    );

    if ($missing->isNotEmpty()) {
        $this->warn('Eksik kayit ornekleri');
        $this->table(
            ['Plaka', 'Yon', 'Gecis Zamani', 'Kaynak', 'Ref'],
            $missing->take($sample)->map(fn (array $row): array => [
                $row['plate'], $row['direction'], $row['passed_at'], $row['source'], $row['ref'],
            ])->all()
        );
    }

    if ($extra->isNotEmpty()) {
        $this->warn('Fazla kayit ornekleri');
        $this->table(
            ['Plaka', 'Yon', 'Gecis Zamani', 'Kaynak', 'Ref'],
            $extra->take($sample)->map(fn (array $row): array => [
                $row['plate'], $row['direction'], $row['passed_at'], $row['source'], $row['ref'],
            ])->all()
        );
    }

    if ($missing->isEmpty() && $extra->isEmpty()) {
        $this->info('VehiclePass ve dashboard kaynagi son 7 gun icin uyumlu gorunuyor.');
    }

    return $missing->isEmpty() && $extra->isEmpty() ? 0 : 1;
})->purpose('Compare VehiclePass records with current dashboard data sources');

Artisan::command('otokantar:health-check {--json : Output raw JSON}', function (SystemHealthService $health) {
    $report = $health->report();
    $summary = function (string $name, array $check): string {
        return match ($name) {
            'database' => (string) ($check['connection'] ?? '-'),
            'runtime' => (string) ($check['path'] ?? '-'),
            'queue' => 'pending='.($check['pending_jobs'] ?? '?').', failed='.($check['failed_jobs'] ?? '?'),
            'disk' => 'free_mb='.($check['free_mb'] ?? '?'),
            'backup' => 'latest='.($check['latest_backup'] ?? '-'),
            'ingest' => 'latest='.($check['latest_vehicle_pass']['plate'] ?? '-'),
            default => (string) ($check['env'] ?? $check['message'] ?? '-'),
        };
    };

    if ((bool) $this->option('json')) {
        $this->line(json_encode($report, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE) ?: '{}');

        return ($report['status'] ?? 'critical') === 'critical' ? 1 : 0;
    }

    $this->info('OtoKantar sistem durumu: '.strtoupper((string) ($report['status'] ?? 'unknown')));
    $this->line('Uretim zamani: '.($report['generated_at'] ?? '-'));

    $rows = [];
    foreach (($report['checks'] ?? []) as $name => $check) {
        $rows[] = [
            $name,
            strtoupper((string) ($check['status'] ?? 'unknown')),
            $summary((string) $name, is_array($check) ? $check : []),
        ];
    }

    $this->table(['Kontrol', 'Durum', 'Ozet'], $rows);

    return ($report['status'] ?? 'critical') === 'critical' ? 1 : 0;
})->purpose('Show product health diagnostics for support and monitoring');

Artisan::command('otokantar:health-alerts {--force : Send even when alerts are disabled or health is OK} {--dry-run : Show alert payload without sending}', function (HealthAlertService $alerts) {
    $result = $alerts->checkAndNotify(
        force: (bool) $this->option('force'),
        dryRun: (bool) $this->option('dry-run'),
    );

    $this->line(json_encode([
        'status' => $result['status'] ?? 'unknown',
        'reason' => $result['reason'] ?? null,
        'overall_status' => $result['payload']['overall_status'] ?? null,
        'health_score' => $result['payload']['health_score'] ?? null,
        'queue' => $result['payload']['queue'] ?? [],
        'problems' => $result['payload']['problems'] ?? [],
        'sent' => $result['sent'] ?? [],
    ], JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE) ?: '{}');

    return in_array($result['status'] ?? null, ['sent', 'skipped', 'dry-run', 'not_configured'], true) ? 0 : 1;
})->purpose('Send n8n/Telegram alerts from the current health report');

Artisan::command('otokantar:support-bundle {--path= : Bundle destination root}', function (SystemHealthService $health, QueueOperationsService $queue) {
    $root = $this->option('path') ?: storage_path('app/support-bundles');
    if (! (str_starts_with((string) $root, '/') || preg_match('/^[A-Za-z]:[\/\\\\]/', (string) $root) === 1)) {
        $root = base_path((string) $root);
    }

    $bundleDir = rtrim((string) $root, '\\/').DIRECTORY_SEPARATOR.'support_'.now()->format('Ymd_His');
    \Illuminate\Support\Facades\File::ensureDirectoryExists($bundleDir, 0750);

    $healthReport = $health->report();
    $queueReport = $queue->supportSummary();
    $failedJobs = $queueReport['failed_job_sample'] ?? [];

    file_put_contents($bundleDir.DIRECTORY_SEPARATOR.'health.json', json_encode($healthReport, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE));
    file_put_contents($bundleDir.DIRECTORY_SEPARATOR.'queue.json', json_encode($queueReport, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE));
    file_put_contents($bundleDir.DIRECTORY_SEPARATOR.'failed_jobs.json', json_encode($failedJobs, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE));
    file_put_contents($bundleDir.DIRECTORY_SEPARATOR.'summary.txt', implode(PHP_EOL, [
        'OtoKantar Support Bundle',
        'Generated at: '.($healthReport['generated_at'] ?? now()->toIso8601String()),
        'Overall status: '.($healthReport['overall_status'] ?? 'Unknown'),
        'Health score: '.($healthReport['health_score'] ?? 'n/a'),
        'Queue status: '.($queueReport['status'] ?? 'Unknown'),
        'Pending jobs: '.($queueReport['metrics']['pending_jobs'] ?? 'n/a'),
        'Failed jobs: '.($queueReport['metrics']['failed_jobs'] ?? 'n/a'),
        'Pending temp images: '.($queueReport['metrics']['pending_temp_images'] ?? 'n/a'),
        '',
    ]));

    $this->info('Support bundle hazir: '.$bundleDir);

    return 0;
})->purpose('Create a support bundle with health, queue and failed job summaries');

Artisan::command('backup:run {--connection= : Database connection to dump} {--path= : Backup destination directory} {--keep= : Number of backup folders to keep} {--without-legacy-runtime : Skip legacy runtime file archive} {--sync-remote : Copy this backup to configured remote storage after creation}', function () {
    $manifest = app(ProjectBackupService::class)->run(
        connection: $this->option('connection') ?: null,
        destination: $this->option('path') ?: null,
        includeLegacyRuntime: ! (bool) $this->option('without-legacy-runtime'),
        keep: $this->option('keep') !== null ? (int) $this->option('keep') : null,
        syncRemote: (bool) $this->option('sync-remote'),
    );

    $this->info('Backup hazir: '.$manifest['backup_dir']);

    $database = is_array($manifest['database'] ?? null) ? $manifest['database'] : [];
    $this->line('Database: '.($database['status'] ?? 'unknown').' '.($database['file'] ?? ''));

    $runtime = is_array($manifest['legacy_runtime'] ?? null) ? $manifest['legacy_runtime'] : [];
    if ($runtime !== []) {
        $this->line('Legacy runtime: '.($runtime['status'] ?? 'unknown').' '.($runtime['file'] ?? ''));
    }

    foreach (($manifest['warnings'] ?? []) as $warning) {
        $this->warn((string) $warning);
    }

    return 0;
})->purpose('Create a local database and runtime backup bundle');

Artisan::command('backup:list {--path= : Backup root directory}', function () {
    $backups = app(ProjectBackupService::class)->list($this->option('path') ?: null);

    if ($backups === []) {
        $this->warn('Backup bulunamadi.');

        return 0;
    }

    $this->table(
        ['Ad', 'Tarih', 'Database', 'Runtime', 'MB', 'Path'],
        collect($backups)->map(fn (array $backup): array => [
            $backup['name'],
            $backup['created_at'],
            $backup['database'],
            $backup['runtime'],
            number_format(((int) $backup['bytes']) / 1024 / 1024, 2),
            $backup['path'],
        ])->all()
    );

    return 0;
})->purpose('List available local backup bundles');

Artisan::command('backup:restore {--backup=latest : Backup folder name, full path, or latest} {--connection= : Target database connection} {--database-only : Restore only database, skip legacy runtime files} {--force : Required confirmation flag for destructive restore}', function () {
    if (! (bool) $this->option('force')) {
        $this->error('Restore mevcut database/runtime dosyalarini ezebilir. Calistirmak icin --force ekle.');

        return 1;
    }

    $result = app(ProjectBackupService::class)->restore(
        backup: (string) $this->option('backup'),
        connection: $this->option('connection') ?: null,
        restoreRuntime: ! (bool) $this->option('database-only'),
    );

    $this->info('Restore tamamlandi: '.$result['backup_dir']);
    $database = is_array($result['database'] ?? null) ? $result['database'] : [];
    $this->line('Database: '.($database['status'] ?? 'unknown').' '.($database['target'] ?? ''));

    $runtime = is_array($result['legacy_runtime'] ?? null) ? $result['legacy_runtime'] : [];
    if ($runtime !== []) {
        $this->line('Legacy runtime: '.($runtime['status'] ?? 'unknown').' '.($runtime['target'] ?? ($runtime['reason'] ?? '')));
    }

    return 0;
})->purpose('Restore database and runtime files from a local backup bundle');

Artisan::command('backup:sync {backup=latest : Backup folder name, full path, or latest}', function () {
    if (! (bool) config('backup.remote.enabled')) {
        $this->error('Remote backup kapali. BACKUP_REMOTE_ENABLED=true ve BACKUP_REMOTE_DESTINATION ayarla.');

        return 1;
    }

    $result = app(ProjectBackupService::class)->sync((string) $this->argument('backup'));

    $this->info('Remote backup sync tamamlandi.');
    $this->line('Source: '.$result['source']);
    $this->line('Destination: '.$result['destination']);

    return 0;
})->purpose('Copy a local backup bundle to configured remote storage');
