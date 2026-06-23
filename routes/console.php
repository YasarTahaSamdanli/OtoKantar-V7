<?php

use Illuminate\Foundation\Inspiring;
use Illuminate\Support\Facades\Artisan;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Hash;
use App\Models\VehiclePass;
use App\Models\User;

Artisan::command('inspire', function () {
    $this->comment(Inspiring::quote());
})->purpose('Display an inspiring quote');

Artisan::command('otokantar:ensure-admin', function () {
    $email = env('ADMIN_EMAIL');
    $password = env('ADMIN_PASSWORD');
    $name = env('ADMIN_NAME', 'Admin');

    if (!$email || !$password) {
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
        if ($root === '' || !(str_starts_with($root, '/') || preg_match('/^[A-Za-z]:[\/\\\\]/', $root) === 1)) {
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
                if (!is_array($decoded)) {
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
