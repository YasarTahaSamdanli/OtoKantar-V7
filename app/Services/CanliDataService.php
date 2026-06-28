<?php

namespace App\Services;

use App\Models\VehiclePass;
use PDO;
use Throwable;

class CanliDataService
{
    public function __construct(
        private readonly CanliAgirlikService $agirlikService,
    ) {}

    public function legacyPath(string $name): string
    {
        $root = rtrim((string) config('services.legacy_runtime.path', base_path('legacy')), '\\/');
        if (!$this->isAbsolutePath($root)) {
            $root = base_path($root);
        }

        return $root.DIRECTORY_SEPARATOR.$name;
    }

    public function durumOkuVeyaFallback(?PDO $pdo = null): array
    {
        $jsonDurumDosya = $this->legacyPath('canli_durum.json');

        if (is_file($jsonDurumDosya)) {
            $raw = file_get_contents($jsonDurumDosya);
            if ($raw !== false) {
                $json = json_decode($raw, true);
                if (is_array($json)) {
                    $sonGuncelleme = isset($json['son_guncelleme']) ? strtotime((string) $json['son_guncelleme']) : false;
                    $json['_sunucu_zaman'] = date('Y-m-d\TH:i:s');
                    $json['_dosya_mtime'] = date('Y-m-d\TH:i:s', (int) filemtime($jsonDurumDosya));
                    $json['_durum_yasi_saniye'] = $sonGuncelleme ? max(0, time() - $sonGuncelleme) : null;
                    if (!isset($json['sistem']) || !is_array($json['sistem'])) {
                        $json['sistem'] = [];
                    }
                    $json['sistem']['mimari'] = 'MySQL + JSON durum';
                    return $json;
                }
            }
        }

        if ($pdo !== null) {
            return $this->dbDurumFallback($pdo);
        }

        return [
            'son_guncelleme' => null,
            'kantar_kg' => null,
            'kantar_sabit' => false,
            'seans_kilitli' => false,
            'plaka_buffer' => null,
            'plaka_buffer_detay' => null,
            'fps' => null,
            'yakalama_fps' => null,
            'son_kayit' => null,
            'son_10' => [],
            'sistem' => [
                'surum' => 'V7',
                'mimari' => 'JSON-only demo',
                'kanallar' => ['remote ingest: canli_durum.json + canli_kare.jpg'],
                'simulasyon_modu' => null,
                'ocr_backend' => null,
                'ocr_fallback' => null,
                'ocr_kare_atlama' => null,
                'canli_kare_aralik' => null,
                'calisiyor' => false,
            ],
            '_sunucu_zaman' => date('Y-m-d\TH:i:s'),
            '_dosya_mtime' => null,
            '_durum_yasi_saniye' => null,
        ];
    }

    public function jsonOnlyPanelPayload(int $limit, array $filters = []): array
    {
        $durum = $this->durumOkuVeyaFallback();
        $history = $this->historyKayitlariVeToplam($limit, $filters);
        $kayitlar = $history['kayitlar'];

        $kayitlar = array_values(array_filter(
            $kayitlar,
            fn (mixed $row): bool => is_array($row) && $this->kayitFiltreyeUyar($row, $filters)
        ));

        return [
            'durum' => $durum,
            'toplam' => $history['toplam'] > 0 ? $history['toplam'] : count($kayitlar),
            'limit' => $limit,
            'filtre' => $this->normalizeFilters($filters),
            'kayitlar' => $kayitlar,
            'ozet' => $this->jsonOzetGetir($kayitlar),
            '_sunucu_zaman' => date('Y-m-d\TH:i:s'),
            '_demo_modu' => true,
        ];
    }

    public function vehiclePassHasRecords(array $filters = []): bool
    {
        try {
            return $this->vehiclePassQuery($filters)->exists();
        } catch (Throwable) {
            return false;
        }
    }

    public function vehiclePassPanelPayload(int $limit, array $filters = []): array
    {
        $query = $this->vehiclePassQuery($filters);
        $total = (clone $query)->count();
        $kayitlar = $query
            ->orderByDesc('passed_at')
            ->orderByDesc('id')
            ->limit($limit)
            ->get()
            ->map(fn (VehiclePass $pass): array => $this->vehiclePassKaydiniNormalizeEt($pass))
            ->all();
        $durum = $this->vehiclePassDurumPayload($kayitlar[0] ?? null);

        return [
            'durum' => $durum,
            'toplam' => $total,
            'limit' => $limit,
            'filtre' => $this->normalizeFilters($filters),
            'kayitlar' => $kayitlar,
            'ozet' => $this->vehiclePassOzetGetir($filters),
            '_sunucu_zaman' => date('Y-m-d\TH:i:s'),
            '_source' => 'vehicle_passes',
        ];
    }

    public function vehiclePassLiveTickerPayload(): array
    {
        return $this->vehiclePassPanelPayload(5);
    }

    public function vehiclePassArchivePayload(int $page, int $perPage, array $filters = []): array
    {
        $page = max(1, $page);
        $perPage = max(1, min(100, $perPage));
        $query = $this->vehiclePassQuery($filters);
        $total = (clone $query)->count();
        $lastPage = max(1, (int) ceil($total / $perPage));
        $page = min($page, $lastPage);
        $kayitlar = $query
            ->orderByDesc('passed_at')
            ->orderByDesc('id')
            ->offset(($page - 1) * $perPage)
            ->limit($perPage)
            ->get()
            ->map(fn (VehiclePass $pass): array => $this->vehiclePassKaydiniNormalizeEt($pass))
            ->all();

        return [
            'kayitlar' => $kayitlar,
            'toplam' => $total,
            'limit' => $perPage,
            'filtre' => $this->normalizeFilters($filters),
            'pagination' => [
                'current_page' => $page,
                'per_page' => $perPage,
                'total' => $total,
                'last_page' => $lastPage,
            ],
            '_sunucu_zaman' => date('Y-m-d\TH:i:s'),
            '_source' => 'vehicle_passes',
        ];
    }

    public function vehiclePassCsvIcerikOlustur(array $filters = []): array
    {
        $passes = $this->vehiclePassQuery($filters)
            ->orderByDesc('passed_at')
            ->orderByDesc('id')
            ->limit(5000)
            ->get();

        $filename = 'kantar_raporu_vehicle_passes_' . $this->filterSlug($filters) . '_' . date('Ymd_His') . '.csv';
        $header = ['Plaka', 'Yon', 'GecisZamani', 'GirisKg', 'CikisKg', 'AracKg', 'MalzemeKg', 'NetKg', 'Guven', 'Snapshot'];

        $out = fopen('php://temp', 'w+');
        fwrite($out, "\xEF\xBB\xBF");
        fputcsv($out, $header, ';');
        foreach ($passes as $pass) {
            $record = $this->vehiclePassKaydiniNormalizeEt($pass);
            fputcsv($out, [
                $record['plaka'] ?? '',
                $record['tip'] ?? '',
                $record['gecis_zamani'] ?? '',
                $record['giris_agirlik'] ?? '',
                $record['cikis_agirlik'] ?? '',
                $record['arac_agirlik'] ?? '',
                $record['malzeme_agirlik'] ?? '',
                $record['net_agirlik'] ?? '',
                $record['guven'] ?? '',
                $pass->snapshot_url ?: $pass->snapshot_path,
            ], ';');
        }
        rewind($out);
        $csv = stream_get_contents($out) ?: '';
        fclose($out);

        return ['content' => $csv, 'filename' => $filename, 'row_count' => $passes->count()];
    }

    public function dbPanelPayload(PDO $pdo, int $limit, array $filters = []): array
    {
        $jsonIndex = $this->agirlikService->jsonAgirlikIndexiGetir($this->legacyPath('canli_durum.json'));
        $csvIndex = $this->agirlikService->csvAgirlikIndexiGetir($this->legacyPath('kantar_raporu.csv'));
        $kayitlar = $this->dbKayitlariGetir($pdo, $limit, $jsonIndex, $csvIndex, $filters);
        $dbToplam = $this->dbKayitSayisi($pdo, $filters);
        $durum = $this->durumOkuVeyaFallback($pdo);

        return [
            'durum' => $durum,
            'toplam' => $dbToplam,
            'limit' => $limit,
            'filtre' => $this->normalizeFilters($filters),
            'kayitlar' => $kayitlar,
            'ozet' => $this->dbOzetGetir($pdo),
            '_sunucu_zaman' => date('Y-m-d\TH:i:s'),
        ];
    }

    public function dbLiveTickerPayload(PDO $pdo): array
    {
        $jsonIndex = $this->agirlikService->jsonAgirlikIndexiGetir($this->legacyPath('canli_durum.json'));
        $csvIndex = $this->agirlikService->csvAgirlikIndexiGetir($this->legacyPath('kantar_raporu.csv'));
        $kayitlar = $this->dbKayitlariGetir($pdo, 5, $jsonIndex, $csvIndex);
        $durum = $this->durumOkuVeyaFallback($pdo);

        return [
            'durum' => $durum,
            'toplam' => count($kayitlar),
            'limit' => 5,
            'kayitlar' => $kayitlar,
            'ozet' => $this->dbOzetGetir($pdo),
            '_sunucu_zaman' => date('Y-m-d\TH:i:s'),
        ];
    }

    public function dbArchivePayload(PDO $pdo, int $page, int $perPage, array $filters = []): array
    {
        $page = max(1, $page);
        $perPage = max(1, min(100, $perPage));
        $total = $this->dbKayitSayisi($pdo, $filters);
        $lastPage = max(1, (int) ceil($total / $perPage));
        $page = min($page, $lastPage);
        $jsonIndex = $this->agirlikService->jsonAgirlikIndexiGetir($this->legacyPath('canli_durum.json'));
        $csvIndex = $this->agirlikService->csvAgirlikIndexiGetir($this->legacyPath('kantar_raporu.csv'));
        $kayitlar = $this->dbKayitlariGetir($pdo, $perPage, $jsonIndex, $csvIndex, $filters, ($page - 1) * $perPage);

        return [
            'kayitlar' => $kayitlar,
            'toplam' => $total,
            'limit' => $perPage,
            'filtre' => $this->normalizeFilters($filters),
            'pagination' => [
                'current_page' => $page,
                'per_page' => $perPage,
                'total' => $total,
                'last_page' => $lastPage,
            ],
            '_sunucu_zaman' => date('Y-m-d\TH:i:s'),
        ];
    }

    public function jsonOnlyArchivePayload(int $page, int $perPage, array $filters = []): array
    {
        $page = max(1, $page);
        $perPage = max(1, min(100, $perPage));
        $history = $this->historyKayitlariVeToplam(PHP_INT_MAX, $filters);
        $total = (int) $history['toplam'];
        $lastPage = max(1, (int) ceil($total / $perPage));
        $page = min($page, $lastPage);

        return [
            'kayitlar' => array_slice($history['kayitlar'], ($page - 1) * $perPage, $perPage),
            'toplam' => $total,
            'limit' => $perPage,
            'filtre' => $this->normalizeFilters($filters),
            'pagination' => [
                'current_page' => $page,
                'per_page' => $perPage,
                'total' => $total,
                'last_page' => $lastPage,
            ],
            '_sunucu_zaman' => date('Y-m-d\TH:i:s'),
            '_demo_modu' => true,
        ];
    }

    public function csvIcerikOlustur(PDO $pdo, array $filters = []): array
    {
        [$where, $params] = $this->dateWhereSql($filters, 'g');
        $stmt = $pdo->prepare(
            "SELECT a.plaka, g.yon, g.gecis_zamani, g.guven
             FROM gecisler g
             INNER JOIN araclar a ON a.id = g.id
             {$where}
             ORDER BY g.gecis_zamani DESC
             LIMIT 5000"
        );
        $this->bindDateParams($stmt, $params);
        $stmt->execute();
        $rows = $stmt->fetchAll(PDO::FETCH_ASSOC);

        $filename = 'kantar_raporu_mysql_' . $this->filterSlug($filters) . '_' . date('Ymd_His') . '.csv';
        $header = ['Plaka', 'Yon', 'GecisZamani', 'Guven'];

        $out = fopen('php://temp', 'w+');
        fwrite($out, "\xEF\xBB\xBF");
        fputcsv($out, $header, ';');
        foreach ($rows as $r) {
            fputcsv($out, [
                $r['plaka'] ?? '',
                $r['yon'] ?? '',
                $r['gecis_zamani'] ?? '',
                $r['guven'] ?? '',
            ], ';');
        }
        rewind($out);
        $csv = stream_get_contents($out) ?: '';
        fclose($out);

        return ['content' => $csv, 'filename' => $filename, 'row_count' => count($rows)];
    }

    public function csvDosyaIcerikOlustur(string $csvDosya, array $filters = []): array
    {
        $filename = 'kantar_raporu_dosya_' . $this->filterSlug($filters) . '_' . date('Ymd_His') . '.csv';
        $header = ['Tarih', 'Saat', 'Plaka', 'Tip', 'Guven', 'Operator'];
        $rows = [];

        if (is_file($csvDosya)) {
            $handle = fopen($csvDosya, 'r');
            if ($handle !== false) {
                $csvHeader = fgetcsv($handle, 0, ';') ?: [];
                $csvHeader = array_map(fn ($value): string => preg_replace('/^\xEF\xBB\xBF/', '', trim((string) $value)) ?? '', $csvHeader);
                if ($csvHeader !== []) {
                    $header = $csvHeader;
                }

                while (($row = fgetcsv($handle, 0, ';')) !== false) {
                    $assoc = [];
                    foreach ($header as $index => $key) {
                        $assoc[$key] = $row[$index] ?? '';
                    }

                    if ($this->csvSatiriFiltreyeUyar($assoc, $filters)) {
                        $rows[] = $row;
                    }
                }
                fclose($handle);
            }
        }

        $out = fopen('php://temp', 'w+');
        fwrite($out, "\xEF\xBB\xBF");
        fputcsv($out, $header, ';');
        foreach ($rows as $row) {
            fputcsv($out, $row, ';');
        }
        rewind($out);
        $csv = stream_get_contents($out) ?: '';
        fclose($out);

        return ['content' => $csv, 'filename' => $filename, 'row_count' => count($rows)];
    }

    public function jsonCsvIcerikOlustur(array $filters = []): array
    {
        $history = $this->historyKayitlariVeToplam(5000, $filters);
        $records = $history['kayitlar'];

        $rows = [];
        foreach ($records as $record) {
            if (!is_array($record) || !$this->kayitFiltreyeUyar($record, $filters)) {
                continue;
            }

            $tip = $this->kayitTipi($record);
            $date = $tip === 'CIKIS'
                ? (string) ($record['cikis_tarih'] ?? $record['tarih'] ?? $record['giris_tarih'] ?? '')
                : (string) ($record['giris_tarih'] ?? $record['tarih'] ?? $record['cikis_tarih'] ?? '');
            $time = $tip === 'CIKIS'
                ? (string) ($record['cikis_saat'] ?? $record['saat'] ?? $record['giris_saat'] ?? '')
                : (string) ($record['giris_saat'] ?? $record['saat'] ?? $record['cikis_saat'] ?? '');
            $timestamp = trim($date.' '.$time);

            $rows[] = [
                (string) ($record['plaka'] ?? ''),
                $tip,
                $timestamp,
                $record['guven'] ?? '',
            ];
        }

        $filename = 'kantar_raporu_json_' . $this->filterSlug($filters) . '_' . date('Ymd_His') . '.csv';
        $out = fopen('php://temp', 'w+');
        fwrite($out, "\xEF\xBB\xBF");
        fputcsv($out, ['Plaka', 'Yon', 'GecisZamani', 'Guven'], ';');
        foreach ($rows as $row) {
            fputcsv($out, $row, ';');
        }
        rewind($out);
        $csv = stream_get_contents($out) ?: '';
        fclose($out);

        return ['content' => $csv, 'filename' => $filename, 'row_count' => count($rows)];
    }

    private function historyKayitlariVeToplam(int $limit, array $filters = []): array
    {
        $path = $this->legacyPath('gecis_gecmisi.jsonl');
        $kayitlar = [];

        if (is_file($path)) {
            $lines = file($path, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
            if (is_array($lines)) {
                foreach ($lines as $line) {
                    $row = json_decode((string) $line, true);
                    if (!is_array($row)) {
                        continue;
                    }

                    $record = $this->historyKaydiniNormalizeEt($row);
                    if (!$this->kayitFiltreyeUyar($record, $filters)) {
                        continue;
                    }

                    $kayitlar[] = $record;
                }
            }
        }

        $kayitlar = $this->kayitlariBirlestir($kayitlar, $this->csvGecmisKayitlari($filters), PHP_INT_MAX);

        usort($kayitlar, function (array $a, array $b): int {
            return strcmp((string) ($b['gecis_zamani'] ?? ''), (string) ($a['gecis_zamani'] ?? ''));
        });

        return [
            'kayitlar' => array_slice($kayitlar, 0, $limit),
            'toplam' => count($kayitlar),
        ];
    }

    private function csvGecmisKayitlari(array $filters = []): array
    {
        $path = $this->legacyPath('kantar_raporu.csv');
        if (!is_file($path)) {
            return [];
        }

        $handle = fopen($path, 'r');
        if ($handle === false) {
            return [];
        }

        $records = [];
        try {
            $header = fgetcsv($handle, 0, ';') ?: [];
            $header = array_map(fn ($value): string => preg_replace('/^\xEF\xBB\xBF/', '', trim((string) $value)) ?? '', $header);
            if ($header === []) {
                return [];
            }

            while (($row = fgetcsv($handle, 0, ';')) !== false) {
                $assoc = [];
                foreach ($header as $index => $key) {
                    $assoc[$key] = $row[$index] ?? '';
                }

                $record = $this->csvKaydiniNormalizeEt($assoc);
                if (!$this->kayitFiltreyeUyar($record, $filters)) {
                    continue;
                }

                $records[] = $record;
            }
        } finally {
            fclose($handle);
        }

        return $records;
    }

    private function csvKaydiniNormalizeEt(array $row): array
    {
        $durum = strtoupper(trim((string) ($row['Durum'] ?? 'GIRIS')));
        $tip = ($durum === 'TAMAMLANDI' || $durum === 'CIKIS') ? 'CIKIS' : 'GIRIS';
        $date = $tip === 'CIKIS'
            ? (string) ($row['CikisTarih'] ?? $row['GirisTarih'] ?? '')
            : (string) ($row['GirisTarih'] ?? $row['CikisTarih'] ?? '');
        $time = $tip === 'CIKIS'
            ? (string) ($row['CikisSaat'] ?? $row['GirisSaat'] ?? '')
            : (string) ($row['GirisSaat'] ?? $row['CikisSaat'] ?? '');
        $timestamp = strtotime(trim($date.' '.$time));

        $entryWeight = $this->agirlikService->parseAgirlik($row['GirisAgirlik(kg)'] ?? null);
        $exitWeight = $this->agirlikService->parseAgirlik($row['CikisAgirlik(kg)'] ?? null);
        $netWeight = $this->agirlikService->parseAgirlik($row['NetAgirlik(kg)'] ?? null);

        return array_merge([
            'arac_id' => 0,
            'plaka' => (string) ($row['Plaka'] ?? ''),
            'durum' => $tip,
            'tip' => $tip,
            'giris_tarih' => (string) ($row['GirisTarih'] ?? ''),
            'giris_saat' => (string) ($row['GirisSaat'] ?? ''),
            'giris_agirlik' => $entryWeight,
            'cikis_tarih' => (string) ($row['CikisTarih'] ?? ''),
            'cikis_saat' => (string) ($row['CikisSaat'] ?? ''),
            'cikis_agirlik' => $exitWeight,
            'net_agirlik' => $netWeight,
            'guven' => $this->parseGuven($row['Guven'] ?? null),
            'kara_liste' => false,
            'gecis_zamani' => $timestamp !== false ? date('Y-m-d H:i:s', $timestamp) : null,
        ], $this->agirlikDetayi($tip, $entryWeight, $exitWeight, $netWeight));
    }

    private function kayitlariBirlestir(array $primary, array $secondary, int $limit): array
    {
        $merged = [];
        $seen = [];

        foreach (array_merge($primary, $secondary) as $record) {
            if (!is_array($record)) {
                continue;
            }

            $key = $this->kayitAnahtari($record);
            if (isset($seen[$key])) {
                continue;
            }

            $seen[$key] = true;
            $merged[] = $record;
        }

        usort($merged, function (array $a, array $b): int {
            return strcmp((string) ($b['gecis_zamani'] ?? ''), (string) ($a['gecis_zamani'] ?? ''));
        });

        return array_slice($merged, 0, $limit);
    }

    private function ortakKayitSayisi(array $primary, array $secondary): int
    {
        $keys = [];
        foreach ($primary as $record) {
            if (is_array($record)) {
                $keys[$this->kayitAnahtari($record)] = true;
            }
        }

        $count = 0;
        foreach ($secondary as $record) {
            if (is_array($record) && isset($keys[$this->kayitAnahtari($record)])) {
                $count++;
            }
        }

        return $count;
    }

    private function kayitAnahtari(array $record): string
    {
        $tip = $this->kayitTipi($record);
        $date = $tip === 'CIKIS'
            ? (string) ($record['cikis_tarih'] ?? $record['tarih'] ?? $record['giris_tarih'] ?? '')
            : (string) ($record['giris_tarih'] ?? $record['tarih'] ?? $record['cikis_tarih'] ?? '');
        $time = $tip === 'CIKIS'
            ? (string) ($record['cikis_saat'] ?? $record['saat'] ?? $record['giris_saat'] ?? '')
            : (string) ($record['giris_saat'] ?? $record['saat'] ?? $record['cikis_saat'] ?? '');

        if (($date === '' || $time === '') && !empty($record['gecis_zamani'])) {
            $timestamp = strtotime((string) $record['gecis_zamani']);
            if ($timestamp !== false) {
                $date = date('Y-m-d', $timestamp);
                $time = date('H:i:s', $timestamp);
            }
        }

        return strtoupper(trim((string) ($record['plaka'] ?? ''))).'|'.$tip.'|'.$date.'|'.$time;
    }

    private function historyKaydiniNormalizeEt(array $row): array
    {
        $tip = $this->kayitTipi($row);
        $timestamp = strtotime((string) ($row['gecis_zamani'] ?? ''));

        $date = $tip === 'CIKIS'
            ? (string) ($row['cikis_tarih'] ?? $row['tarih'] ?? '')
            : (string) ($row['giris_tarih'] ?? $row['tarih'] ?? '');
        $time = $tip === 'CIKIS'
            ? (string) ($row['cikis_saat'] ?? $row['saat'] ?? '')
            : (string) ($row['giris_saat'] ?? $row['saat'] ?? '');

        if (($date === '' || $time === '') && $timestamp !== false) {
            $date = $date !== '' ? $date : date('Y-m-d', $timestamp);
            $time = $time !== '' ? $time : date('H:i:s', $timestamp);
        }

        $entryWeight = $this->agirlikService->parseAgirlik($row['giris_agirlik'] ?? null);
        $exitWeight = $this->agirlikService->parseAgirlik($row['cikis_agirlik'] ?? null);
        $netWeight = $this->agirlikService->parseAgirlik($row['net_agirlik'] ?? null);

        return array_merge([
            'arac_id' => (int) ($row['arac_id'] ?? 0),
            'plaka' => (string) ($row['plaka'] ?? ''),
            'durum' => $tip,
            'tip' => $tip,
            'giris_tarih' => $tip === 'GIRIS' ? $date : (string) ($row['giris_tarih'] ?? ''),
            'giris_saat' => $tip === 'GIRIS' ? $time : (string) ($row['giris_saat'] ?? ''),
            'giris_agirlik' => $entryWeight,
            'cikis_tarih' => $tip === 'CIKIS' ? $date : (string) ($row['cikis_tarih'] ?? ''),
            'cikis_saat' => $tip === 'CIKIS' ? $time : (string) ($row['cikis_saat'] ?? ''),
            'cikis_agirlik' => $exitWeight,
            'net_agirlik' => $netWeight,
            'guven' => $this->parseGuven($row['guven'] ?? null),
            'kara_liste' => (bool) ($row['kara_liste'] ?? false),
            'gecis_zamani' => $timestamp !== false ? date('Y-m-d H:i:s', $timestamp) : null,
        ], $this->agirlikDetayi($tip, $entryWeight, $exitWeight, $netWeight));
    }

    private function agirlikDetayi(string $tip, ?float $entryWeight, ?float $exitWeight, ?float $netWeight): array
    {
        if ($tip !== 'CIKIS') {
            return [
                'arac_agirlik' => null,
                'malzeme_agirlik' => null,
            ];
        }

        if ($entryWeight !== null && $entryWeight <= 0) {
            $entryWeight = null;
        }
        if ($exitWeight !== null && $exitWeight <= 0) {
            $exitWeight = null;
        }

        $materialWeight = $netWeight;
        if ($materialWeight === null && $entryWeight !== null && $exitWeight !== null) {
            $materialWeight = $exitWeight - $entryWeight;
        }

        return [
            'arac_agirlik' => $entryWeight !== null && $exitWeight !== null ? min($entryWeight, $exitWeight) : null,
            'malzeme_agirlik' => $materialWeight,
            'net_agirlik' => $materialWeight,
        ];
    }

    private function isAbsolutePath(string $path): bool
    {
        return $path !== '' && (str_starts_with($path, '/') || preg_match('/^[A-Za-z]:[\/\\\\]/', $path) === 1);
    }

    private function vehiclePassQuery(array $filters = [])
    {
        $filters = $this->normalizeFilters($filters);
        $query = VehiclePass::query();

        match ($filters['period']) {
            'day' => $query
                ->where('passed_at', '>=', $filters['date'].' 00:00:00')
                ->where('passed_at', '<', date('Y-m-d', strtotime($filters['date'].' +1 day')).' 00:00:00'),
            'month' => $query
                ->where('passed_at', '>=', $filters['month'].'-01 00:00:00')
                ->where('passed_at', '<', date('Y-m', strtotime($filters['month'].'-01 +1 month')).'-01 00:00:00'),
            'year' => $query
                ->where('passed_at', '>=', $filters['year'].'-01-01 00:00:00')
                ->where('passed_at', '<', ((int) $filters['year'] + 1).'-01-01 00:00:00'),
            default => null,
        };

        if ($filters['plate'] !== '') {
            $query->where('plate', 'like', '%'.$filters['plate'].'%');
        }

        return $query;
    }

    private function vehiclePassKaydiniNormalizeEt(VehiclePass $pass): array
    {
        $direction = strtoupper(trim((string) $pass->direction));
        if ($direction !== 'CIKIS') {
            $direction = 'GIRIS';
        }

        $passedAt = $pass->passed_at;
        $entryAt = $pass->entry_at ?: ($direction === 'GIRIS' ? $passedAt : null);
        $exitAt = $pass->exit_at ?: ($direction === 'CIKIS' ? $passedAt : null);

        $entryWeight = $pass->entry_weight_kg !== null ? (float) $pass->entry_weight_kg : null;
        $exitWeight = $pass->exit_weight_kg !== null ? (float) $pass->exit_weight_kg : null;
        $netWeight = $pass->net_weight_kg !== null ? (float) $pass->net_weight_kg : null;

        return array_merge([
            'arac_id' => $pass->legacy_vehicle_id ?: $pass->id,
            'plaka' => (string) $pass->plate,
            'durum' => $direction,
            'tip' => $direction,
            'giris_tarih' => $entryAt ? $entryAt->format('Y-m-d') : '',
            'giris_saat' => $entryAt ? $entryAt->format('H:i:s') : '',
            'giris_agirlik' => $entryWeight,
            'cikis_tarih' => $exitAt ? $exitAt->format('Y-m-d') : '',
            'cikis_saat' => $exitAt ? $exitAt->format('H:i:s') : '',
            'cikis_agirlik' => $exitWeight,
            'net_agirlik' => $netWeight,
            'guven' => $this->parseGuven($pass->confidence),
            'kara_liste' => (bool) $pass->is_blacklisted,
            'gecis_zamani' => $passedAt ? $passedAt->format('Y-m-d H:i:s') : null,
            'snapshot' => $pass->snapshot_url ?: $pass->snapshot_path,
        ], $this->agirlikDetayi($direction, $entryWeight, $exitWeight, $netWeight));
    }

    private function vehiclePassDurumPayload(?array $sonKayit): array
    {
        $durum = $this->durumOkuVeyaFallback();
        if (!isset($durum['sistem']) || !is_array($durum['sistem'])) {
            $durum['sistem'] = [];
        }

        $durum['sistem']['mimari'] = 'VehiclePass + JSON durum';
        $durum['sistem']['kanallar'] = ['vehicle_passes', 'canli_durum.json', 'canli_kare.jpg'];

        if ($sonKayit !== null) {
            $durum['son_kayit'] = $sonKayit;
            $durum['plaka_buffer'] = (string) ($sonKayit['plaka'] ?? '');
            $durum['plaka_buffer_detay'] = ['plaka' => (string) ($sonKayit['plaka'] ?? '')];
            $durum['son_guncelleme'] = $sonKayit['gecis_zamani'] ?? ($durum['son_guncelleme'] ?? null);
        }

        return $durum;
    }

    private function vehiclePassOzetGetir(array $filters = []): array
    {
        $today = date('Y-m-d');
        $todayQuery = $this->vehiclePassQuery(['period' => 'day', 'date' => $today]);
        $filteredQuery = $this->vehiclePassQuery($filters);

        $bugun = (clone $todayQuery)->count();
        $sonSaat = (clone $filteredQuery)->where('passed_at', '>=', date('Y-m-d H:i:s', time() - 3600))->count();
        $tamamlanan = (clone $todayQuery)->where('direction', 'CIKIS')->count();
        $girisBugun = (clone $todayQuery)->where('direction', 'GIRIS')->count();
        $avg = $this->parseGuven((clone $filteredQuery)->avg('confidence'));

        return [
            'bugun_kayit' => $bugun,
            'son_saat_kayit' => $sonSaat,
            'aktif_seans' => max(0, $girisBugun - $tamamlanan),
            'tamamlanan' => $tamamlanan,
            'ortalama_guven' => $avg !== null ? round($avg, 3) : null,
            'ortalama_guven_yuzde' => $avg !== null ? (int) round($avg * 100) : null,
        ];
    }

    private function jsonOzetGetir(array $kayitlar): array
    {
        $today = date('Y-m-d');
        $oneHourAgo = time() - 3600;
        $todayCount = 0;
        $lastHourCount = 0;
        $completed = 0;
        $latestByPlate = [];
        $guven = [];

        foreach ($kayitlar as $row) {
            if (!is_array($row)) {
                continue;
            }

            $date = $this->kayitTarihi($row);
            $ts = $this->kayitTimestamp($row);
            $tip = $this->kayitTipi($row);

            if ($date === $today) {
                $todayCount++;
            }
            if ($ts !== false && $ts >= $oneHourAgo) {
                $lastHourCount++;
            }
            if ($tip === 'CIKIS') {
                $completed++;
            }

            $plate = strtoupper(trim((string) ($row['plaka'] ?? '')));
            if ($plate !== '' && $ts !== false && (!isset($latestByPlate[$plate]) || $ts > $latestByPlate[$plate]['ts'])) {
                $latestByPlate[$plate] = ['ts' => $ts, 'tip' => $tip];
            }

            $parsedGuven = $this->parseGuven($row['guven'] ?? null);
            if ($parsedGuven !== null) {
                $guven[] = $parsedGuven;
            }
        }

        $avg = $guven !== [] ? array_sum($guven) / count($guven) : null;
        $active = count(array_filter($latestByPlate, fn (array $row): bool => $row['tip'] === 'GIRIS'));

        return [
            'bugun_kayit' => $todayCount,
            'son_saat_kayit' => $lastHourCount,
            'aktif_seans' => $active,
            'tamamlanan' => $completed,
            'ortalama_guven' => $avg !== null ? round($avg, 3) : null,
            'ortalama_guven_yuzde' => $avg !== null ? (int) round($avg * 100) : null,
        ];
    }

    private function parseGuven(mixed $value): ?float
    {
        if ($value === null || $value === '') {
            return null;
        }
        $num = (float) $value;
        if ($num > 1.0) {
            $num = $num / 100.0;
        }
        return max(0.0, min(1.0, $num));
    }

    private function dbKayitlariGetir(PDO $pdo, int $limit, array $agirlikIndex = [], array $csvAgirlikIndex = [], array $filters = [], int $offset = 0): array
    {
        [$where, $params] = $this->dateWhereSql($filters, 'g');
        $stmt = $pdo->prepare(
            "SELECT g.id AS arac_id, a.plaka, a.kara_liste, g.yon, g.gecis_zamani, g.guven
             FROM gecisler g
             INNER JOIN araclar a ON a.id = g.id
             {$where}
             ORDER BY g.gecis_zamani DESC
             LIMIT :limit OFFSET :offset"
        );
        $this->bindDateParams($stmt, $params);
        $stmt->bindValue(':limit', $limit, PDO::PARAM_INT);
        $stmt->bindValue(':offset', max(0, $offset), PDO::PARAM_INT);
        $stmt->execute();
        $rows = $stmt->fetchAll(PDO::FETCH_ASSOC);

        $kayitlar = [];
        foreach ($rows as $r) {
            $dt = strtotime((string) ($r['gecis_zamani'] ?? ''));
            $tarih = $dt ? date('Y-m-d', $dt) : '';
            $saat = $dt ? date('H:i:s', $dt) : '';
            $tip = strtoupper(trim((string) ($r['yon'] ?? 'GIRIS')));
            if ($tip !== 'CIKIS') {
                $tip = 'GIRIS';
            }
            $plaka = strtoupper(trim((string) ($r['plaka'] ?? '')));
            $weights = $this->agirlikService->agirlikBul($plaka, $tip, $tarih, $saat, $agirlikIndex, $csvAgirlikIndex);
            $entryWeight = is_array($weights) ? ($weights['giris_agirlik'] ?? null) : null;
            $exitWeight = is_array($weights) ? ($weights['cikis_agirlik'] ?? null) : null;
            $netWeight = is_array($weights) ? ($weights['net_agirlik'] ?? null) : null;

            $kayitlar[] = array_merge([
                'arac_id' => (int) ($r['arac_id'] ?? 0),
                'plaka' => (string) ($r['plaka'] ?? ''),
                'durum' => $tip,
                'tip' => $tip,
                'giris_tarih' => $tip === 'GIRIS' ? $tarih : '',
                'giris_saat' => $tip === 'GIRIS' ? $saat : '',
                'giris_agirlik' => $entryWeight,
                'cikis_tarih' => $tip === 'CIKIS' ? $tarih : '',
                'cikis_saat' => $tip === 'CIKIS' ? $saat : '',
                'cikis_agirlik' => $exitWeight,
                'net_agirlik' => $netWeight,
                'guven' => $this->parseGuven($r['guven'] ?? null),
                'kara_liste' => (bool) ($r['kara_liste'] ?? false),
                'gecis_zamani' => $r['gecis_zamani'] ?? null,
            ], $this->agirlikDetayi($tip, $entryWeight, $exitWeight, $netWeight));
        }

        return $kayitlar;
    }

    private function dbKayitSayisi(PDO $pdo, array $filters = []): int
    {
        [$where, $params] = $this->dateWhereSql($filters, 'g');
        $stmt = $pdo->prepare(
            "SELECT COUNT(*)
             FROM gecisler g
             INNER JOIN araclar a ON a.id = g.id
             {$where}"
        );
        $this->bindDateParams($stmt, $params);
        $stmt->execute();

        return (int) $stmt->fetchColumn();
    }

    private function normalizeFilters(array $filters): array
    {
        $period = strtolower((string) ($filters['period'] ?? 'all'));
        if (!in_array($period, ['all', 'day', 'month', 'year'], true)) {
            $period = 'all';
        }

        return [
            'period' => $period,
            'date' => preg_match('/^\d{4}-\d{2}-\d{2}$/', (string) ($filters['date'] ?? '')) === 1
                ? (string) $filters['date']
                : date('Y-m-d'),
            'month' => preg_match('/^\d{4}-\d{2}$/', (string) ($filters['month'] ?? '')) === 1
                ? (string) $filters['month']
                : date('Y-m'),
            'year' => preg_match('/^\d{4}$/', (string) ($filters['year'] ?? '')) === 1
                ? (string) $filters['year']
                : date('Y'),
            'plate' => strtoupper(trim((string) ($filters['plate'] ?? $filters['plaka'] ?? ''))),
        ];
    }

    private function dateWhereSql(array $filters, string $alias): array
    {
        $filters = $this->normalizeFilters($filters);
        $column = $alias.'.gecis_zamani';

        [$dateSql, $params] = match ($filters['period']) {
            'day' => ["DATE({$column}) = :filter_date", ['filter_date' => $filters['date']]],
            'month' => ["DATE_FORMAT({$column}, '%Y-%m') = :filter_month", ['filter_month' => $filters['month']]],
            'year' => ["YEAR({$column}) = :filter_year", ['filter_year' => (int) $filters['year']]],
            default => ['', []],
        };

        $conditions = [];
        if ($dateSql !== '') {
            $conditions[] = $dateSql;
        }
        if ($filters['plate'] !== '') {
            $conditions[] = 'UPPER(a.plaka) LIKE :filter_plate';
            $params['filter_plate'] = '%'.$filters['plate'].'%';
        }

        return [
            $conditions === [] ? '' : 'WHERE '.implode(' AND ', $conditions),
            $params,
        ];
    }

    private function bindDateParams(\PDOStatement $stmt, array $params): void
    {
        foreach ($params as $key => $value) {
            $stmt->bindValue(':'.$key, $value, is_int($value) ? PDO::PARAM_INT : PDO::PARAM_STR);
        }
    }

    private function filterSlug(array $filters): string
    {
        $filters = $this->normalizeFilters($filters);

        return match ($filters['period']) {
            'day' => 'gunluk_'.$filters['date'],
            'month' => 'aylik_'.$filters['month'],
            'year' => 'yillik_'.$filters['year'],
            default => 'tum_kayitlar',
        };
    }

    private function csvSatiriFiltreyeUyar(array $row, array $filters): bool
    {
        $filters = $this->normalizeFilters($filters);
        if ($filters['plate'] !== '' && !str_contains(strtoupper((string) ($row['Plaka'] ?? $row['plaka'] ?? '')), $filters['plate'])) {
            return false;
        }
        if ($filters['period'] === 'all') {
            return true;
        }

        $date = $this->csvSatirTarihi($row);
        if ($date === null) {
            return false;
        }

        return match ($filters['period']) {
            'day' => $date === $filters['date'],
            'month' => str_starts_with($date, $filters['month'].'-'),
            'year' => str_starts_with($date, $filters['year'].'-'),
            default => true,
        };
    }

    private function kayitFiltreyeUyar(array $row, array $filters): bool
    {
        $filters = $this->normalizeFilters($filters);
        if ($filters['plate'] !== '' && !str_contains(strtoupper((string) ($row['plaka'] ?? '')), $filters['plate'])) {
            return false;
        }
        if ($filters['period'] === 'all') {
            return true;
        }

        $date = $this->kayitTarihi($row);
        if ($date === null) {
            return false;
        }

        return match ($filters['period']) {
            'day' => $date === $filters['date'],
            'month' => str_starts_with($date, $filters['month'].'-'),
            'year' => str_starts_with($date, $filters['year'].'-'),
            default => true,
        };
    }

    private function kayitTarihi(array $row): ?string
    {
        $tip = $this->kayitTipi($row);
        $value = $tip === 'CIKIS'
            ? ($row['cikis_tarih'] ?? $row['tarih'] ?? $row['giris_tarih'] ?? null)
            : ($row['giris_tarih'] ?? $row['tarih'] ?? $row['cikis_tarih'] ?? null);

        if ($value === null || trim((string) $value) === '') {
            $value = $row['gecis_zamani'] ?? $row['GecisZamani'] ?? null;
        }

        if ($value === null || trim((string) $value) === '') {
            return null;
        }

        $timestamp = strtotime((string) $value);
        return $timestamp === false ? null : date('Y-m-d', $timestamp);
    }

    private function kayitTipi(array $row): string
    {
        $raw = strtoupper(trim((string) ($row['tip'] ?? $row['durum'] ?? $row['yon'] ?? 'GIRIS')));

        return str_contains($raw, 'CIKIS') || str_contains($raw, 'TAMAMLANDI') ? 'CIKIS' : 'GIRIS';
    }

    private function csvSatirTarihi(array $row): ?string
    {
        $tip = $this->kayitTipi([
            'tip' => $row['Tip'] ?? null,
            'durum' => $row['Durum'] ?? $row['Yon'] ?? null,
        ]);
        $value = $tip === 'CIKIS'
            ? ($row['CikisTarih'] ?? $row['Tarih'] ?? $row['GecisZamani'] ?? $row['GirisTarih'] ?? null)
            : ($row['GirisTarih'] ?? $row['Tarih'] ?? $row['GecisZamani'] ?? $row['CikisTarih'] ?? null);

        if ($value === null || trim((string) $value) === '') {
            return null;
        }

        $timestamp = strtotime((string) $value);
        return $timestamp === false ? null : date('Y-m-d', $timestamp);
    }

    private function dbOzetGetir(PDO $pdo): array
    {
        $bugun = (int) $pdo->query(
            'SELECT COUNT(*) FROM gecisler WHERE DATE(gecis_zamani)=CURDATE()'
        )->fetchColumn();

        $sonSaat = (int) $pdo->query(
            'SELECT COUNT(*) FROM gecisler WHERE gecis_zamani >= (NOW() - INTERVAL 1 HOUR)'
        )->fetchColumn();

        $tamamlanan = (int) $pdo->query(
            "SELECT COUNT(*) FROM gecisler
             WHERE DATE(gecis_zamani)=CURDATE() AND UPPER(yon)='CIKIS'"
        )->fetchColumn();

        $aktifIc = (int) $pdo->query(
            "SELECT COUNT(*) FROM (
                SELECT g1.id
                FROM gecisler g1
                INNER JOIN (
                    SELECT id, MAX(gecis_zamani) AS son_zaman
                    FROM gecisler
                    GROUP BY id
                ) s ON s.id = g1.id AND s.son_zaman = g1.gecis_zamani
                WHERE UPPER(g1.yon)='GIRIS'
            ) x"
        )->fetchColumn();

        $ortGuvenNorm = $this->parseGuven($pdo->query('SELECT AVG(guven) FROM gecisler')->fetchColumn());

        return [
            'bugun_kayit' => $bugun,
            'son_saat_kayit' => $sonSaat,
            'aktif_seans' => $aktifIc,
            'tamamlanan' => $tamamlanan,
            'ortalama_guven' => $ortGuvenNorm !== null ? round($ortGuvenNorm, 3) : null,
            'ortalama_guven_yuzde' => $ortGuvenNorm !== null ? (int) round($ortGuvenNorm * 100) : null,
        ];
    }

    private function dbDurumFallback(PDO $pdo): array
    {
        $stmt = $pdo->query(
            "SELECT a.plaka, g.yon, g.gecis_zamani, g.guven
             FROM gecisler g
             INNER JOIN araclar a ON a.id = g.id
             ORDER BY g.gecis_zamani DESC
             LIMIT 1"
        );
        $son = $stmt->fetch(PDO::FETCH_ASSOC);

        $durum = [
            'son_guncelleme' => null,
            'kantar_kg' => null,
            'kantar_sabit' => false,
            'seans_kilitli' => false,
            'plaka_buffer' => null,
            'plaka_buffer_detay' => null,
            'fps' => null,
            'yakalama_fps' => null,
            'son_kayit' => null,
            'son_10' => [],
            'sistem' => [
                'surum' => 'V7',
                'mimari' => 'MySQL',
                'kanallar' => ['MySQL:otokantar.gecisler+araclar', 'canli_durum.json (opsiyonel)', 'canli_kare.jpg'],
                'simulasyon_modu' => null,
                'ocr_backend' => null,
                'ocr_fallback' => null,
                'ocr_kare_atlama' => null,
                'canli_kare_aralik' => null,
                'calisiyor' => $son ? true : false,
            ],
            '_sunucu_zaman' => date('Y-m-d\TH:i:s'),
            '_dosya_mtime' => null,
            '_durum_yasi_saniye' => null,
        ];

        if ($son) {
            $ts = strtotime((string) ($son['gecis_zamani'] ?? ''));
            $durum['son_guncelleme'] = $ts ? date('c', $ts) : null;
            $durum['_durum_yasi_saniye'] = $ts ? max(0, time() - $ts) : null;
            $durum['plaka_buffer'] = (string) ($son['plaka'] ?? '');
            $durum['plaka_buffer_detay'] = ['plaka' => (string) ($son['plaka'] ?? '')];

            $tip = strtoupper(trim((string) ($son['yon'] ?? 'GIRIS')));
            if ($tip !== 'CIKIS') {
                $tip = 'GIRIS';
            }

            $durum['son_kayit'] = [
                'plaka' => (string) ($son['plaka'] ?? ''),
                'durum' => $tip,
                'tip' => $tip,
                'giris_tarih' => $tip === 'GIRIS' && $ts ? date('Y-m-d', $ts) : '',
                'giris_saat' => $tip === 'GIRIS' && $ts ? date('H:i:s', $ts) : '',
                'cikis_tarih' => $tip === 'CIKIS' && $ts ? date('Y-m-d', $ts) : '',
                'cikis_saat' => $tip === 'CIKIS' && $ts ? date('H:i:s', $ts) : '',
                'guven' => $this->parseGuven($son['guven'] ?? null),
                'gecis_zamani' => $ts ? date('Y-m-d H:i:s', $ts) : null,
            ];
        }

        return $durum;
    }

    private function kayitTimestamp(array $row): int|false
    {
        if (!empty($row['gecis_zamani'])) {
            $timestamp = strtotime((string) $row['gecis_zamani']);
            if ($timestamp !== false) {
                return $timestamp;
            }
        }

        $tip = $this->kayitTipi($row);
        $date = $tip === 'CIKIS'
            ? (string) ($row['cikis_tarih'] ?? $row['tarih'] ?? $row['giris_tarih'] ?? '')
            : (string) ($row['giris_tarih'] ?? $row['tarih'] ?? $row['cikis_tarih'] ?? '');
        $time = $tip === 'CIKIS'
            ? (string) ($row['cikis_saat'] ?? $row['saat'] ?? $row['giris_saat'] ?? '')
            : (string) ($row['giris_saat'] ?? $row['saat'] ?? $row['cikis_saat'] ?? '');

        return strtotime(trim($date.' '.$time));
    }
}
