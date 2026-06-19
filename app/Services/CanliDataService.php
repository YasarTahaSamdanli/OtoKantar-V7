<?php

namespace App\Services;

use PDO;

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
        $kayitlar = [];

        if (isset($durum['son_10']) && is_array($durum['son_10'])) {
            $kayitlar = array_slice(array_reverse($durum['son_10']), 0, $limit);
        }

        if ($kayitlar === [] && isset($durum['son_kayit']) && is_array($durum['son_kayit'])) {
            $kayitlar = [$durum['son_kayit']];
        }

        $kayitlar = array_values(array_filter(
            $kayitlar,
            fn (mixed $row): bool => is_array($row) && $this->kayitFiltreyeUyar($row, $filters)
        ));

        return [
            'durum' => $durum,
            'toplam' => count($kayitlar),
            'limit' => $limit,
            'filtre' => $this->normalizeFilters($filters),
            'kayitlar' => $kayitlar,
            'ozet' => $this->jsonOzetGetir($kayitlar),
            '_sunucu_zaman' => date('Y-m-d\TH:i:s'),
            '_demo_modu' => true,
        ];
    }

    public function dbPanelPayload(PDO $pdo, int $limit, array $filters = []): array
    {
        $jsonIndex = $this->agirlikService->jsonAgirlikIndexiGetir($this->legacyPath('canli_durum.json'));
        $csvIndex = $this->agirlikService->csvAgirlikIndexiGetir($this->legacyPath('kantar_raporu.csv'));
        $kayitlar = $this->dbKayitlariGetir($pdo, $limit, $jsonIndex, $csvIndex, $filters);
        $durum = $this->durumOkuVeyaFallback($pdo);

        return [
            'durum' => $durum,
            'toplam' => $this->dbKayitSayisi($pdo, $filters),
            'limit' => $limit,
            'filtre' => $this->normalizeFilters($filters),
            'kayitlar' => $kayitlar,
            'ozet' => $this->dbOzetGetir($pdo),
            '_sunucu_zaman' => date('Y-m-d\TH:i:s'),
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
        $durum = $this->durumOkuVeyaFallback();
        $rows = [];

        $records = [];
        if (isset($durum['son_10']) && is_array($durum['son_10'])) {
            $records = array_reverse($durum['son_10']);
        }
        if ($records === [] && isset($durum['son_kayit']) && is_array($durum['son_kayit'])) {
            $records = [$durum['son_kayit']];
        }

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

    private function isAbsolutePath(string $path): bool
    {
        return $path !== '' && (str_starts_with($path, '/') || preg_match('/^[A-Za-z]:[\/\\\\]/', $path) === 1);
    }

    private function jsonOzetGetir(array $kayitlar): array
    {
        $today = date('Y-m-d');
        $oneHourAgo = time() - 3600;
        $todayCount = 0;
        $lastHourCount = 0;
        $completed = 0;
        $active = 0;
        $guven = [];

        foreach ($kayitlar as $row) {
            if (!is_array($row)) {
                continue;
            }

            $date = (string) ($row['giris_tarih'] ?? $row['tarih'] ?? $row['cikis_tarih'] ?? '');
            $time = (string) ($row['giris_saat'] ?? $row['saat'] ?? $row['cikis_saat'] ?? '');
            $ts = strtotime(trim($date.' '.$time));
            $tip = strtoupper((string) ($row['durum'] ?? $row['tip'] ?? ''));

            if ($date === $today) {
                $todayCount++;
            }
            if ($ts !== false && $ts >= $oneHourAgo) {
                $lastHourCount++;
            }
            if ($tip === 'TAMAMLANDI' || $tip === 'CIKIS') {
                $completed++;
            } else {
                $active++;
            }

            $parsedGuven = $this->parseGuven($row['guven'] ?? null);
            if ($parsedGuven !== null) {
                $guven[] = $parsedGuven;
            }
        }

        $avg = $guven !== [] ? array_sum($guven) / count($guven) : null;

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

    private function dbKayitlariGetir(PDO $pdo, int $limit, array $agirlikIndex = [], array $csvAgirlikIndex = [], array $filters = []): array
    {
        [$where, $params] = $this->dateWhereSql($filters, 'g');
        $stmt = $pdo->prepare(
            "SELECT g.id AS arac_id, a.plaka, a.kara_liste, g.yon, g.gecis_zamani, g.guven
             FROM gecisler g
             INNER JOIN araclar a ON a.id = g.id
             {$where}
             ORDER BY g.gecis_zamani DESC
             LIMIT :limit"
        );
        $this->bindDateParams($stmt, $params);
        $stmt->bindValue(':limit', $limit, PDO::PARAM_INT);
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

            $kayitlar[] = [
                'arac_id' => (int) ($r['arac_id'] ?? 0),
                'plaka' => (string) ($r['plaka'] ?? ''),
                'durum' => $tip,
                'tip' => $tip,
                'giris_tarih' => $tip === 'GIRIS' ? $tarih : '',
                'giris_saat' => $tip === 'GIRIS' ? $saat : '',
                'giris_agirlik' => is_array($weights) ? ($weights['giris_agirlik'] ?? null) : null,
                'cikis_tarih' => $tip === 'CIKIS' ? $tarih : '',
                'cikis_saat' => $tip === 'CIKIS' ? $saat : '',
                'cikis_agirlik' => is_array($weights) ? ($weights['cikis_agirlik'] ?? null) : null,
                'net_agirlik' => is_array($weights) ? ($weights['net_agirlik'] ?? null) : null,
                'guven' => $this->parseGuven($r['guven'] ?? null),
                'kara_liste' => (bool) ($r['kara_liste'] ?? false),
                'gecis_zamani' => $r['gecis_zamani'] ?? null,
            ];
        }

        return $kayitlar;
    }

    private function dbKayitSayisi(PDO $pdo, array $filters = []): int
    {
        [$where, $params] = $this->dateWhereSql($filters, 'g');
        $stmt = $pdo->prepare("SELECT COUNT(*) FROM gecisler g {$where}");
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
        ];
    }

    private function dateWhereSql(array $filters, string $alias): array
    {
        $filters = $this->normalizeFilters($filters);
        $column = $alias.'.gecis_zamani';

        return match ($filters['period']) {
            'day' => ["WHERE DATE({$column}) = :filter_date", ['filter_date' => $filters['date']]],
            'month' => ["WHERE DATE_FORMAT({$column}, '%Y-%m') = :filter_month", ['filter_month' => $filters['month']]],
            'year' => ["WHERE YEAR({$column}) = :filter_year", ['filter_year' => (int) $filters['year']]],
            default => ['', []],
        };
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
        $value = $row['Tarih']
            ?? $row['GirisTarih']
            ?? $row['CikisTarih']
            ?? $row['GecisZamani']
            ?? null;

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
                'giris_tarih' => $ts ? date('Y-m-d', $ts) : '',
                'giris_saat' => $ts ? date('H:i:s', $ts) : '',
                'guven' => $this->parseGuven($son['guven'] ?? null),
            ];
        }

        return $durum;
    }
}
