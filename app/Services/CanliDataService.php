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

    public function jsonOnlyPanelPayload(int $limit): array
    {
        $durum = $this->durumOkuVeyaFallback();
        $kayitlar = [];

        if (isset($durum['son_10']) && is_array($durum['son_10'])) {
            $kayitlar = array_slice(array_reverse($durum['son_10']), 0, $limit);
        }

        if ($kayitlar === [] && isset($durum['son_kayit']) && is_array($durum['son_kayit'])) {
            $kayitlar = [$durum['son_kayit']];
        }

        return [
            'durum' => $durum,
            'toplam' => count($kayitlar),
            'limit' => $limit,
            'kayitlar' => $kayitlar,
            'ozet' => $this->jsonOzetGetir($kayitlar),
            '_sunucu_zaman' => date('Y-m-d\TH:i:s'),
            '_demo_modu' => true,
        ];
    }

    public function dbPanelPayload(PDO $pdo, int $limit): array
    {
        $jsonIndex = $this->agirlikService->jsonAgirlikIndexiGetir($this->legacyPath('canli_durum.json'));
        $csvIndex = $this->agirlikService->csvAgirlikIndexiGetir($this->legacyPath('kantar_raporu.csv'));
        $kayitlar = $this->dbKayitlariGetir($pdo, $limit, $jsonIndex, $csvIndex);
        $durum = $this->durumOkuVeyaFallback($pdo);

        return [
            'durum' => $durum,
            'toplam' => (int) $pdo->query('SELECT COUNT(*) FROM gecisler')->fetchColumn(),
            'limit' => $limit,
            'kayitlar' => $kayitlar,
            'ozet' => $this->dbOzetGetir($pdo),
            '_sunucu_zaman' => date('Y-m-d\TH:i:s'),
        ];
    }

    public function csvIcerikOlustur(PDO $pdo): array
    {
        $stmt = $pdo->query(
            "SELECT a.plaka, g.yon, g.gecis_zamani, g.guven
             FROM gecisler g
             INNER JOIN araclar a ON a.id = g.id
             ORDER BY g.gecis_zamani DESC
             LIMIT 5000"
        );
        $rows = $stmt->fetchAll(PDO::FETCH_ASSOC);

        $filename = 'kantar_raporu_mysql_' . date('Ymd_His') . '.csv';
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

        return ['content' => $csv, 'filename' => $filename];
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

    private function dbKayitlariGetir(PDO $pdo, int $limit, array $agirlikIndex = [], array $csvAgirlikIndex = []): array
    {
        $stmt = $pdo->prepare(
            "SELECT g.id AS arac_id, a.plaka, a.kara_liste, g.yon, g.gecis_zamani, g.guven
             FROM gecisler g
             INNER JOIN araclar a ON a.id = g.id
             ORDER BY g.gecis_zamani DESC
             LIMIT :limit"
        );
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
