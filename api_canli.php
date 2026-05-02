<?php
declare(strict_types=1);

header('Cross-Origin-Resource-Policy: same-origin');
header('X-Content-Type-Options: nosniff');

$action = $_GET['action'] ?? 'panel';
$limit = min(200, max(1, (int) ($_GET['limit'] ?? 40)));
$jsonDurumDosya = __DIR__ . '/canli_durum.json';
$csvRaporDosya = __DIR__ . '/kantar_raporu.csv';

function json_yanit(array $payload, int $status = 200): void
{
    http_response_code($status);
    header('Content-Type: application/json; charset=utf-8');
    header('Cache-Control: no-store, no-cache, must-revalidate');
    header('Pragma: no-cache');
    echo json_encode($payload, JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT);
    exit;
}

function pdo_baglan(): PDO
{
    $dsn = 'mysql:host=localhost;port=3306;dbname=otokantar;charset=utf8mb4';
    return new PDO($dsn, 'root', '', [
        PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
        PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
    ]);
}

function parse_guven($value): ?float
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

function parse_agirlik($value): ?float
{
    if ($value === null) {
        return null;
    }
    $text = trim((string) $value);
    if ($text === '') {
        return null;
    }
    $text = str_replace(',', '.', $text);
    return is_numeric($text) ? (float) $text : null;
}

function json_agirlik_indexi_getir(string $jsonDurumDosya): array
{
    if (!is_file($jsonDurumDosya)) {
        return [];
    }
    $raw = file_get_contents($jsonDurumDosya);
    if ($raw === false) {
        return [];
    }
    $data = json_decode($raw, true);
    if (!is_array($data) || !isset($data['son_10']) || !is_array($data['son_10'])) {
        return [];
    }

    $index = [];
    foreach ($data['son_10'] as $row) {
        if (!is_array($row)) {
            continue;
        }
        $plaka = strtoupper(trim((string) ($row['plaka'] ?? '')));
        if ($plaka === '') {
            continue;
        }
        $durumRaw = strtoupper(trim((string) ($row['durum'] ?? $row['tip'] ?? 'GIRIS')));
        $tip = ($durumRaw === 'TAMAMLANDI' || $durumRaw === 'CIKIS') ? 'CIKIS' : 'GIRIS';
        $tarih = trim((string) ($row['giris_tarih'] ?? $row['tarih'] ?? ''));
        $saat = trim((string) ($row['giris_saat'] ?? $row['saat'] ?? ''));
        if ($tip === 'CIKIS' && trim((string) ($row['cikis_tarih'] ?? '')) !== '') {
            $tarih = trim((string) $row['cikis_tarih']);
        }
        if ($tip === 'CIKIS' && trim((string) ($row['cikis_saat'] ?? '')) !== '') {
            $saat = trim((string) $row['cikis_saat']);
        }
        if ($tarih === '' || $saat === '') {
            continue;
        }

        $key = $plaka . '|' . $tip . '|' . $tarih . '|' . $saat;
        $index[$key] = [
            'giris_agirlik' => parse_agirlik($row['giris_agirlik'] ?? null),
            'cikis_agirlik' => parse_agirlik($row['cikis_agirlik'] ?? null),
            'net_agirlik' => parse_agirlik($row['net_agirlik'] ?? null),
        ];
    }
    return $index;
}

function csv_agirlik_indexi_getir(string $csvDosya): array
{
    $exact = [];
    $minute = [];
    if (!is_file($csvDosya)) {
        return ['exact' => $exact, 'minute' => $minute];
    }
    $lines = file($csvDosya, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
    if (!is_array($lines)) {
        return ['exact' => $exact, 'minute' => $minute];
    }

    foreach ($lines as $line) {
        $row = str_getcsv((string) $line, ';');
        if (!is_array($row) || count($row) < 10) {
            continue;
        }
        $plaka = strtoupper(trim((string) ($row[0] ?? '')));
        $durum = strtoupper(trim((string) ($row[1] ?? '')));
        if ($plaka === '' || ($durum !== 'ICERIDE' && $durum !== 'TAMAMLANDI' && $durum !== 'GIRIS' && $durum !== 'CIKIS')) {
            continue;
        }
        $tip = ($durum === 'TAMAMLANDI' || $durum === 'CIKIS') ? 'CIKIS' : 'GIRIS';
        $tarih = trim((string) (($tip === 'CIKIS' ? ($row[5] ?? '') : ($row[2] ?? ''))));
        $saat = trim((string) (($tip === 'CIKIS' ? ($row[6] ?? '') : ($row[3] ?? ''))));
        if ($tarih === '' || $saat === '') {
            continue;
        }

        $weights = [
            'giris_agirlik' => parse_agirlik($row[4] ?? null),
            'cikis_agirlik' => parse_agirlik($row[7] ?? null),
            'net_agirlik' => parse_agirlik($row[8] ?? null),
        ];
        $exact[$plaka . '|' . $tip . '|' . $tarih . '|' . $saat] = $weights;
        $minute[$plaka . '|' . $tip . '|' . $tarih . '|' . substr($saat, 0, 5)] = $weights;
    }
    return ['exact' => $exact, 'minute' => $minute];
}

function agirlik_bul(string $plaka, string $tip, string $tarih, string $saat, array $jsonIndex, array $csvIndex): ?array
{
    $exactKey = $plaka . '|' . $tip . '|' . $tarih . '|' . $saat;
    if (isset($jsonIndex[$exactKey]) && is_array($jsonIndex[$exactKey])) {
        return $jsonIndex[$exactKey];
    }
    if (isset($csvIndex['exact'][$exactKey]) && is_array($csvIndex['exact'][$exactKey])) {
        return $csvIndex['exact'][$exactKey];
    }

    $minuteKey = $plaka . '|' . $tip . '|' . $tarih . '|' . substr($saat, 0, 5);
    if (isset($csvIndex['minute'][$minuteKey]) && is_array($csvIndex['minute'][$minuteKey])) {
        return $csvIndex['minute'][$minuteKey];
    }
    return null;
}

function db_kayitlari_getir(PDO $pdo, int $limit, array $agirlikIndex = [], array $csvAgirlikIndex = []): array
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
    $rows = $stmt->fetchAll();

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
        $weights = agirlik_bul($plaka, $tip, $tarih, $saat, $agirlikIndex, $csvAgirlikIndex);

        $girisAgirlik = null;
        $cikisAgirlik = null;
        $netAgirlik = null;
        if (is_array($weights)) {
            $girisAgirlik = $weights['giris_agirlik'] ?? null;
            $cikisAgirlik = $weights['cikis_agirlik'] ?? null;
            $netAgirlik = $weights['net_agirlik'] ?? null;
        }

        $kayitlar[] = [
            'arac_id' => (int) $r['arac_id'],
            'plaka' => (string) ($r['plaka'] ?? ''),
            'durum' => $tip,
            'tip' => $tip,
            'giris_tarih' => $tip === 'GIRIS' ? $tarih : '',
            'giris_saat' => $tip === 'GIRIS' ? $saat : '',
            'giris_agirlik' => $girisAgirlik,
            'cikis_tarih' => $tip === 'CIKIS' ? $tarih : '',
            'cikis_saat' => $tip === 'CIKIS' ? $saat : '',
            'cikis_agirlik' => $cikisAgirlik,
            'net_agirlik' => $netAgirlik,
            'guven' => parse_guven($r['guven'] ?? null),
            'kara_liste' => (bool) ($r['kara_liste'] ?? false),
            'gecis_zamani' => $r['gecis_zamani'] ?? null,
        ];
    }

    return $kayitlar;
}

function db_ozet_getir(PDO $pdo): array
{
    $bugun = (int) $pdo->query(
        "SELECT COUNT(*) FROM gecisler WHERE DATE(gecis_zamani)=CURDATE()"
    )->fetchColumn();

    $sonSaat = (int) $pdo->query(
        "SELECT COUNT(*) FROM gecisler WHERE gecis_zamani >= (NOW() - INTERVAL 1 HOUR)"
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

    $ortGuven = $pdo->query("SELECT AVG(guven) FROM gecisler")->fetchColumn();
    $ortGuvenNorm = parse_guven($ortGuven);

    return [
        'bugun_kayit' => $bugun,
        'son_saat_kayit' => $sonSaat,
        'aktif_seans' => $aktifIc,
        'tamamlanan' => $tamamlanan,
        'ortalama_guven' => $ortGuvenNorm !== null ? round($ortGuvenNorm, 3) : null,
        'ortalama_guven_yuzde' => $ortGuvenNorm !== null ? (int) round($ortGuvenNorm * 100) : null,
    ];
}

function db_durum_fallback(PDO $pdo): array
{
    $stmt = $pdo->query(
        "SELECT a.plaka, g.yon, g.gecis_zamani, g.guven
         FROM gecisler g
         INNER JOIN araclar a ON a.id = g.id
         ORDER BY g.gecis_zamani DESC
         LIMIT 1"
    );
    $son = $stmt->fetch();

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
        $ts = strtotime((string) $son['gecis_zamani']);
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
            'guven' => parse_guven($son['guven'] ?? null),
        ];
    }

    return $durum;
}

function durum_oku_veya_fallback(string $jsonDurumDosya, PDO $pdo): array
{
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

    return db_durum_fallback($pdo);
}

function csv_indir_mysql(PDO $pdo): void
{
    $stmt = $pdo->query(
        "SELECT a.plaka, g.yon, g.gecis_zamani, g.guven
         FROM gecisler g
         INNER JOIN araclar a ON a.id = g.id
         ORDER BY g.gecis_zamani DESC
         LIMIT 5000"
    );
    $rows = $stmt->fetchAll();

    $dosyaAdi = 'kantar_raporu_mysql_' . date('Ymd_His') . '.csv';
    header('Content-Type: text/csv; charset=utf-8');
    header('Content-Disposition: attachment; filename="' . $dosyaAdi . '"');
    header('Cache-Control: no-store');

    $out = fopen('php://output', 'w');
    if ($out === false) {
        exit;
    }
    fwrite($out, "\xEF\xBB\xBF");
    fputcsv($out, ['Plaka', 'Yon', 'GecisZamani', 'Guven'], ';');
    foreach ($rows as $r) {
        fputcsv($out, [
            $r['plaka'] ?? '',
            $r['yon'] ?? '',
            $r['gecis_zamani'] ?? '',
            $r['guven'] ?? '',
        ], ';');
    }
    fclose($out);
    exit;
}

try {
    $pdo = pdo_baglan();

    if ($action === 'durum') {
        json_yanit(durum_oku_veya_fallback($jsonDurumDosya, $pdo));
    }

    if ($action === 'panel') {
        $agirlikIndex = json_agirlik_indexi_getir($jsonDurumDosya);
        $csvAgirlikIndex = csv_agirlik_indexi_getir($csvRaporDosya);
        $kayitlar = db_kayitlari_getir($pdo, $limit, $agirlikIndex, $csvAgirlikIndex);
        $durum = durum_oku_veya_fallback($jsonDurumDosya, $pdo);
        json_yanit([
            'durum' => $durum,
            'toplam' => (int) $pdo->query("SELECT COUNT(*) FROM gecisler")->fetchColumn(),
            'limit' => $limit,
            'kayitlar' => $kayitlar,
            'ozet' => db_ozet_getir($pdo),
            '_sunucu_zaman' => date('Y-m-d\TH:i:s'),
        ]);
    }

    if ($action === 'csv_indir') {
        csv_indir_mysql($pdo);
    }

    json_yanit([
        'hata' => 'Gecersiz action parametresi',
        'gecerli_actionlar' => ['durum', 'panel', 'csv_indir'],
    ], 400);
} catch (Throwable $e) {
    json_yanit([
        'hata' => 'MySQL baglanti/sorgu hatasi',
        'mesaj' => $e->getMessage(),
    ], 500);
}
