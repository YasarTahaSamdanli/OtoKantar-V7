<?php
/**
 * OtoKantar V7 - canli panel endpoint'i
 *
 * Bu katman Python surecine dokunmaz.
 * Yalnizca ayni klasordeki JSON ve CSV dosyalarini okur.
 *
 * Action'lar:
 *   ?action=durum
 *   ?action=csv&limit=50
 *   ?action=panel&limit=30
 *   ?action=csv_indir
 */

declare(strict_types=1);

header('Cross-Origin-Resource-Policy: same-origin');
header('X-Content-Type-Options: nosniff');

$action = $_GET['action'] ?? 'durum';
$JSON_DOSYA = __DIR__ . '/canli_durum.json';
$CSV_DOSYA = __DIR__ . '/kantar_raporu.csv';

function json_yanit(array $payload, int $status = 200): void
{
    http_response_code($status);
    header('Content-Type: application/json; charset=utf-8');
    header('Cache-Control: no-store, no-cache, must-revalidate');
    header('Pragma: no-cache');
    echo json_encode($payload, JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT);
    exit;
}

function trim_bom(string $value): string
{
    return preg_replace('/^\xEF\xBB\xBF/', '', $value) ?? $value;
}

function tarih_iso(?int $timestamp): ?string
{
    return $timestamp ? date('Y-m-d\TH:i:s', $timestamp) : null;
}

function parse_float($value): ?float
{
    if ($value === null) {
        return null;
    }
    $value = trim((string) $value);
    if ($value === '') {
        return null;
    }
    $value = str_replace(',', '.', $value);
    return is_numeric($value) ? (float) $value : null;
}

function durum_bos(): array
{
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
            'surum' => null,
            'mimari' => 'Loose Coupling',
            'kanallar' => ['canli_durum.json', 'kantar_raporu.csv', 'canli_kare.jpg'],
            'simulasyon_modu' => null,
            'ocr_backend' => null,
            'ocr_fallback' => null,
            'ocr_kare_atlama' => null,
            'canli_kare_aralik' => null,
            'calisiyor' => false,
        ],
    ];
}

function durum_oku(string $jsonDosya, bool $strict = true): array
{
    if (!is_file($jsonDosya)) {
        if ($strict) {
            json_yanit([
                'hata' => 'canli_durum.json bulunamadi',
                'yol' => $jsonDosya,
            ], 404);
        }
        $bos = durum_bos();
        $bos['_sunucu_zaman'] = date('Y-m-d\TH:i:s');
        $bos['_dosya_mtime'] = null;
        $bos['_durum_yasi_saniye'] = null;
        return $bos;
    }

    $icerik = file_get_contents($jsonDosya);
    if ($icerik === false) {
        if ($strict) {
            json_yanit(['hata' => 'canli_durum.json okunamadi'], 500);
        }
        $bos = durum_bos();
        $bos['_sunucu_zaman'] = date('Y-m-d\TH:i:s');
        $bos['_dosya_mtime'] = tarih_iso(@filemtime($jsonDosya) ?: null);
        $bos['_durum_yasi_saniye'] = null;
        return $bos;
    }

    $veri = json_decode($icerik, true);
    if (!is_array($veri)) {
        if ($strict) {
            json_yanit([
                'hata' => 'JSON parse hatasi',
                'json_hata' => json_last_error_msg(),
            ], 500);
        }
        $veri = [];
    }

    $veri = array_replace_recursive(durum_bos(), $veri);
    $veri['_sunucu_zaman'] = date('Y-m-d\TH:i:s');
    $veri['_dosya_mtime'] = tarih_iso(@filemtime($jsonDosya) ?: null);

    $yas = null;
    if (!empty($veri['son_guncelleme'])) {
        $ts = strtotime((string) $veri['son_guncelleme']);
        if ($ts !== false) {
            $yas = max(0, time() - $ts);
        }
    }
    $veri['_durum_yasi_saniye'] = $yas;

    return $veri;
}

function csv_baslik_key(string $value): string
{
    $value = trim_bom(trim($value));
    $value = strtolower($value);
    $value = strtr($value, [
        'i' => 'i',
        'ı' => 'i',
        'ğ' => 'g',
        'ü' => 'u',
        'ş' => 's',
        'ö' => 'o',
        'ç' => 'c',
        '(' => '',
        ')' => '',
        '[' => '',
        ']' => '',
        '{' => '',
        '}' => '',
        '-' => '',
        '_' => '',
        ' ' => '',
        '/' => '',
        '\\' => '',
        '.' => '',
    ]);
    return preg_replace('/[^a-z0-9]/', '', $value) ?? '';
}

function kayit_bos(): array
{
    return [
        'plaka' => '',
        'durum' => '',
        'giris_tarih' => '',
        'giris_saat' => '',
        'giris_agirlik' => null,
        'cikis_tarih' => '',
        'cikis_saat' => '',
        'cikis_agirlik' => null,
        'net_agirlik' => null,
        'guven' => null,
        'operator' => '',
        'firma_adi' => '',
        'sofor_adi' => '',
        'sofor_tel' => '',
        'malzeme_cinsi' => '',
        'irsaliye_no' => '',
    ];
}

function normalize_assoc(array $assoc): array
{
    $kayit = kayit_bos();
    $kayit['plaka'] = trim((string) ($assoc['plaka'] ?? ''));
    $kayit['durum'] = strtoupper(trim((string) ($assoc['durum'] ?? ($assoc['tip'] ?? ''))));
    $kayit['giris_tarih'] = trim((string) ($assoc['giristarih'] ?? ($assoc['tarih'] ?? '')));
    $kayit['giris_saat'] = trim((string) ($assoc['girissaat'] ?? ($assoc['saat'] ?? '')));
    $kayit['giris_agirlik'] = parse_float($assoc['girisagirlikkg'] ?? ($assoc['agirlik'] ?? null));
    $kayit['cikis_tarih'] = trim((string) ($assoc['cikistarih'] ?? ''));
    $kayit['cikis_saat'] = trim((string) ($assoc['cikissaat'] ?? ''));
    $kayit['cikis_agirlik'] = parse_float($assoc['cikisagirlikkg'] ?? null);
    $kayit['net_agirlik'] = parse_float($assoc['netagirlikkg'] ?? null);
    $kayit['guven'] = parse_float($assoc['guven'] ?? null);
    $kayit['operator'] = trim((string) ($assoc['operator'] ?? ''));
    $kayit['firma_adi'] = trim((string) ($assoc['firmaadi'] ?? ''));
    $kayit['sofor_adi'] = trim((string) ($assoc['soforadi'] ?? ''));
    $kayit['sofor_tel'] = trim((string) ($assoc['sofortel'] ?? ''));
    $kayit['malzeme_cinsi'] = trim((string) ($assoc['malzemecinsi'] ?? ''));
    $kayit['irsaliye_no'] = trim((string) ($assoc['irsaliyeno'] ?? ''));
    return $kayit;
}

function normalize_row(array $headerKeys, array $row): ?array
{
    $row = array_map(static function ($value) {
        return is_string($value) ? trim_bom(trim($value)) : $value;
    }, $row);

    $row = array_values(array_filter($row, static function ($value) {
        return $value !== null;
    }));

    if (count($row) === 0) {
        return null;
    }

    if ($headerKeys && count($row) === count($headerKeys)) {
        $assoc = array_combine($headerKeys, $row);
        if (is_array($assoc)) {
            return normalize_assoc($assoc);
        }
    }

    $count = count($row);
    if ($count >= 16) {
        return [
            'plaka' => trim((string) $row[0]),
            'durum' => strtoupper(trim((string) $row[1])),
            'giris_tarih' => trim((string) $row[2]),
            'giris_saat' => trim((string) $row[3]),
            'giris_agirlik' => parse_float($row[4]),
            'cikis_tarih' => trim((string) $row[5]),
            'cikis_saat' => trim((string) $row[6]),
            'cikis_agirlik' => parse_float($row[7]),
            'net_agirlik' => parse_float($row[8]),
            'guven' => parse_float($row[9]),
            'operator' => trim((string) $row[10]),
            'firma_adi' => trim((string) $row[11]),
            'sofor_adi' => trim((string) $row[12]),
            'sofor_tel' => trim((string) $row[13]),
            'malzeme_cinsi' => trim((string) $row[14]),
            'irsaliye_no' => trim((string) $row[15]),
        ];
    }

    if ($count >= 11) {
        return [
            'plaka' => trim((string) $row[0]),
            'durum' => strtoupper(trim((string) $row[1])),
            'giris_tarih' => trim((string) $row[2]),
            'giris_saat' => trim((string) $row[3]),
            'giris_agirlik' => parse_float($row[4]),
            'cikis_tarih' => trim((string) $row[5]),
            'cikis_saat' => trim((string) $row[6]),
            'cikis_agirlik' => parse_float($row[7]),
            'net_agirlik' => parse_float($row[8]),
            'guven' => parse_float($row[9]),
            'operator' => trim((string) $row[10]),
            'firma_adi' => '',
            'sofor_adi' => '',
            'sofor_tel' => '',
            'malzeme_cinsi' => '',
            'irsaliye_no' => '',
        ];
    }

    if ($count >= 6) {
        $tip = strtoupper(trim((string) $row[3]));
        return [
            'plaka' => trim((string) $row[2]),
            'durum' => $tip,
            'giris_tarih' => trim((string) $row[0]),
            'giris_saat' => trim((string) $row[1]),
            'giris_agirlik' => null,
            'cikis_tarih' => '',
            'cikis_saat' => '',
            'cikis_agirlik' => null,
            'net_agirlik' => null,
            'guven' => parse_float($row[4]),
            'operator' => trim((string) $row[5]),
            'firma_adi' => '',
            'sofor_adi' => '',
            'sofor_tel' => '',
            'malzeme_cinsi' => '',
            'irsaliye_no' => '',
        ];
    }

    return null;
}

function csv_ozet(array $kayitlar): array
{
    $bugun = date('Y-m-d');
    $birSaatOnce = time() - 3600;
    $bugunKayit = 0;
    $sonSaatKayit = 0;
    $aktifSeans = 0;
    $tamamlanan = 0;
    $guvenToplam = 0.0;
    $guvenAdet = 0;

    foreach ($kayitlar as $kayit) {
        $durum = strtoupper((string) ($kayit['durum'] ?? ''));
        $tarih = (string) ($kayit['giris_tarih'] ?? '');
        $saat = (string) ($kayit['giris_saat'] ?? '');

        if ($tarih === $bugun) {
            $bugunKayit++;
        }

        $ts = strtotime(trim($tarih . ' ' . $saat));
        if ($ts !== false && $ts >= $birSaatOnce) {
            $sonSaatKayit++;
        }

        if ($durum === 'ICERIDE' || $durum === 'GIRIS') {
            $aktifSeans++;
        }
        if ($durum === 'TAMAMLANDI' || $durum === 'CIKIS') {
            $tamamlanan++;
        }

        if (isset($kayit['guven']) && $kayit['guven'] !== null) {
            $guvenToplam += (float) $kayit['guven'];
            $guvenAdet++;
        }
    }

    return [
        'bugun_kayit' => $bugunKayit,
        'son_saat_kayit' => $sonSaatKayit,
        'aktif_seans' => $aktifSeans,
        'tamamlanan' => $tamamlanan,
        'ortalama_guven' => $guvenAdet > 0 ? round($guvenToplam / $guvenAdet, 3) : null,
        'ortalama_guven_yuzde' => $guvenAdet > 0 ? (int) round(($guvenToplam / $guvenAdet) * 100) : null,
    ];
}

function csv_oku(string $csvDosya, int $limit = 50, bool $strict = true): array
{
    $bos = [
        'toplam' => 0,
        'limit' => $limit,
        'kayitlar' => [],
        'ozet' => csv_ozet([]),
        '_dosya_mtime' => null,
    ];

    if (!is_file($csvDosya)) {
        if ($strict) {
            json_yanit(['hata' => 'kantar_raporu.csv bulunamadi'], 404);
        }
        return $bos;
    }

    $ornek = file($csvDosya, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
    if ($ornek === false || count($ornek) === 0) {
        return $bos;
    }

    $ilkSatir = trim_bom((string) $ornek[0]);
    $delimiter = substr_count($ilkSatir, ';') >= substr_count($ilkSatir, ',') ? ';' : ',';

    $fp = fopen($csvDosya, 'r');
    if (!$fp) {
        if ($strict) {
            json_yanit(['hata' => 'CSV okunamadi'], 500);
        }
        return $bos;
    }

    $header = fgetcsv($fp, 0, $delimiter);
    $header = is_array($header) ? array_map('csv_baslik_key', $header) : [];
    $tumKayitlar = [];

    while (($row = fgetcsv($fp, 0, $delimiter)) !== false) {
        $normalized = normalize_row($header, $row);
        if ($normalized !== null && $normalized['plaka'] !== '') {
            $tumKayitlar[] = $normalized;
        }
    }
    fclose($fp);

    return [
        'toplam' => count($tumKayitlar),
        'limit' => $limit,
        'kayitlar' => array_reverse(array_slice($tumKayitlar, -$limit)),
        'ozet' => csv_ozet($tumKayitlar),
        '_dosya_mtime' => tarih_iso(@filemtime($csvDosya) ?: null),
    ];
}

$limit = min(200, max(1, (int) ($_GET['limit'] ?? 30)));

if ($action === 'durum') {
    json_yanit(durum_oku($JSON_DOSYA, true));
}

if ($action === 'csv') {
    json_yanit(csv_oku($CSV_DOSYA, $limit, true));
}

if ($action === 'panel') {
    $durum = durum_oku($JSON_DOSYA, false);
    $csv = csv_oku($CSV_DOSYA, $limit, false);
    json_yanit([
        'durum' => $durum,
        'toplam' => $csv['toplam'],
        'limit' => $csv['limit'],
        'kayitlar' => $csv['kayitlar'],
        'ozet' => $csv['ozet'],
        '_sunucu_zaman' => date('Y-m-d\TH:i:s'),
        '_durum_dosya_mtime' => $durum['_dosya_mtime'] ?? null,
        '_csv_dosya_mtime' => $csv['_dosya_mtime'],
    ]);
}

if ($action === 'csv_indir') {
    if (!is_file($CSV_DOSYA)) {
        http_response_code(404);
        echo 'kantar_raporu.csv bulunamadi';
        exit;
    }

    $dosyaAdi = 'kantar_raporu_' . date('Ymd_His') . '.csv';
    header('Content-Type: text/csv; charset=utf-8');
    header('Content-Disposition: attachment; filename="' . $dosyaAdi . '"');
    header('Content-Length: ' . filesize($CSV_DOSYA));
    header('Cache-Control: no-store');
    readfile($CSV_DOSYA);
    exit;
}

json_yanit([
    'hata' => 'Gecersiz action parametresi',
    'gecerli_actionlar' => ['durum', 'csv', 'panel', 'csv_indir'],
    'kullanim' => [
        'Canli durum' => 'api_canli.php?action=durum',
        'CSV JSON' => 'api_canli.php?action=csv&limit=50',
        'Tek istek panel' => 'api_canli.php?action=panel&limit=30',
        'CSV indir' => 'api_canli.php?action=csv_indir',
    ],
], 400);
