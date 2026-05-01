<?php
/**
 * OtoKantar V7 — api_canli.php
 * AJAX endpoint: JavaScript'in fetch() ile her 1 saniyede çağırdığı PHP servisi.
 *
 * Desteklenen action'lar:
 *   ?action=durum     → canli_durum.json içeriğini JSON olarak döner
 *   ?action=csv       → kantar_raporu.csv'den son N kaydı JSON olarak döner
 *   ?action=csv_indir → kantar_raporu.csv'yi tarayıcıya indirir
 *
 * NOT: Python backend'e hiç dokunulmaz. Bu dosya sadece okuma yapar.
 */

// ─── Güvenlik: sadece aynı sunucudan gelen isteklere izin ver ─────────────────
// İhtiyaç duyarsanız aşağıdaki satırı kaldırabilirsiniz.
header('Access-Control-Allow-Origin: same-origin');
header('X-Content-Type-Options: nosniff');

$action = $_GET['action'] ?? 'durum';

// ─── Dosya yolları ────────────────────────────────────────────────────────────
$JSON_DOSYA = __DIR__ . '/canli_durum.json';
$CSV_DOSYA  = __DIR__ . '/kantar_raporu.csv';

// ═══════════════════════════════════════════════════════════════════════════════
// ACTION: durum — canli_durum.json'ı oku ve döndür
// ═══════════════════════════════════════════════════════════════════════════════
if ($action === 'durum') {
    header('Content-Type: application/json; charset=utf-8');
    header('Cache-Control: no-store, no-cache, must-revalidate');
    header('Pragma: no-cache');

    if (!file_exists($JSON_DOSYA)) {
        http_response_code(404);
        echo json_encode([
            'hata' => 'canli_durum.json bulunamadı',
            'yol'  => $JSON_DOSYA,
        ], JSON_UNESCAPED_UNICODE);
        exit;
    }

    $icerik = file_get_contents($JSON_DOSYA);
    if ($icerik === false) {
        http_response_code(500);
        echo json_encode(['hata' => 'Dosya okunamadı'], JSON_UNESCAPED_UNICODE);
        exit;
    }

    $veri = json_decode($icerik, true);
    if (json_last_error() !== JSON_ERROR_NONE) {
        http_response_code(500);
        echo json_encode([
            'hata'        => 'JSON parse hatası',
            'json_hata'   => json_last_error_msg(),
        ], JSON_UNESCAPED_UNICODE);
        exit;
    }

    // Sunucu zaman damgası ekle (client-side gecikme tespiti için)
    $veri['_sunucu_zaman'] = date('Y-m-d\TH:i:s');
    $veri['_dosya_mtime']  = date('Y-m-d\TH:i:s', filemtime($JSON_DOSYA));

    echo json_encode($veri, JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT);
    exit;
}

// ═══════════════════════════════════════════════════════════════════════════════
// ACTION: csv — kantar_raporu.csv'den son N kaydı JSON olarak döndür
// ═══════════════════════════════════════════════════════════════════════════════
if ($action === 'csv') {
    header('Content-Type: application/json; charset=utf-8');
    header('Cache-Control: no-store, no-cache, must-revalidate');

    $limit = min(100, max(1, intval($_GET['limit'] ?? 50)));

    if (!file_exists($CSV_DOSYA)) {
        http_response_code(404);
        echo json_encode(['hata' => 'kantar_raporu.csv bulunamadı'], JSON_UNESCAPED_UNICODE);
        exit;
    }

    $fp = fopen($CSV_DOSYA, 'r');
    if (!$fp) {
        http_response_code(500);
        echo json_encode(['hata' => 'CSV okunamadı'], JSON_UNESCAPED_UNICODE);
        exit;
    }

    $baslik = fgetcsv($fp);
    $tumSatirlar = [];

    while (($satir = fgetcsv($fp)) !== false) {
        if ($baslik && count($satir) === count($baslik)) {
            $tumSatirlar[] = array_combine($baslik, $satir);
        } else {
            $tumSatirlar[] = $satir;
        }
    }
    fclose($fp);

    $toplam   = count($tumSatirlar);
    $sonKayitlar = array_reverse(array_slice($tumSatirlar, -$limit));

    echo json_encode([
        'toplam'       => $toplam,
        'limit'        => $limit,
        'baslik'       => $baslik,
        'kayitlar'     => $sonKayitlar,
        '_sunucu_zaman' => date('Y-m-d\TH:i:s'),
    ], JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT);
    exit;
}

// ═══════════════════════════════════════════════════════════════════════════════
// ACTION: csv_indir — CSV dosyasını tarayıcıya indirir
// ═══════════════════════════════════════════════════════════════════════════════
if ($action === 'csv_indir') {
    if (!file_exists($CSV_DOSYA)) {
        http_response_code(404);
        echo 'kantar_raporu.csv bulunamadı';
        exit;
    }

    $dosyaAdi = 'kantar_raporu_' . date('Ymd_His') . '.csv';
    header('Content-Type: text/csv; charset=utf-8');
    header('Content-Disposition: attachment; filename="' . $dosyaAdi . '"');
    header('Content-Length: ' . filesize($CSV_DOSYA));
    header('Cache-Control: no-store');

    // UTF-8 BOM ekle (Excel Türkçe karakter uyumu için)
    echo "\xEF\xBB\xBF";
    readfile($CSV_DOSYA);
    exit;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Bilinmeyen action
// ═══════════════════════════════════════════════════════════════════════════════
http_response_code(400);
header('Content-Type: application/json; charset=utf-8');
echo json_encode([
    'hata'            => 'Geçersiz action parametresi',
    'gecerli_actionlar' => ['durum', 'csv', 'csv_indir'],
    'kullanim'        => [
        'Canlı durum' => 'api_canli.php?action=durum',
        'CSV JSON'    => 'api_canli.php?action=csv&limit=50',
        'CSV indir'   => 'api_canli.php?action=csv_indir',
    ],
], JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT);
exit;