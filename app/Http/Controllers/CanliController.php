<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Response;
use PDO;
use Throwable;

class CanliController extends Controller
{
    public function view()
    {
        return view('panel');
    }

    public function api(Request $request)
    {
        $action = (string) $request->query('action', 'panel');
        $limit = min(200, max(1, (int) $request->query('limit', 40)));

        try {
            $cacheKey = $this->canliCacheKey($request, $action, $limit);
            $ttlSeconds = $action === 'panel' ? 2 : 1;

            $cached = Cache::get($cacheKey);
            if (is_array($cached)) {
                return response()->json($cached);
            }

            $pdo = DB::connection('legacy')->getPdo();

            if ($action === 'durum') {
                $payload = $this->durumOkuVeyaFallback($pdo);
                Cache::put($cacheKey, $payload, $ttlSeconds);
                return response()->json($payload);
            }

            if ($action === 'panel') {
                $jsonIndex = $this->jsonAgirlikIndexiGetir($this->legacyPath('canli_durum.json'));
                $csvIndex = $this->csvAgirlikIndexiGetir($this->legacyPath('kantar_raporu.csv'));
                $kayitlar = $this->dbKayitlariGetir($pdo, $limit, $jsonIndex, $csvIndex);
                $durum = $this->durumOkuVeyaFallback($pdo);

                $payload = [
                    'durum' => $durum,
                    'toplam' => (int) $pdo->query('SELECT COUNT(*) FROM gecisler')->fetchColumn(),
                    'limit' => $limit,
                    'kayitlar' => $kayitlar,
                    'ozet' => $this->dbOzetGetir($pdo),
                    '_sunucu_zaman' => date('Y-m-d\TH:i:s'),
                ];

                Cache::put($cacheKey, $payload, $ttlSeconds);
                return response()->json($payload);
            }

            return response()->json([
                'hata' => 'Gecersiz action parametresi',
                'gecerli_actionlar' => ['durum', 'panel'],
            ], 400);
        } catch (Throwable $e) {
            return response()->json([
                'hata' => 'MySQL baglanti/sorgu hatasi',
                'mesaj' => $e->getMessage(),
            ], 500);
        }
    }

    public function csv()
    {
        try {
            $pdo = DB::connection('legacy')->getPdo();
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

            return Response::make($csv, 200, [
                'Content-Type' => 'text/csv; charset=utf-8',
                'Content-Disposition' => 'attachment; filename="'.$filename.'"',
                'Cache-Control' => 'no-store',
            ]);
        } catch (Throwable $e) {
            abort(500, $e->getMessage());
        }
    }

    public function kare()
    {
        $path = $this->legacyPath('canli_kare.jpg');
        abort_unless(is_file($path), 404);

        return response()->file($path, [
            'Cache-Control' => 'no-store, no-cache, must-revalidate, max-age=0',
            'Pragma' => 'no-cache',
        ]);
    }

    private function legacyPath(string $name): string
    {
        return base_path('legacy'.DIRECTORY_SEPARATOR.$name);
    }

    private function canliCacheKey(Request $request, string $action, int $limit): string
    {
        $userPart = $request->user()?->id ? ('u:'.$request->user()->id) : ('ip:'.$request->ip());
        $qs = (string) $request->getQueryString();
        return 'canli:'.$userPart.':'.$action.':'.$limit.':'.sha1($qs);
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

    private function parseAgirlik(mixed $value): ?float
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

    private function jsonAgirlikIndexiGetir(string $jsonDurumDosya): array
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

            $key = $plaka.'|'.$tip.'|'.$tarih.'|'.$saat;
            $index[$key] = [
                'giris_agirlik' => $this->parseAgirlik($row['giris_agirlik'] ?? null),
                'cikis_agirlik' => $this->parseAgirlik($row['cikis_agirlik'] ?? null),
                'net_agirlik' => $this->parseAgirlik($row['net_agirlik'] ?? null),
            ];
        }
        return $index;
    }

    private function csvAgirlikIndexiGetir(string $csvDosya): array
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
                'giris_agirlik' => $this->parseAgirlik($row[4] ?? null),
                'cikis_agirlik' => $this->parseAgirlik($row[7] ?? null),
                'net_agirlik' => $this->parseAgirlik($row[8] ?? null),
            ];
            $exact[$plaka.'|'.$tip.'|'.$tarih.'|'.$saat] = $weights;
            $minute[$plaka.'|'.$tip.'|'.$tarih.'|'.substr($saat, 0, 5)] = $weights;
        }

        return ['exact' => $exact, 'minute' => $minute];
    }

    private function agirlikBul(string $plaka, string $tip, string $tarih, string $saat, array $jsonIndex, array $csvIndex): ?array
    {
        $exactKey = $plaka.'|'.$tip.'|'.$tarih.'|'.$saat;
        if (isset($jsonIndex[$exactKey]) && is_array($jsonIndex[$exactKey])) {
            return $jsonIndex[$exactKey];
        }
        if (isset($csvIndex['exact'][$exactKey]) && is_array($csvIndex['exact'][$exactKey])) {
            return $csvIndex['exact'][$exactKey];
        }

        $minuteKey = $plaka.'|'.$tip.'|'.$tarih.'|'.substr($saat, 0, 5);
        if (isset($csvIndex['minute'][$minuteKey]) && is_array($csvIndex['minute'][$minuteKey])) {
            return $csvIndex['minute'][$minuteKey];
        }
        return null;
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
            $weights = $this->agirlikBul($plaka, $tip, $tarih, $saat, $agirlikIndex, $csvAgirlikIndex);

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

    private function durumOkuVeyaFallback(PDO $pdo): array
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

        return $this->dbDurumFallback($pdo);
    }
}

