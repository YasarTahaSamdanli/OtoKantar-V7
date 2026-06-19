<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Str;
use PDO;
use Symfony\Component\HttpFoundation\Response;
use Throwable;

class LiveIngestController extends Controller
{
    public function store(Request $request)
    {
        $expectedToken = (string) config('services.legacy_runtime.api_token', '');
        $givenToken = $request->bearerToken() ?: (string) $request->header('X-API-Token', '');

        if ($expectedToken === '' || !hash_equals($expectedToken, $givenToken)) {
            return response()->json(['hata' => 'Yetkisiz istek'], Response::HTTP_UNAUTHORIZED);
        }

        $root = $this->runtimeRoot();

        try {
            $wrote = [];

            $payload = $this->extractJsonPayload($request);
            if ($payload !== null) {
                $payload['_remote_ingest_at'] = now()->toIso8601String();
                $this->atomicWrite($root.DIRECTORY_SEPARATOR.'canli_durum.json', json_encode(
                    $payload,
                    JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES | JSON_PRETTY_PRINT
                ) ?: '{}');
                $wrote[] = 'canli_durum.json';
            }

            $imageBytes = $this->extractImageBytes($request);
            if ($imageBytes !== null) {
                if ($this->isTransitionEvent($payload)) {
                    $this->atomicWrite($root.DIRECTORY_SEPARATOR.'canli_kare.jpg', $imageBytes);
                    $wrote[] = 'canli_kare.jpg';
                } elseif ($payload === null) {
                    return response()->json([
                        'hata' => 'JPG yalnizca GIRIS/CIKIS event payload ile kabul edilir.',
                    ], Response::HTTP_UNPROCESSABLE_ENTITY);
                }
            }

            if ($this->isTransitionEvent($payload)) {
                try {
                    if ($this->storeTransitionEvent($payload)) {
                        $wrote[] = 'legacy_db';
                    }
                } catch (Throwable $e) {
                    Log::warning('Live ingest DB kaydi atlandi', ['exception' => $e]);
                }
            }

            if ($wrote === []) {
                return response()->json([
                    'hata' => 'JSON veya JPG verisi bulunamadi.',
                ], Response::HTTP_UNPROCESSABLE_ENTITY);
            }

            return response()->json([
                'ok' => true,
                'yazilanlar' => $wrote,
                'runtime_path' => $root,
            ]);
        } catch (Throwable $e) {
            Log::error('Live ingest kayit hatasi', ['exception' => $e]);

            return response()->json([
                'hata' => 'Canli veri kaydedilemedi.',
            ], Response::HTTP_INTERNAL_SERVER_ERROR);
        }
    }

    private function runtimeRoot(): string
    {
        $root = rtrim((string) config('services.legacy_runtime.path'), '\\/');
        if (!$this->isAbsolutePath($root)) {
            $root = base_path($root);
        }

        if (!is_dir($root)) {
            mkdir($root, 0775, true);
        }

        return $root;
    }

    private function isAbsolutePath(string $path): bool
    {
        return $path !== '' && (str_starts_with($path, '/') || preg_match('/^[A-Za-z]:[\/\\\\]/', $path) === 1);
    }

    private function extractJsonPayload(Request $request): ?array
    {
        $json = $request->input('json');

        if (is_string($json) && trim($json) !== '') {
            $decoded = json_decode($json, true);

            return is_array($decoded) ? $decoded : null;
        }

        $payload = $request->input('payload');
        if (is_array($payload)) {
            return $payload;
        }

        $body = $request->json()->all();

        return $body !== [] ? $body : null;
    }

    private function extractImageBytes(Request $request): ?string
    {
        if ($request->hasFile('image')) {
            $file = $request->file('image');

            return $file?->isValid() ? file_get_contents($file->getRealPath()) ?: null : null;
        }

        $imageBase64 = $request->input('image_base64');
        if (is_string($imageBase64) && trim($imageBase64) !== '') {
            $imageBase64 = preg_replace('/^data:image\/[a-zA-Z0-9.+-]+;base64,/', '', $imageBase64) ?: $imageBase64;
            $decoded = base64_decode($imageBase64, true);

            return $decoded === false ? null : $decoded;
        }

        return null;
    }

    private function isTransitionEvent(?array $payload): bool
    {
        if ($payload === null) {
            return false;
        }

        $eventType = strtoupper(trim((string) (
            $payload['event_type']
            ?? $payload['olay_tipi']
            ?? $payload['_event_type']
            ?? ''
        )));

        return in_array($eventType, ['GIRIS', 'CIKIS'], true);
    }

    private function storeTransitionEvent(?array $payload): bool
    {
        if ($payload === null) {
            return false;
        }

        $record = is_array($payload['son_kayit'] ?? null) ? $payload['son_kayit'] : $payload;
        $plate = strtoupper(trim((string) ($record['plaka'] ?? $payload['plaka'] ?? '')));
        if ($plate === '') {
            return false;
        }

        $direction = strtoupper(trim((string) (
            $payload['event_type']
            ?? $payload['olay_tipi']
            ?? $payload['_event_type']
            ?? $record['tip']
            ?? $record['durum']
            ?? 'GIRIS'
        )));
        $direction = $direction === 'CIKIS' ? 'CIKIS' : 'GIRIS';

        $date = (string) ($direction === 'CIKIS'
            ? ($record['cikis_tarih'] ?? $record['tarih'] ?? '')
            : ($record['giris_tarih'] ?? $record['tarih'] ?? ''));
        $time = (string) ($direction === 'CIKIS'
            ? ($record['cikis_saat'] ?? $record['saat'] ?? '')
            : ($record['giris_saat'] ?? $record['saat'] ?? ''));

        $timestamp = strtotime(trim($date.' '.$time));
        if ($timestamp === false) {
            $timestamp = time();
        }
        $gecisZamani = date('Y-m-d H:i:s', $timestamp);

        $confidence = $this->normalizeConfidence($record['guven'] ?? $payload['guven'] ?? null);
        $pdo = DB::connection('legacy')->getPdo();
        $this->ensureLegacyTables($pdo);

        $pdo->beginTransaction();
        try {
            $select = $pdo->prepare('SELECT id FROM araclar WHERE plaka = :plaka LIMIT 1');
            $select->execute(['plaka' => $plate]);
            $vehicleId = $select->fetchColumn();

            if (!$vehicleId) {
                $insertVehicle = $pdo->prepare('INSERT INTO araclar (plaka) VALUES (:plaka)');
                $insertVehicle->execute(['plaka' => $plate]);
                $vehicleId = (int) $pdo->lastInsertId();
            } else {
                $vehicleId = (int) $vehicleId;
            }

            $duplicate = $pdo->prepare(
                'SELECT COUNT(*) FROM gecisler WHERE id = :id AND yon = :yon AND gecis_zamani = :gecis_zamani'
            );
            $duplicate->execute([
                'id' => $vehicleId,
                'yon' => $direction,
                'gecis_zamani' => $gecisZamani,
            ]);

            if ((int) $duplicate->fetchColumn() === 0) {
                $insertPass = $pdo->prepare(
                    'INSERT INTO gecisler (id, yon, gecis_zamani, guven)
                     VALUES (:id, :yon, :gecis_zamani, :guven)'
                );
                $insertPass->bindValue(':id', $vehicleId, PDO::PARAM_INT);
                $insertPass->bindValue(':yon', $direction);
                $insertPass->bindValue(':gecis_zamani', $gecisZamani);
                $insertPass->bindValue(':guven', $confidence);
                $insertPass->execute();
            }

            $pdo->commit();
            return true;
        } catch (Throwable $e) {
            if ($pdo->inTransaction()) {
                $pdo->rollBack();
            }
            throw $e;
        }
    }

    private function ensureLegacyTables(PDO $pdo): void
    {
        $pdo->exec(
            'CREATE TABLE IF NOT EXISTS araclar (
                id INT AUTO_INCREMENT PRIMARY KEY,
                plaka VARCHAR(20) NOT NULL UNIQUE,
                kara_liste BOOLEAN NOT NULL DEFAULT FALSE,
                ilk_kayit TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB'
        );

        $pdo->exec(
            'CREATE TABLE IF NOT EXISTS gecisler (
                id INT NOT NULL,
                yon VARCHAR(10) NOT NULL,
                gecis_zamani TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                guven FLOAT,
                CONSTRAINT fk_gecis_arac
                  FOREIGN KEY (id) REFERENCES araclar(id)
                  ON DELETE RESTRICT ON UPDATE CASCADE,
                PRIMARY KEY (id, gecis_zamani, yon),
                INDEX idx_gecis_arac_zaman (id, gecis_zamani),
                INDEX idx_gecis_yon (yon)
            ) ENGINE=InnoDB'
        );
    }

    private function normalizeConfidence(mixed $value): ?float
    {
        if ($value === null || $value === '') {
            return null;
        }

        $confidence = (float) $value;
        if ($confidence > 1.0) {
            $confidence /= 100.0;
        }

        return max(0.0, min(1.0, $confidence));
    }

    private function atomicWrite(string $path, string $contents): void
    {
        $directory = dirname($path);
        if (!is_dir($directory)) {
            mkdir($directory, 0775, true);
        }

        $tmp = $path.'.'.Str::random(10).'.tmp';
        file_put_contents($tmp, $contents, LOCK_EX);

        for ($attempt = 0; $attempt < 5; $attempt++) {
            if (@rename($tmp, $path)) {
                return;
            }

            if (PHP_OS_FAMILY === 'Windows' && is_file($path)) {
                @unlink($path);
            }

            usleep(50_000);
        }

        if (!@copy($tmp, $path)) {
            @unlink($tmp);
            throw new \RuntimeException('Dosya atomik olarak yazilamadi: '.$path);
        }

        @unlink($tmp);
    }
}
