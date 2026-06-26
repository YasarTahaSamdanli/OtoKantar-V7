<?php

namespace App\Services;

use App\Models\VehiclePass;
use Illuminate\Database\QueryException;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Str;
use PDO;
use RuntimeException;
use Throwable;

class LiveIngestProcessor
{
    public function __construct(
        private readonly VehicleProfileService $vehicleProfiles,
    ) {}

    public function process(?array $payload, ?string $imageBytes): array
    {
        $root = $this->runtimeRoot();
        $wrote = [];

        if ($this->isTransitionEvent($payload)) {
            $payload = $this->enrichTransitionWeights($payload);
            $identity = $this->eventIdentity($payload);
            $payload['event_id'] = $identity['event_id'];

            if ($this->alreadyProcessed($identity['event_id'], $identity['legacy_pass_key'])) {
                return [
                    'wrote' => ['duplicate_skipped'],
                    'runtime_path' => $root,
                    'event_type' => $payload['event_type'] ?? $payload['olay_tipi'] ?? $payload['_event_type'] ?? null,
                    'event_id' => $identity['event_id'],
                    'duplicate' => true,
                ];
            }
        }

        if ($payload !== null) {
            $payload['_remote_ingest_at'] = now()->toIso8601String();
            $this->atomicWrite($root.DIRECTORY_SEPARATOR.'canli_durum.json', json_encode(
                $payload,
                JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES | JSON_PRETTY_PRINT
            ) ?: '{}');
            $wrote[] = 'canli_durum.json';
        }

        if ($imageBytes !== null && $this->isTransitionEvent($payload)) {
            $this->atomicWrite($root.DIRECTORY_SEPARATOR.'canli_kare.jpg', $imageBytes);
            $wrote[] = 'canli_kare.jpg';
        }

        if ($this->isTransitionEvent($payload)) {
            if ($this->storeTransitionHistory($payload, $root)) {
                $wrote[] = 'gecis_gecmisi.jsonl';
            }

            try {
                if ($this->storeTransitionEvent($payload)) {
                    $wrote[] = 'legacy_db';
                }
            } catch (Throwable $e) {
                Log::warning('Live ingest DB kaydi atlandi', ['exception' => $e]);
            }

            try {
                if ($this->storeVehiclePass($payload, $root, $imageBytes !== null)) {
                    $wrote[] = 'vehicle_passes';
                }
            } catch (Throwable $e) {
                Log::warning('VehiclePass dual-write atlandi', ['exception' => $e]);
            }
        }

        if ($wrote === []) {
            throw new RuntimeException('JSON veya JPG verisi bulunamadi.');
        }

        return [
            'wrote' => $wrote,
            'runtime_path' => $root,
            'event_type' => $payload['event_type'] ?? $payload['olay_tipi'] ?? $payload['_event_type'] ?? null,
            'event_id' => $payload['event_id'] ?? null,
            'duplicate' => false,
        ];
    }

    private function runtimeRoot(): string
    {
        $root = rtrim((string) config('services.legacy_runtime.path'), '\\/');
        if (! $this->isAbsolutePath($root)) {
            $root = base_path($root);
        }

        if (! is_dir($root)) {
            mkdir($root, 0775, true);
        }

        return $root;
    }

    private function isAbsolutePath(string $path): bool
    {
        return $path !== '' && (str_starts_with($path, '/') || preg_match('/^[A-Za-z]:[\/\\\\]/', $path) === 1);
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

            if (! $vehicleId) {
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

    private function enrichTransitionWeights(?array $payload): ?array
    {
        if ($payload === null) {
            return null;
        }

        $record = is_array($payload['son_kayit'] ?? null) ? $payload['son_kayit'] : $payload;
        $direction = strtoupper(trim((string) (
            $payload['event_type']
            ?? $payload['olay_tipi']
            ?? $payload['_event_type']
            ?? $record['tip']
            ?? $record['durum']
            ?? 'GIRIS'
        )));
        $direction = $direction === 'CIKIS' ? 'CIKIS' : 'GIRIS';

        $scaleWeight = $this->positiveWeight($payload['kantar_kg'] ?? null);
        $entryWeight = $this->positiveWeight($record['giris_agirlik'] ?? null);
        $exitWeight = $this->positiveWeight($record['cikis_agirlik'] ?? null);

        if ($direction === 'GIRIS') {
            if ($entryWeight === null && $scaleWeight !== null) {
                $record['giris_agirlik'] = $scaleWeight;
            }

            $record['net_agirlik'] = null;
            $record['malzeme_agirlik'] = null;
            $record['arac_agirlik'] = null;
        } else {
            if ($exitWeight === null && $scaleWeight !== null) {
                $exitWeight = $scaleWeight;
                $record['cikis_agirlik'] = $scaleWeight;
            }

            if ($entryWeight !== null && $exitWeight !== null) {
                $netWeight = abs($exitWeight - $entryWeight);
                $record['net_agirlik'] = $netWeight;
                $record['malzeme_agirlik'] = $netWeight;
                $record['arac_agirlik'] = min($entryWeight, $exitWeight);
            }
        }

        if (is_array($payload['son_kayit'] ?? null)) {
            $payload['son_kayit'] = $record;
        } else {
            $payload = array_merge($payload, $record);
        }

        return $payload;
    }

    private function positiveWeight(mixed $value): ?float
    {
        $weight = $this->parseWeight($value);

        return $weight !== null && $weight > 0 ? $weight : null;
    }

    private function storeVehiclePass(?array $payload, string $root, bool $snapshotWritten): bool
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
            ? ($record['cikis_tarih'] ?? $record['tarih'] ?? $record['giris_tarih'] ?? '')
            : ($record['giris_tarih'] ?? $record['tarih'] ?? $record['cikis_tarih'] ?? ''));
        $time = (string) ($direction === 'CIKIS'
            ? ($record['cikis_saat'] ?? $record['saat'] ?? $record['giris_saat'] ?? '')
            : ($record['giris_saat'] ?? $record['saat'] ?? $record['cikis_saat'] ?? ''));

        $timestamp = strtotime(trim($date.' '.$time));
        if ($timestamp === false) {
            $timestamp = time();
        }

        $passedAt = date('Y-m-d H:i:s', $timestamp);
        $identity = $this->eventIdentity($payload, $plate, $direction, $timestamp);
        $eventId = $identity['event_id'];
        $legacyPassKey = $identity['legacy_pass_key'];
        $snapshotPath = $snapshotWritten ? $root.DIRECTORY_SEPARATOR.'canli_kare.jpg' : null;

        $values = [
            'event_id' => $eventId !== '' ? $eventId : null,
            'plate' => $plate,
            'direction' => $direction,
            'status' => $record['durum'] ?? $record['tip'] ?? null,
            'passed_at' => $passedAt,
            'entry_at' => $this->recordDateTime($record, 'giris'),
            'exit_at' => $this->recordDateTime($record, 'cikis'),
            'entry_weight_kg' => $this->parseWeight($record['giris_agirlik'] ?? null),
            'exit_weight_kg' => $this->parseWeight($record['cikis_agirlik'] ?? null),
            'net_weight_kg' => $this->parseWeight($record['net_agirlik'] ?? null),
            'scale_weight_kg' => $this->parseWeight($payload['kantar_kg'] ?? null),
            'confidence' => $this->normalizeConfidence($record['guven'] ?? $payload['guven'] ?? null),
            'snapshot_disk' => $snapshotPath !== null ? 'legacy_runtime' : null,
            'snapshot_path' => $snapshotPath,
            'snapshot_url' => null,
            'source' => 'remote_ingest',
            'source_payload' => $payload,
            'legacy_vehicle_id' => null,
            'legacy_pass_key' => $legacyPassKey,
            'is_blacklisted' => (bool) ($record['kara_liste'] ?? false),
            'operator' => $record['operator'] ?? null,
            'company_name' => $record['firma_adi'] ?? $record['firma'] ?? null,
            'driver_name' => $record['sofor_adi'] ?? null,
            'driver_phone' => $record['sofor_tel'] ?? null,
            'material_type' => $record['malzeme_cinsi'] ?? null,
            'dispatch_no' => $record['irsaliye_no'] ?? null,
        ];

        $vehiclePass = $this->upsertVehiclePass($eventId, $legacyPassKey, $values);

        $this->vehicleProfiles->syncForPass($vehiclePass, $vehiclePass->wasRecentlyCreated);

        return true;
    }

    private function upsertVehiclePass(string $eventId, string $legacyPassKey, array $values): VehiclePass
    {
        try {
            return VehiclePass::updateOrCreate(['event_id' => $eventId], $values);
        } catch (QueryException $e) {
            if (! $this->isUniqueConstraintViolation($e)) {
                throw $e;
            }

            $existing = VehiclePass::query()
                ->where('event_id', $eventId)
                ->orWhere('legacy_pass_key', $legacyPassKey)
                ->first();

            if ($existing !== null) {
                return $existing;
            }

            throw $e;
        }
    }

    private function alreadyProcessed(string $eventId, string $legacyPassKey): bool
    {
        try {
            return VehiclePass::query()
                ->where('event_id', $eventId)
                ->orWhere('legacy_pass_key', $legacyPassKey)
                ->exists();
        } catch (QueryException $e) {
            Log::warning('VehiclePass idempotency kontrolu atlandi', ['exception' => $e]);

            return false;
        }
    }

    private function eventIdentity(
        ?array $payload,
        ?string $plate = null,
        ?string $direction = null,
        ?int $timestamp = null,
    ): array {
        $record = is_array($payload['son_kayit'] ?? null) ? $payload['son_kayit'] : ($payload ?? []);
        $plate = strtoupper(trim((string) ($plate ?? $record['plaka'] ?? $payload['plaka'] ?? '')));
        $direction = strtoupper(trim((string) (
            $direction
            ?? $payload['event_type']
            ?? $payload['olay_tipi']
            ?? $payload['_event_type']
            ?? $record['tip']
            ?? $record['durum']
            ?? 'GIRIS'
        )));
        $direction = $direction === 'CIKIS' ? 'CIKIS' : 'GIRIS';

        if ($timestamp === null) {
            $date = (string) ($direction === 'CIKIS'
                ? ($record['cikis_tarih'] ?? $record['tarih'] ?? $record['giris_tarih'] ?? '')
                : ($record['giris_tarih'] ?? $record['tarih'] ?? $record['cikis_tarih'] ?? ''));
            $time = (string) ($direction === 'CIKIS'
                ? ($record['cikis_saat'] ?? $record['saat'] ?? $record['giris_saat'] ?? '')
                : ($record['giris_saat'] ?? $record['saat'] ?? $record['cikis_saat'] ?? ''));
            $parsed = strtotime(trim($date.' '.$time));
            $timestamp = $parsed === false ? time() : $parsed;
        }

        $legacyPassKey = $this->transitionEventKey($plate, $direction, $timestamp);
        $givenEventId = trim((string) ($payload['event_id'] ?? ''));

        return [
            'event_id' => $givenEventId !== '' ? $givenEventId : $this->canonicalEventId($plate, $direction, $timestamp),
            'legacy_pass_key' => $legacyPassKey,
        ];
    }

    private function canonicalEventId(string $plate, string $direction, int $timestamp): string
    {
        return 'otokantar:v1:'.strtoupper(trim($plate)).':'.$direction.':'.date('YmdHis', $timestamp);
    }

    private function isUniqueConstraintViolation(QueryException $e): bool
    {
        $sqlState = (string) ($e->errorInfo[0] ?? '');
        $driverCode = (string) ($e->errorInfo[1] ?? '');

        return in_array($sqlState, ['23000', '23505'], true)
            || in_array($driverCode, ['1062', '1555', '2067'], true);
    }

    private function recordDateTime(array $record, string $prefix): ?string
    {
        $date = trim((string) ($record[$prefix.'_tarih'] ?? ''));
        $time = trim((string) ($record[$prefix.'_saat'] ?? ''));
        $timestamp = strtotime(trim($date.' '.$time));

        return $timestamp === false ? null : date('Y-m-d H:i:s', $timestamp);
    }

    private function parseWeight(mixed $value): ?float
    {
        if ($value === null || $value === '') {
            return null;
        }

        $value = str_replace(',', '.', (string) $value);

        return is_numeric($value) ? (float) $value : null;
    }

    private function transitionEventKey(string $plate, string $direction, int $timestamp): string
    {
        return sha1($plate.'|'.$direction.'|'.date('Y-m-d H:i:s', $timestamp));
    }

    private function storeTransitionHistory(?array $payload, string $root): bool
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
            ? ($record['cikis_tarih'] ?? $record['tarih'] ?? $record['giris_tarih'] ?? '')
            : ($record['giris_tarih'] ?? $record['tarih'] ?? $record['cikis_tarih'] ?? ''));
        $time = (string) ($direction === 'CIKIS'
            ? ($record['cikis_saat'] ?? $record['saat'] ?? $record['giris_saat'] ?? '')
            : ($record['giris_saat'] ?? $record['saat'] ?? $record['cikis_saat'] ?? ''));

        $timestamp = strtotime(trim($date.' '.$time));
        if ($timestamp === false) {
            $timestamp = time();
        }

        $eventId = trim((string) ($payload['event_id'] ?? ''));
        if ($eventId === '') {
            $eventId = $this->transitionEventKey($plate, $direction, $timestamp);
        }

        $historyRecord = $record;
        $historyRecord['plaka'] = $plate;
        $historyRecord['durum'] = $direction;
        $historyRecord['tip'] = $direction;
        $historyRecord['guven'] = $this->normalizeConfidence($record['guven'] ?? $payload['guven'] ?? null);
        $historyRecord['gecis_zamani'] = date('Y-m-d H:i:s', $timestamp);
        $historyRecord['_event_id'] = $eventId;
        $historyRecord['_remote_ingest_at'] = now()->toIso8601String();

        $path = $root.DIRECTORY_SEPARATOR.'gecis_gecmisi.jsonl';
        $directory = dirname($path);
        if (! is_dir($directory)) {
            mkdir($directory, 0775, true);
        }

        $line = json_encode($historyRecord, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES);
        if ($line === false) {
            return false;
        }

        $handle = fopen($path, 'c+');
        if ($handle === false) {
            return false;
        }

        try {
            flock($handle, LOCK_EX);
            rewind($handle);
            while (($existing = fgets($handle)) !== false) {
                $decoded = json_decode($existing, true);
                if (is_array($decoded) && (string) ($decoded['_event_id'] ?? '') === $eventId) {
                    return false;
                }
            }
            fseek($handle, 0, SEEK_END);
            fwrite($handle, $line.PHP_EOL);
            fflush($handle);

            return true;
        } finally {
            flock($handle, LOCK_UN);
            fclose($handle);
        }
    }

    private function atomicWrite(string $path, string $contents): void
    {
        $directory = dirname($path);
        if (! is_dir($directory)) {
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

        if (! @copy($tmp, $path)) {
            @unlink($tmp);
            throw new RuntimeException('Dosya atomik olarak yazilamadi: '.$path);
        }

        @unlink($tmp);
    }
}
