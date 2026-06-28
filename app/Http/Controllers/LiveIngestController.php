<?php

namespace App\Http\Controllers;

use App\Jobs\ProcessLiveIngest;
use App\Services\AuditLogService;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Str;
use Symfony\Component\HttpFoundation\Response;
use Throwable;

class LiveIngestController extends Controller
{
    public function __construct(
        private readonly AuditLogService $audit,
    ) {}

    public function store(Request $request)
    {
        $expectedToken = (string) config('services.legacy_runtime.api_token', '');
        $givenToken = $request->bearerToken() ?: (string) $request->header('X-API-Token', '');

        if ($expectedToken === '' || ! hash_equals($expectedToken, $givenToken)) {
            Log::channel('stress_live_ingest')->warning('live_ingest_rejected', [
                'reason' => $expectedToken === '' ? 'token_not_configured' : 'invalid_token',
                'ip' => $request->ip(),
                'content_length' => $request->server('CONTENT_LENGTH'),
            ]);
            $this->audit->record('live_ingest.rejected', $request, metadata: [
                'reason' => $expectedToken === '' ? 'token_not_configured' : 'invalid_token',
                'has_json' => $request->input('json') !== null || $request->input('payload') !== null || $request->json()->all() !== [],
                'has_image' => $request->hasFile('image') || $request->input('image_base64') !== null,
            ]);

            return response()->json(['hata' => 'Yetkisiz istek'], Response::HTTP_UNAUTHORIZED);
        }

        if ($this->requestBodyTooLarge($request)) {
            Log::channel('stress_live_ingest')->warning('live_ingest_rejected', [
                'reason' => 'request_too_large',
                'ip' => $request->ip(),
                'content_length' => $request->server('CONTENT_LENGTH'),
            ]);
            $this->audit->record('live_ingest.rejected', $request, metadata: [
                'reason' => 'request_too_large',
                'content_length' => $request->server('CONTENT_LENGTH'),
            ]);

            return response()->json(['hata' => 'Istek boyutu cok buyuk.'], Response::HTTP_REQUEST_ENTITY_TOO_LARGE);
        }

        try {
            $payload = $this->extractJsonPayload($request);
            if ($payload === false) {
                Log::channel('stress_live_ingest')->warning('live_ingest_rejected', [
                    'reason' => 'invalid_or_too_large_json',
                    'ip' => $request->ip(),
                ]);
                $this->audit->record('live_ingest.rejected', $request, metadata: [
                    'reason' => 'invalid_or_too_large_json',
                ]);

                return response()->json(['hata' => 'JSON verisi gecersiz veya cok buyuk.'], Response::HTTP_UNPROCESSABLE_ENTITY);
            }

            $imageBytes = $this->extractImageBytes($request);
            if ($imageBytes === false) {
                Log::channel('stress_live_ingest')->warning('live_ingest_rejected', [
                    'reason' => 'invalid_or_too_large_image',
                    'ip' => $request->ip(),
                ]);
                $this->audit->record('live_ingest.rejected', $request, metadata: [
                    'reason' => 'invalid_or_too_large_image',
                ]);

                return response()->json(['hata' => 'Gorsel gecersiz, JPG degil veya cok buyuk.'], Response::HTTP_UNPROCESSABLE_ENTITY);
            }

            if ($imageBytes !== null && $payload === null) {
                return response()->json([
                    'hata' => 'JPG yalnizca GIRIS/CIKIS event payload ile kabul edilir.',
                ], Response::HTTP_UNPROCESSABLE_ENTITY);
            }

            if ($payload === null && $imageBytes === null) {
                Log::channel('stress_live_ingest')->warning('live_ingest_rejected', [
                    'reason' => 'empty_payload',
                    'ip' => $request->ip(),
                ]);
                $this->audit->record('live_ingest.rejected', $request, metadata: [
                    'reason' => 'empty_payload',
                ]);

                return response()->json([
                    'hata' => 'JSON veya JPG verisi bulunamadi.',
                ], Response::HTTP_UNPROCESSABLE_ENTITY);
            }

            $imagePath = $this->storePendingImage($imageBytes);
            $queue = (string) config('services.legacy_runtime.queue', 'live-ingest');

            try {
                ProcessLiveIngest::dispatch($payload, $imagePath)->onQueue($queue);
            } catch (Throwable $e) {
                $this->deletePendingImage($imagePath);

                throw $e;
            }

            Log::channel('stress_live_ingest')->info('live_ingest_accepted', [
                'queued' => true,
                'queue' => $queue,
                'event_type' => $payload['event_type'] ?? $payload['olay_tipi'] ?? $payload['_event_type'] ?? null,
                'ip' => $request->ip(),
            ]);

            return response()->json([
                'ok' => true,
                'queued' => true,
                'queue' => $queue,
            ]);
        } catch (Throwable $e) {
            Log::error('Live ingest kayit hatasi', ['exception' => $e]);
            Log::channel('stress_exceptions')->error('live_ingest_exception', [
                'exception' => $e::class,
                'message' => $e->getMessage(),
                'file' => $e->getFile(),
                'line' => $e->getLine(),
            ]);

            return response()->json([
                'hata' => 'Canli veri kaydedilemedi.',
            ], Response::HTTP_INTERNAL_SERVER_ERROR);
        }
    }

    private function requestBodyTooLarge(Request $request): bool
    {
        $contentLength = (int) $request->server('CONTENT_LENGTH', 0);
        if ($contentLength <= 0) {
            return false;
        }

        $max = $this->maxJsonBytes() + $this->maxImageBytes() + 65536;

        return $contentLength > $max;
    }

    private function extractJsonPayload(Request $request): array|false|null
    {
        $json = $request->input('json');

        if (is_string($json) && trim($json) !== '') {
            if (strlen($json) > $this->maxJsonBytes()) {
                return false;
            }

            $decoded = json_decode($json, true, 64);

            return is_array($decoded) ? $decoded : false;
        }

        $payload = $request->input('payload');
        if (is_array($payload)) {
            if (strlen(json_encode($payload, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES) ?: '') > $this->maxJsonBytes()) {
                return false;
            }

            return $payload;
        }

        $body = $request->json()->all();
        if ($body !== [] && strlen((string) $request->getContent()) > $this->maxJsonBytes()) {
            return false;
        }

        return $body !== [] ? $body : null;
    }

    private function extractImageBytes(Request $request): string|false|null
    {
        if ($request->hasFile('image')) {
            $file = $request->file('image');

            if (! $file?->isValid() || $file->getSize() > $this->maxImageBytes()) {
                return false;
            }

            $bytes = file_get_contents($file->getRealPath()) ?: null;

            return $this->validJpegBytes($bytes) ? $bytes : false;
        }

        $imageBase64 = $request->input('image_base64');
        if (is_string($imageBase64) && trim($imageBase64) !== '') {
            if (strlen($imageBase64) > (int) ceil($this->maxImageBytes() * 1.4)) {
                return false;
            }

            $imageBase64 = preg_replace('/^data:image\/[a-zA-Z0-9.+-]+;base64,/', '', $imageBase64) ?: $imageBase64;
            $decoded = base64_decode($imageBase64, true);

            return $this->validJpegBytes($decoded) ? $decoded : false;
        }

        return null;
    }

    private function storePendingImage(?string $imageBytes): ?string
    {
        if ($imageBytes === null) {
            return null;
        }

        $directory = storage_path('app/live-ingest-pending');
        if (! is_dir($directory)) {
            mkdir($directory, 0775, true);
        }

        $path = $directory.DIRECTORY_SEPARATOR.Str::uuid().'.jpg';
        file_put_contents($path, $imageBytes, LOCK_EX);

        return $path;
    }

    private function deletePendingImage(?string $path): void
    {
        if ($path !== null && is_file($path)) {
            @unlink($path);
        }
    }

    private function validJpegBytes(?string $bytes): bool
    {
        if ($bytes === null || strlen($bytes) < 4 || strlen($bytes) > $this->maxImageBytes()) {
            return false;
        }

        return str_starts_with($bytes, "\xFF\xD8") && str_ends_with($bytes, "\xFF\xD9");
    }

    private function maxJsonBytes(): int
    {
        return max(1024, (int) config('services.legacy_runtime.max_json_bytes', 262144));
    }

    private function maxImageBytes(): int
    {
        return max(1024, (int) config('services.legacy_runtime.max_image_bytes', 2097152));
    }
}
