<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Str;
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
                $this->atomicWrite($root.DIRECTORY_SEPARATOR.'canli_kare.jpg', $imageBytes);
                $wrote[] = 'canli_kare.jpg';
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

    private function atomicWrite(string $path, string $contents): void
    {
        $directory = dirname($path);
        if (!is_dir($directory)) {
            mkdir($directory, 0775, true);
        }

        $tmp = $path.'.'.Str::random(10).'.tmp';
        file_put_contents($tmp, $contents, LOCK_EX);
        rename($tmp, $path);
    }
}
