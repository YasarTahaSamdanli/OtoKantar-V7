<?php

namespace App\Http\Middleware;

use Closure;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Log;
use Symfony\Component\HttpFoundation\Response;
use Throwable;

class StressTestRequestLogger
{
    public function handle(Request $request, Closure $next): Response
    {
        if (! $this->enabled()) {
            return $next($request);
        }

        $started = microtime(true);

        try {
            $response = $next($request);
        } catch (Throwable $e) {
            $this->logException($request, $e, $started);
            throw $e;
        }

        $this->logRequest($request, $response, $started);

        return $response;
    }

    private function enabled(): bool
    {
        return filter_var(env('STRESS_TEST_LOGGING', true), FILTER_VALIDATE_BOOL);
    }

    private function logRequest(Request $request, Response $response, float $started): void
    {
        $context = [
            'ts' => now()->toIso8601String(),
            'method' => $request->method(),
            'path' => $request->path(),
            'route' => optional($request->route())->getName(),
            'status' => $response->getStatusCode(),
            'duration_ms' => round((microtime(true) - $started) * 1000, 2),
            'ip' => $request->ip(),
            'user_id' => $request->user()?->id,
            'query' => $request->query(),
            'content_length' => $request->server('CONTENT_LENGTH'),
            'has_json' => $request->isJson() || $request->input('json') !== null || $request->input('payload') !== null,
            'has_image' => $request->hasFile('image') || $request->input('image_base64') !== null,
        ];

        Log::channel('stress_laravel')->info('laravel_request', $context);

        if ($request->is('api/*') || str_starts_with((string) optional($request->route())->getName(), 'api.')) {
            Log::channel('stress_api')->info('api_request', $context);
        }

        if ($request->is('api/live-ingest')) {
            Log::channel('stress_live_ingest')->info('live_ingest_request', $context);
        }

        if ($response->getStatusCode() >= 400) {
            Log::channel('stress_exceptions')->warning('http_error_response', $context);
        }
    }

    private function logException(Request $request, Throwable $e, float $started): void
    {
        Log::channel('stress_exceptions')->error('request_exception', [
            'ts' => now()->toIso8601String(),
            'method' => $request->method(),
            'path' => $request->path(),
            'route' => optional($request->route())->getName(),
            'duration_ms' => round((microtime(true) - $started) * 1000, 2),
            'ip' => $request->ip(),
            'user_id' => $request->user()?->id,
            'exception' => $e::class,
            'message' => $e->getMessage(),
            'file' => $e->getFile(),
            'line' => $e->getLine(),
        ]);
    }
}
