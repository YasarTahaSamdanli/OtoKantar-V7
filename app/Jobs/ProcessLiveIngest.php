<?php

namespace App\Jobs;

use App\Models\LiveIngestJobRun;
use App\Services\LiveIngestProcessor;
use Illuminate\Contracts\Queue\ShouldQueue;
use Illuminate\Foundation\Queue\Queueable;
use Illuminate\Support\Facades\Log;
use Throwable;

class ProcessLiveIngest implements ShouldQueue
{
    use Queueable;

    public int $tries = 3;

    public int $backoff = 10;

    public function __construct(
        public readonly ?array $payload,
        public readonly ?string $imagePath,
    ) {}

    public function handle(LiveIngestProcessor $processor): void
    {
        $started = microtime(true);
        $startedAt = now();
        $status = 'success';
        $exception = null;
        $imageBytes = $this->readImageBytes();

        try {
            $processor->process($this->payload, $imageBytes);
        } catch (Throwable $e) {
            $status = 'failed';
            $exception = $e;

            throw $e;
        } finally {
            $this->recordRun($status, $started, $startedAt, $exception);
            $this->deleteImagePath();
        }
    }

    private function recordRun(string $status, float $started, \Illuminate\Support\Carbon $startedAt, ?Throwable $exception): void
    {
        try {
            LiveIngestJobRun::create([
                'event_id' => is_array($this->payload) ? ($this->payload['event_id'] ?? null) : null,
                'queue' => (string) config('services.legacy_runtime.queue', 'live-ingest'),
                'status' => $status,
                'duration_ms' => (int) round((microtime(true) - $started) * 1000),
                'exception_class' => $exception ? $exception::class : null,
                'exception_message' => $exception ? mb_substr($exception->getMessage(), 0, 500) : null,
                'started_at' => $startedAt,
                'finished_at' => now(),
            ]);
        } catch (Throwable $e) {
            Log::warning('Live ingest job run metriği yazilamadi', ['exception' => $e]);
        }
    }

    private function readImageBytes(): ?string
    {
        if ($this->imagePath === null) {
            return null;
        }

        $bytes = @file_get_contents($this->imagePath);

        return $bytes === false ? null : $bytes;
    }

    private function deleteImagePath(): void
    {
        if ($this->imagePath === null || ! is_file($this->imagePath)) {
            return;
        }

        try {
            @unlink($this->imagePath);
        } catch (Throwable $e) {
            Log::warning('Live ingest gecici gorsel silinemedi', [
                'path' => $this->imagePath,
                'exception' => $e,
            ]);
        }
    }
}
