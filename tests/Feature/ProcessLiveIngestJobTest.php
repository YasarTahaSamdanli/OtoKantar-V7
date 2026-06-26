<?php

namespace Tests\Feature;

use App\Jobs\ProcessLiveIngest;
use App\Services\LiveIngestProcessor;
use Illuminate\Foundation\Testing\RefreshDatabase;
use Illuminate\Support\Facades\File;
use Mockery;
use RuntimeException;
use Tests\TestCase;

class ProcessLiveIngestJobTest extends TestCase
{
    use RefreshDatabase;

    protected function tearDown(): void
    {
        File::deleteDirectory(storage_path('framework/testing/live-ingest-job'));

        parent::tearDown();
    }

    public function test_job_processes_payload_and_temp_image_then_removes_temp_file(): void
    {
        $runtimePath = storage_path('framework/testing/live-ingest-job/runtime');
        $tempPath = storage_path('framework/testing/live-ingest-job/tmp/canli.jpg');
        File::ensureDirectoryExists(dirname($tempPath));
        File::put($tempPath, $this->tinyJpeg());

        config(['services.legacy_runtime.path' => $runtimePath]);

        (new ProcessLiveIngest([
            'event_type' => 'GIRIS',
            'son_guncelleme' => '2026-06-08T12:00:00',
            'kantar_kg' => 1234.5,
            'son_kayit' => [
                'plaka' => '06JOB123',
                'durum' => 'GIRIS',
                'giris_tarih' => '2026-06-08',
                'giris_saat' => '12:00:00',
                'giris_agirlik' => 1234.5,
                'guven' => 0.9,
            ],
        ], $tempPath))->handle(app(\App\Services\LiveIngestProcessor::class));

        $this->assertFileExists($runtimePath.DIRECTORY_SEPARATOR.'canli_durum.json');
        $this->assertFileExists($runtimePath.DIRECTORY_SEPARATOR.'canli_kare.jpg');
        $this->assertFileExists($runtimePath.DIRECTORY_SEPARATOR.'gecis_gecmisi.jsonl');
        $this->assertFileDoesNotExist($tempPath);
        $this->assertDatabaseHas('vehicle_passes', ['plate' => '06JOB123']);
        $this->assertDatabaseHas('live_ingest_job_runs', ['status' => 'success']);
    }

    public function test_duplicate_event_id_is_processed_only_once(): void
    {
        $runtimePath = storage_path('framework/testing/live-ingest-job/runtime');
        config(['services.legacy_runtime.path' => $runtimePath]);

        $payload = $this->payload([
            'event_id' => 'otokantar:v1:06DUP123:GIRIS:20260608120000',
            'son_kayit' => [
                'plaka' => '06DUP123',
                'durum' => 'GIRIS',
                'giris_tarih' => '2026-06-08',
                'giris_saat' => '12:00:00',
                'giris_agirlik' => 1234.5,
                'guven' => 0.9,
            ],
        ]);

        $firstImage = $this->tempImagePath('first.jpg');
        $secondImage = $this->tempImagePath('second.jpg');

        (new ProcessLiveIngest($payload, $firstImage))->handle(app(LiveIngestProcessor::class));
        (new ProcessLiveIngest($payload, $secondImage))->handle(app(LiveIngestProcessor::class));

        $this->assertDatabaseCount('vehicle_passes', 1);
        $this->assertSame(1, count(file($runtimePath.DIRECTORY_SEPARATOR.'gecis_gecmisi.jsonl', FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES) ?: []));
        $this->assertFileDoesNotExist($firstImage);
        $this->assertFileDoesNotExist($secondImage);
    }

    public function test_legacy_payload_without_event_id_gets_canonical_event_id_and_is_idempotent(): void
    {
        $runtimePath = storage_path('framework/testing/live-ingest-job/runtime');
        config(['services.legacy_runtime.path' => $runtimePath]);

        $payload = $this->payload([
            'son_kayit' => [
                'plaka' => '06LEG123',
                'durum' => 'GIRIS',
                'giris_tarih' => '2026-06-08',
                'giris_saat' => '12:00:00',
                'giris_agirlik' => 1234.5,
                'guven' => 0.9,
            ],
        ]);

        (new ProcessLiveIngest($payload, null))->handle(app(LiveIngestProcessor::class));
        (new ProcessLiveIngest($payload, null))->handle(app(LiveIngestProcessor::class));

        $this->assertDatabaseCount('vehicle_passes', 1);
        $this->assertDatabaseHas('vehicle_passes', [
            'plate' => '06LEG123',
            'event_id' => 'otokantar:v1:06LEG123:GIRIS:20260608120000',
        ]);
    }

    public function test_temp_image_is_removed_when_job_fails(): void
    {
        $tempPath = $this->tempImagePath('failing.jpg');
        $processor = Mockery::mock(LiveIngestProcessor::class);
        $processor->shouldReceive('process')->once()->andThrow(new RuntimeException('forced failure'));

        try {
            (new ProcessLiveIngest($this->payload(), $tempPath))->handle($processor);
            $this->fail('Job exception was not thrown.');
        } catch (RuntimeException $e) {
            $this->assertSame('forced failure', $e->getMessage());
        }

        $this->assertFileDoesNotExist($tempPath);
        $this->assertDatabaseHas('live_ingest_job_runs', ['status' => 'failed']);
    }

    private function payload(array $overrides = []): array
    {
        return array_replace_recursive([
            'event_type' => 'GIRIS',
            'son_guncelleme' => '2026-06-08T12:00:00',
            'kantar_kg' => 1234.5,
            'son_kayit' => [
                'plaka' => '06JOB123',
                'durum' => 'GIRIS',
                'giris_tarih' => '2026-06-08',
                'giris_saat' => '12:00:00',
                'giris_agirlik' => 1234.5,
                'guven' => 0.9,
            ],
        ], $overrides);
    }

    private function tempImagePath(string $name): string
    {
        $tempPath = storage_path('framework/testing/live-ingest-job/tmp/'.$name);
        File::ensureDirectoryExists(dirname($tempPath));
        File::put($tempPath, $this->tinyJpeg());

        return $tempPath;
    }

    private function tinyJpeg(): string
    {
        return base64_decode('/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAX/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAH/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAEFAqf/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAEDAQE/ASP/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAECAQE/ASP/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAY/Al//xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAE/IV//2gAMAwEAAgADAAAAEP/EABQRAQAAAAAAAAAAAAAAAAAAABD/2gAIAQMBAT8QH//EABQRAQAAAAAAAAAAAAAAAAAAABD/2gAIAQIBAT8QH//EABQQAQAAAAAAAAAAAAAAAAAAABD/2gAIAQEAAT8QH//Z', true) ?: "\xFF\xD8\xFF\xD9";
    }
}
