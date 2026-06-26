<?php

namespace Tests\Feature;

use Illuminate\Foundation\Testing\RefreshDatabase;
use Illuminate\Support\Facades\File;
use Tests\TestCase;

class LiveIngestIdempotencyTest extends TestCase
{
    use RefreshDatabase;

    protected function tearDown(): void
    {
        File::deleteDirectory(storage_path('framework/testing/live-ingest-idempotency'));
        File::deleteDirectory(storage_path('app/live-ingest-pending'));

        parent::tearDown();
    }

    public function test_duplicate_post_with_same_event_id_creates_one_vehicle_pass_and_one_history_line(): void
    {
        $runtimePath = storage_path('framework/testing/live-ingest-idempotency/runtime');

        config([
            'services.legacy_runtime.api_token' => 'secret-token',
            'services.legacy_runtime.path' => $runtimePath,
        ]);

        $body = [
            'json' => json_encode([
                'event_type' => 'GIRIS',
                'event_id' => 'otokantar:v1:06API123:GIRIS:20260608120000',
                'son_guncelleme' => '2026-06-08T12:00:00',
                'kantar_kg' => 1234.5,
                'son_kayit' => [
                    'plaka' => '06API123',
                    'durum' => 'GIRIS',
                    'giris_tarih' => '2026-06-08',
                    'giris_saat' => '12:00:00',
                    'giris_agirlik' => 1234.5,
                    'guven' => 0.9,
                ],
            ]),
            'image_base64' => base64_encode($this->tinyJpeg()),
        ];

        $this->withHeader('Authorization', 'Bearer secret-token')
            ->post('/api/live-ingest', $body)
            ->assertOk()
            ->assertJsonPath('queued', true);

        $this->withHeader('Authorization', 'Bearer secret-token')
            ->post('/api/live-ingest', $body)
            ->assertOk()
            ->assertJsonPath('queued', true);

        $this->assertDatabaseCount('vehicle_passes', 1);
        $this->assertDatabaseHas('vehicle_passes', [
            'plate' => '06API123',
            'event_id' => 'otokantar:v1:06API123:GIRIS:20260608120000',
        ]);

        $historyPath = $runtimePath.DIRECTORY_SEPARATOR.'gecis_gecmisi.jsonl';
        $this->assertFileExists($historyPath);
        $this->assertSame(1, count(file($historyPath, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES) ?: []));
        $this->assertSame([], File::files(storage_path('app/live-ingest-pending')));
    }

    private function tinyJpeg(): string
    {
        return base64_decode('/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAX/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAH/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAEFAqf/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAEDAQE/ASP/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAECAQE/ASP/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAY/Al//xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAE/IV//2gAMAwEAAgADAAAAEP/EABQRAQAAAAAAAAAAAAAAAAAAABD/2gAIAQMBAT8QH//EABQRAQAAAAAAAAAAAAAAAAAAABD/2gAIAQIBAT8QH//EABQQAQAAAAAAAAAAAAAAAAAAABD/2gAIAQEAAT8QH//Z', true) ?: "\xFF\xD8\xFF\xD9";
    }
}
