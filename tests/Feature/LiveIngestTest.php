<?php

namespace Tests\Feature;

use Illuminate\Support\Facades\File;
use Tests\TestCase;

class LiveIngestTest extends TestCase
{
    protected function tearDown(): void
    {
        File::deleteDirectory(storage_path('framework/testing/live-ingest'));

        parent::tearDown();
    }

    public function test_live_ingest_requires_token(): void
    {
        config(['services.legacy_runtime.api_token' => 'secret-token']);

        $this->postJson('/api/live-ingest', [
            'son_guncelleme' => now()->toIso8601String(),
        ])->assertUnauthorized();
    }

    public function test_live_ingest_writes_json_and_entry_exit_image(): void
    {
        $runtimePath = storage_path('framework/testing/live-ingest/'.__FUNCTION__);

        config([
            'services.legacy_runtime.api_token' => 'secret-token',
            'services.legacy_runtime.path' => $runtimePath,
        ]);

        $this->withHeader('Authorization', 'Bearer secret-token')
            ->post('/api/live-ingest', [
                'json' => json_encode([
                    'event_type' => 'GIRIS',
                    'son_guncelleme' => '2026-06-08T12:00:00',
                    'kantar_kg' => 1234.5,
                    'son_10' => [],
                ]),
                'image_base64' => base64_encode('fake-jpg'),
            ])
            ->assertOk()
            ->assertJsonPath('ok', true);

        $this->assertFileExists($runtimePath.DIRECTORY_SEPARATOR.'canli_durum.json');
        $this->assertFileExists($runtimePath.DIRECTORY_SEPARATOR.'canli_kare.jpg');
        $this->assertStringContainsString('1234.5', File::get($runtimePath.DIRECTORY_SEPARATOR.'canli_durum.json'));
    }

    public function test_live_ingest_ignores_non_event_image_when_status_json_exists(): void
    {
        $runtimePath = storage_path('framework/testing/live-ingest/'.__FUNCTION__);

        config([
            'services.legacy_runtime.api_token' => 'secret-token',
            'services.legacy_runtime.path' => $runtimePath,
        ]);

        $this->withHeader('Authorization', 'Bearer secret-token')
            ->post('/api/live-ingest', [
                'json' => json_encode([
                    'son_guncelleme' => '2026-06-08T12:00:00',
                    'kantar_kg' => 1234.5,
                ]),
                'image_base64' => base64_encode('fake-jpg'),
            ])
            ->assertOk()
            ->assertJsonMissing(['canli_kare.jpg']);

        $this->assertFileExists($runtimePath.DIRECTORY_SEPARATOR.'canli_durum.json');
        $this->assertFileDoesNotExist($runtimePath.DIRECTORY_SEPARATOR.'canli_kare.jpg');
    }

    public function test_live_ingest_rejects_image_without_event_payload(): void
    {
        $runtimePath = storage_path('framework/testing/live-ingest/'.__FUNCTION__);

        config([
            'services.legacy_runtime.api_token' => 'secret-token',
            'services.legacy_runtime.path' => $runtimePath,
        ]);

        $this->withHeader('Authorization', 'Bearer secret-token')
            ->post('/api/live-ingest', [
                'image_base64' => base64_encode('fake-jpg'),
            ])
            ->assertUnprocessable();

        $this->assertFileDoesNotExist($runtimePath.DIRECTORY_SEPARATOR.'canli_kare.jpg');
    }
}
