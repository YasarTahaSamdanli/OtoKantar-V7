<?php

namespace Tests\Feature;

use Illuminate\Http\UploadedFile;
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

    public function test_live_ingest_writes_json_and_image(): void
    {
        $runtimePath = storage_path('framework/testing/live-ingest');

        config([
            'services.legacy_runtime.api_token' => 'secret-token',
            'services.legacy_runtime.path' => $runtimePath,
        ]);

        $this->withHeader('Authorization', 'Bearer secret-token')
            ->post('/api/live-ingest', [
                'json' => json_encode([
                    'son_guncelleme' => '2026-06-08T12:00:00',
                    'kantar_kg' => 1234.5,
                    'son_10' => [],
                ]),
                'image' => UploadedFile::fake()->create('canli_kare.jpg', 1, 'image/jpeg'),
            ])
            ->assertOk()
            ->assertJsonPath('ok', true);

        $this->assertFileExists($runtimePath.DIRECTORY_SEPARATOR.'canli_durum.json');
        $this->assertFileExists($runtimePath.DIRECTORY_SEPARATOR.'canli_kare.jpg');
        $this->assertStringContainsString('1234.5', File::get($runtimePath.DIRECTORY_SEPARATOR.'canli_durum.json'));
    }
}
