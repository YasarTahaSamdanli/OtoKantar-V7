<?php

namespace Tests\Feature;

use Illuminate\Support\Facades\File;
use App\Services\CanliDataService;
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
                    'son_kayit' => [
                        'plaka' => '06TST123',
                        'durum' => 'GIRIS',
                        'giris_tarih' => '2026-06-08',
                        'giris_saat' => '12:00:00',
                        'giris_agirlik' => 1234.5,
                        'guven' => 0.9,
                    ],
                    'son_10' => [],
                ]),
                'image_base64' => base64_encode('fake-jpg'),
            ])
            ->assertOk()
            ->assertJsonPath('ok', true);

        $this->assertFileExists($runtimePath.DIRECTORY_SEPARATOR.'canli_durum.json');
        $this->assertFileExists($runtimePath.DIRECTORY_SEPARATOR.'canli_kare.jpg');
        $this->assertFileExists($runtimePath.DIRECTORY_SEPARATOR.'gecis_gecmisi.jsonl');
        $this->assertStringContainsString('1234.5', File::get($runtimePath.DIRECTORY_SEPARATOR.'canli_durum.json'));
        $this->assertStringContainsString('"plaka"', File::get($runtimePath.DIRECTORY_SEPARATOR.'gecis_gecmisi.jsonl'));
    }

    public function test_json_panel_fallback_reads_full_history_not_only_last_10(): void
    {
        $runtimePath = storage_path('framework/testing/live-ingest/'.__FUNCTION__);

        config([
            'services.legacy_runtime.path' => $runtimePath,
        ]);

        File::ensureDirectoryExists($runtimePath);
        File::put($runtimePath.DIRECTORY_SEPARATOR.'canli_durum.json', json_encode([
            'son_guncelleme' => '2026-06-19T12:00:00',
            'son_10' => [],
        ]));

        $lines = [];
        for ($i = 1; $i <= 12; $i++) {
            $day = str_pad((string) $i, 2, '0', STR_PAD_LEFT);
            $plate = '06TST'.str_pad((string) $i, 3, '0', STR_PAD_LEFT);
            $lines[] = json_encode([
                'plaka' => $plate,
                'durum' => 'GIRIS',
                'tip' => 'GIRIS',
                'giris_tarih' => '2026-06-'.$day,
                'giris_saat' => '10:00:00',
                'giris_agirlik' => 1000 + $i,
                'guven' => 0.9,
                'gecis_zamani' => '2026-06-'.$day.' 10:00:00',
                '_event_id' => 'event-'.$i,
            ], JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES);
        }
        File::put($runtimePath.DIRECTORY_SEPARATOR.'gecis_gecmisi.jsonl', implode(PHP_EOL, $lines).PHP_EOL);

        $payload = $this->app->make(CanliDataService::class)->jsonOnlyPanelPayload(200, [
            'period' => 'month',
            'month' => '2026-06',
        ]);

        $this->assertSame(12, $payload['toplam']);
        $this->assertCount(12, $payload['kayitlar']);
        $this->assertSame('06TST012', $payload['kayitlar'][0]['plaka']);
    }

    public function test_live_ingest_fills_zero_entry_weight_from_scale_and_keeps_net_empty(): void
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
                    'kantar_kg' => 18500,
                    'son_kayit' => [
                        'plaka' => '06ZERO01',
                        'durum' => 'GIRIS',
                        'giris_tarih' => '2026-06-08',
                        'giris_saat' => '12:00:00',
                        'giris_agirlik' => 0,
                        'net_agirlik' => 999,
                        'guven' => 0.9,
                    ],
                ]),
            ])
            ->assertOk();

        $durum = json_decode(File::get($runtimePath.DIRECTORY_SEPARATOR.'canli_durum.json'), true);

        $this->assertSame(18500, $durum['son_kayit']['giris_agirlik']);
        $this->assertNull($durum['son_kayit']['net_agirlik']);
    }

    public function test_live_ingest_calculates_net_vehicle_and_material_weight_on_exit(): void
    {
        $runtimePath = storage_path('framework/testing/live-ingest/'.__FUNCTION__);

        config([
            'services.legacy_runtime.api_token' => 'secret-token',
            'services.legacy_runtime.path' => $runtimePath,
        ]);

        $this->withHeader('Authorization', 'Bearer secret-token')
            ->post('/api/live-ingest', [
                'json' => json_encode([
                    'event_type' => 'CIKIS',
                    'son_guncelleme' => '2026-06-08T14:00:00',
                    'kantar_kg' => 12000,
                    'son_kayit' => [
                        'plaka' => '06NET001',
                        'durum' => 'CIKIS',
                        'giris_tarih' => '2026-06-08',
                        'giris_saat' => '12:00:00',
                        'giris_agirlik' => 42000,
                        'cikis_tarih' => '2026-06-08',
                        'cikis_saat' => '14:00:00',
                        'cikis_agirlik' => 0,
                        'guven' => 0.9,
                    ],
                ]),
            ])
            ->assertOk();

        $durum = json_decode(File::get($runtimePath.DIRECTORY_SEPARATOR.'canli_durum.json'), true);

        $this->assertSame(12000, $durum['son_kayit']['cikis_agirlik']);
        $this->assertSame(12000, $durum['son_kayit']['arac_agirlik']);
        $this->assertSame(30000, $durum['son_kayit']['malzeme_agirlik']);
        $this->assertSame(30000, $durum['son_kayit']['net_agirlik']);
    }

    public function test_json_panel_fallback_reads_csv_history_when_jsonl_is_missing(): void
    {
        $runtimePath = storage_path('framework/testing/live-ingest/'.__FUNCTION__);

        config([
            'services.legacy_runtime.path' => $runtimePath,
        ]);

        File::ensureDirectoryExists($runtimePath);
        File::put($runtimePath.DIRECTORY_SEPARATOR.'canli_durum.json', json_encode([
            'son_guncelleme' => '2026-06-19T12:00:00',
            'son_10' => [],
        ]));
        File::put($runtimePath.DIRECTORY_SEPARATOR.'kantar_raporu.csv', implode(PHP_EOL, [
            'Plaka;Durum;GirisTarih;GirisSaat;GirisAgirlik(kg);CikisTarih;CikisSaat;CikisAgirlik(kg);NetAgirlik(kg);Guven;Operator;FirmaAdi;SoforAdi;SoforTel;MalzemeCinsi;IrsaliyeNo',
            '06CSV001;ICERIDE;2026-06-18;10:00:00;12000;;;;;0.90;AUTO;;;;;',
            '06CSV002;TAMAMLANDI;2026-06-19;09:00:00;42000;2026-06-19;11:00:00;12000;30000;0.95;AUTO;;;;;',
        ]).PHP_EOL);

        $payload = $this->app->make(CanliDataService::class)->jsonOnlyPanelPayload(200, [
            'period' => 'month',
            'month' => '2026-06',
        ]);

        $this->assertSame(2, $payload['toplam']);
        $this->assertCount(2, $payload['kayitlar']);
        $this->assertSame('06CSV002', $payload['kayitlar'][0]['plaka']);
        $this->assertSame(30000.0, $payload['kayitlar'][0]['net_agirlik']);
    }

    public function test_json_summary_uses_exit_time_and_latest_plate_state(): void
    {
        $today = date('Y-m-d');
        $yesterday = date('Y-m-d', strtotime($today.' -1 day'));
        $runtimePath = storage_path('framework/testing/live-ingest/'.__FUNCTION__);

        config([
            'services.legacy_runtime.path' => $runtimePath,
        ]);

        File::ensureDirectoryExists($runtimePath);
        File::put($runtimePath.DIRECTORY_SEPARATOR.'canli_durum.json', json_encode([
            'son_guncelleme' => $today.'T12:00:00',
            'son_10' => [],
        ]));
        File::put($runtimePath.DIRECTORY_SEPARATOR.'gecis_gecmisi.jsonl', implode(PHP_EOL, [
            json_encode([
                'plaka' => '34SUM001',
                'durum' => 'GIRIS',
                'tip' => 'GIRIS',
                'giris_tarih' => $yesterday,
                'giris_saat' => '10:00:00',
                'giris_agirlik' => 42000,
                'guven' => 0.9,
                'gecis_zamani' => $yesterday.' 10:00:00',
                '_event_id' => 'summary-entry',
            ], JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES),
            json_encode([
                'plaka' => '34SUM001',
                'durum' => 'CIKIS',
                'tip' => 'CIKIS',
                'giris_tarih' => $yesterday,
                'giris_saat' => '10:00:00',
                'giris_agirlik' => 42000,
                'cikis_tarih' => $today,
                'cikis_saat' => '11:00:00',
                'cikis_agirlik' => 12000,
                'net_agirlik' => 30000,
                'guven' => 0.95,
                'gecis_zamani' => $today.' 11:00:00',
                '_event_id' => 'summary-exit',
            ], JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES),
        ]).PHP_EOL);

        $payload = $this->app->make(CanliDataService::class)->jsonOnlyPanelPayload(200);

        $this->assertSame(1, $payload['ozet']['bugun_kayit']);
        $this->assertSame(0, $payload['ozet']['aktif_seans']);
        $this->assertSame(1, $payload['ozet']['tamamlanan']);
    }

    public function test_csv_file_export_filters_completed_rows_by_exit_date(): void
    {
        $runtimePath = storage_path('framework/testing/live-ingest/'.__FUNCTION__);
        $csvPath = $runtimePath.DIRECTORY_SEPARATOR.'kantar_raporu.csv';

        File::ensureDirectoryExists($runtimePath);
        File::put($csvPath, implode(PHP_EOL, [
            'Plaka;Durum;GirisTarih;GirisSaat;GirisAgirlik(kg);CikisTarih;CikisSaat;CikisAgirlik(kg);NetAgirlik(kg);Guven;Operator;FirmaAdi;SoforAdi;SoforTel;MalzemeCinsi;IrsaliyeNo',
            '34CSV999;TAMAMLANDI;2026-06-22;23:55:00;42000;2026-06-23;00:10:00;12000;30000;0.95;AUTO;;;;;',
        ]).PHP_EOL);

        $export = $this->app->make(CanliDataService::class)->csvDosyaIcerikOlustur($csvPath, [
            'period' => 'day',
            'date' => '2026-06-23',
        ]);

        $this->assertSame(1, $export['row_count']);
        $this->assertStringContainsString('34CSV999', $export['content']);
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
