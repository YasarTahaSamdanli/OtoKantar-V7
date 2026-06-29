<?php

namespace Tests\Feature;

use Illuminate\Support\Facades\File;
use RuntimeException;
use Tests\TestCase;
use ZipArchive;

class BackupCommandTest extends TestCase
{
    protected function tearDown(): void
    {
        File::deleteDirectory(storage_path('framework/testing/backups'));

        parent::tearDown();
    }

    public function test_backup_command_copies_sqlite_database_and_writes_manifest(): void
    {
        $databasePath = storage_path('framework/testing/backups/database.sqlite');
        $backupPath = storage_path('framework/testing/backups/output');

        File::ensureDirectoryExists(dirname($databasePath));
        File::put($databasePath, 'sqlite-backup-fixture');

        config([
            'database.connections.backup_test' => [
                'driver' => 'sqlite',
                'database' => $databasePath,
                'prefix' => '',
            ],
        ]);

        $this->artisan('backup:run', [
            '--connection' => 'backup_test',
            '--path' => $backupPath,
            '--without-legacy-runtime' => true,
            '--keep' => 2,
        ])->assertSuccessful();

        $backupDirs = File::directories($backupPath);

        $this->assertCount(1, $backupDirs);
        $this->assertFileExists($backupDirs[0].DIRECTORY_SEPARATOR.'database.sqlite');
        $this->assertFileExists($backupDirs[0].DIRECTORY_SEPARATOR.'manifest.json');

        $manifest = json_decode(File::get($backupDirs[0].DIRECTORY_SEPARATOR.'manifest.json'), true);

        $this->assertSame('ok', $manifest['database']['status']);
        $this->assertSame('sqlite', $manifest['database']['driver']);
    }

    public function test_backup_restore_replaces_sqlite_database_from_selected_backup(): void
    {
        $databasePath = storage_path('framework/testing/backups/restore.sqlite');
        $backupPath = storage_path('framework/testing/backups/restore-output');

        File::ensureDirectoryExists(dirname($databasePath));
        File::put($databasePath, 'before-restore');

        config([
            'database.connections.restore_test' => [
                'driver' => 'sqlite',
                'database' => $databasePath,
                'prefix' => '',
            ],
        ]);

        $this->artisan('backup:run', [
            '--connection' => 'restore_test',
            '--path' => $backupPath,
            '--without-legacy-runtime' => true,
            '--keep' => 2,
        ])->assertSuccessful();

        File::put($databasePath, 'after-restore');
        $backupDir = File::directories($backupPath)[0];

        $this->artisan('backup:restore', [
            '--backup' => $backupDir,
            '--connection' => 'restore_test',
            '--database-only' => true,
            '--force' => true,
        ])->assertSuccessful();

        $this->assertSame('before-restore', File::get($databasePath));
    }

    public function test_backup_restore_requires_force_flag(): void
    {
        $this->artisan('backup:restore', [
            '--backup' => 'latest',
        ])->assertFailed();
    }

    public function test_backup_restore_extracts_safe_runtime_zip(): void
    {
        if (! class_exists(ZipArchive::class)) {
            $this->markTestSkipped('ZipArchive extension is not available.');
        }

        $backupDir = storage_path('framework/testing/backups/safe-runtime');
        $databasePath = storage_path('framework/testing/backups/safe-restore.sqlite');
        $runtimePath = storage_path('framework/testing/backups/safe-runtime-target');
        $zipPath = $backupDir.DIRECTORY_SEPARATOR.'legacy_runtime.zip';

        File::ensureDirectoryExists($backupDir);
        File::put($backupDir.DIRECTORY_SEPARATOR.'database.sqlite', 'before-restore');
        File::put($databasePath, 'after-restore');

        $zip = new ZipArchive;
        $this->assertTrue($zip->open($zipPath, ZipArchive::CREATE | ZipArchive::OVERWRITE));
        $zip->addFromString('nested/safe.txt', 'safe runtime file');
        $zip->close();

        File::put($backupDir.DIRECTORY_SEPARATOR.'manifest.json', json_encode([
            'database' => [
                'connection' => 'safe_restore_test',
                'driver' => 'sqlite',
                'file' => $backupDir.DIRECTORY_SEPARATOR.'database.sqlite',
                'status' => 'ok',
            ],
            'legacy_runtime' => [
                'file' => $zipPath,
                'status' => 'ok',
            ],
        ], JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES) ?: '{}');

        config([
            'database.connections.safe_restore_test' => [
                'driver' => 'sqlite',
                'database' => $databasePath,
                'prefix' => '',
            ],
            'services.legacy_runtime.path' => $runtimePath,
        ]);

        $this->artisan('backup:restore', [
            '--backup' => $backupDir,
            '--connection' => 'safe_restore_test',
            '--force' => true,
        ])->assertSuccessful();

        $this->assertSame('before-restore', File::get($databasePath));
        $this->assertSame('safe runtime file', File::get($runtimePath.DIRECTORY_SEPARATOR.'nested'.DIRECTORY_SEPARATOR.'safe.txt'));
    }

    public function test_backup_restore_rejects_runtime_zip_path_traversal(): void
    {
        if (! class_exists(ZipArchive::class)) {
            $this->markTestSkipped('ZipArchive extension is not available.');
        }

        $backupDir = storage_path('framework/testing/backups/malicious-runtime');
        $databasePath = storage_path('framework/testing/backups/restore.sqlite');
        $runtimePath = storage_path('framework/testing/backups/runtime-target');
        $escapePath = storage_path('framework/testing/backups/escape.txt');
        $zipPath = $backupDir.DIRECTORY_SEPARATOR.'legacy_runtime.zip';

        File::ensureDirectoryExists($backupDir);
        File::put($backupDir.DIRECTORY_SEPARATOR.'database.sqlite', 'before-restore');
        File::put($databasePath, 'after-restore');

        $zip = new ZipArchive;
        $this->assertTrue($zip->open($zipPath, ZipArchive::CREATE | ZipArchive::OVERWRITE));
        $zip->addFromString('../escape.txt', 'escaped');
        $zip->addFromString('safe.txt', 'safe');
        $zip->close();

        File::put($backupDir.DIRECTORY_SEPARATOR.'manifest.json', json_encode([
            'database' => [
                'connection' => 'restore_test',
                'driver' => 'sqlite',
                'file' => $backupDir.DIRECTORY_SEPARATOR.'database.sqlite',
                'status' => 'ok',
            ],
            'legacy_runtime' => [
                'file' => $zipPath,
                'status' => 'ok',
            ],
        ], JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES) ?: '{}');

        config([
            'database.connections.restore_test' => [
                'driver' => 'sqlite',
                'database' => $databasePath,
                'prefix' => '',
            ],
            'services.legacy_runtime.path' => $runtimePath,
        ]);

        try {
            $this->artisan('backup:restore', [
                '--backup' => $backupDir,
                '--connection' => 'restore_test',
                '--force' => true,
            ]);

            $this->fail('Unsafe runtime zip entry should fail restore.');
        } catch (RuntimeException $exception) {
            $this->assertStringContainsString('Unsafe runtime zip entry path', $exception->getMessage());
        }

        $this->assertFileDoesNotExist($escapePath);
        $this->assertFileDoesNotExist($runtimePath.DIRECTORY_SEPARATOR.'safe.txt');
    }
}
