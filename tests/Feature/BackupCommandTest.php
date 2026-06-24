<?php

namespace Tests\Feature;

use Illuminate\Support\Facades\File;
use Tests\TestCase;

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
}
