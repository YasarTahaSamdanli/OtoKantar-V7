<?php

namespace App\Services;

use Illuminate\Support\Facades\File;
use Illuminate\Support\Str;
use RuntimeException;
use Symfony\Component\Process\Process;
use ZipArchive;

class ProjectBackupService
{
    /**
     * @return array<string, mixed>
     */
    public function run(
        ?string $connection = null,
        ?string $destination = null,
        ?bool $includeLegacyRuntime = null,
        ?int $keep = null,
        bool $syncRemote = false,
    ): array {
        $backupRoot = $this->absolutePath($destination ?: (string) config('backup.path'));
        File::ensureDirectoryExists($backupRoot, 0750);

        $backupDir = $backupRoot.DIRECTORY_SEPARATOR.now()->format('Ymd_His');
        File::ensureDirectoryExists($backupDir, 0750);

        $manifest = [
            'created_at' => now()->toIso8601String(),
            'app_env' => app()->environment(),
            'backup_dir' => $backupDir,
            'database' => null,
            'legacy_runtime' => null,
            'warnings' => [],
        ];

        $this->backupDatabase($connection ?: (string) config('database.default'), $backupDir, $manifest);

        if ($includeLegacyRuntime ?? (bool) config('backup.include_legacy_runtime')) {
            $this->backupLegacyRuntime($backupDir, $manifest);
        }

        $manifestPath = $backupDir.DIRECTORY_SEPARATOR.'manifest.json';
        File::put($manifestPath, json_encode($manifest, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE) ?: '{}');

        $this->prune($backupRoot, $keep ?? (int) config('backup.keep', 14), $backupDir);

        if ($syncRemote || (bool) config('backup.remote.sync_after_run')) {
            $manifest['remote_sync'] = $this->sync($backupDir);
            File::put($manifestPath, json_encode($manifest, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE) ?: '{}');
        }

        return $manifest;
    }

    /**
     * @return array<int, array<string, mixed>>
     */
    public function list(?string $root = null): array
    {
        $backupRoot = $this->absolutePath($root ?: (string) config('backup.path'));
        if (! is_dir($backupRoot)) {
            return [];
        }

        return collect(File::directories($backupRoot))
            ->map(function (string $dir): array {
                $manifest = $this->readManifest($dir);

                return [
                    'name' => basename($dir),
                    'path' => $dir,
                    'created_at' => $manifest['created_at'] ?? date('c', filemtime($dir) ?: time()),
                    'database' => $manifest['database']['status'] ?? 'unknown',
                    'runtime' => $manifest['legacy_runtime']['status'] ?? 'unknown',
                    'bytes' => $this->directoryBytes($dir),
                ];
            })
            ->sortByDesc('created_at')
            ->values()
            ->all();
    }

    /**
     * @return array<string, mixed>
     */
    public function restore(string $backup, ?string $connection = null, bool $restoreRuntime = true): array
    {
        $backupDir = $this->resolveBackupDir($backup);
        $manifest = $this->readManifest($backupDir);

        if ($manifest === []) {
            throw new RuntimeException("Backup manifest not found: {$backupDir}");
        }

        $result = [
            'backup_dir' => $backupDir,
            'database' => $this->restoreDatabase($manifest, $backupDir, $connection),
            'legacy_runtime' => null,
        ];

        if ($restoreRuntime) {
            $result['legacy_runtime'] = $this->restoreLegacyRuntime($manifest, $backupDir);
        }

        return $result;
    }

    /**
     * @return array<string, mixed>
     */
    public function sync(?string $backup = null): array
    {
        $backupDir = $backup ? $this->resolveBackupDir($backup) : $this->latestBackupDir();
        $destination = trim((string) config('backup.remote.destination'));
        if ($destination === '') {
            throw new RuntimeException('BACKUP_REMOTE_DESTINATION is not configured.');
        }

        $binary = (string) config('backup.remote.binary', 'rclone');
        $process = new Process([
            $binary,
            'copy',
            $backupDir,
            rtrim($destination, '/').'/'.basename($backupDir),
            '--create-empty-src-dirs',
        ], base_path());
        $process->setTimeout(900);
        $process->run();

        if (! $process->isSuccessful()) {
            throw new RuntimeException('Remote backup sync failed: '.$process->getErrorOutput());
        }

        return [
            'status' => 'ok',
            'source' => $backupDir,
            'destination' => rtrim($destination, '/').'/'.basename($backupDir),
        ];
    }

    /**
     * @param  array<string, mixed>  $manifest
     */
    private function backupDatabase(string $connection, string $backupDir, array &$manifest): void
    {
        $config = config('database.connections.'.$connection);
        if (! is_array($config)) {
            throw new RuntimeException("Database connection not found: {$connection}");
        }

        $driver = (string) ($config['driver'] ?? '');
        $manifest['database'] = [
            'connection' => $connection,
            'driver' => $driver,
            'file' => null,
            'status' => 'skipped',
        ];

        match ($driver) {
            'sqlite' => $this->backupSqlite($config, $backupDir, $manifest),
            'mysql', 'mariadb' => $this->backupMysql($config, $backupDir, $manifest),
            'pgsql' => $this->backupPostgres($config, $backupDir, $manifest),
            default => $manifest['warnings'][] = "Unsupported database driver for backup: {$driver}",
        };
    }

    /**
     * @param  array<string, mixed>  $config
     * @param  array<string, mixed>  $manifest
     */
    private function backupSqlite(array $config, string $backupDir, array &$manifest): void
    {
        $database = (string) ($config['database'] ?? '');
        if ($database === ':memory:' || ! is_file($database)) {
            $manifest['warnings'][] = 'SQLite database is in-memory or missing; database backup skipped.';

            return;
        }

        $target = $backupDir.DIRECTORY_SEPARATOR.'database.sqlite';
        File::copy($database, $target);

        $manifest['database']['file'] = $target;
        $manifest['database']['status'] = 'ok';
        $manifest['database']['bytes'] = filesize($target) ?: 0;
    }

    /**
     * @param  array<string, mixed>  $config
     * @param  array<string, mixed>  $manifest
     */
    private function backupMysql(array $config, string $backupDir, array &$manifest): void
    {
        $database = (string) ($config['database'] ?? '');
        $target = $backupDir.DIRECTORY_SEPARATOR.Str::slug($database ?: 'database').'.mysql.sql';
        $binary = (string) config('backup.mysql_dump_binary', 'mysqldump');

        $args = array_values(array_filter([
            $binary,
            '--single-transaction',
            '--quick',
            '--routines',
            '--triggers',
            '--no-tablespaces',
            '--host='.(string) ($config['host'] ?? '127.0.0.1'),
            '--port='.(string) ($config['port'] ?? '3306'),
            '--user='.(string) ($config['username'] ?? ''),
            $database,
        ], fn (string $value): bool => $value !== '--user='));

        $process = new Process($args, base_path(), [
            'MYSQL_PWD' => (string) ($config['password'] ?? ''),
        ]);
        $process->setTimeout(300);

        $handle = fopen($target, 'wb');
        if ($handle === false) {
            throw new RuntimeException('MySQL backup file could not be opened for writing.');
        }

        try {
            $process->run(function (string $type, string $buffer) use ($handle): void {
                if ($type === Process::OUT) {
                    fwrite($handle, $buffer);
                }
            });
        } finally {
            fclose($handle);
        }

        if (! $process->isSuccessful()) {
            File::delete($target);
            throw new RuntimeException('mysqldump failed: '.$process->getErrorOutput());
        }

        $manifest['database']['file'] = $target;
        $manifest['database']['status'] = 'ok';
        $manifest['database']['bytes'] = filesize($target) ?: 0;
    }

    /**
     * @param  array<string, mixed>  $config
     * @param  array<string, mixed>  $manifest
     */
    private function backupPostgres(array $config, string $backupDir, array &$manifest): void
    {
        $database = (string) ($config['database'] ?? '');
        $target = $backupDir.DIRECTORY_SEPARATOR.Str::slug($database ?: 'database').'.pgsql.dump';
        $binary = (string) config('backup.pg_dump_binary', 'pg_dump');

        $args = [
            $binary,
            '--format=custom',
            '--file='.$target,
            '--host='.(string) ($config['host'] ?? '127.0.0.1'),
            '--port='.(string) ($config['port'] ?? '5432'),
            '--username='.(string) ($config['username'] ?? ''),
            $database,
        ];

        $process = new Process($args, base_path(), [
            'PGPASSWORD' => (string) ($config['password'] ?? ''),
        ]);
        $process->setTimeout(300);
        $process->run();

        if (! $process->isSuccessful()) {
            throw new RuntimeException('pg_dump failed: '.$process->getErrorOutput());
        }

        $manifest['database']['file'] = $target;
        $manifest['database']['status'] = 'ok';
        $manifest['database']['bytes'] = filesize($target) ?: 0;
    }

    /**
     * @param  array<string, mixed>  $manifest
     * @return array<string, mixed>
     */
    private function restoreDatabase(array $manifest, string $backupDir, ?string $connection): array
    {
        $database = is_array($manifest['database'] ?? null) ? $manifest['database'] : [];
        $source = $this->manifestFilePath((string) ($database['file'] ?? ''), $backupDir);
        if ($source === '' || ! is_file($source)) {
            throw new RuntimeException('Database backup file is missing.');
        }

        $connectionName = $connection ?: (string) ($database['connection'] ?? config('database.default'));
        $config = config('database.connections.'.$connectionName);
        if (! is_array($config)) {
            throw new RuntimeException("Database connection not found: {$connectionName}");
        }

        $driver = (string) ($config['driver'] ?? '');

        return match ($driver) {
            'sqlite' => $this->restoreSqlite($source, $config, $connectionName),
            'mysql', 'mariadb' => $this->restoreMysql($source, $config, $connectionName),
            'pgsql' => $this->restorePostgres($source, $config, $connectionName),
            default => throw new RuntimeException("Unsupported database driver for restore: {$driver}"),
        };
    }

    /**
     * @param  array<string, mixed>  $config
     * @return array<string, mixed>
     */
    private function restoreSqlite(string $source, array $config, string $connection): array
    {
        $database = (string) ($config['database'] ?? '');
        if ($database === '' || $database === ':memory:') {
            throw new RuntimeException('SQLite restore target is in-memory or missing.');
        }

        File::ensureDirectoryExists(dirname($database), 0750);
        File::copy($source, $database);

        return [
            'status' => 'ok',
            'connection' => $connection,
            'driver' => 'sqlite',
            'target' => $database,
        ];
    }

    /**
     * @param  array<string, mixed>  $config
     * @return array<string, mixed>
     */
    private function restoreMysql(string $source, array $config, string $connection): array
    {
        $database = (string) ($config['database'] ?? '');
        $binary = (string) config('backup.mysql_binary', 'mysql');
        $args = array_values(array_filter([
            $binary,
            '--host='.(string) ($config['host'] ?? '127.0.0.1'),
            '--port='.(string) ($config['port'] ?? '3306'),
            '--user='.(string) ($config['username'] ?? ''),
            $database,
        ], fn (string $value): bool => $value !== '--user='));

        $handle = fopen($source, 'rb');
        if ($handle === false) {
            throw new RuntimeException('MySQL backup file could not be opened.');
        }

        $process = new Process($args, base_path(), [
            'MYSQL_PWD' => (string) ($config['password'] ?? ''),
        ]);
        $process->setInput($handle);
        $process->setTimeout(900);
        $process->run();
        fclose($handle);

        if (! $process->isSuccessful()) {
            throw new RuntimeException('mysql restore failed: '.$process->getErrorOutput());
        }

        return [
            'status' => 'ok',
            'connection' => $connection,
            'driver' => 'mysql',
            'target' => $database,
        ];
    }

    /**
     * @param  array<string, mixed>  $config
     * @return array<string, mixed>
     */
    private function restorePostgres(string $source, array $config, string $connection): array
    {
        $database = (string) ($config['database'] ?? '');
        $binary = (string) config('backup.pg_restore_binary', 'pg_restore');
        $process = new Process([
            $binary,
            '--clean',
            '--if-exists',
            '--no-owner',
            '--dbname='.$database,
            '--host='.(string) ($config['host'] ?? '127.0.0.1'),
            '--port='.(string) ($config['port'] ?? '5432'),
            '--username='.(string) ($config['username'] ?? ''),
            $source,
        ], base_path(), [
            'PGPASSWORD' => (string) ($config['password'] ?? ''),
        ]);
        $process->setTimeout(900);
        $process->run();

        if (! $process->isSuccessful()) {
            throw new RuntimeException('pg_restore failed: '.$process->getErrorOutput());
        }

        return [
            'status' => 'ok',
            'connection' => $connection,
            'driver' => 'pgsql',
            'target' => $database,
        ];
    }

    /**
     * @param  array<string, mixed>  $manifest
     * @return array<string, mixed>
     */
    private function restoreLegacyRuntime(array $manifest, string $backupDir): array
    {
        $runtime = is_array($manifest['legacy_runtime'] ?? null) ? $manifest['legacy_runtime'] : [];
        $source = $this->manifestFilePath((string) ($runtime['file'] ?? ''), $backupDir);
        if ($source === '' || ! is_file($source)) {
            return ['status' => 'skipped', 'reason' => 'missing_runtime_archive'];
        }

        if (! class_exists(ZipArchive::class)) {
            throw new RuntimeException('PHP zip extension is missing; runtime restore cannot continue.');
        }

        $target = $this->absolutePath((string) config('services.legacy_runtime.path', base_path('legacy_python')));
        File::ensureDirectoryExists($target, 0750);

        $zip = new ZipArchive;
        if ($zip->open($source) !== true) {
            throw new RuntimeException('Runtime zip backup could not be opened.');
        }

        $zip->extractTo($target);
        $zip->close();

        return [
            'status' => 'ok',
            'source' => $source,
            'target' => $target,
        ];
    }

    /**
     * @param  array<string, mixed>  $manifest
     */
    private function backupLegacyRuntime(string $backupDir, array &$manifest): void
    {
        $root = $this->absolutePath((string) config('services.legacy_runtime.path', base_path('legacy_python')));
        $manifest['legacy_runtime'] = [
            'source' => $root,
            'file' => null,
            'status' => 'skipped',
        ];

        if (! is_dir($root)) {
            $manifest['warnings'][] = 'Legacy runtime path is missing; runtime backup skipped.';

            return;
        }

        if (! class_exists(ZipArchive::class)) {
            $manifest['warnings'][] = 'PHP zip extension is missing; runtime backup skipped.';

            return;
        }

        $target = $backupDir.DIRECTORY_SEPARATOR.'legacy_runtime.zip';
        $zip = new ZipArchive;
        if ($zip->open($target, ZipArchive::CREATE | ZipArchive::OVERWRITE) !== true) {
            throw new RuntimeException('Runtime zip backup could not be created.');
        }

        $files = File::allFiles($root);
        foreach ($files as $file) {
            $path = $file->getPathname();
            if ($this->shouldSkipRuntimeFile($path)) {
                continue;
            }

            $relative = str_replace('\\', '/', ltrim(Str::after($path, $root), DIRECTORY_SEPARATOR));
            $zip->addFile($path, $relative);
        }

        $zip->close();

        $manifest['legacy_runtime']['file'] = $target;
        $manifest['legacy_runtime']['status'] = 'ok';
        $manifest['legacy_runtime']['bytes'] = filesize($target) ?: 0;
    }

    private function shouldSkipRuntimeFile(string $path): bool
    {
        $normalized = str_replace('\\', '/', $path);

        return str_contains($normalized, '/__pycache__/')
            || str_ends_with($normalized, '.pyc')
            || str_ends_with($normalized, '.log');
    }

    private function prune(string $backupRoot, int $keep, string $currentBackupDir): void
    {
        if ($keep <= 0) {
            return;
        }

        $directories = collect(File::directories($backupRoot))
            ->reject(fn (string $dir): bool => $dir === $currentBackupDir)
            ->sortByDesc(fn (string $dir): int => filemtime($dir) ?: 0)
            ->values();

        $directories->slice(max(0, $keep - 1))->each(fn (string $dir) => File::deleteDirectory($dir));
    }

    /**
     * @return array<string, mixed>
     */
    private function readManifest(string $backupDir): array
    {
        $manifestPath = $backupDir.DIRECTORY_SEPARATOR.'manifest.json';
        if (! is_file($manifestPath)) {
            return [];
        }

        $decoded = json_decode((string) File::get($manifestPath), true);

        return is_array($decoded) ? $decoded : [];
    }

    private function manifestFilePath(string $path, string $backupDir): string
    {
        if ($path !== '' && is_file($path)) {
            return $path;
        }

        if ($path === '') {
            return '';
        }

        $candidate = $backupDir.DIRECTORY_SEPARATOR.basename($path);

        return is_file($candidate) ? $candidate : $path;
    }

    private function resolveBackupDir(string $backup): string
    {
        $backup = rtrim($backup, '\\/');
        if ($backup === 'latest') {
            return $this->latestBackupDir();
        }

        if (is_dir($backup)) {
            return $backup;
        }

        $candidate = $this->absolutePath((string) config('backup.path')).DIRECTORY_SEPARATOR.$backup;
        if (is_dir($candidate)) {
            return $candidate;
        }

        throw new RuntimeException("Backup directory not found: {$backup}");
    }

    private function latestBackupDir(): string
    {
        $backupRoot = $this->absolutePath((string) config('backup.path'));
        $latest = collect(File::directories($backupRoot))
            ->sortByDesc(fn (string $dir): int => filemtime($dir) ?: 0)
            ->first();

        if (! is_string($latest)) {
            throw new RuntimeException('No backups found.');
        }

        return $latest;
    }

    private function directoryBytes(string $dir): int
    {
        return collect(File::allFiles($dir))->sum(fn ($file): int => $file->getSize());
    }

    private function absolutePath(string $path): string
    {
        $path = rtrim($path, '\\/');
        if ($path !== '' && (str_starts_with($path, '/') || preg_match('/^[A-Za-z]:[\/\\\\]/', $path) === 1)) {
            return $path;
        }

        return base_path($path);
    }
}
