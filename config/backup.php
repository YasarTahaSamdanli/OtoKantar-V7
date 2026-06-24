<?php

return [
    'path' => env('BACKUP_PATH', storage_path('app/backups')),
    'keep' => env('BACKUP_KEEP', 14),
    'mysql_dump_binary' => env('BACKUP_MYSQLDUMP_BINARY', PHP_OS_FAMILY === 'Windows' ? 'C:\\xampp\\mysql\\bin\\mysqldump.exe' : 'mysqldump'),
    'mysql_binary' => env('BACKUP_MYSQL_BINARY', PHP_OS_FAMILY === 'Windows' ? 'C:\\xampp\\mysql\\bin\\mysql.exe' : 'mysql'),
    'pg_dump_binary' => env('BACKUP_PG_DUMP_BINARY', 'pg_dump'),
    'pg_restore_binary' => env('BACKUP_PG_RESTORE_BINARY', 'pg_restore'),
    'include_legacy_runtime' => env('BACKUP_INCLUDE_LEGACY_RUNTIME', true),
    'remote' => [
        'enabled' => env('BACKUP_REMOTE_ENABLED', false),
        'binary' => env('BACKUP_RCLONE_BINARY', 'rclone'),
        'destination' => env('BACKUP_REMOTE_DESTINATION', ''),
        'sync_after_run' => env('BACKUP_REMOTE_SYNC_AFTER_RUN', false),
    ],
];
