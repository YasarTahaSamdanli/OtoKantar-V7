<?php

return [
    'runtime_files' => [
        'canli_durum.json',
        'canli_kare.jpg',
        'gecis_gecmisi.jsonl',
        'kantar_raporu.csv',
    ],

    'stale_after_seconds' => env('SYSTEM_HEALTH_STALE_AFTER_SECONDS', 120),
    'disk_warning_free_mb' => env('SYSTEM_HEALTH_DISK_WARNING_FREE_MB', 1024),

    'queue' => [
        'name' => env('LIVE_INGEST_QUEUE', 'live-ingest'),
        'pending_warning' => env('SYSTEM_HEALTH_QUEUE_PENDING_WARNING', 10),
        'pending_critical' => env('SYSTEM_HEALTH_QUEUE_PENDING_CRITICAL', 50),
        'oldest_pending_warning_seconds' => env('SYSTEM_HEALTH_QUEUE_OLDEST_PENDING_WARNING_SECONDS', 60),
        'oldest_pending_critical_seconds' => env('SYSTEM_HEALTH_QUEUE_OLDEST_PENDING_CRITICAL_SECONDS', 300),
        'failed_warning' => env('SYSTEM_HEALTH_FAILED_JOBS_WARNING_COUNT', 1),
        'failed_critical' => env('SYSTEM_HEALTH_FAILED_JOBS_CRITICAL_COUNT', 5),
        'temp_images_warning' => env('SYSTEM_HEALTH_TEMP_IMAGES_WARNING', 5),
        'temp_images_critical' => env('SYSTEM_HEALTH_TEMP_IMAGES_CRITICAL', 25),
        'avg_duration_warning_ms' => env('SYSTEM_HEALTH_AVG_JOB_DURATION_WARNING_MS', 2000),
        'avg_duration_critical_ms' => env('SYSTEM_HEALTH_AVG_JOB_DURATION_CRITICAL_MS', 10000),
        'worker_down_after_seconds' => env('SYSTEM_HEALTH_WORKER_DOWN_AFTER_SECONDS', 300),
        'history_hours' => env('SYSTEM_HEALTH_QUEUE_HISTORY_HOURS', 24),
        'failed_job_sample' => env('SYSTEM_HEALTH_FAILED_JOB_SAMPLE', 10),
    ],

    'alerts' => [
        'enabled' => env('SYSTEM_HEALTH_ALERTS_ENABLED', false),
        'customer_name' => env('SYSTEM_HEALTH_ALERT_CUSTOMER', env('APP_NAME', 'OtoKantar')),
        'min_level' => env('SYSTEM_HEALTH_ALERT_MIN_LEVEL', 'warning'),
        'cooldown_minutes' => env('SYSTEM_HEALTH_ALERT_COOLDOWN_MINUTES', 15),
        'timeout_seconds' => env('SYSTEM_HEALTH_ALERT_TIMEOUT_SECONDS', 5),
        'telegram' => [
            'bot_token' => env('TELEGRAM_BOT_TOKEN'),
            'chat_id' => env('TELEGRAM_CHAT_ID'),
        ],
        'n8n' => [
            'webhook_url' => env('N8N_HEALTH_WEBHOOK_URL'),
        ],
    ],
];
