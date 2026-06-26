<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class LiveIngestJobRun extends Model
{
    protected $fillable = [
        'event_id',
        'queue',
        'status',
        'duration_ms',
        'exception_class',
        'exception_message',
        'started_at',
        'finished_at',
    ];

    protected function casts(): array
    {
        return [
            'duration_ms' => 'integer',
            'started_at' => 'datetime',
            'finished_at' => 'datetime',
        ];
    }
}
