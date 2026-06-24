<?php

namespace App\Services;

use App\Models\AuditLog;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Log;
use Throwable;

class AuditLogService
{
    public function record(
        string $action,
        ?Request $request = null,
        ?Model $subject = null,
        array $metadata = [],
        ?int $userId = null,
    ): void {
        try {
            AuditLog::create([
                'user_id' => $userId ?? $request?->user()?->id,
                'action' => $action,
                'subject_type' => $subject ? $subject::class : null,
                'subject_id' => $subject ? (string) $subject->getKey() : null,
                'metadata' => $metadata === [] ? null : $metadata,
                'ip_address' => $request?->ip(),
                'user_agent' => $request?->userAgent(),
                'created_at' => now(),
            ]);
        } catch (Throwable $e) {
            Log::warning('Audit log yazilamadi', [
                'action' => $action,
                'exception' => $e,
            ]);
        }
    }
}
