<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Models\AuditLog;
use App\Services\AuditLogService;
use Illuminate\Http\Request;

class AuditLogController extends Controller
{
    public function __construct(
        private readonly AuditLogService $audit,
    ) {}

    public function index(Request $request)
    {
        $action = trim((string) $request->query('action', ''));
        $user = trim((string) $request->query('user', ''));

        $logs = AuditLog::query()
            ->with('user:id,name,email,role')
            ->when($action !== '', fn ($query) => $query->where('action', $action))
            ->when($user !== '', function ($query) use ($user) {
                $query->whereHas('user', function ($userQuery) use ($user) {
                    $userQuery
                        ->where('name', 'like', '%'.$user.'%')
                        ->orWhere('email', 'like', '%'.$user.'%');
                });
            })
            ->latest('created_at')
            ->paginate(30)
            ->withQueryString();

        return view('admin.audit-logs.index', [
            'logs' => $logs,
            'actions' => AuditLog::query()->distinct()->orderBy('action')->pluck('action'),
            'filters' => [
                'action' => $action,
                'user' => $user,
            ],
        ]);
    }

    public function destroyAcceptedLiveIngest(Request $request)
    {
        $deleted = AuditLog::query()
            ->where('action', 'live_ingest.accepted')
            ->delete();

        $this->audit->record('admin.audit_logs.cleaned', $request, metadata: [
            'action' => 'live_ingest.accepted',
            'deleted' => $deleted,
        ]);

        return redirect()
            ->route('admin.audit-logs.index')
            ->with('status', "{$deleted} live_ingest.accepted kaydi silindi.");
    }
}
