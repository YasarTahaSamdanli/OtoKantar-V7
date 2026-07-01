<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Services\LiveDataResetService;
use App\Services\QueueOperationsService;
use App\Services\SystemHealthService;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;

class OperationsController extends Controller
{
    public function __invoke(SystemHealthService $health, QueueOperationsService $queue)
    {
        return view('admin.operations.index', [
            'health' => $health->report(),
            'queue' => $queue->metrics(),
        ]);
    }

    public function resetLiveData(Request $request, LiveDataResetService $reset): RedirectResponse
    {
        $data = $request->validate([
            'confirm' => ['required', 'string', 'in:SIFIRLA'],
        ]);

        $result = $reset->reset();

        return redirect()
            ->route('admin.operations.index')
            ->with('status', 'Canli test verileri sifirlandi. Silinen tablolar: '.implode(', ', $result['tables']));
    }
}
