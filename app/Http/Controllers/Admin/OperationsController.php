<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Services\QueueOperationsService;
use App\Services\SystemHealthService;

class OperationsController extends Controller
{
    public function __invoke(SystemHealthService $health, QueueOperationsService $queue)
    {
        return view('admin.operations.index', [
            'health' => $health->report(),
            'queue' => $queue->metrics(),
        ]);
    }
}
