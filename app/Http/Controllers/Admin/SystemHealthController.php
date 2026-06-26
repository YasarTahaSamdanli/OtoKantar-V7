<?php

namespace App\Http\Controllers\Admin;

use App\Http\Controllers\Controller;
use App\Services\SystemHealthService;

class SystemHealthController extends Controller
{
    public function __invoke(SystemHealthService $health)
    {
        return response()->json($health->report());
    }
}
