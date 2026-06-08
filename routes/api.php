<?php

use App\Http\Controllers\LiveIngestController;
use Illuminate\Support\Facades\Route;

Route::post('/live-ingest', [LiveIngestController::class, 'store'])
    ->middleware('throttle:live-ingest')
    ->name('api.live-ingest');
