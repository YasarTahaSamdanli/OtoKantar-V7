<?php

use App\Http\Controllers\ProfileController;
use App\Http\Controllers\Admin\UserController as AdminUserController;
use App\Http\Controllers\Admin\AuditLogController as AdminAuditLogController;
use App\Http\Controllers\Admin\OperationsController as AdminOperationsController;
use App\Http\Controllers\Admin\SystemHealthController as AdminSystemHealthController;
use App\Http\Controllers\CanliController;
use App\Http\Controllers\DashboardController;
use App\Http\Controllers\VehicleProfileController;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Route;

Route::middleware(['auth', 'role:admin,employee'])->group(function () {
    Route::get('/', function () {
        return auth()->user()->isAdmin()
            ? redirect()->route('dashboard')
            : redirect()->route('canli.view');
    });

    // Legacy entrypoints (old PHP URLs) -> new protected Laravel routes
    Route::get('/index.php', fn () => redirect()->route('canli.view'));
    Route::get('/api_canli.php', function (Request $request) {
        if ($request->query('action') === 'csv_indir') {
            return redirect()->route('canli.csv');
        }

        $qs = $request->getQueryString();
        $url = route('canli.api') . ($qs ? ('?'.$qs) : '');
        return redirect()->to($url);
    });
    Route::get('/canli_kare.jpg', function (Request $request) {
        $qs = $request->getQueryString();
        $url = route('canli.kare') . ($qs ? ('?'.$qs) : '');
        return redirect()->to($url);
    });

    Route::get('/canli', [CanliController::class, 'view'])->name('canli.view');
    Route::get('/canli/api', [CanliController::class, 'api'])->middleware('throttle:canli-api')->name('canli.api');
    Route::get('/canli/live-ticker', [CanliController::class, 'getLiveTicker'])->middleware('throttle:canli-api')->name('canli.live-ticker');
    Route::get('/canli/archive', [CanliController::class, 'getArchive'])->middleware('throttle:canli-api')->name('canli.archive');
    Route::get('/canli/csv', [CanliController::class, 'csv'])->name('canli.csv');
    Route::get('/canli/kare', [CanliController::class, 'kare'])->middleware('throttle:canli-kare')->name('canli.kare');
    Route::get('/araclar', [VehicleProfileController::class, 'index'])->name('vehicle-profiles.index');
    Route::get('/araclar/{vehicleProfile}', [VehicleProfileController::class, 'show'])->name('vehicle-profiles.show');
    Route::patch('/araclar/{vehicleProfile}', [VehicleProfileController::class, 'update'])->name('vehicle-profiles.update');
    Route::get('/dashboard', DashboardController::class)->middleware('verified')->name('dashboard');
});

Route::middleware('auth')->group(function () {
    Route::get('/profile', [ProfileController::class, 'edit'])->name('profile.edit');
    Route::patch('/profile', [ProfileController::class, 'update'])->name('profile.update');
    Route::delete('/profile', [ProfileController::class, 'destroy'])->name('profile.destroy');
});

Route::middleware(['auth', 'role:admin'])->prefix('admin')->name('admin.')->group(function () {
    Route::resource('users', AdminUserController::class)->only(['index', 'create', 'store']);
    Route::get('operations', AdminOperationsController::class)->name('operations.index');
    Route::get('audit-logs', [AdminAuditLogController::class, 'index'])->name('audit-logs.index');
    Route::delete('audit-logs/live-ingest-accepted', [AdminAuditLogController::class, 'destroyAcceptedLiveIngest'])
        ->name('audit-logs.destroy-live-ingest-accepted');
    Route::get('system-health', AdminSystemHealthController::class)->name('system-health');
});

require __DIR__.'/auth.php';
