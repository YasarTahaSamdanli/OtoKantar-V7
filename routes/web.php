<?php

use App\Http\Controllers\ProfileController;
use App\Http\Controllers\Admin\UserController as AdminUserController;
use App\Http\Controllers\CanliController;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Route;

Route::middleware(['auth', 'role:admin'])->group(function () {
    Route::get('/', function () {
        return redirect()->route('dashboard');
    });

    Route::get('/dashboard', function () {
        return view('dashboard');
    })->middleware('verified')->name('dashboard');

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
    Route::get('/canli/csv', [CanliController::class, 'csv'])->name('canli.csv');
    Route::get('/canli/kare', [CanliController::class, 'kare'])->middleware('throttle:canli-kare')->name('canli.kare');
});

Route::middleware('auth')->group(function () {
    Route::get('/profile', [ProfileController::class, 'edit'])->name('profile.edit');
    Route::patch('/profile', [ProfileController::class, 'update'])->name('profile.update');
    Route::delete('/profile', [ProfileController::class, 'destroy'])->name('profile.destroy');
});

Route::middleware(['auth', 'role:admin'])->prefix('admin')->name('admin.')->group(function () {
    Route::resource('users', AdminUserController::class)->only(['index', 'create', 'store']);
});

require __DIR__.'/auth.php';
