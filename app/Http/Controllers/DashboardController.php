<?php

namespace App\Http\Controllers;

use App\Models\VehiclePass;
use App\Models\VehicleProfile;
use Illuminate\View\View;

class DashboardController extends Controller
{
    public function __invoke(): View
    {
        $today = now()->startOfDay();

        return view('dashboard', [
            'summary' => [
                'total_profiles' => VehicleProfile::count(),
                'new_today' => VehicleProfile::where('first_seen_at', '>=', $today)->count(),
                'recognized_passes_today' => VehiclePass::where('vehicle_recognition_status', 'TANINAN_ARAC')
                    ->where('passed_at', '>=', $today)
                    ->count(),
                'total_net_ton' => ((float) VehicleProfile::sum('total_net_weight_kg')) / 1000,
            ],
            'newVehicles' => VehicleProfile::query()
                ->latest('first_seen_at')
                ->limit(8)
                ->get(),
            'frequentVehicles' => VehicleProfile::query()
                ->orderByDesc('total_entry_count')
                ->orderByDesc('last_seen_at')
                ->limit(8)
                ->get(),
            'recentVehicles' => VehicleProfile::query()
                ->latest('last_seen_at')
                ->limit(8)
                ->get(),
        ]);
    }
}
