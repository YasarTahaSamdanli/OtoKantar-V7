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
        $tomorrow = $today->copy()->addDay();

        return view('dashboard', [
            'summary' => [
                'total_profiles' => VehicleProfile::count(),
                'new_today' => VehicleProfile::where('first_seen_at', '>=', $today)
                    ->where('first_seen_at', '<', $tomorrow)
                    ->count(),
                'recognized_passes_today' => VehiclePass::where('vehicle_recognition_status', 'TANINAN_ARAC')
                    ->where('passed_at', '>=', $today)
                    ->where('passed_at', '<', $tomorrow)
                    ->count(),
                'total_net_ton' => ((float) VehicleProfile::sum('total_net_weight_kg')) / 1000,
            ],
            'newVehicles' => VehicleProfile::query()
                ->where('first_seen_at', '>=', $today)
                ->where('first_seen_at', '<', $tomorrow)
                ->latest('first_seen_at')
                ->limit(3)
                ->get(),
            'frequentVehicles' => VehicleProfile::query()
                ->orderByDesc('total_entry_count')
                ->orderByDesc('last_seen_at')
                ->limit(3)
                ->get(),
            'recentVehicles' => VehicleProfile::query()
                ->latest('last_seen_at')
                ->limit(3)
                ->get(),
        ]);
    }
}
