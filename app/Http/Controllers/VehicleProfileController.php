<?php

namespace App\Http\Controllers;

use App\Models\Company;
use App\Models\VehicleProfile;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;
use Illuminate\View\View;

class VehicleProfileController extends Controller
{
    public function index(Request $request): View
    {
        $tab = (string) $request->query('tab', 'recent');
        $search = trim((string) $request->query('q', ''));
        $today = now()->startOfDay();
        $tomorrow = $today->copy()->addDay();

        $query = VehicleProfile::query();

        if ($search !== '') {
            $query->where(function ($query) use ($search): void {
                $query->where('plate', 'like', '%'.$search.'%')
                    ->orWhere('company_name', 'like', '%'.$search.'%')
                    ->orWhere('driver_name', 'like', '%'.$search.'%');
            });
        }

        match ($tab) {
            'new' => $query->where('first_seen_at', '>=', $today)
                ->where('first_seen_at', '<', $tomorrow)
                ->latest('first_seen_at'),
            'frequent' => $query->orderByDesc('total_entry_count')->orderByDesc('last_seen_at'),
            default => $query->latest('last_seen_at'),
        };

        return view('vehicle-profiles.index', [
            'profiles' => $query->paginate(20)->withQueryString(),
            'tab' => in_array($tab, ['new', 'frequent', 'recent'], true) ? $tab : 'recent',
            'search' => $search,
        ]);
    }

    public function show(VehicleProfile $vehicleProfile): View
    {
        return view('vehicle-profiles.show', [
            'profile' => $vehicleProfile,
            'companies' => Company::query()->where('is_active', true)->orderBy('name')->get(),
            'passes' => $vehicleProfile->passes()
                ->latest('passed_at')
                ->paginate(20),
        ]);
    }

    public function update(Request $request, VehicleProfile $vehicleProfile): RedirectResponse
    {
        $validated = $request->validate([
            'company_id' => ['nullable', 'exists:companies,id'],
            'company_name' => ['nullable', 'string', 'max:255'],
            'driver_name' => ['nullable', 'string', 'max:255'],
            'notes' => ['nullable', 'string', 'max:5000'],
        ]);

        if (! empty($validated['company_id'])) {
            $validated['company_name'] = Company::query()->findOrFail($validated['company_id'])->name;
        }

        $vehicleProfile->update($validated);

        $vehicleProfile->passes()->update([
            'company_id' => $vehicleProfile->company_id,
            'company_name' => $vehicleProfile->company_name,
            'driver_name' => $vehicleProfile->driver_name,
        ]);

        return redirect()
            ->route('vehicle-profiles.show', $vehicleProfile)
            ->with('status', 'Arac karti guncellendi.');
    }
}
