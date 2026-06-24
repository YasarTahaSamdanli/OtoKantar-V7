<?php

namespace App\Services;

use App\Models\VehiclePass;
use App\Models\VehicleProfile;
use Illuminate\Support\Facades\DB;

class VehicleProfileService
{
    public const STATUS_NEW = 'YENI_ARAC';
    public const STATUS_RECOGNIZED = 'TANINAN_ARAC';

    public function syncForPass(VehiclePass $pass, bool $isNewPass): VehicleProfile
    {
        return DB::transaction(function () use ($pass, $isNewPass): VehicleProfile {
            $plate = strtoupper(trim($pass->plate));
            $profile = VehicleProfile::query()->where('plate', $plate)->lockForUpdate()->first();
            $status = (!$isNewPass && $pass->vehicle_recognition_status)
                ? $pass->vehicle_recognition_status
                : ($profile === null ? self::STATUS_NEW : self::STATUS_RECOGNIZED);

            if ($profile === null) {
                $profile = VehicleProfile::create([
                    'plate' => $plate,
                    'company_name' => $pass->company_name,
                    'driver_name' => $pass->driver_name,
                    'first_seen_at' => $pass->passed_at,
                    'last_seen_at' => $pass->passed_at,
                    'total_entry_count' => 0,
                    'total_net_weight_kg' => null,
                ]);
            }

            if ($isNewPass) {
                $profile->forceFill([
                    'company_name' => $pass->company_name ?: $profile->company_name,
                    'driver_name' => $pass->driver_name ?: $profile->driver_name,
                    'first_seen_at' => $profile->first_seen_at
                        ? ($profile->first_seen_at->lte($pass->passed_at) ? $profile->first_seen_at : $pass->passed_at)
                        : $pass->passed_at,
                    'last_seen_at' => $profile->last_seen_at
                        ? ($profile->last_seen_at->gte($pass->passed_at) ? $profile->last_seen_at : $pass->passed_at)
                        : $pass->passed_at,
                    'total_entry_count' => $profile->total_entry_count + 1,
                    'total_net_weight_kg' => $this->addNullableDecimal(
                        $profile->total_net_weight_kg,
                        $pass->net_weight_kg
                    ),
                ])->save();
            } else {
                $this->recalculateProfile($profile, $pass);
            }

            $pass->forceFill([
                'vehicle_profile_id' => $profile->id,
                'vehicle_recognition_status' => $status,
                'plate' => $plate,
            ])->save();

            return $profile;
        });
    }

    private function recalculateProfile(VehicleProfile $profile, VehiclePass $pass): void
    {
        $summary = VehiclePass::query()
            ->where('plate', $profile->plate)
            ->selectRaw('
                MIN(passed_at) as first_seen_at,
                MAX(passed_at) as last_seen_at,
                COUNT(*) as total_entry_count,
                SUM(net_weight_kg) as total_net_weight_kg
            ')
            ->first();

        $profile->forceFill([
            'company_name' => $pass->company_name ?: $profile->company_name,
            'driver_name' => $pass->driver_name ?: $profile->driver_name,
            'first_seen_at' => $summary?->first_seen_at,
            'last_seen_at' => $summary?->last_seen_at,
            'total_entry_count' => (int) ($summary?->total_entry_count ?? 0),
            'total_net_weight_kg' => $summary?->total_net_weight_kg,
        ])->save();
    }

    private function addNullableDecimal(mixed $current, mixed $amount): ?float
    {
        if ($amount === null || $amount === '') {
            return $current === null ? null : (float) $current;
        }

        return (float) ($current ?? 0) + (float) $amount;
    }
}
