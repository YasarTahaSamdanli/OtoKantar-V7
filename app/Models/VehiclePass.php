<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

class VehiclePass extends Model
{
    protected $fillable = [
        'event_id',
        'vehicle_profile_id',
        'company_id',
        'vehicle_recognition_status',
        'plate',
        'direction',
        'status',
        'passed_at',
        'entry_at',
        'exit_at',
        'entry_weight_kg',
        'exit_weight_kg',
        'net_weight_kg',
        'scale_weight_kg',
        'confidence',
        'snapshot_disk',
        'snapshot_path',
        'snapshot_url',
        'source',
        'source_payload',
        'legacy_vehicle_id',
        'legacy_pass_key',
        'is_blacklisted',
        'operator',
        'company_name',
        'driver_name',
        'driver_phone',
        'material_type',
        'dispatch_no',
    ];

    protected function casts(): array
    {
        return [
            'passed_at' => 'datetime',
            'entry_at' => 'datetime',
            'exit_at' => 'datetime',
            'entry_weight_kg' => 'decimal:3',
            'exit_weight_kg' => 'decimal:3',
            'net_weight_kg' => 'decimal:3',
            'scale_weight_kg' => 'decimal:3',
            'confidence' => 'decimal:4',
            'source_payload' => 'array',
            'legacy_vehicle_id' => 'integer',
            'is_blacklisted' => 'boolean',
        ];
    }

    public function vehicleProfile(): BelongsTo
    {
        return $this->belongsTo(VehicleProfile::class);
    }

    public function company(): BelongsTo
    {
        return $this->belongsTo(Company::class);
    }
}
