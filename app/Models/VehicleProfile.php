<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Relations\HasMany;

class VehicleProfile extends Model
{
    protected $fillable = [
        'plate',
        'company_id',
        'company_name',
        'driver_name',
        'first_seen_at',
        'last_seen_at',
        'total_entry_count',
        'total_net_weight_kg',
        'notes',
    ];

    protected function casts(): array
    {
        return [
            'first_seen_at' => 'datetime',
            'last_seen_at' => 'datetime',
            'total_entry_count' => 'integer',
            'total_net_weight_kg' => 'decimal:3',
        ];
    }

    public function passes(): HasMany
    {
        return $this->hasMany(VehiclePass::class);
    }

    public function company(): BelongsTo
    {
        return $this->belongsTo(Company::class);
    }
}
