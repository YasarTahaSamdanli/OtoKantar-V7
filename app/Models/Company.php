<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\HasMany;

class Company extends Model
{
    public const TYPES = [
        'supplier' => 'Tedarikci',
        'carrier' => 'Nakliye',
        'customer' => 'Musteri',
        'contractor' => 'Taseron',
        'other' => 'Diger',
    ];

    protected $fillable = [
        'name',
        'type',
        'tax_number',
        'contact_name',
        'phone',
        'email',
        'is_active',
        'notes',
    ];

    protected function casts(): array
    {
        return [
            'is_active' => 'boolean',
        ];
    }

    public function vehicleProfiles(): HasMany
    {
        return $this->hasMany(VehicleProfile::class);
    }

    public function vehiclePasses(): HasMany
    {
        return $this->hasMany(VehiclePass::class);
    }

    public function typeLabel(): string
    {
        return self::TYPES[$this->type] ?? self::TYPES['other'];
    }
}
