<?php

use Illuminate\Foundation\Inspiring;
use Illuminate\Support\Facades\Artisan;
use Illuminate\Support\Facades\Hash;
use App\Models\VehiclePass;
use App\Models\User;

Artisan::command('inspire', function () {
    $this->comment(Inspiring::quote());
})->purpose('Display an inspiring quote');

Artisan::command('otokantar:ensure-admin', function () {
    $email = env('ADMIN_EMAIL');
    $password = env('ADMIN_PASSWORD');
    $name = env('ADMIN_NAME', 'Admin');

    if (!$email || !$password) {
        $this->warn('ADMIN_EMAIL veya ADMIN_PASSWORD tanimli degil; admin olusturulmadi.');

        return 0;
    }

    User::updateOrCreate(
        ['email' => $email],
        [
            'name' => $name,
            'password' => Hash::make($password),
            'role' => 'admin',
        ]
    );

    $this->info('Admin kullanici hazir: '.$email);

    return 0;
})->purpose('Create or update the production admin user from environment variables');

Artisan::command('vehicle-passes:latest {--limit=10 : Number of latest records to show}', function () {
    $limit = max(1, min(100, (int) $this->option('limit')));
    $total = VehiclePass::count();
    $latest = VehiclePass::query()
        ->latest('passed_at')
        ->limit($limit)
        ->get([
            'id',
            'event_id',
            'plate',
            'direction',
            'passed_at',
            'entry_weight_kg',
            'exit_weight_kg',
            'net_weight_kg',
            'confidence',
            'source',
            'legacy_pass_key',
        ]);

    $this->info('VehiclePass toplam kayit: '.$total);

    if ($latest->isEmpty()) {
        $this->warn('VehiclePass kaydi bulunamadi.');

        return 0;
    }

    $this->table(
        ['ID', 'Plaka', 'Yon', 'Gecis Zamani', 'Giris Kg', 'Cikis Kg', 'Net Kg', 'Guven', 'Kaynak', 'Event/Legacy Key'],
        $latest->map(fn (VehiclePass $pass): array => [
            $pass->id,
            $pass->plate,
            $pass->direction,
            optional($pass->passed_at)->toDateTimeString(),
            $pass->entry_weight_kg,
            $pass->exit_weight_kg,
            $pass->net_weight_kg,
            $pass->confidence,
            $pass->source,
            $pass->event_id ?: $pass->legacy_pass_key,
        ])->all()
    );

    return 0;
})->purpose('Show latest central vehicle pass records');
