<?php

use Illuminate\Foundation\Inspiring;
use Illuminate\Support\Facades\Artisan;
use Illuminate\Support\Facades\Hash;
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
