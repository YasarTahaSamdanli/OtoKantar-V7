<?php

namespace App\Providers;

use Illuminate\Cache\RateLimiting\Limit;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\RateLimiter;
use Illuminate\Support\Facades\URL;
use Illuminate\Support\ServiceProvider;

class AppServiceProvider extends ServiceProvider
{
    /**
     * Register any application services.
     */
    public function register(): void
    {
        //
    }

    /**
     * Bootstrap any application services.
     */
    public function boot(): void
    {
        if (app()->isProduction()) {
            URL::forceScheme('https');
        }

        RateLimiter::for('canli-api', function (Request $request) {
            $key = $request->user()?->id ? ('u:'.$request->user()->id) : ('ip:'.$request->ip());
            return Limit::perMinute(120)->by($key);
        });

        RateLimiter::for('canli-kare', function (Request $request) {
            $key = $request->user()?->id ? ('u:'.$request->user()->id) : ('ip:'.$request->ip());
            return Limit::perMinute(240)->by($key);
        });

        RateLimiter::for('live-ingest', function (Request $request) {
            return Limit::perMinute(240)->by($request->ip());
        });
    }
}
