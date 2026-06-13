<!DOCTYPE html>
<html lang="{{ str_replace('_', '-', app()->getLocale()) }}">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <meta name="csrf-token" content="{{ csrf_token() }}">

        <title>{{ config('app.name', 'OtoKantar V7') }}</title>

        <!-- Fonts -->
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;600;700&display=swap" rel="stylesheet">

        <!-- Scripts -->
        @vite(['resources/css/app.css', 'resources/js/app.js'])
    </head>
    <body class="text-slate-100 antialiased" style="font-family: Syne, sans-serif;">
        <div class="min-h-screen flex flex-col sm:justify-center items-center px-4 pt-6 sm:pt-0 bg-[radial-gradient(circle_at_top_right,rgba(33,209,159,.09),transparent_24%),linear-gradient(160deg,#0d1015,#151b25)]">
            <div class="text-center">
                <a href="/" class="inline-flex items-center justify-center rounded-2xl border border-emerald-400/30 bg-emerald-400/10 p-4 text-emerald-300">
                    <x-application-logo class="h-10 w-10 fill-current" />
                </a>
                <div class="mt-4 text-sm font-semibold uppercase tracking-[.18em] text-slate-200">OtoKantar V7</div>
                <div class="mt-1 text-xs text-slate-400" style="font-family: 'JetBrains Mono', monospace;">Yonetim ve canli panel</div>
            </div>

            <div class="w-full sm:max-w-md mt-6 rounded-2xl border border-white/10 bg-[#181f2b]/95 px-6 py-6 shadow-[0_24px_70px_rgba(0,0,0,.35)]">
                {{ $slot }}
            </div>
        </div>
    </body>
</html>
