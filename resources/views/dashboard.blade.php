<x-app-layout>
    <x-slot name="header">
        <div class="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
            <div>
                <p class="text-xs font-semibold uppercase tracking-[.18em] text-emerald-300">OtoKantar</p>
                <h2 class="mt-2 text-2xl font-semibold text-slate-100">Isletme Ozeti</h2>
            </div>
            <a href="{{ route('canli.view') }}"
               class="inline-flex items-center justify-center rounded-full border border-emerald-400/30 bg-emerald-400/10 px-4 py-2 text-sm font-semibold text-emerald-200 transition hover:bg-emerald-400/15">
                Canli Panele Git
            </a>
        </div>
    </x-slot>

    <div class="min-h-[calc(100vh-8rem)] py-8">
        <div class="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <div class="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                <section class="rounded-xl border border-white/10 bg-[#181f2b]/95 p-5 shadow-[0_20px_48px_rgba(0,0,0,.22)]">
                    <div class="text-sm text-slate-400">Kayitli Arac</div>
                    <div class="mt-3 text-3xl font-semibold text-slate-100">{{ number_format($summary['total_profiles']) }}</div>
                </section>
                <section class="rounded-xl border border-white/10 bg-[#181f2b]/95 p-5 shadow-[0_20px_48px_rgba(0,0,0,.22)]">
                    <div class="text-sm text-slate-400">Bugun Yeni Plaka</div>
                    <div class="mt-3 text-3xl font-semibold text-emerald-300">{{ number_format($summary['new_today']) }}</div>
                </section>
                <section class="rounded-xl border border-white/10 bg-[#181f2b]/95 p-5 shadow-[0_20px_48px_rgba(0,0,0,.22)]">
                    <div class="text-sm text-slate-400">Bugun Taninan Gecis</div>
                    <div class="mt-3 text-3xl font-semibold text-slate-100">{{ number_format($summary['recognized_passes_today']) }}</div>
                </section>
                <section class="rounded-xl border border-white/10 bg-[#181f2b]/95 p-5 shadow-[0_20px_48px_rgba(0,0,0,.22)]">
                    <div class="text-sm text-slate-400">Toplam Net Tonaj</div>
                    <div class="mt-3 text-3xl font-semibold text-slate-100">{{ number_format($summary['total_net_ton'], 1, ',', '.') }}</div>
                </section>
            </div>

            <div class="mt-6 grid items-start gap-5 xl:grid-cols-3">
                @include('vehicle-profiles.partials.profile-list', [
                    'title' => 'Yeni Araclar',
                    'profiles' => $newVehicles,
                    'empty' => 'Henuz yeni arac kaydi yok.',
                    'tab' => 'new',
                ])

                @include('vehicle-profiles.partials.profile-list', [
                    'title' => 'En Sik Gelen Araclar',
                    'profiles' => $frequentVehicles,
                    'empty' => 'Henuz sik gelen arac olusmadi.',
                    'tab' => 'frequent',
                ])

                @include('vehicle-profiles.partials.profile-list', [
                    'title' => 'Son Gorulen Araclar',
                    'profiles' => $recentVehicles,
                    'empty' => 'Henuz arac gecisi yok.',
                    'tab' => 'recent',
                ])
            </div>
        </div>
    </div>
</x-app-layout>
