<x-app-layout>
    <x-slot name="header">
        <div class="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
            <div>
                <p class="text-xs font-semibold uppercase tracking-[.18em] text-emerald-300">Queue ve operasyon</p>
                <h2 class="mt-2 text-2xl font-semibold text-slate-100">Operations Dashboard</h2>
            </div>
            <div class="rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-300">
                Health Score: <span class="font-semibold text-emerald-300">{{ $health['health_score'] ?? 0 }}</span>
                <span class="mx-2 text-slate-600">/</span>
                {{ $health['overall_status'] ?? 'Unknown' }}
            </div>
        </div>
    </x-slot>

    @php
        $metrics = $queue['metrics'] ?? [];
        $levels = $queue['levels'] ?? [];
        $alertsEnabled = (bool) config('system_health.alerts.enabled', false);
        $badge = function (string $level): string {
            return match ($level) {
                'CRITICAL' => 'border-rose-400/30 bg-rose-400/10 text-rose-200',
                'WARNING' => 'border-amber-400/30 bg-amber-400/10 text-amber-200',
                default => 'border-emerald-400/30 bg-emerald-400/10 text-emerald-200',
            };
        };
        $metricRows = [
            ['Live ingest pending', $metrics['pending_jobs'] ?? 0, $levels['pending_jobs'] ?? 'OK'],
            ['Failed jobs', $metrics['failed_jobs'] ?? 0, $levels['failed_jobs'] ?? 'OK'],
            ['Son basarili ingest', $metrics['last_successful_ingest_at'] ?? '-', 'OK'],
            ['Son basarisiz ingest', $metrics['last_failed_ingest_at'] ?? '-', ($metrics['last_failed_ingest_at'] ?? null) ? 'WARNING' : 'OK'],
            ['Ortalama job suresi', ($metrics['avg_job_duration_ms_24h'] ?? '-') . ' ms', $levels['avg_job_duration_ms_24h'] ?? 'OK'],
            ['En eski pending yas', ($metrics['oldest_pending_job_age_seconds'] ?? '-') . ' sn', $levels['oldest_pending_job_age_seconds'] ?? 'OK'],
            ['Pending temp image', $metrics['pending_temp_images'] ?? 0, $levels['pending_temp_images'] ?? 'OK'],
        ];
    @endphp

    <div class="min-h-[calc(100vh-8rem)] py-8">
        <div class="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            @if (session('status'))
                <div class="mb-6 rounded-2xl border border-emerald-400/25 bg-emerald-400/10 px-5 py-4 text-sm text-emerald-200">
                    {{ session('status') }}
                </div>
            @endif

            <section class="mb-6 grid gap-4 lg:grid-cols-[1fr_2fr]">
                <div class="rounded-2xl border border-white/10 bg-[#181f2b]/95 p-6 shadow-[0_20px_48px_rgba(0,0,0,.24)]">
                    <div class="text-xs font-semibold uppercase tracking-[.16em] text-slate-500">Overall</div>
                    <div class="mt-4 text-5xl font-semibold text-slate-100">{{ $health['health_score'] ?? 0 }}</div>
                    <div class="mt-3 inline-flex rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-[.12em] {{ $badge($queue['status'] ?? 'OK') }}">
                        {{ $health['overall_status'] ?? 'Unknown' }}
                    </div>
                    <div class="mt-5 text-sm text-slate-400">
                        {{ $queue['worker']['message'] ?? 'Queue worker durumu okunamadi.' }}
                    </div>
                </div>

                <div class="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                    @foreach($metricRows as [$label, $value, $level])
                        <div class="rounded-2xl border border-white/10 bg-[#181f2b]/95 p-5">
                            <div class="flex items-start justify-between gap-3">
                                <div class="text-xs font-semibold uppercase tracking-[.14em] text-slate-500">{{ $label }}</div>
                                <span class="rounded-full border px-2 py-0.5 text-[10px] font-semibold {{ $badge($level) }}">{{ $level }}</span>
                            </div>
                            <div class="mt-4 break-words text-2xl font-semibold text-slate-100" style="font-family: 'JetBrains Mono', monospace;">{{ $value }}</div>
                        </div>
                    @endforeach
                </div>
            </section>

            <section class="mb-6 rounded-2xl border border-rose-400/20 bg-rose-950/20 p-6">
                <div class="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
                    <div class="max-w-2xl">
                        <div class="text-sm font-semibold text-rose-100">Canli test verilerini sifirla</div>
                        <div class="mt-2 text-sm leading-6 text-rose-200/75">
                            Kantar gecisleri, arac kartlari, ingest job/cache kayitlari ve runtime dosyalari temizlenir. Kullanici hesaplari korunur.
                        </div>
                    </div>
                    <form method="POST" action="{{ route('admin.operations.reset-live-data') }}" class="grid gap-3 sm:grid-cols-[1fr_auto] lg:min-w-[440px]">
                        @csrf
                        <label class="block">
                            <span class="mb-2 block text-xs font-semibold uppercase tracking-[.14em] text-rose-200/70">Onay</span>
                            <input name="confirm" value="{{ old('confirm') }}" placeholder="SIFIRLA"
                                   class="w-full rounded-xl border-white/10 bg-[#0f151f] px-4 py-2.5 text-sm text-slate-100 placeholder:text-slate-600 focus:border-rose-300 focus:ring-rose-300">
                            @error('confirm')
                                <span class="mt-2 block text-xs text-rose-200">{{ $message }}</span>
                            @enderror
                        </label>
                        <button type="submit"
                                class="self-end rounded-xl border border-rose-300/30 bg-rose-400/15 px-5 py-2.5 text-xs font-semibold uppercase tracking-[.14em] text-rose-100 hover:bg-rose-400/20">
                            Sifirla
                        </button>
                    </form>
                </div>
            </section>

            <section class="mb-6 rounded-2xl border border-white/10 bg-[#181f2b]/95 p-6">
                <div class="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                    <div>
                        <div class="text-sm font-semibold text-slate-100">Alarm sistemi</div>
                        <div class="mt-1 text-sm text-slate-500">Health score ve queue metrikleri n8n/Telegram bildirimleri icin kullanilir.</div>
                    </div>
                    <div class="grid gap-3 sm:grid-cols-3">
                        <div class="rounded-xl border border-white/10 bg-white/[.03] px-4 py-3">
                            <div class="text-[10px] font-semibold uppercase tracking-[.14em] text-slate-500">Durum</div>
                            <div class="mt-1 text-sm font-semibold {{ $alertsEnabled ? 'text-emerald-300' : 'text-slate-400' }}">{{ $alertsEnabled ? 'ACTIVE' : 'DISABLED' }}</div>
                        </div>
                        <div class="rounded-xl border border-white/10 bg-white/[.03] px-4 py-3">
                            <div class="text-[10px] font-semibold uppercase tracking-[.14em] text-slate-500">Esik</div>
                            <div class="mt-1 text-sm font-semibold text-slate-100">{{ strtoupper((string) config('system_health.alerts.min_level', 'warning')) }}</div>
                        </div>
                        <div class="rounded-xl border border-white/10 bg-white/[.03] px-4 py-3">
                            <div class="text-[10px] font-semibold uppercase tracking-[.14em] text-slate-500">Cooldown</div>
                            <div class="mt-1 text-sm font-semibold text-slate-100">{{ config('system_health.alerts.cooldown_minutes', 15) }} dk</div>
                        </div>
                    </div>
                </div>
            </section>

            <section class="mb-6 rounded-2xl border border-white/10 bg-[#181f2b]/95 p-6">
                <div class="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
                    <div>
                        <div class="text-sm font-semibold text-slate-100">Son {{ $queue['history_hours'] ?? 24 }} saat</div>
                        <div class="mt-1 text-sm text-slate-500">Queue isleme ozeti ve worker backlog sinyali.</div>
                    </div>
                    <div class="text-sm text-slate-400" style="font-family: 'JetBrains Mono', monospace;">
                        queue={{ $queue['queue'] ?? 'live-ingest' }}
                    </div>
                </div>
                <div class="mt-5 grid gap-4 sm:grid-cols-3">
                    <div class="rounded-xl border border-white/10 bg-white/[.03] p-4">
                        <div class="text-xs uppercase tracking-[.14em] text-slate-500">Basarili</div>
                        <div class="mt-2 text-2xl font-semibold text-emerald-300">{{ $metrics['successful_jobs_24h'] ?? 0 }}</div>
                    </div>
                    <div class="rounded-xl border border-white/10 bg-white/[.03] p-4">
                        <div class="text-xs uppercase tracking-[.14em] text-slate-500">Run hatasi</div>
                        <div class="mt-2 text-2xl font-semibold text-rose-300">{{ $metrics['failed_runs_24h'] ?? 0 }}</div>
                    </div>
                    <div class="rounded-xl border border-white/10 bg-white/[.03] p-4">
                        <div class="text-xs uppercase tracking-[.14em] text-slate-500">Worker</div>
                        <div class="mt-2 text-2xl font-semibold text-slate-100">{{ $queue['worker']['status'] ?? 'UNKNOWN' }}</div>
                    </div>
                </div>
            </section>

            <section class="overflow-hidden rounded-2xl border border-white/10 bg-[#181f2b]/95">
                <div class="border-b border-white/10 px-6 py-5">
                    <div class="text-sm font-semibold text-slate-100">Failed job gecmisi</div>
                    <div class="mt-1 text-sm text-slate-500">Son failed job ornekleri, alarm entegrasyonu icin ayni veri health raporunda da bulunur.</div>
                </div>
                <div class="overflow-x-auto">
                    <table class="min-w-full text-sm">
                        <thead class="bg-white/[.03] text-left text-xs uppercase tracking-[.14em] text-slate-500">
                        <tr>
                            <th class="px-6 py-4 font-semibold">Zaman</th>
                            <th class="px-6 py-4 font-semibold">UUID</th>
                            <th class="px-6 py-4 font-semibold">Ozet</th>
                        </tr>
                        </thead>
                        <tbody class="divide-y divide-white/10">
                        @forelse(($queue['failed_job_sample'] ?? []) as $failed)
                            <tr class="align-top text-slate-300">
                                <td class="whitespace-nowrap px-6 py-4 text-slate-500" style="font-family: 'JetBrains Mono', monospace;">{{ $failed['failed_at'] ?? '-' }}</td>
                                <td class="px-6 py-4 text-slate-400" style="font-family: 'JetBrains Mono', monospace;">{{ $failed['uuid'] ?? '-' }}</td>
                                <td class="px-6 py-4 text-slate-400">{{ $failed['exception_summary'] ?? '-' }}</td>
                            </tr>
                        @empty
                            <tr>
                                <td colspan="3" class="px-6 py-10 text-center text-sm text-slate-500">Failed job yok.</td>
                            </tr>
                        @endforelse
                        </tbody>
                    </table>
                </div>
            </section>
        </div>
    </div>
</x-app-layout>
