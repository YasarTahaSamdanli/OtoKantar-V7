<details class="rounded-xl border border-white/10 bg-[#181f2b]/95 shadow-[0_20px_48px_rgba(0,0,0,.22)]">
    <summary class="flex cursor-pointer list-none items-center justify-between gap-3 px-5 py-4 [&::-webkit-details-marker]:hidden">
        <span>
            <span class="block text-base font-semibold text-slate-100">{{ $title }}</span>
            <span class="mt-1 block text-sm text-slate-500">{{ $profiles->count() }} kayit hazir</span>
        </span>
        <span class="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-semibold text-slate-300">Ac / Kapat</span>
    </summary>

    <div class="divide-y divide-white/10 border-t border-white/10">
        @forelse ($profiles as $profile)
            <a href="{{ route('vehicle-profiles.show', $profile) }}" class="block px-5 py-4 transition hover:bg-white/[.03]">
                <div class="flex items-start justify-between gap-3">
                    <div>
                        <div class="text-lg font-semibold text-slate-100">{{ $profile->plate }}</div>
                        <div class="mt-1 text-sm text-slate-400">
                            {{ $profile->company_name ?: 'Firma yok' }} @if($profile->driver_name) / {{ $profile->driver_name }} @endif
                        </div>
                    </div>
                    <span class="shrink-0 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-300">
                        {{ number_format($profile->total_entry_count) }} gecis
                    </span>
                </div>
                <div class="mt-3 flex flex-wrap gap-2 text-xs text-slate-500">
                    <span>Ilk: {{ optional($profile->first_seen_at)->format('d.m.Y H:i') ?: '-' }}</span>
                    <span>Son: {{ optional($profile->last_seen_at)->format('d.m.Y H:i') ?: '-' }}</span>
                </div>
            </a>
        @empty
            <div class="px-5 py-10 text-center text-sm text-slate-500">{{ $empty }}</div>
        @endforelse
    </div>

    <div class="border-t border-white/10 px-5 py-4">
        <a href="{{ route('vehicle-profiles.index', ['tab' => $tab]) }}" class="text-sm font-semibold text-emerald-300 hover:text-emerald-200">
            Tumunu Gor
        </a>
    </div>
</details>
