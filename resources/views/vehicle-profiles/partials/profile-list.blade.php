<section data-profile-panel="{{ $tab }}"
         class="profile-panel rounded-xl border border-white/10 bg-[#181f2b]/95 shadow-[0_20px_48px_rgba(0,0,0,.22)]">
    <button type="button"
            data-profile-toggle
            class="flex w-full cursor-pointer items-center justify-between gap-3 px-5 py-4 text-left"
            aria-expanded="false"
            aria-controls="profile-panel-{{ $tab }}">
        <span>
            <span class="block text-base font-semibold text-slate-100">{{ $title }}</span>
            <span class="mt-1 block text-sm text-slate-500">{{ $profiles->count() }} kayit hazir</span>
        </span>
        <span class="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-semibold text-slate-300">
            <span data-profile-toggle-label>Ac</span>
            <svg class="profile-panel-chevron h-3.5 w-3.5 transition-transform duration-300" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                <path fill-rule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 11.17l3.71-3.94a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z" clip-rule="evenodd" />
            </svg>
        </span>
    </button>

    <div id="profile-panel-{{ $tab }}"
         data-profile-body
         class="profile-panel-body"
         aria-hidden="true">
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
    </div>
</section>
