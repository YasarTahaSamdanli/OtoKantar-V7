<x-app-layout>
    <x-slot name="header">
        <div class="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
            <div>
                <p class="text-xs font-semibold uppercase tracking-[.18em] text-emerald-300">Plaka Takibi</p>
                <h2 class="mt-2 text-2xl font-semibold text-slate-100">Arac Kartlari</h2>
            </div>
            <form method="GET" action="{{ route('vehicle-profiles.index') }}" class="flex w-full gap-2 sm:w-auto">
                <input type="hidden" name="tab" value="{{ $tab }}">
                <input name="q" value="{{ $search }}" placeholder="Plaka, firma veya sofor ara"
                       class="min-w-0 flex-1 rounded-full border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-100 placeholder:text-slate-500 focus:border-emerald-400 focus:ring-emerald-400 sm:w-80">
                <button class="rounded-full border border-emerald-400/30 bg-emerald-400/10 px-5 py-2 text-sm font-semibold text-emerald-200 transition hover:bg-emerald-400/15">
                    Ara
                </button>
            </form>
        </div>
    </x-slot>

    <div class="min-h-[calc(100vh-8rem)] py-8">
        <div class="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <div class="mb-5 flex flex-wrap gap-2">
                <a href="{{ route('vehicle-profiles.index', ['tab' => 'recent', 'q' => $search]) }}"
                   class="rounded-full border px-4 py-2 text-sm font-semibold {{ $tab === 'recent' ? 'border-emerald-400/30 bg-emerald-400/10 text-emerald-300' : 'border-white/10 bg-white/5 text-slate-300 hover:bg-white/10' }}">
                    Son Gorulenler
                </a>
                <a href="{{ route('vehicle-profiles.index', ['tab' => 'new', 'q' => $search]) }}"
                   class="rounded-full border px-4 py-2 text-sm font-semibold {{ $tab === 'new' ? 'border-emerald-400/30 bg-emerald-400/10 text-emerald-300' : 'border-white/10 bg-white/5 text-slate-300 hover:bg-white/10' }}">
                    Yeni Araclar
                </a>
                <a href="{{ route('vehicle-profiles.index', ['tab' => 'frequent', 'q' => $search]) }}"
                   class="rounded-full border px-4 py-2 text-sm font-semibold {{ $tab === 'frequent' ? 'border-emerald-400/30 bg-emerald-400/10 text-emerald-300' : 'border-white/10 bg-white/5 text-slate-300 hover:bg-white/10' }}">
                    En Sik Gelenler
                </a>
            </div>

            <section class="overflow-hidden rounded-xl border border-white/10 bg-[#181f2b]/95 shadow-[0_20px_48px_rgba(0,0,0,.22)]">
                <div class="overflow-x-auto">
                    <table class="min-w-full divide-y divide-white/10">
                        <thead class="bg-white/[.03]">
                            <tr>
                                <th class="px-5 py-3 text-left text-xs font-semibold uppercase tracking-[.14em] text-slate-500">Plaka</th>
                                <th class="px-5 py-3 text-left text-xs font-semibold uppercase tracking-[.14em] text-slate-500">Firma / Sofor</th>
                                <th class="px-5 py-3 text-left text-xs font-semibold uppercase tracking-[.14em] text-slate-500">Ilk Gecis</th>
                                <th class="px-5 py-3 text-left text-xs font-semibold uppercase tracking-[.14em] text-slate-500">Son Gecis</th>
                                <th class="px-5 py-3 text-right text-xs font-semibold uppercase tracking-[.14em] text-slate-500">Toplam</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-white/10">
                            @forelse ($profiles as $profile)
                                <tr class="transition hover:bg-white/[.03]">
                                    <td class="px-5 py-4">
                                        <a href="{{ route('vehicle-profiles.show', $profile) }}" class="text-base font-semibold text-emerald-300 hover:text-emerald-200">
                                            {{ $profile->plate }}
                                        </a>
                                    </td>
                                    <td class="px-5 py-4 text-sm text-slate-300">
                                        <div>{{ $profile->company_name ?: 'Firma yok' }}</div>
                                        <div class="mt-1 text-slate-500">{{ $profile->driver_name ?: 'Sofor yok' }}</div>
                                    </td>
                                    <td class="px-5 py-4 text-sm text-slate-400">{{ optional($profile->first_seen_at)->format('d.m.Y H:i') ?: '-' }}</td>
                                    <td class="px-5 py-4 text-sm text-slate-400">{{ optional($profile->last_seen_at)->format('d.m.Y H:i') ?: '-' }}</td>
                                    <td class="px-5 py-4 text-right text-sm font-semibold text-slate-100">{{ number_format($profile->total_entry_count) }}</td>
                                </tr>
                            @empty
                                <tr>
                                    <td colspan="5" class="px-5 py-12 text-center text-sm text-slate-500">Arac karti bulunamadi.</td>
                                </tr>
                            @endforelse
                        </tbody>
                    </table>
                </div>
            </section>

            <div class="mt-5">
                {{ $profiles->links() }}
            </div>
        </div>
    </div>
</x-app-layout>
