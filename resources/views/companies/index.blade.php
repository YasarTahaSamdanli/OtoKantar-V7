<x-app-layout>
    <x-slot name="header">
        <div class="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
            <div>
                <p class="text-xs font-semibold uppercase tracking-[.18em] text-emerald-300">Firma Kartlari</p>
                <h2 class="mt-2 text-3xl font-semibold text-slate-100">Firmalar</h2>
            </div>
            <a href="{{ route('companies.create') }}"
               class="inline-flex items-center justify-center rounded-full border border-emerald-400/30 bg-emerald-400/10 px-4 py-2 text-sm font-semibold text-emerald-200 transition hover:bg-emerald-400/15">
                Yeni Firma
            </a>
        </div>
    </x-slot>

    <div class="min-h-[calc(100vh-8rem)] py-8">
        <div class="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <form method="GET" class="mb-5 grid gap-3 rounded-xl border border-white/10 bg-[#181f2b]/95 p-4 sm:grid-cols-[1fr_220px_auto]">
                <input name="q" value="{{ $search }}" placeholder="Firma, vergi no, yetkili veya telefon ara"
                       class="rounded-xl border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-100 placeholder:text-slate-500 focus:border-emerald-400 focus:ring-emerald-400">
                <select name="type"
                        class="rounded-xl border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-100 focus:border-emerald-400 focus:ring-emerald-400">
                    <option value="all" @selected($type === 'all')>Tum tipler</option>
                    @foreach ($types as $value => $label)
                        <option value="{{ $value }}" @selected($type === $value)>{{ $label }}</option>
                    @endforeach
                </select>
                <button class="rounded-full border border-white/10 bg-white/5 px-5 py-2 text-sm font-semibold text-slate-200 transition hover:bg-white/10">
                    Filtrele
                </button>
            </form>

            <section class="overflow-hidden rounded-xl border border-white/10 bg-[#181f2b]/95 shadow-[0_20px_48px_rgba(0,0,0,.22)]">
                <div class="overflow-x-auto">
                    <table class="min-w-full divide-y divide-white/10">
                        <thead class="bg-white/[.03]">
                            <tr>
                                <th class="px-5 py-3 text-left text-xs font-semibold uppercase tracking-[.14em] text-slate-500">Firma</th>
                                <th class="px-5 py-3 text-left text-xs font-semibold uppercase tracking-[.14em] text-slate-500">Tip</th>
                                <th class="px-5 py-3 text-right text-xs font-semibold uppercase tracking-[.14em] text-slate-500">Arac</th>
                                <th class="px-5 py-3 text-right text-xs font-semibold uppercase tracking-[.14em] text-slate-500">Net Ton</th>
                                <th class="px-5 py-3 text-right text-xs font-semibold uppercase tracking-[.14em] text-slate-500">Durum</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-white/10">
                            @forelse ($companies as $company)
                                <tr class="transition hover:bg-white/[.03]">
                                    <td class="px-5 py-4">
                                        <a href="{{ route('companies.show', $company) }}" class="font-semibold text-slate-100 hover:text-emerald-200">{{ $company->name }}</a>
                                        <div class="mt-1 text-sm text-slate-500">{{ $company->contact_name ?: 'Yetkili yok' }} @if($company->phone) / {{ $company->phone }} @endif</div>
                                    </td>
                                    <td class="px-5 py-4 text-sm text-slate-300">{{ $company->typeLabel() }}</td>
                                    <td class="px-5 py-4 text-right text-sm text-slate-300">{{ number_format($company->vehicle_profiles_count) }}</td>
                                    <td class="px-5 py-4 text-right text-sm text-slate-300">{{ number_format(((float) $company->vehicle_passes_sum_net_weight_kg) / 1000, 2, ',', '.') }}</td>
                                    <td class="px-5 py-4 text-right">
                                        <span class="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-300">
                                            {{ $company->is_active ? 'Aktif' : 'Pasif' }}
                                        </span>
                                    </td>
                                </tr>
                            @empty
                                <tr>
                                    <td colspan="5" class="px-5 py-12 text-center text-sm text-slate-500">Henuz firma karti yok.</td>
                                </tr>
                            @endforelse
                        </tbody>
                    </table>
                </div>
            </section>

            <div class="mt-5">{{ $companies->links() }}</div>
        </div>
    </div>
</x-app-layout>
