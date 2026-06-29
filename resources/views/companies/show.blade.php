<x-app-layout>
    <x-slot name="header">
        <div class="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
            <div>
                <p class="text-xs font-semibold uppercase tracking-[.18em] text-emerald-300">Firma Karti</p>
                <h2 class="mt-2 text-3xl font-semibold text-slate-100">{{ $company->name }}</h2>
            </div>
            <div class="flex flex-wrap gap-2">
                <a href="{{ route('companies.csv', $company) }}?{{ http_build_query($filters) }}"
                   class="inline-flex items-center justify-center rounded-full border border-emerald-400/30 bg-emerald-400/10 px-4 py-2 text-sm font-semibold text-emerald-200 transition hover:bg-emerald-400/15">
                    CSV Indir
                </a>
                <a href="{{ route('companies.edit', $company) }}"
                   class="inline-flex items-center justify-center rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm font-semibold text-slate-300 transition hover:bg-white/10">
                    Duzenle
                </a>
            </div>
        </div>
    </x-slot>

    <div class="min-h-[calc(100vh-8rem)] py-8">
        <div class="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            @if (session('status'))
                <div class="mb-5 rounded-xl border border-emerald-400/25 bg-emerald-400/10 px-5 py-4 text-sm text-emerald-200">
                    {{ session('status') }}
                </div>
            @endif

            <section class="mb-5 grid gap-4 sm:grid-cols-3">
                <div class="rounded-xl border border-white/10 bg-[#181f2b]/95 p-5">
                    <div class="text-sm text-slate-400">Gecis</div>
                    <div class="mt-3 text-2xl font-semibold text-slate-100">{{ number_format($summary['pass_count']) }}</div>
                </div>
                <div class="rounded-xl border border-white/10 bg-[#181f2b]/95 p-5">
                    <div class="text-sm text-slate-400">Tekil Plaka</div>
                    <div class="mt-3 text-2xl font-semibold text-slate-100">{{ number_format($summary['vehicle_count']) }}</div>
                </div>
                <div class="rounded-xl border border-white/10 bg-[#181f2b]/95 p-5">
                    <div class="text-sm text-slate-400">Net Tonaj</div>
                    <div class="mt-3 text-2xl font-semibold text-slate-100">{{ number_format($summary['net_weight_kg'] / 1000, 2, ',', '.') }}</div>
                </div>
            </section>

            <div class="grid gap-5 lg:grid-cols-[1fr_320px]">
                <div class="space-y-5">
                    <form method="GET" class="grid gap-3 rounded-xl border border-white/10 bg-[#181f2b]/95 p-4 md:grid-cols-5">
                        <input name="date_from" type="date" value="{{ $filters['date_from'] }}"
                               class="rounded-xl border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-100 focus:border-emerald-400 focus:ring-emerald-400">
                        <input name="date_to" type="date" value="{{ $filters['date_to'] }}"
                               class="rounded-xl border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-100 focus:border-emerald-400 focus:ring-emerald-400">
                        <input name="plate" value="{{ $filters['plate'] }}" placeholder="Plaka"
                               class="rounded-xl border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-100 placeholder:text-slate-500 focus:border-emerald-400 focus:ring-emerald-400">
                        <input name="material" value="{{ $filters['material'] }}" placeholder="Malzeme"
                               class="rounded-xl border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-100 placeholder:text-slate-500 focus:border-emerald-400 focus:ring-emerald-400">
                        <button class="rounded-full border border-white/10 bg-white/5 px-5 py-2 text-sm font-semibold text-slate-200 transition hover:bg-white/10">
                            Filtrele
                        </button>
                    </form>

                    <section class="overflow-hidden rounded-xl border border-white/10 bg-[#181f2b]/95 shadow-[0_20px_48px_rgba(0,0,0,.22)]">
                        <div class="border-b border-white/10 px-5 py-4">
                            <h3 class="text-base font-semibold text-slate-100">Firma Hareketleri</h3>
                        </div>
                        <div class="overflow-x-auto">
                            <table class="min-w-full divide-y divide-white/10">
                                <thead class="bg-white/[.03]">
                                    <tr>
                                        <th class="px-5 py-3 text-left text-xs font-semibold uppercase tracking-[.14em] text-slate-500">Zaman</th>
                                        <th class="px-5 py-3 text-left text-xs font-semibold uppercase tracking-[.14em] text-slate-500">Plaka</th>
                                        <th class="px-5 py-3 text-left text-xs font-semibold uppercase tracking-[.14em] text-slate-500">Malzeme</th>
                                        <th class="px-5 py-3 text-right text-xs font-semibold uppercase tracking-[.14em] text-slate-500">Giris</th>
                                        <th class="px-5 py-3 text-right text-xs font-semibold uppercase tracking-[.14em] text-slate-500">Cikis</th>
                                        <th class="px-5 py-3 text-right text-xs font-semibold uppercase tracking-[.14em] text-slate-500">Net</th>
                                    </tr>
                                </thead>
                                <tbody class="divide-y divide-white/10">
                                    @forelse ($passes as $pass)
                                        <tr>
                                            <td class="px-5 py-4 text-sm text-slate-400">{{ optional($pass->passed_at)->format('d.m.Y H:i') ?: '-' }}</td>
                                            <td class="px-5 py-4 text-sm font-semibold text-slate-100">{{ $pass->plate }}</td>
                                            <td class="px-5 py-4 text-sm text-slate-300">{{ $pass->material_type ?: '-' }}</td>
                                            <td class="px-5 py-4 text-right text-sm text-slate-300">{{ $pass->entry_weight_kg !== null ? number_format((float) $pass->entry_weight_kg, 0, ',', '.') : '-' }}</td>
                                            <td class="px-5 py-4 text-right text-sm text-slate-300">{{ $pass->exit_weight_kg !== null ? number_format((float) $pass->exit_weight_kg, 0, ',', '.') : '-' }}</td>
                                            <td class="px-5 py-4 text-right text-sm text-slate-300">{{ $pass->net_weight_kg !== null ? number_format((float) $pass->net_weight_kg, 0, ',', '.') : '-' }}</td>
                                        </tr>
                                    @empty
                                        <tr>
                                            <td colspan="6" class="px-5 py-12 text-center text-sm text-slate-500">Bu filtrelerde firma hareketi yok.</td>
                                        </tr>
                                    @endforelse
                                </tbody>
                            </table>
                        </div>
                    </section>

                    {{ $passes->links() }}
                </div>

                <aside class="rounded-xl border border-white/10 bg-[#181f2b]/95 p-5 shadow-[0_20px_48px_rgba(0,0,0,.22)]">
                    <h3 class="text-base font-semibold text-slate-100">Firma Bilgileri</h3>
                    <dl class="mt-5 space-y-4 text-sm">
                        <div><dt class="text-slate-500">Tip</dt><dd class="mt-1 text-slate-200">{{ $company->typeLabel() }}</dd></div>
                        <div><dt class="text-slate-500">Yetkili</dt><dd class="mt-1 text-slate-200">{{ $company->contact_name ?: '-' }}</dd></div>
                        <div><dt class="text-slate-500">Telefon</dt><dd class="mt-1 text-slate-200">{{ $company->phone ?: '-' }}</dd></div>
                        <div><dt class="text-slate-500">E-posta</dt><dd class="mt-1 text-slate-200">{{ $company->email ?: '-' }}</dd></div>
                        <div><dt class="text-slate-500">Vergi No</dt><dd class="mt-1 text-slate-200">{{ $company->tax_number ?: '-' }}</dd></div>
                        <div><dt class="text-slate-500">Not</dt><dd class="mt-1 whitespace-pre-line text-slate-300">{{ $company->notes ?: '-' }}</dd></div>
                    </dl>
                </aside>
            </div>
        </div>
    </div>
</x-app-layout>
