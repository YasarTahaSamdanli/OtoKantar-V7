<x-app-layout>
    <x-slot name="header">
        <div class="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
            <div>
                <p class="text-xs font-semibold uppercase tracking-[.18em] text-emerald-300">Arac Karti</p>
                <h2 class="mt-2 text-3xl font-semibold text-slate-100">{{ $profile->plate }}</h2>
            </div>
            <a href="{{ route('vehicle-profiles.index') }}"
               class="inline-flex items-center justify-center rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm font-semibold text-slate-300 transition hover:bg-white/10">
                Listeye Don
            </a>
        </div>
    </x-slot>

    <div class="min-h-[calc(100vh-8rem)] py-8">
        <div class="mx-auto grid max-w-7xl gap-5 px-4 sm:px-6 lg:grid-cols-[1fr_360px] lg:px-8">
            <div class="space-y-5">
                @if (session('status'))
                    <div class="rounded-xl border border-emerald-400/25 bg-emerald-400/10 px-5 py-4 text-sm text-emerald-200">
                        {{ session('status') }}
                    </div>
                @endif

                <section class="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                    <div class="rounded-xl border border-white/10 bg-[#181f2b]/95 p-5">
                        <div class="text-sm text-slate-400">Ilk Gecis</div>
                        <div class="mt-3 text-lg font-semibold text-slate-100">{{ optional($profile->first_seen_at)->format('d.m.Y H:i') ?: '-' }}</div>
                    </div>
                    <div class="rounded-xl border border-white/10 bg-[#181f2b]/95 p-5">
                        <div class="text-sm text-slate-400">Son Gecis</div>
                        <div class="mt-3 text-lg font-semibold text-slate-100">{{ optional($profile->last_seen_at)->format('d.m.Y H:i') ?: '-' }}</div>
                    </div>
                    <div class="rounded-xl border border-white/10 bg-[#181f2b]/95 p-5">
                        <div class="text-sm text-slate-400">Toplam Gecis</div>
                        <div class="mt-3 text-lg font-semibold text-slate-100">{{ number_format($profile->total_entry_count) }}</div>
                    </div>
                    <div class="rounded-xl border border-white/10 bg-[#181f2b]/95 p-5">
                        <div class="text-sm text-slate-400">Net Tonaj</div>
                        <div class="mt-3 text-lg font-semibold text-slate-100">
                            {{ $profile->total_net_weight_kg !== null ? number_format(((float) $profile->total_net_weight_kg) / 1000, 2, ',', '.') : '-' }}
                        </div>
                    </div>
                </section>

                <section class="overflow-hidden rounded-xl border border-white/10 bg-[#181f2b]/95 shadow-[0_20px_48px_rgba(0,0,0,.22)]">
                    <div class="border-b border-white/10 px-5 py-4">
                        <h3 class="text-base font-semibold text-slate-100">Gecis Gecmisi</h3>
                    </div>
                    <div class="overflow-x-auto">
                        <table class="min-w-full divide-y divide-white/10">
                            <thead class="bg-white/[.03]">
                                <tr>
                                    <th class="px-5 py-3 text-left text-xs font-semibold uppercase tracking-[.14em] text-slate-500">Durum</th>
                                    <th class="px-5 py-3 text-left text-xs font-semibold uppercase tracking-[.14em] text-slate-500">Zaman</th>
                                    <th class="px-5 py-3 text-left text-xs font-semibold uppercase tracking-[.14em] text-slate-500">Tanima</th>
                                    <th class="px-5 py-3 text-right text-xs font-semibold uppercase tracking-[.14em] text-slate-500">Net Kg</th>
                                </tr>
                            </thead>
                            <tbody class="divide-y divide-white/10">
                                @forelse ($passes as $pass)
                                    <tr>
                                        <td class="px-5 py-4 text-sm font-semibold text-slate-100">{{ $pass->direction }}</td>
                                        <td class="px-5 py-4 text-sm text-slate-400">{{ optional($pass->passed_at)->format('d.m.Y H:i') ?: '-' }}</td>
                                        <td class="px-5 py-4">
                                            <span class="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-300">
                                                {{ $pass->vehicle_recognition_status === 'YENI_ARAC' ? 'Yeni Arac' : 'Taninmis Arac' }}
                                            </span>
                                        </td>
                                        <td class="px-5 py-4 text-right text-sm text-slate-300">
                                            {{ $pass->net_weight_kg !== null ? number_format((float) $pass->net_weight_kg, 0, ',', '.') : '-' }}
                                        </td>
                                    </tr>
                                @empty
                                    <tr>
                                        <td colspan="4" class="px-5 py-12 text-center text-sm text-slate-500">Bu arac icin gecis yok.</td>
                                    </tr>
                                @endforelse
                            </tbody>
                        </table>
                    </div>
                </section>

                {{ $passes->links() }}
            </div>

            <aside class="rounded-xl border border-white/10 bg-[#181f2b]/95 p-5 shadow-[0_20px_48px_rgba(0,0,0,.22)]">
                <h3 class="text-base font-semibold text-slate-100">Profil Bilgileri</h3>
                <form method="POST" action="{{ route('vehicle-profiles.update', $profile) }}" class="mt-5 space-y-4">
                    @csrf
                    @method('PATCH')

                    <div>
                        <label class="mb-2 block text-sm text-slate-400" for="company_id">Firma Karti</label>
                        <select id="company_id" name="company_id"
                                class="w-full rounded-xl border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-100 focus:border-emerald-400 focus:ring-emerald-400">
                            <option value="">Firma karti secme</option>
                            @foreach ($companies as $company)
                                <option value="{{ $company->id }}" @selected((string) old('company_id', $profile->company_id) === (string) $company->id)>
                                    {{ $company->name }}
                                </option>
                            @endforeach
                        </select>
                        @error('company_id') <div class="mt-2 text-sm text-red-300">{{ $message }}</div> @enderror
                    </div>

                    <div>
                        <label class="mb-2 block text-sm text-slate-400" for="company_name">Firma Adi / Serbest Metin</label>
                        <input id="company_name" name="company_name" value="{{ old('company_name', $profile->company_name) }}"
                               class="w-full rounded-xl border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-100 focus:border-emerald-400 focus:ring-emerald-400">
                        @error('company_name') <div class="mt-2 text-sm text-red-300">{{ $message }}</div> @enderror
                    </div>

                    <div>
                        <label class="mb-2 block text-sm text-slate-400" for="driver_name">Sofor Adi</label>
                        <input id="driver_name" name="driver_name" value="{{ old('driver_name', $profile->driver_name) }}"
                               class="w-full rounded-xl border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-100 focus:border-emerald-400 focus:ring-emerald-400">
                        @error('driver_name') <div class="mt-2 text-sm text-red-300">{{ $message }}</div> @enderror
                    </div>

                    <div>
                        <label class="mb-2 block text-sm text-slate-400" for="notes">Notlar</label>
                        <textarea id="notes" name="notes" rows="7"
                                  class="w-full rounded-xl border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-100 focus:border-emerald-400 focus:ring-emerald-400">{{ old('notes', $profile->notes) }}</textarea>
                        @error('notes') <div class="mt-2 text-sm text-red-300">{{ $message }}</div> @enderror
                    </div>

                    <button class="w-full rounded-full border border-emerald-400/30 bg-emerald-400/10 px-5 py-2 text-sm font-semibold text-emerald-200 transition hover:bg-emerald-400/15">
                        Kaydet
                    </button>
                </form>
            </aside>
        </div>
    </div>
</x-app-layout>
