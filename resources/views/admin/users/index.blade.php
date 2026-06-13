<x-app-layout>
    <x-slot name="header">
        <div class="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
            <div>
                <p class="text-xs font-semibold uppercase tracking-[.18em] text-emerald-300">Hesap yonetimi</p>
                <h2 class="mt-2 text-2xl font-semibold text-slate-100">
                    Kullanicilar
                </h2>
            </div>
            <a href="{{ route('admin.users.create') }}"
               class="inline-flex items-center justify-center rounded-xl border border-emerald-400/30 bg-emerald-400/15 px-4 py-2.5 text-xs font-semibold uppercase tracking-[.14em] text-emerald-200 hover:bg-emerald-400/20">
                Yeni Kullanici
            </a>
        </div>
    </x-slot>

    <div class="min-h-[calc(100vh-8rem)] py-8">
        <div class="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            @if (session('status'))
                <div class="mb-4 rounded-2xl border border-emerald-400/20 bg-emerald-400/10 px-4 py-3 text-sm text-emerald-200">
                    {{ session('status') }}
                </div>
            @endif

            <section class="overflow-hidden rounded-2xl border border-white/10 bg-[#181f2b]/95 shadow-[0_20px_48px_rgba(0,0,0,.24)]">
                <div class="flex flex-col gap-2 border-b border-white/10 px-6 py-5 sm:flex-row sm:items-center sm:justify-between">
                    <div>
                        <div class="text-sm font-semibold text-slate-100">Sisteme girebilen hesaplar</div>
                        <div class="mt-1 text-sm text-slate-500">Admin ve calisan rollerini buradan takip et.</div>
                    </div>
                    <div class="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-400" style="font-family: 'JetBrains Mono', monospace;">
                        {{ $users->total() }} kayit
                    </div>
                </div>

                <div class="overflow-x-auto">
                    <table class="min-w-full text-sm">
                        <thead class="bg-white/[.03] text-left text-xs uppercase tracking-[.14em] text-slate-500">
                        <tr>
                            <th class="px-6 py-4 font-semibold">ID</th>
                            <th class="px-6 py-4 font-semibold">Ad</th>
                            <th class="px-6 py-4 font-semibold">Email</th>
                            <th class="px-6 py-4 font-semibold">Rol</th>
                            <th class="px-6 py-4 font-semibold">Olusturma</th>
                        </tr>
                        </thead>
                        <tbody class="divide-y divide-white/10">
                        @foreach($users as $user)
                            <tr class="text-slate-300 hover:bg-white/[.03]">
                                <td class="px-6 py-4 text-slate-500" style="font-family: 'JetBrains Mono', monospace;">{{ $user->id }}</td>
                                <td class="px-6 py-4 font-semibold text-slate-100">{{ $user->name }}</td>
                                <td class="px-6 py-4 text-slate-400">{{ $user->email }}</td>
                                <td class="px-6 py-4">
                                    <span class="inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold {{ $user->role === 'admin' ? 'border border-emerald-400/20 bg-emerald-400/10 text-emerald-300' : 'border border-white/10 bg-white/5 text-slate-300' }}">
                                        {{ $user->role }}
                                    </span>
                                </td>
                                <td class="px-6 py-4 text-slate-500" style="font-family: 'JetBrains Mono', monospace;">{{ $user->created_at?->format('Y-m-d H:i') }}</td>
                            </tr>
                        @endforeach
                        </tbody>
                    </table>
                </div>

                <div class="flex flex-col gap-3 border-t border-white/10 px-6 py-4 sm:flex-row sm:items-center sm:justify-between">
                    <div class="text-sm text-slate-500">
                        {{ $users->firstItem() ?? 0 }}-{{ $users->lastItem() ?? 0 }} / {{ $users->total() }}
                    </div>
                    <div class="flex gap-2">
                        @if ($users->onFirstPage())
                            <span class="rounded-xl border border-white/10 px-4 py-2 text-sm text-slate-600">Onceki</span>
                        @else
                            <a href="{{ $users->previousPageUrl() }}" class="rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-sm font-semibold text-slate-300 hover:bg-white/10">Onceki</a>
                        @endif

                        @if ($users->hasMorePages())
                            <a href="{{ $users->nextPageUrl() }}" class="rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-sm font-semibold text-slate-300 hover:bg-white/10">Sonraki</a>
                        @else
                            <span class="rounded-xl border border-white/10 px-4 py-2 text-sm text-slate-600">Sonraki</span>
                        @endif
                    </div>
                </div>
            </section>
        </div>
    </div>
</x-app-layout>
