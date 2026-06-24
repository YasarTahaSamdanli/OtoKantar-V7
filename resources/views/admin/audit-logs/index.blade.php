<x-app-layout>
    <x-slot name="header">
        <div class="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
            <div>
                <p class="text-xs font-semibold uppercase tracking-[.18em] text-emerald-300">Guvenlik ve izlenebilirlik</p>
                <h2 class="mt-2 text-2xl font-semibold text-slate-100">
                    Audit Log
                </h2>
            </div>
            <a href="{{ route('admin.audit-logs.index') }}"
               class="inline-flex items-center justify-center rounded-xl border border-white/10 bg-white/5 px-4 py-2.5 text-xs font-semibold uppercase tracking-[.14em] text-slate-300 hover:bg-white/10">
                Filtreleri Temizle
            </a>
        </div>
    </x-slot>

    <div class="min-h-[calc(100vh-8rem)] py-8">
        <div class="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <section class="mb-4 rounded-2xl border border-white/10 bg-[#181f2b]/95 p-4 shadow-[0_20px_48px_rgba(0,0,0,.24)]">
                <form method="GET" action="{{ route('admin.audit-logs.index') }}" class="grid gap-3 md:grid-cols-[1fr_1fr_auto] md:items-end">
                    <label class="block">
                        <span class="text-xs font-semibold uppercase tracking-[.14em] text-slate-500">Aksiyon</span>
                        <select name="action" class="mt-2 w-full rounded-xl border-white/10 bg-[#0f151f] text-sm text-slate-200 focus:border-emerald-400 focus:ring-emerald-400">
                            <option value="">Tum aksiyonlar</option>
                            @foreach($actions as $action)
                                <option value="{{ $action }}" @selected($filters['action'] === $action)>{{ $action }}</option>
                            @endforeach
                        </select>
                    </label>

                    <label class="block">
                        <span class="text-xs font-semibold uppercase tracking-[.14em] text-slate-500">Kullanici</span>
                        <input name="user" value="{{ $filters['user'] }}" type="search" placeholder="Ad veya email"
                               class="mt-2 w-full rounded-xl border-white/10 bg-[#0f151f] text-sm text-slate-200 placeholder:text-slate-600 focus:border-emerald-400 focus:ring-emerald-400">
                    </label>

                    <button type="submit"
                            class="rounded-xl border border-emerald-400/30 bg-emerald-400/15 px-4 py-2.5 text-xs font-semibold uppercase tracking-[.14em] text-emerald-200 hover:bg-emerald-400/20">
                        Uygula
                    </button>
                </form>
            </section>

            <section class="overflow-hidden rounded-2xl border border-white/10 bg-[#181f2b]/95 shadow-[0_20px_48px_rgba(0,0,0,.24)]">
                <div class="flex flex-col gap-2 border-b border-white/10 px-6 py-5 sm:flex-row sm:items-center sm:justify-between">
                    <div>
                        <div class="text-sm font-semibold text-slate-100">Sistem olaylari</div>
                        <div class="mt-1 text-sm text-slate-500">Giris, cikis, CSV export, kullanici yonetimi ve ingest olaylari.</div>
                    </div>
                    <div class="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-400" style="font-family: 'JetBrains Mono', monospace;">
                        {{ $logs->total() }} kayit
                    </div>
                </div>

                <div class="overflow-x-auto">
                    <table class="min-w-full text-sm">
                        <thead class="bg-white/[.03] text-left text-xs uppercase tracking-[.14em] text-slate-500">
                        <tr>
                            <th class="px-6 py-4 font-semibold">Zaman</th>
                            <th class="px-6 py-4 font-semibold">Aksiyon</th>
                            <th class="px-6 py-4 font-semibold">Kullanici</th>
                            <th class="px-6 py-4 font-semibold">Hedef</th>
                            <th class="px-6 py-4 font-semibold">IP</th>
                            <th class="px-6 py-4 font-semibold">Detay</th>
                        </tr>
                        </thead>
                        <tbody class="divide-y divide-white/10">
                        @forelse($logs as $log)
                            <tr class="align-top text-slate-300 hover:bg-white/[.03]">
                                <td class="whitespace-nowrap px-6 py-4 text-slate-500" style="font-family: 'JetBrains Mono', monospace;">
                                    {{ $log->created_at?->format('Y-m-d H:i:s') }}
                                </td>
                                <td class="px-6 py-4">
                                    <span class="inline-flex items-center rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1 text-xs font-semibold text-emerald-300">
                                        {{ $log->action }}
                                    </span>
                                </td>
                                <td class="px-6 py-4">
                                    @if($log->user)
                                        <div class="font-semibold text-slate-100">{{ $log->user->name }}</div>
                                        <div class="mt-1 text-xs text-slate-500">{{ $log->user->email }}</div>
                                    @else
                                        <span class="text-slate-600">Sistem / anonim</span>
                                    @endif
                                </td>
                                <td class="px-6 py-4 text-slate-500" style="font-family: 'JetBrains Mono', monospace;">
                                    @if($log->subject_type)
                                        {{ class_basename($log->subject_type) }} #{{ $log->subject_id }}
                                    @else
                                        -
                                    @endif
                                </td>
                                <td class="whitespace-nowrap px-6 py-4 text-slate-500" style="font-family: 'JetBrains Mono', monospace;">
                                    {{ $log->ip_address ?: '-' }}
                                </td>
                                <td class="px-6 py-4">
                                    @if($log->metadata)
                                        <details class="max-w-md rounded-xl border border-white/10 bg-white/[.03] px-3 py-2">
                                            <summary class="cursor-pointer text-xs font-semibold uppercase tracking-[.12em] text-slate-400">Metadata</summary>
                                            <pre class="mt-3 whitespace-pre-wrap break-words text-xs text-slate-400" style="font-family: 'JetBrains Mono', monospace;">{{ json_encode($log->metadata, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES) }}</pre>
                                        </details>
                                    @else
                                        <span class="text-slate-600">-</span>
                                    @endif
                                </td>
                            </tr>
                        @empty
                            <tr>
                                <td colspan="6" class="px-6 py-10 text-center text-sm text-slate-500">
                                    Audit log kaydi bulunamadi.
                                </td>
                            </tr>
                        @endforelse
                        </tbody>
                    </table>
                </div>

                <div class="flex flex-col gap-3 border-t border-white/10 px-6 py-4 sm:flex-row sm:items-center sm:justify-between">
                    <div class="text-sm text-slate-500">
                        {{ $logs->firstItem() ?? 0 }}-{{ $logs->lastItem() ?? 0 }} / {{ $logs->total() }}
                    </div>
                    <div>
                        {{ $logs->links() }}
                    </div>
                </div>
            </section>
        </div>
    </div>
</x-app-layout>
