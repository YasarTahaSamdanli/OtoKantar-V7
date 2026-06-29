<x-app-layout>
    <x-slot name="header">
        <div class="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
            <div>
                <p class="text-xs font-semibold uppercase tracking-[.18em] text-emerald-300">Firma Karti</p>
                <h2 class="mt-2 text-3xl font-semibold text-slate-100">{{ $company->name }}</h2>
            </div>
            <a href="{{ route('companies.show', $company) }}"
               class="inline-flex items-center justify-center rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm font-semibold text-slate-300 transition hover:bg-white/10">
                Karta Don
            </a>
        </div>
    </x-slot>

    <div class="min-h-[calc(100vh-8rem)] py-8">
        <div class="mx-auto max-w-3xl px-4 sm:px-6 lg:px-8">
            <form method="POST" action="{{ route('companies.update', $company) }}"
                  class="space-y-4 rounded-xl border border-white/10 bg-[#181f2b]/95 p-5 shadow-[0_20px_48px_rgba(0,0,0,.22)]">
                @method('PATCH')
                @include('companies._form')
            </form>
        </div>
    </div>
</x-app-layout>
