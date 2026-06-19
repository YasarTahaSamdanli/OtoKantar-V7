<x-app-layout>
    <x-slot name="header">
        <div class="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
            <div>
                <p class="text-xs font-semibold uppercase tracking-[.18em] text-emerald-300">OtoKantar V7</p>
                <h2 class="mt-2 text-2xl font-semibold text-slate-100">
                    Yonetim Paneli
                </h2>
            </div>
            <div class="rounded-full border border-emerald-400/20 bg-emerald-400/10 px-4 py-2 text-xs font-semibold uppercase tracking-[.14em] text-emerald-300">
                Admin
            </div>
        </div>
    </x-slot>

    <div class="min-h-[calc(100vh-8rem)] py-8">
        <div class="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <div class="grid gap-4 lg:grid-cols-3">
                <a href="{{ route('canli.view') }}"
                   class="group rounded-2xl border border-white/10 bg-[#181f2b]/95 p-6 shadow-[0_20px_48px_rgba(0,0,0,.24)] transition hover:border-emerald-400/30 hover:bg-[#1b2431]">
                    <div class="flex items-start justify-between gap-4">
                        <div>
                            <div class="text-xs font-semibold uppercase tracking-[.16em] text-slate-500">Operasyon</div>
                            <div class="mt-3 text-2xl font-semibold text-slate-100">Canli Panel</div>
                        </div>
                        <span class="rounded-full border border-emerald-400/30 bg-emerald-400/10 px-3 py-1 text-xs text-emerald-300">Acilir</span>
                    </div>
                    <p class="mt-4 text-sm leading-6 text-slate-400">Kantar durumu, canli kare, son gecisler ve CSV raporu.</p>
                </a>

                <a href="{{ route('admin.users.index') }}"
                   class="group rounded-2xl border border-white/10 bg-[#181f2b]/95 p-6 shadow-[0_20px_48px_rgba(0,0,0,.24)] transition hover:border-emerald-400/30 hover:bg-[#1b2431]">
                    <div class="flex items-start justify-between gap-4">
                        <div>
                            <div class="text-xs font-semibold uppercase tracking-[.16em] text-slate-500">Yetki</div>
                            <div class="mt-3 text-2xl font-semibold text-slate-100">Kullanicilar</div>
                        </div>
                        <span class="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-300">Liste</span>
                    </div>
                    <p class="mt-4 text-sm leading-6 text-slate-400">Admin ve calisan hesaplarini tek ekrandan kontrol et.</p>
                </a>

                <a href="{{ route('admin.users.create') }}"
                   class="group rounded-2xl border border-white/10 bg-[#181f2b]/95 p-6 shadow-[0_20px_48px_rgba(0,0,0,.24)] transition hover:border-emerald-400/30 hover:bg-[#1b2431]">
                    <div class="flex items-start justify-between gap-4">
                        <div>
                            <div class="text-xs font-semibold uppercase tracking-[.16em] text-slate-500">Hesap</div>
                            <div class="mt-3 text-2xl font-semibold text-slate-100">Yeni Kullanici</div>
                        </div>
                        <span class="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-300">Form</span>
                    </div>
                    <p class="mt-4 text-sm leading-6 text-slate-400">Yeni calisan veya yeni admin hesabi olustur.</p>
                </a>
            </div>

            <div class="mt-6 grid gap-4">
                <section class="rounded-2xl border border-white/10 bg-[#181f2b]/95 p-6 shadow-[0_20px_48px_rgba(0,0,0,.24)]">
                    <div class="text-xs font-semibold uppercase tracking-[.16em] text-slate-500">Roller</div>
                    <div class="mt-4 space-y-3">
                        <div class="flex items-start justify-between gap-4 rounded-xl border border-white/10 bg-white/[.03] p-4">
                            <div>
                                <div class="font-semibold text-slate-100">Admin</div>
                                <p class="mt-1 text-sm text-slate-400">Canli paneli acar, kullanici olusturur ve hesaplari gorur.</p>
                            </div>
                            <span class="rounded-full bg-emerald-400/10 px-3 py-1 text-xs text-emerald-300">aktif</span>
                        </div>
                        <div class="flex items-start justify-between gap-4 rounded-xl border border-white/10 bg-white/[.03] p-4">
                            <div>
                                <div class="font-semibold text-slate-100">Calisan</div>
                                <p class="mt-1 text-sm text-slate-400">Su an panel ve yonetim ekranlari admin rolune kapali.</p>
                            </div>
                            <span class="rounded-full bg-white/5 px-3 py-1 text-xs text-slate-400">sinirli</span>
                        </div>
                    </div>
                </section>
            </div>
        </div>
    </div>
</x-app-layout>
