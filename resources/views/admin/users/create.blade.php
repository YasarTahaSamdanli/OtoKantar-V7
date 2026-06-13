<x-app-layout>
    <x-slot name="header">
        <div class="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
            <div>
                <p class="text-xs font-semibold uppercase tracking-[.18em] text-emerald-300">Yeni hesap</p>
                <h2 class="mt-2 text-2xl font-semibold text-slate-100">
                    Kullanici Olustur
                </h2>
            </div>
            <a href="{{ route('admin.users.index') }}"
               class="inline-flex items-center justify-center rounded-xl border border-white/10 bg-white/5 px-4 py-2.5 text-xs font-semibold uppercase tracking-[.14em] text-slate-200 hover:bg-white/10">
                Listeye Don
            </a>
        </div>
    </x-slot>

    <div class="min-h-[calc(100vh-8rem)] py-8">
        <div class="mx-auto max-w-3xl px-4 sm:px-6 lg:px-8">
            <section class="rounded-2xl border border-white/10 bg-[#181f2b]/95 p-6 shadow-[0_20px_48px_rgba(0,0,0,.24)]">
                <form method="POST" action="{{ route('admin.users.store') }}" class="space-y-6">
                    @csrf

                    <div class="grid gap-5 sm:grid-cols-2">
                        <div class="sm:col-span-2">
                            <x-input-label for="name" value="Ad Soyad" />
                            <x-text-input id="name" name="name" type="text" class="mt-2 block w-full" :value="old('name')" required autofocus />
                            <x-input-error :messages="$errors->get('name')" class="mt-2" />
                        </div>

                        <div class="sm:col-span-2">
                            <x-input-label for="email" value="Email" />
                            <x-text-input id="email" name="email" type="email" class="mt-2 block w-full" :value="old('email')" required />
                            <x-input-error :messages="$errors->get('email')" class="mt-2" />
                        </div>

                        <div>
                            <x-input-label for="password" value="Sifre" />
                            <x-text-input id="password" name="password" type="password" class="mt-2 block w-full" required autocomplete="new-password" />
                            <x-input-error :messages="$errors->get('password')" class="mt-2" />
                        </div>

                        <div>
                            <x-input-label for="password_confirmation" value="Sifre Tekrar" />
                            <x-text-input id="password_confirmation" name="password_confirmation" type="password" class="mt-2 block w-full" required autocomplete="new-password" />
                        </div>

                        <div class="sm:col-span-2">
                            <x-input-label for="role" value="Rol" />
                            <select id="role"
                                    name="role"
                                    class="mt-2 block w-full rounded-xl border-white/10 bg-[#0f151f] text-slate-100 shadow-sm focus:border-emerald-400/50 focus:ring-emerald-400/30"
                                    required>
                                <option value="employee" @selected(old('role', 'employee') === 'employee')>Calisan</option>
                                <option value="admin" @selected(old('role') === 'admin')>Admin</option>
                            </select>
                            <x-input-error :messages="$errors->get('role')" class="mt-2" />
                        </div>
                    </div>

                    <div class="flex flex-col-reverse gap-3 border-t border-white/10 pt-6 sm:flex-row sm:items-center sm:justify-between">
                        <a href="{{ route('admin.users.index') }}" class="inline-flex items-center justify-center rounded-xl border border-white/10 bg-white/5 px-4 py-2.5 text-xs font-semibold uppercase tracking-[.14em] text-slate-200 hover:bg-white/10">
                            Vazgec
                        </a>
                        <x-primary-button>
                            Olustur
                        </x-primary-button>
                    </div>
                </form>
            </section>
        </div>
    </div>
</x-app-layout>
