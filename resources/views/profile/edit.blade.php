<x-app-layout>
    <x-slot name="header">
        <div>
            <p class="text-xs font-semibold uppercase tracking-[.18em] text-emerald-300">Hesap</p>
            <h2 class="mt-2 text-2xl font-semibold text-slate-100">
                Profil
            </h2>
        </div>
    </x-slot>

    <div class="min-h-[calc(100vh-8rem)] py-8">
        <div class="mx-auto max-w-7xl space-y-6 px-4 sm:px-6 lg:px-8">
            <div class="rounded-2xl border border-white/10 bg-[#181f2b]/95 p-6 shadow-[0_20px_48px_rgba(0,0,0,.24)] sm:p-8">
                <div class="max-w-xl">
                    @include('profile.partials.update-profile-information-form')
                </div>
            </div>

            <div class="rounded-2xl border border-white/10 bg-[#181f2b]/95 p-6 shadow-[0_20px_48px_rgba(0,0,0,.24)] sm:p-8">
                <div class="max-w-xl">
                    @include('profile.partials.update-password-form')
                </div>
            </div>

            <div class="rounded-2xl border border-rose-400/20 bg-[#181f2b]/95 p-6 shadow-[0_20px_48px_rgba(0,0,0,.24)] sm:p-8">
                <div class="max-w-xl">
                    @include('profile.partials.delete-user-form')
                </div>
            </div>
        </div>
    </div>
</x-app-layout>
