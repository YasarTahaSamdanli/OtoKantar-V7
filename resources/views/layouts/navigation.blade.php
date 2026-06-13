<nav x-data="{ open: false }" class="sticky top-0 z-40 border-b border-white/10 bg-[#0a0e14]/90 backdrop-blur">
    <div class="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div class="flex min-h-16 items-center justify-between gap-4 py-3">
            <div class="flex items-center gap-5">
                <a href="{{ Auth::user()->isAdmin() ? route('dashboard') : route('profile.edit') }}" class="flex items-center gap-3">
                    <span class="grid h-10 w-10 place-items-center rounded-2xl border border-emerald-400/30 bg-emerald-400/10 text-emerald-300">
                        <x-application-logo class="h-5 w-5 fill-current" />
                    </span>
                    <span>
                        <span class="block text-sm font-semibold uppercase tracking-[.18em] text-slate-100">OtoKantar</span>
                        <span class="block text-[11px] text-slate-500" style="font-family: 'JetBrains Mono', monospace;">V7 yonetim</span>
                    </span>
                </a>

                @if (Auth::user()->isAdmin())
                    <div class="hidden items-center gap-2 lg:flex">
                        <x-nav-link :href="route('dashboard')" :active="request()->routeIs('dashboard')">
                            Yonetim
                        </x-nav-link>
                        <x-nav-link :href="route('canli.view')" :active="request()->routeIs('canli.*')">
                            Canli Panel
                        </x-nav-link>
                        <x-nav-link :href="route('admin.users.index')" :active="request()->routeIs('admin.users.*')">
                            Kullanicilar
                        </x-nav-link>
                    </div>
                @endif
            </div>

            <div class="hidden items-center gap-3 sm:flex">
                <div class="text-right">
                    <div class="text-sm font-semibold text-slate-200">{{ Auth::user()->name }}</div>
                    <div class="text-xs text-slate-500" style="font-family: 'JetBrains Mono', monospace;">{{ Auth::user()->role }}</div>
                </div>

                <x-dropdown align="right" width="48">
                    <x-slot name="trigger">
                        <button class="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-2 text-sm font-medium text-slate-300 transition hover:bg-white/10 hover:text-slate-100 focus:outline-none">
                            Hesap
                            <svg class="h-4 w-4 fill-current" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                                <path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd" />
                            </svg>
                        </button>
                    </x-slot>

                    <x-slot name="content">
                        <x-dropdown-link :href="route('profile.edit')">
                            Profil
                        </x-dropdown-link>

                        <form method="POST" action="{{ route('logout') }}">
                            @csrf
                            <x-dropdown-link :href="route('logout')"
                                    onclick="event.preventDefault(); this.closest('form').submit();">
                                Cikis
                            </x-dropdown-link>
                        </form>
                    </x-slot>
                </x-dropdown>
            </div>

            <div class="flex items-center sm:hidden">
                <button @click="open = ! open" class="inline-flex items-center justify-center rounded-xl border border-white/10 bg-white/5 p-2 text-slate-400 transition hover:bg-white/10 hover:text-slate-100 focus:outline-none">
                    <svg class="h-6 w-6" stroke="currentColor" fill="none" viewBox="0 0 24 24">
                        <path :class="{'hidden': open, 'inline-flex': ! open }" class="inline-flex" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16" />
                        <path :class="{'hidden': ! open, 'inline-flex': open }" class="hidden" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                </button>
            </div>
        </div>
    </div>

    <div :class="{'block': open, 'hidden': ! open}" class="hidden border-t border-white/10 bg-[#0f151f] lg:hidden">
        @if (Auth::user()->isAdmin())
            <div class="space-y-1 py-2">
                <x-responsive-nav-link :href="route('dashboard')" :active="request()->routeIs('dashboard')">
                    Yonetim
                </x-responsive-nav-link>
                <x-responsive-nav-link :href="route('canli.view')" :active="request()->routeIs('canli.*')">
                    Canli Panel
                </x-responsive-nav-link>
                <x-responsive-nav-link :href="route('admin.users.index')" :active="request()->routeIs('admin.users.*')">
                    Kullanicilar
                </x-responsive-nav-link>
            </div>
        @endif

        <div class="border-t border-white/10 px-4 py-4">
            <div class="font-semibold text-slate-200">{{ Auth::user()->name }}</div>
            <div class="text-sm text-slate-500">{{ Auth::user()->email }}</div>
            <div class="mt-3 space-y-1">
                <x-responsive-nav-link :href="route('profile.edit')">
                    Profil
                </x-responsive-nav-link>
                <form method="POST" action="{{ route('logout') }}">
                    @csrf
                    <x-responsive-nav-link :href="route('logout')"
                            onclick="event.preventDefault(); this.closest('form').submit();">
                        Cikis
                    </x-responsive-nav-link>
                </form>
            </div>
        </div>
    </div>
</nav>
