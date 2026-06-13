<button {{ $attributes->merge(['type' => 'button', 'class' => 'inline-flex items-center justify-center rounded-xl border border-white/10 bg-white/5 px-4 py-2.5 text-xs font-semibold uppercase tracking-[.14em] text-slate-200 hover:bg-white/10 focus:outline-none focus:ring-2 focus:ring-emerald-400/40 disabled:opacity-25 transition ease-in-out duration-150']) }}>
    {{ $slot }}
</button>
