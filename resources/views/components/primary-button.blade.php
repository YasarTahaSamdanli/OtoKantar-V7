<button {{ $attributes->merge(['type' => 'submit', 'class' => 'inline-flex items-center justify-center rounded-xl border border-emerald-400/30 bg-emerald-400/15 px-4 py-2.5 text-xs font-semibold uppercase tracking-[.14em] text-emerald-200 hover:bg-emerald-400/20 focus:outline-none focus:ring-2 focus:ring-emerald-400/40 transition ease-in-out duration-150']) }}>
    {{ $slot }}
</button>
