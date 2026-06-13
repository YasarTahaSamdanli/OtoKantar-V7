<button {{ $attributes->merge(['type' => 'submit', 'class' => 'inline-flex items-center justify-center rounded-xl border border-rose-400/30 bg-rose-500/15 px-4 py-2.5 text-xs font-semibold uppercase tracking-[.14em] text-rose-200 hover:bg-rose-500/20 focus:outline-none focus:ring-2 focus:ring-rose-400/40 transition ease-in-out duration-150']) }}>
    {{ $slot }}
</button>
