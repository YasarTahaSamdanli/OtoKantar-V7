@props(['active'])

@php
$classes = ($active ?? false)
            ? 'inline-flex items-center rounded-full border border-emerald-400/30 bg-emerald-400/10 px-4 py-2 text-xs font-semibold uppercase tracking-[.14em] text-emerald-300 transition duration-150 ease-in-out'
            : 'inline-flex items-center rounded-full border border-white/10 px-4 py-2 text-xs font-semibold uppercase tracking-[.14em] text-slate-400 hover:border-white/20 hover:bg-white/5 hover:text-slate-100 transition duration-150 ease-in-out';
@endphp

<a {{ $attributes->merge(['class' => $classes]) }}>
    {{ $slot }}
</a>
