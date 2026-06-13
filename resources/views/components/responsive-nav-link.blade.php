@props(['active'])

@php
$classes = ($active ?? false)
            ? 'block w-full border-l-4 border-emerald-400 bg-emerald-400/10 py-3 ps-4 pe-4 text-start text-sm font-semibold uppercase tracking-[.12em] text-emerald-300 transition duration-150 ease-in-out'
            : 'block w-full border-l-4 border-transparent py-3 ps-4 pe-4 text-start text-sm font-semibold uppercase tracking-[.12em] text-slate-400 hover:border-white/20 hover:bg-white/5 hover:text-slate-100 transition duration-150 ease-in-out';
@endphp

<a {{ $attributes->merge(['class' => $classes]) }}>
    {{ $slot }}
</a>
