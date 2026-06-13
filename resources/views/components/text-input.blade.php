@props(['disabled' => false])

<input @disabled($disabled) {{ $attributes->merge(['class' => 'rounded-xl border-white/10 bg-[#0f151f] text-slate-100 shadow-sm placeholder:text-slate-500 focus:border-emerald-400/50 focus:ring-emerald-400/30']) }}>
