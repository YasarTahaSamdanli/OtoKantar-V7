@csrf

<div>
    <label class="mb-2 block text-sm text-slate-400" for="name">Firma Adi</label>
    <input id="name" name="name" value="{{ old('name', $company->name) }}"
           class="w-full rounded-xl border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-100 focus:border-emerald-400 focus:ring-emerald-400">
    @error('name') <div class="mt-2 text-sm text-red-300">{{ $message }}</div> @enderror
</div>

<div class="grid gap-4 sm:grid-cols-2">
    <div>
        <label class="mb-2 block text-sm text-slate-400" for="type">Firma Tipi</label>
        <select id="type" name="type"
                class="w-full rounded-xl border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-100 focus:border-emerald-400 focus:ring-emerald-400">
            @foreach ($types as $value => $label)
                <option value="{{ $value }}" @selected(old('type', $company->type) === $value)>{{ $label }}</option>
            @endforeach
        </select>
        @error('type') <div class="mt-2 text-sm text-red-300">{{ $message }}</div> @enderror
    </div>

    <div>
        <label class="mb-2 block text-sm text-slate-400" for="tax_number">Vergi No</label>
        <input id="tax_number" name="tax_number" value="{{ old('tax_number', $company->tax_number) }}"
               class="w-full rounded-xl border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-100 focus:border-emerald-400 focus:ring-emerald-400">
        @error('tax_number') <div class="mt-2 text-sm text-red-300">{{ $message }}</div> @enderror
    </div>
</div>

<div class="grid gap-4 sm:grid-cols-2">
    <div>
        <label class="mb-2 block text-sm text-slate-400" for="contact_name">Yetkili</label>
        <input id="contact_name" name="contact_name" value="{{ old('contact_name', $company->contact_name) }}"
               class="w-full rounded-xl border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-100 focus:border-emerald-400 focus:ring-emerald-400">
        @error('contact_name') <div class="mt-2 text-sm text-red-300">{{ $message }}</div> @enderror
    </div>

    <div>
        <label class="mb-2 block text-sm text-slate-400" for="phone">Telefon</label>
        <input id="phone" name="phone" value="{{ old('phone', $company->phone) }}"
               class="w-full rounded-xl border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-100 focus:border-emerald-400 focus:ring-emerald-400">
        @error('phone') <div class="mt-2 text-sm text-red-300">{{ $message }}</div> @enderror
    </div>
</div>

<div>
    <label class="mb-2 block text-sm text-slate-400" for="email">E-posta</label>
    <input id="email" name="email" type="email" value="{{ old('email', $company->email) }}"
           class="w-full rounded-xl border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-100 focus:border-emerald-400 focus:ring-emerald-400">
    @error('email') <div class="mt-2 text-sm text-red-300">{{ $message }}</div> @enderror
</div>

<label class="flex items-center gap-3 rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-slate-300">
    <input type="hidden" name="is_active" value="0">
    <input type="checkbox" name="is_active" value="1" @checked(old('is_active', $company->is_active))
           class="rounded border-white/10 bg-white/10 text-emerald-400 focus:ring-emerald-400">
    Aktif firma
</label>

<div>
    <label class="mb-2 block text-sm text-slate-400" for="notes">Notlar</label>
    <textarea id="notes" name="notes" rows="6"
              class="w-full rounded-xl border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-100 focus:border-emerald-400 focus:ring-emerald-400">{{ old('notes', $company->notes) }}</textarea>
    @error('notes') <div class="mt-2 text-sm text-red-300">{{ $message }}</div> @enderror
</div>

<button class="w-full rounded-full border border-emerald-400/30 bg-emerald-400/10 px-5 py-2 text-sm font-semibold text-emerald-200 transition hover:bg-emerald-400/15">
    Kaydet
</button>
