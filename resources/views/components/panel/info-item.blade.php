@props([
    'label',
    'value',
    'valueId' => null,
])

<div>
    <div class="info-k">{{ $label }}</div>
    <div class="info-v" @if ($valueId) id="{{ $valueId }}" @endif>{{ $value }}</div>
</div>
