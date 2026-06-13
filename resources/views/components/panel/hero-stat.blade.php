@props([
    'title',
    'value' => '--',
    'valueId' => null,
    'subtitle' => null,
    'subtitleId' => null,
])

<div class="panel">
    <div class="title">{{ $title }}</div>
    <div class="v" @if ($valueId) id="{{ $valueId }}" @endif>{{ $value }}</div>
    @if ($subtitle !== null)
        <div class="metric-s" @if ($subtitleId) id="{{ $subtitleId }}" @endif>{{ $subtitle }}</div>
    @endif
</div>
