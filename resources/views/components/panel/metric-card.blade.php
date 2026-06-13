@props([
    'title',
    'value' => '0',
    'valueId' => null,
    'valueClass' => '',
    'subtitle' => null,
    'subtitleId' => null,
])

<section class="card">
    <div class="title">{{ $title }}</div>
    <div class="metric-v{{ $valueClass ? ' '.$valueClass : '' }}" @if ($valueId) id="{{ $valueId }}" @endif>{{ $value }}</div>
    @if ($subtitle !== null)
        <div class="metric-s" @if ($subtitleId) id="{{ $subtitleId }}" @endif>{{ $subtitle }}</div>
    @endif
</section>
