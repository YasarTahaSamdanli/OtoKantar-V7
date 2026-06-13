@props([
    'title',
    'badge',
    'badgeId' => null,
])

<div class="head">
    <div class="title">{{ $title }}</div>
    <div class="badge" @if ($badgeId) id="{{ $badgeId }}" @endif>{{ $badge }}</div>
</div>
