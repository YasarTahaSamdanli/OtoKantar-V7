@props([
    'tab',
    'active' => false,
])

<button class="view-btn{{ $active ? ' active' : '' }}" type="button" data-tab="{{ $tab }}">
    {{ $slot }}
</button>
