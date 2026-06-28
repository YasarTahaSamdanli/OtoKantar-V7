<?php

namespace App\Services\Canli;

class CanliRuntimePathResolver
{
    public function path(string $name): string
    {
        $root = rtrim((string) config('services.legacy_runtime.path', base_path('legacy')), '\\/');
        if (! $this->isAbsolutePath($root)) {
            $root = base_path($root);
        }

        return $root.DIRECTORY_SEPARATOR.$name;
    }

    private function isAbsolutePath(string $path): bool
    {
        return $path !== '' && (str_starts_with($path, '/') || preg_match('/^[A-Za-z]:[\/\\\\]/', $path) === 1);
    }
}
