<?php

namespace Tests\Unit;

use App\Http\Controllers\CanliController;
use App\Services\AuditLogService;
use App\Services\CanliDataService;
use Illuminate\Http\Request;
use ReflectionMethod;
use Tests\TestCase;

class CanliCacheKeyTest extends TestCase
{
    public function test_cache_busting_query_parameters_do_not_change_cache_key(): void
    {
        $first = $this->cacheKeyFor([
            'action' => 'panel',
            'limit' => '40',
            't' => '1710000000000',
            '_' => 'abc',
            'timestamp' => '2026-06-13T12:00:00',
        ]);

        $second = $this->cacheKeyFor([
            'action' => 'panel',
            'limit' => '40',
            't' => '1710000000999',
            '_' => 'def',
            'timestamp' => '2026-06-13T12:00:01',
        ]);

        $this->assertSame($first, $second);
    }

    public function test_meaningful_query_parameters_still_change_cache_key(): void
    {
        $first = $this->cacheKeyFor([
            'action' => 'panel',
            'limit' => '40',
            'filter' => 'today',
        ]);

        $second = $this->cacheKeyFor([
            'action' => 'panel',
            'limit' => '40',
            'filter' => 'all',
        ]);

        $this->assertNotSame($first, $second);
    }

    public function test_query_parameter_order_does_not_change_cache_key(): void
    {
        $first = $this->cacheKeyFor([
            'action' => 'panel',
            'limit' => '40',
            'filter' => 'today',
            'sort' => 'desc',
        ]);

        $second = $this->cacheKeyFor([
            'sort' => 'desc',
            'filter' => 'today',
            'limit' => '40',
            'action' => 'panel',
        ]);

        $this->assertSame($first, $second);
    }

    private function cacheKeyFor(array $query): string
    {
        $request = Request::create('/canli/api', 'GET', $query);
        $request->setUserResolver(fn (): object => (object) ['id' => 7]);

        $method = new ReflectionMethod(CanliController::class, 'canliCacheKey');
        $method->setAccessible(true);

        return $method->invoke(
            new CanliController(
                $this->app->make(CanliDataService::class),
                $this->app->make(AuditLogService::class),
            ),
            $request,
            (string) ($query['action'] ?? 'panel'),
            min(200, max(1, (int) ($query['limit'] ?? 40)))
        );
    }
}
