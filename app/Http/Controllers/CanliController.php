<?php

namespace App\Http\Controllers;

use App\Services\CanliDataService;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Facades\Response;
use Symfony\Component\HttpFoundation\Response as SymfonyResponse;
use Throwable;

class CanliController extends Controller
{
    public function __construct(
        private readonly CanliDataService $canliData,
    ) {}

    public function view(Request $request)
    {
        if ($guard = $this->guardCanliAccess($request)) {
            return $guard;
        }

        return view('panel');
    }

    public function api(Request $request)
    {
        if ($guard = $this->guardCanliAccess($request)) {
            return $guard;
        }

        $action = (string) $request->query('action', 'panel');
        $limit = min(200, max(1, (int) $request->query('limit', 40)));
        $filters = $this->recordFilters($request);

        try {
            $cacheKey = $this->canliCacheKey($request, $action, $limit);
            $ttlSeconds = $action === 'panel' ? 2 : 1;

            $cached = Cache::get($cacheKey);
            if (is_array($cached)) {
                return response()->json($cached);
            }

            if ($action === 'durum') {
                try {
                    $payload = $this->canliData->durumOkuVeyaFallback(DB::connection('legacy')->getPdo());
                } catch (Throwable) {
                    $payload = $this->canliData->durumOkuVeyaFallback();
                }
                Cache::put($cacheKey, $payload, $ttlSeconds);
                return response()->json($payload);
            }

            if ($action === 'panel') {
                try {
                    $pdo = DB::connection('legacy')->getPdo();
                } catch (Throwable $e) {
                    Log::warning('Legacy DB yok, JSON-only panel payload kullaniliyor', ['exception' => $e]);
                    $payload = $this->canliData->jsonOnlyPanelPayload($limit);
                    Cache::put($cacheKey, $payload, $ttlSeconds);

                    return response()->json($payload);
                }

                $payload = $this->canliData->dbPanelPayload($pdo, $limit, $filters);
                Cache::put($cacheKey, $payload, $ttlSeconds);
                return response()->json($payload);
            }

            return response()->json([
                'hata' => 'Gecersiz action parametresi',
                'gecerli_actionlar' => ['durum', 'panel'],
            ], 400);
        } catch (Throwable $e) {
            Log::error('Canli API hatasi', [
                'action' => $action,
                'limit' => $limit,
                'exception' => $e,
            ]);

            return response()->json([
                'hata' => 'MySQL baglanti/sorgu hatasi',
                'mesaj' => 'Canli veri kaynagi su anda yanit vermiyor.',
            ], 500);
        }
    }

    public function csv(Request $request)
    {
        if ($guard = $this->guardCanliAccess($request)) {
            return $guard;
        }

        abort_unless($request->user()?->isAdmin(), 403);

        try {
            $pdo = DB::connection('legacy')->getPdo();
            $export = $this->canliData->csvIcerikOlustur($pdo, $this->recordFilters($request));

            return Response::make($export['content'], 200, [
                'Content-Type' => 'text/csv; charset=utf-8',
                'Content-Disposition' => 'attachment; filename="'.$export['filename'].'"',
                'Cache-Control' => 'no-store',
            ]);
        } catch (Throwable $e) {
            Log::error('Canli CSV olusturma hatasi', ['exception' => $e]);
            abort(500, 'CSV raporu su anda olusturulamiyor.');
        }
    }

    public function kare(Request $request)
    {
        if ($guard = $this->guardCanliAccess($request)) {
            return $guard;
        }

        $path = $this->canliData->legacyPath('canli_kare.jpg');
        abort_unless(is_file($path), 404);

        return response()->file($path, [
            'Cache-Control' => 'no-store, no-cache, must-revalidate, max-age=0',
            'Pragma' => 'no-cache',
        ]);
    }

    private function guardCanliAccess(Request $request): ?SymfonyResponse
    {
        $user = $request->user();

        if (!$user) {
            return $request->expectsJson()
                ? response()->json(['message' => 'Unauthenticated.'], 401)
                : redirect()->guest(route('login'));
        }

        if (!in_array($user->role, ['admin', 'employee'], true)) {
            abort(403);
        }

        return null;
    }

    private function canliCacheKey(Request $request, string $action, int $limit): string
    {
        $userPart = $request->user()?->id ? ('u:'.$request->user()->id) : ('ip:'.$request->ip());
        $query = $this->cacheRelevantQuery($request);

        return 'canli:'.$userPart.':'.$action.':'.$limit.':'.sha1(http_build_query($query, '', '&', PHP_QUERY_RFC3986));
    }

    private function cacheRelevantQuery(Request $request): array
    {
        $ignored = ['_', 'action', 'limit', 't', 'timestamp'];
        $query = array_filter(
            $request->query(),
            fn (string $key): bool => !in_array(strtolower($key), $ignored, true),
            ARRAY_FILTER_USE_KEY
        );

        $this->sortQueryRecursive($query);

        return $query;
    }

    private function recordFilters(Request $request): array
    {
        $period = strtolower((string) $request->query('period', 'all'));
        if (!in_array($period, ['all', 'day', 'month', 'year'], true)) {
            $period = 'all';
        }

        return [
            'period' => $period,
            'date' => $this->validDate((string) $request->query('date', date('Y-m-d'))) ?? date('Y-m-d'),
            'month' => $this->validMonth((string) $request->query('month', date('Y-m'))) ?? date('Y-m'),
            'year' => $this->validYear((string) $request->query('year', date('Y'))) ?? date('Y'),
        ];
    }

    private function validDate(string $value): ?string
    {
        return preg_match('/^\d{4}-\d{2}-\d{2}$/', $value) === 1 ? $value : null;
    }

    private function validMonth(string $value): ?string
    {
        return preg_match('/^\d{4}-\d{2}$/', $value) === 1 ? $value : null;
    }

    private function validYear(string $value): ?string
    {
        return preg_match('/^\d{4}$/', $value) === 1 ? $value : null;
    }

    private function sortQueryRecursive(array &$query): void
    {
        ksort($query);

        foreach ($query as &$value) {
            if (is_array($value)) {
                $this->sortQueryRecursive($value);
            }
        }
    }
}
