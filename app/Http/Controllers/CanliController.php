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
                if ($this->canliData->vehiclePassHasRecords($filters)) {
                    try {
                        $payload = $this->canliData->vehiclePassPanelPayload($limit, $filters);
                        Cache::put($cacheKey, $payload, $ttlSeconds);

                        return response()->json($payload);
                    } catch (Throwable $e) {
                        Log::warning('VehiclePass panel payload kullanilamadi, legacy kaynaklara dusuluyor', ['exception' => $e]);
                    }
                }

                try {
                    $pdo = DB::connection('legacy')->getPdo();
                } catch (Throwable $e) {
                    Log::warning('Legacy DB yok, JSON-only panel payload kullaniliyor', ['exception' => $e]);
                    $payload = $this->canliData->jsonOnlyPanelPayload($limit, $filters);
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

    public function getLiveTicker(Request $request)
    {
        if ($guard = $this->guardCanliAccess($request)) {
            return $guard;
        }

        try {
            $cacheKey = $this->canliCacheKey($request, 'live-ticker', 5);
            $cached = Cache::get($cacheKey);
            if (is_array($cached)) {
                return response()->json($cached);
            }

            if ($this->canliData->vehiclePassHasRecords()) {
                try {
                    $payload = $this->canliData->vehiclePassLiveTickerPayload();
                    Cache::put($cacheKey, $payload, 1);

                    return response()->json($payload);
                } catch (Throwable $e) {
                    Log::warning('VehiclePass live ticker payload kullanilamadi, legacy kaynaklara dusuluyor', ['exception' => $e]);
                }
            }

            try {
                $payload = $this->canliData->dbLiveTickerPayload(DB::connection('legacy')->getPdo());
            } catch (Throwable $e) {
                Log::warning('Live ticker icin DB kullanilamadi, JSON-only panel payload kullaniliyor', ['exception' => $e]);
                $payload = $this->canliData->jsonOnlyPanelPayload(5);
            }

            Cache::put($cacheKey, $payload, 1);

            return response()->json($payload);
        } catch (Throwable $e) {
            Log::error('Canli ticker API hatasi', ['exception' => $e]);

            return response()->json([
                'hata' => 'Canli ticker verisi okunamadi',
                'mesaj' => 'Canli veri kaynagi su anda yanit vermiyor.',
            ], 500);
        }
    }

    public function getArchive(Request $request)
    {
        if ($guard = $this->guardCanliAccess($request)) {
            return $guard;
        }

        $page = max(1, (int) $request->query('page', 1));
        $perPage = 50;
        $filters = $this->recordFilters($request);

        try {
            $cacheKey = $this->canliCacheKey($request, 'archive:'.$page, $perPage);
            $cached = Cache::get($cacheKey);
            if (is_array($cached)) {
                return response()->json($cached);
            }

            if ($this->canliData->vehiclePassHasRecords($filters)) {
                try {
                    $payload = $this->canliData->vehiclePassArchivePayload($page, $perPage, $filters);
                    Cache::put($cacheKey, $payload, 5);

                    return response()->json($payload);
                } catch (Throwable $e) {
                    Log::warning('VehiclePass arsiv payload kullanilamadi, legacy kaynaklara dusuluyor', ['exception' => $e]);
                }
            }

            try {
                $payload = $this->canliData->dbArchivePayload(DB::connection('legacy')->getPdo(), $page, $perPage, $filters);
            } catch (Throwable $e) {
                Log::warning('Arsiv icin DB kullanilamadi, JSON-only arsiv payload kullaniliyor', ['exception' => $e]);
                $payload = $this->canliData->jsonOnlyArchivePayload($page, $perPage, $filters);
            }

            Cache::put($cacheKey, $payload, 5);

            return response()->json($payload);
        } catch (Throwable $e) {
            Log::error('Canli arsiv API hatasi', [
                'page' => $page,
                'exception' => $e,
            ]);

            return response()->json([
                'hata' => 'Arsiv verisi okunamadi',
                'mesaj' => 'Kayit arsivi su anda yanit vermiyor.',
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
            $filters = $this->recordFilters($request);

            if ($this->canliData->vehiclePassHasRecords($filters)) {
                try {
                    $export = $this->canliData->vehiclePassCsvIcerikOlustur($filters);
                } catch (Throwable $e) {
                    Log::warning('VehiclePass CSV kullanilamadi, legacy kaynaklara dusuluyor', ['exception' => $e]);
                    $export = null;
                }
            }

            if (!isset($export) || ($export['row_count'] ?? 0) === 0) {
                try {
                    $pdo = DB::connection('legacy')->getPdo();
                    $export = $this->canliData->csvIcerikOlustur($pdo, $filters);
                    if (($export['row_count'] ?? 0) === 0) {
                        $fallback = $this->canliData->csvDosyaIcerikOlustur($this->canliData->legacyPath('kantar_raporu.csv'), $filters);
                        $export = (($fallback['row_count'] ?? 0) > 0)
                            ? $fallback
                            : $this->canliData->jsonCsvIcerikOlustur($filters);
                    }
                } catch (Throwable $e) {
                    Log::warning('Canli CSV icin DB kullanilamadi, dosya fallback deneniyor', ['exception' => $e]);
                    $export = $this->canliData->csvDosyaIcerikOlustur($this->canliData->legacyPath('kantar_raporu.csv'), $filters);
                    if (($export['row_count'] ?? 0) === 0) {
                        $export = $this->canliData->jsonCsvIcerikOlustur($filters);
                    }
                }
            }

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

        if (!is_file($path)) {
            return Response::make($this->missingFrameSvg(), 200, [
                'Content-Type' => 'image/svg+xml; charset=utf-8',
                'Cache-Control' => 'no-store, no-cache, must-revalidate, max-age=0',
                'Pragma' => 'no-cache',
            ]);
        }

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

    private function missingFrameSvg(): string
    {
        return <<<'SVG'
<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720" role="img" aria-label="Canli kare bekleniyor">
  <rect width="1280" height="720" fill="#091018"/>
  <rect x="1" y="1" width="1278" height="718" fill="none" stroke="rgba(255,255,255,.12)" stroke-width="2"/>
  <g fill="none" stroke="#21d19f" stroke-width="12" opacity=".75">
    <rect x="520" y="290" width="240" height="118" rx="18"/>
    <circle cx="580" cy="410" r="24" fill="#21d19f" stroke="none"/>
    <circle cx="700" cy="410" r="24" fill="#21d19f" stroke="none"/>
    <path d="M575 290v-30c0-28 22-50 50-50h30c28 0 50 22 50 50v30"/>
  </g>
  <text x="640" y="475" text-anchor="middle" fill="#94a0b1" font-family="Arial, sans-serif" font-size="30">Canli kare bekleniyor</text>
</svg>
SVG;
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
            'plate' => strtoupper(trim((string) $request->query('plate', $request->query('plaka', '')))),
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
