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

                $payload = $this->canliData->dbPanelPayload($pdo, $limit);
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

        try {
            $pdo = DB::connection('legacy')->getPdo();
            $export = $this->canliData->csvIcerikOlustur($pdo);

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
        $qs = (string) $request->getQueryString();
        return 'canli:'.$userPart.':'.$action.':'.$limit.':'.sha1($qs);
    }
}
