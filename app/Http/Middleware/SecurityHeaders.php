<?php

namespace App\Http\Middleware;

use Closure;
use Illuminate\Http\Request;
use Symfony\Component\HttpFoundation\Response;

class SecurityHeaders
{
    public function handle(Request $request, Closure $next): Response
    {
        $response = $next($request);

        if (! config('security.headers.enabled')) {
            return $response;
        }

        $response->headers->set('X-Content-Type-Options', 'nosniff');
        $response->headers->set('X-Frame-Options', (string) config('security.headers.frame_options', 'DENY'));
        $response->headers->set('Referrer-Policy', (string) config('security.headers.referrer_policy', 'strict-origin-when-cross-origin'));
        $response->headers->set('Permissions-Policy', (string) config('security.headers.permissions_policy', 'camera=(), microphone=(), geolocation=(), payment=(), usb=()'));
        $response->headers->set('Cross-Origin-Opener-Policy', (string) config('security.headers.cross_origin_opener_policy', 'same-origin'));

        $csp = trim((string) config('security.headers.content_security_policy', ''));
        if ($csp !== '') {
            $response->headers->set('Content-Security-Policy', $csp);
        }

        if ($request->isSecure() && config('security.headers.hsts.enabled')) {
            $hsts = 'max-age='.(int) config('security.headers.hsts.max_age', 31536000);

            if (config('security.headers.hsts.include_subdomains')) {
                $hsts .= '; includeSubDomains';
            }

            if (config('security.headers.hsts.preload')) {
                $hsts .= '; preload';
            }

            $response->headers->set('Strict-Transport-Security', $hsts);
        }

        return $response;
    }
}
