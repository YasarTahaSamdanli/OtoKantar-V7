#!/usr/bin/env bash
set -e

mkdir -p storage/framework/cache storage/framework/sessions storage/framework/views storage/logs bootstrap/cache
chown -R www-data:www-data storage bootstrap/cache

if [ -z "${APP_KEY:-}" ] && [ -n "${APP_KEY_BASE64:-}" ]; then
  export APP_KEY="base64:${APP_KEY_BASE64}"
fi

php artisan config:clear

if [ "${RUN_MIGRATIONS:-false}" = "true" ]; then
  php artisan migrate --force
fi

if [ "${ENSURE_ADMIN:-false}" = "true" ]; then
  php artisan otokantar:ensure-admin
fi

php artisan config:cache
php artisan route:cache
php artisan view:cache

exec "$@"
