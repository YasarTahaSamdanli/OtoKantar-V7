#!/usr/bin/env bash
set -e

mkdir -p storage/framework/cache storage/framework/sessions storage/framework/views storage/logs storage/logs/stress_test bootstrap/cache
chown -R www-data:www-data storage bootstrap/cache

if [ -z "${APP_KEY:-}" ] && [ -n "${APP_KEY_BASE64:-}" ]; then
  export APP_KEY="base64:${APP_KEY_BASE64}"
fi

export PORT="${PORT:-10000}"
sed -ri "s/^Listen [0-9]+/Listen ${PORT}/" /etc/apache2/ports.conf
sed -ri "s/<VirtualHost \*:[0-9]+>/<VirtualHost *:${PORT}>/" /etc/apache2/sites-available/*.conf

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
