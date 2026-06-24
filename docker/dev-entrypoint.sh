#!/usr/bin/env bash
set -e

mkdir -p storage/framework/cache storage/framework/sessions storage/framework/views storage/logs bootstrap/cache storage/app/legacy_python
chown -R www-data:www-data storage bootstrap/cache || true

if [ ! -f .env ] && [ -f .env.docker.example ]; then
  cp .env.docker.example .env
fi

if [ ! -f vendor/autoload.php ]; then
  composer install --no-interaction --prefer-dist
fi

if [ -f artisan ]; then
  php artisan key:generate --ansi --force --no-interaction >/dev/null 2>&1 || true
  php artisan migrate --force --no-interaction || true
fi

exec "$@"
