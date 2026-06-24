# Local Docker Development

Bu dosya OtoKantar V7'yi XAMPP'e bagli kalmadan Docker ile calistirmak icin hazirlandi.

## Servisler

| Servis | Aciklama | URL / Port |
| --- | --- | --- |
| `app` | Laravel + Apache + PHP 8.3 | http://localhost:8080 |
| `node` | Vite dev server | http://localhost:5173 |
| `queue` | Laravel queue listener | - |
| `mysql` | MySQL 8.4 | host: `localhost:3307`, container: `mysql:3306` |

## Ilk calistirma

```bash
docker compose up --build
```

Uygulama:

```text
http://localhost:8080
```

Health check:

```text
http://localhost:8080/up
```

## Ortam dosyasi

Docker gelistirme icin ornek dosya:

```text
.env.docker.example
```

`docker/dev-entrypoint.sh`, container icinde `.env` yoksa `.env.docker.example` dosyasindan olusturur. Repoda zaten `.env` varsa compose environment degerleri Docker servisleri icin override eder.

## Faydalı komutlar

Container icinde Artisan:

```bash
docker compose exec app php artisan migrate
docker compose exec app php artisan otokantar:ensure-admin
```

Test calistirma:

```bash
docker compose exec -e APP_ENV=testing app php artisan test
```

Composer:

```bash
docker compose exec app composer install
docker compose exec app composer require vendor/package
```

Node:

```bash
docker compose exec node npm install
docker compose exec node npm run build
```

Loglar:

```bash
docker compose logs -f app
docker compose logs -f node
docker compose logs -f mysql
```

Servisleri durdurma:

```bash
docker compose down
```

Veritabani ve named volume'leri tamamen silmek:

```bash
docker compose down -v
```

## Local ingest testi

```bash
curl -X POST http://localhost:8080/api/live-ingest \
  -H "Authorization: Bearer local-dev-token" \
  -H "Content-Type: application/json" \
  -d '{"event_type":"GIRIS","son_guncelleme":"2026-06-24T12:00:00","kantar_kg":12450.5,"son_kayit":{"plaka":"34ABC123","durum":"GIRIS","giris_tarih":"2026-06-24","giris_saat":"12:00:00","giris_agirlik":12450.5,"guven":0.94}}'
```

## Notlar

- `node_modules`, `storage` ve MySQL datasi Docker named volume olarak tutulur.
- `vendor` proje klasorunden kullanilir; hostta yoksa `app` container ilk acilista `composer install` calistirir.
- Host makinedeki proje dosyalari container icine bind mount edilir; PHP ve Blade degisiklikleri aninda gorulur.
- Vite hot reload icin `node` servisi 5173 portunu disari acar.
- Production Dockerfile ayri tutuldu. Local gelistirme `docker/php-dev.Dockerfile` kullanir.
