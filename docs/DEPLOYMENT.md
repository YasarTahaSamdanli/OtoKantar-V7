# OtoKantar Deployment Guide

Bu proje su an Render uzerinde ucretsiz calisabilir; ayni Dockerfile daha sonra DigitalOcean App Platform veya bir VPS/Droplet uzerinde kullanilacak sekilde hazirlandi.

## Ortak production mantigi

- Web server: `php:8.3-apache` Docker imaji
- Public document root: `/var/www/html/public`
- Saglik kontrolu: `/up`
- Otomatik baslangic islemleri: `docker/entrypoint.sh`
- Migration: `RUN_MIGRATIONS=true`
- Admin kullanici olusturma/guncelleme: `ENSURE_ADMIN=true`
- Canli veri tokeni: `LIVE_INGEST_API_TOKEN`
- Kalici ana veri kaynagi: uygulama veritabani (`vehicle_passes`)
- Legacy runtime dosyalari: `LEGACY_RUNTIME_PATH`

`/up`, Laravel'in health endpointidir. Render ve DigitalOcean gibi platformlarda health check path olarak bu endpoint kullanilmali.

## Render'da bugunku durum

`render.yaml` Docker runtime ile calisir ve `/up` health check kullanir. Render uzerinde asagidaki secret/env degerleri panelden girilmeli:

- `APP_KEY` veya Render tarafinda uretilen `APP_KEY_BASE64`
- `APP_URL`
- `ADMIN_EMAIL`
- `ADMIN_PASSWORD`
- `LIVE_INGEST_API_TOKEN`
- `DATABASE_URL` / `DB_URL` Render database bind ile gelir

Deploy sonrasi kontrol:

```bash
curl -I https://senin-render-adresin.onrender.com/up
```

Beklenen sonuc `200 OK`.

## DigitalOcean App Platform'a gecis

DigitalOcean'a gecmek istediginde en kolay yol:

1. Repoyu GitHub'a push et.
2. DigitalOcean App Platform'da yeni app olustur.
3. Dockerfile ile build sec.
4. HTTP port olarak `80` kullan.
5. Health check path olarak `/up` kullan.
6. PostgreSQL Managed Database ekle.
7. Env degerlerini `.env.production.example` dosyasina gore gir.
8. Ilk deployda `RUN_MIGRATIONS=true` ve `ENSURE_ADMIN=true` kalsin.

Bu repo icinde `deploy/digitalocean-app.yaml.example` dosyasi var. App Platform'u `doctl` ile yonetmek istersen bunu kendi GitHub repo, domain ve secret degerlerine gore kopyalayip kullanabilirsin.

Ornek:

```bash
doctl apps create --spec deploy/digitalocean-app.yaml
```

Mevcut app'i guncellemek icin:

```bash
doctl apps update <app-id> --spec deploy/digitalocean-app.yaml
```

## DigitalOcean'a gecmeden once checklist

- `APP_DEBUG=false`
- `APP_ENV=production`
- `APP_URL` gercek domain
- `APP_KEY` sabit ve sakli
- `ADMIN_PASSWORD` guclu ve sakli
- `LIVE_INGEST_API_TOKEN` uzun ve rastgele
- `SESSION_ENCRYPT=true`
- `DB_SSLMODE=require`
- Managed database backup aktif
- Domain DNS kayitlari hazir
- `/up` endpointi 200 donuyor
- `/api/live-ingest` yalniz token ile calisiyor
- Canli panel giris yapmadan acilmiyor

## VPS veya Droplet'e manuel gecis notu

App Platform yerine Droplet kullanirsan ayni Dockerfile ile ilerleyebilirsin:

```bash
docker build -t otokantar-v7 .
docker run -d --name otokantar-v7 \
  --restart unless-stopped \
  -p 80:80 \
  --env-file .env.production \
  otokantar-v7
```

Droplet kullanirken ayrica sunlari kurmak gerekir:

- Reverse proxy ve SSL: Caddy veya Nginx + Certbot
- Database: DigitalOcean Managed PostgreSQL onerilir
- Log takibi: `docker logs -f otokantar-v7`
- Backup: database backup + `LEGACY_RUNTIME_PATH` dosyalari

## Gecis gunu kisa plan

1. Render'daki son database backup'ini al.
2. DigitalOcean database'e import et.
3. DigitalOcean app env degerlerini gir.
4. Ilk deployu yap.
5. `/up`, login, `/canli`, CSV export ve `/api/live-ingest` testlerini calistir.
6. Kamera/legacy Python tarafindaki endpoint URL'sini yeni domaine cevir.
7. DNS'i DigitalOcean'a yonlendir.
8. Render'i bir sure yedek olarak tut, sonra kapat.
