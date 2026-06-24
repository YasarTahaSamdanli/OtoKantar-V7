# OtoKantar V7

OtoKantar V7, araç kantar operasyonlarını dijitalleştirmek için geliştirilen Laravel tabanlı bir canlı izleme, kayıt ve yönetim platformudur. Sistem; saha tarafındaki kamera, plaka tanıma ve kantar akışlarından gelen verileri merkezi bir web panelinde toplar, giriş-çıkış kayıtlarını saklar, raporlama ve kullanıcı yönetimi sağlar.

Bu repo şu an Render üzerinde Docker ile çalışabilecek şekilde hazırlanmıştır. Aynı yapı ileride DigitalOcean App Platform, DigitalOcean Droplet veya benzeri production ortamlarına taşınabilecek şekilde düzenlenmektedir.

## Öne Çıkanlar

- Canlı kantar paneli: ağırlık, plaka tamponu, kamera karesi, sistem durumu ve son kayıtlar
- Araç geçiş kayıtları: giriş, çıkış, net ağırlık, güven skoru, snapshot ve kaynak payload
- Yetkili kullanım: admin ve employee rolleri
- Admin kullanıcı yönetimi
- CSV dışa aktarım
- Legacy uyumluluk: eski PHP URL yönlendirmeleri ve legacy MySQL/JSON/CSV fallback desteği
- Remote ingest API: saha bilgisayarından canlı JSON/JPG verisi alma
- Docker tabanlı deployment
- Render ve DigitalOcean geçiş hazırlığı
- Feature ve unit testleri

## Kullanılan Teknolojiler

- PHP 8.2+
- Laravel 12
- Laravel Breeze
- MySQL/MariaDB veya PostgreSQL
- Vite
- Tailwind CSS
- Docker
- PHPUnit

Legacy saha uygulaması tarafında Python, OpenCV, OCR ve kantar/yazıcı entegrasyonları bulunabilir. Bu Laravel uygulaması, o saha akışının web paneli ve merkezi kayıt katmanı olarak konumlanır.

## Mimari

```text
Saha sistemi
  Kamera / OCR / Kantar / Yazıcı
          |
          |  POST /api/live-ingest
          v
Laravel uygulaması
  Auth + Role kontrolü
  Canlı panel API'leri
  VehiclePass kayıt modeli
  Legacy JSON/JPG/CSV fallback
          |
          v
Database + Runtime dosyaları
  vehicle_passes
  canli_durum.json
  canli_kare.jpg
  gecis_gecmisi.jsonl
```

## Ana Modüller

| Alan | Açıklama |
| --- | --- |
| `app/Http/Controllers/CanliController.php` | Canlı panel, arşiv, CSV ve kamera karesi endpointleri |
| `app/Http/Controllers/LiveIngestController.php` | Saha sisteminden gelen JSON/JPG verisini alır |
| `app/Services/CanliDataService.php` | VehiclePass, legacy DB, JSONL ve CSV kaynaklarından panel payload üretir |
| `app/Models/VehiclePass.php` | Merkezi araç geçiş kaydı modeli |
| `resources/views/panel.blade.php` | Canlı izleme paneli |
| `routes/web.php` | Panel, admin ve auth rotaları |
| `routes/api.php` | Remote ingest API rotası |
| `docker/entrypoint.sh` | Production başlangıç işlemleri |
| `docs/DEPLOYMENT.md` | Render ve DigitalOcean deployment rehberi |

## Roller

| Rol | Yetki |
| --- | --- |
| `admin` | Dashboard, canlı panel, kullanıcı yönetimi, CSV export |
| `employee` | Canlı panel görüntüleme |

Public registration varsayılan olarak kapalı tutulmalıdır. Production ortamında ilk admin kullanıcı `ENSURE_ADMIN=true`, `ADMIN_EMAIL` ve `ADMIN_PASSWORD` ile oluşturulabilir.

## Kurulum

### Gereksinimler

- PHP 8.2 veya üzeri
- Composer
- Node.js 22 veya uyumlu güncel LTS
- MySQL/MariaDB ya da PostgreSQL

### Lokal kurulum

```bash
composer install
npm install
cp .env.example .env
php artisan key:generate
php artisan migrate
npm run build
php artisan serve
```

Geliştirme sırasında Vite için:

```bash
npm run dev
```

Laravel'in hazır geliştirme komutu da kullanılabilir:

```bash
composer run dev
```

## Ortam Değişkenleri

Temel değişkenler:

```env
APP_ENV=production
APP_DEBUG=false
APP_URL=https://example.com

DB_CONNECTION=pgsql
DATABASE_URL=
DB_URL=

ALLOW_PUBLIC_REGISTRATION=false
ENSURE_ADMIN=true
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=strong-password

LEGACY_RUNTIME_PATH=/var/www/html/storage/app/legacy_python
LIVE_INGEST_API_TOKEN=long-random-token
```

Production için örnek şablon:

- [.env.production.example](.env.production.example)

## Canlı Veri Akışı

Saha uygulaması canlı durum ve geçiş olaylarını şu endpoint'e gönderir:

```http
POST /api/live-ingest
Authorization: Bearer <LIVE_INGEST_API_TOKEN>
```

Örnek JSON payload:

```json
{
  "event_type": "GIRIS",
  "son_guncelleme": "2026-06-24T12:00:00",
  "kantar_kg": 12450.5,
  "son_kayit": {
    "plaka": "34ABC123",
    "durum": "GIRIS",
    "giris_tarih": "2026-06-24",
    "giris_saat": "12:00:00",
    "giris_agirlik": 12450.5,
    "guven": 0.94
  }
}
```

`GIRIS` ve `CIKIS` olaylarında sistem:

- `canli_durum.json` dosyasını günceller
- varsa `canli_kare.jpg` dosyasını günceller
- `gecis_gecmisi.jsonl` içine olay geçmişi yazar
- legacy MySQL uygunsa `araclar/gecisler` tablolarına yazar
- merkezi `vehicle_passes` tablosuna kayıt açar veya günceller

## Testler

```bash
php artisan test
```

Windows/XAMPP kullanıyorsan:

```powershell
C:\xampp\php\php.exe artisan test
```

Mevcut test kapsamı auth, kullanıcı yönetimi, canlı panel erişimi, remote ingest, CSV export ve VehiclePass panel okuma akışlarını kapsar.

## Deployment

Bu proje Docker ile deploy edilmeye hazırdır.

Render:

- `render.yaml` Docker runtime kullanır
- Health check path: `/up`
- Database env değerleri Render managed database üzerinden bağlanır

DigitalOcean:

- Örnek app spec: [deploy/digitalocean-app.yaml.example](deploy/digitalocean-app.yaml.example)
- Detaylı rehber: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)

Docker imajı production başlangıcında şunları yapar:

```text
storage/bootstrap cache dizinlerini hazırlar
APP_KEY_BASE64 varsa APP_KEY üretir
config cache temizler
RUN_MIGRATIONS=true ise migrate --force çalıştırır
ENSURE_ADMIN=true ise admin kullanıcısını hazırlar
config, route ve view cache oluşturur
Apache başlatır
```

## Health Check

Laravel health endpoint:

```http
GET /up
```

Dockerfile içinde konteyner healthcheck olarak kullanılır. Render ve DigitalOcean App Platform tarafında da aynı endpoint kullanılmalıdır.

## Faydalı Artisan Komutları

Admin kullanıcı oluşturma/güncelleme:

```bash
php artisan otokantar:ensure-admin
```

Son araç geçiş kayıtları:

```bash
php artisan vehicle-passes:latest --limit=10
```

VehiclePass ile dashboard kaynaklarını karşılaştırma:

```bash
php artisan vehicle-passes:verify
```

## Proje Yapısı

```text
app/
  Http/Controllers/
  Models/
  Services/
database/
  migrations/
  seeders/
deploy/
  digitalocean-app.yaml.example
docs/
  DEPLOYMENT.md
legacy/
legacy_python/
resources/
  css/
  js/
  views/
routes/
  api.php
  web.php
tests/
Dockerfile
render.yaml
```

## Yol Haritası

Kısa vadeli profesyonelleştirme sırası:

1. README ve deployment dokümantasyonu
2. Dashboard arayüzünü kurumsallaştırma
3. Audit log altyapısı
4. Docker ile local geliştirme ortamı
5. GitHub Actions CI iyileştirmesi

## Lisans

Bu repo şu an özel ürün geliştirme projesi olarak ele alınmaktadır. Lisans ve ticari kullanım koşulları ürünleşme aşamasında ayrıca netleştirilmelidir.
