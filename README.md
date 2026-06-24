# OtoKantar V7

OtoKantar V7, kantar sahalarinda arac giris-cikis surecini dijital hale getirmek icin gelistirilen web tabanli bir canli izleme, kayit, raporlama ve yonetim platformudur.

Sistem; kamera, plaka tanima, kantar ve saha bilgisayarindan gelen verileri merkezi bir web panelinde toplar. Operatorler anlik agirlik, plaka, kamera karesi, son gecisler ve kayit durumunu tek ekrandan izler. Yonetim ekibi ise gecmis kayitlara, CSV raporlarina, kullanici rollerine ve denetim kayitlarina ulasabilir.

OtoKantar'in amaci basit: kantar operasyonlarini kagit, manuel takip ve daginik dosyalardan kurtarip guvenli, izlenebilir ve uzaktan yonetilebilir bir sisteme tasimak.

## Neden OtoKantar?

Kantar sahalarinda veri kaybi, hatali plaka kaydi, eksik giris-cikis eslestirmesi, manuel raporlama ve sahadaki bilgisayara bagimli calisma ciddi zaman kaybettirir. OtoKantar V7 bu sorunlari merkezi bir panelde toplar.

- Canli kantar takibi: agirlik, plaka, kamera karesi ve sistem durumu anlik gorulur.
- Giris-cikis kayitlari: arac hareketleri tarih, saat, yon, agirlik ve guven skoru ile saklanir.
- Merkezi veri: saha bilgisayarindan gelen JSON/JPG verileri Laravel uygulamasinda kayda alinir.
- Raporlama: filtreli gecmis kayitlar ve CSV export ile yonetim raporu uretilir.
- Yetkili erisim: admin ve employee rolleri ile panel erisimi kontrol edilir.
- Guvenlik: tokenli remote ingest, rate limit, security header, kapali public registration ve audit log altyapisi bulunur.
- Backup ve geri donus: lokal yedek, secili yedege restore ve rclone ile Google Drive gibi uzak hedeflere kopyalama desteklenir.
- Deploy hazirligi: Docker, Render ve DigitalOcean/VPS gecisine uygun yapi vardir.

## Kimler Icin?

OtoKantar V7 su isletmeler icin uygundur:

- Hafriyat, maden, beton, asfalt, kum-cakil ve lojistik sahalari
- Giris-cikis agirlik takibi yapan kantar isletmeleri
- Plaka tanima ve kamera destekli kayit isteyen tesisler
- Sahadaki veriyi merkeze almak isteyen firmalar
- Rapor, denetim izi ve uzaktan erisim ihtiyaci olan yonetimler

## Temel Ozellikler

| Alan | Aciklama |
| --- | --- |
| Canli panel | Anlik kantar agirligi, plaka tamponu, kamera karesi, son kayitlar ve sistem durumu |
| Arac gecis kayitlari | Giris, cikis, net agirlik, arac agirligi, malzeme agirligi, guven skoru |
| Plaka profilleri | Daha once gorulen araclarin taninmasi ve gecis gecmisi |
| Admin paneli | Kullanici olusturma, rol verme ve audit log izleme |
| CSV export | Yonetim ve muhasebe icin filtreli disari aktarim |
| Remote ingest API | Saha bilgisayarindan tokenli JSON/JPG veri alma |
| Legacy uyumluluk | Eski PHP URL yonlendirmeleri, legacy MySQL, JSONL ve CSV fallback |
| Backup sistemi | Database ve runtime dosyalarini yedekleme, listeleme, restore etme |
| Uzak yedek | rclone ile Google Drive, S3, Backblaze veya baska sunucuya kopyalama |
| Guvenlik | Rate limit, security headers, kapali kayit, role middleware, audit log |

## Mimari

```text
Saha sistemi
  Kamera / OCR / Kantar / Yazici
          |
          |  POST /api/live-ingest
          |  Authorization: Bearer <LIVE_INGEST_API_TOKEN>
          v
Laravel uygulamasi
  Auth + role kontrolu
  Canli panel API'leri
  VehiclePass kayit modeli
  Audit log
  Backup / restore komutlari
          |
          v
Database + runtime dosyalari
  vehicle_passes
  canli_durum.json
  canli_kare.jpg
  gecis_gecmisi.jsonl
  backup paketleri
```

## Guvenlik Yaklasimi

OtoKantar V7 guvenlikte tek bir onleme dayanmaz; birden fazla katman kullanir.

- Uygulama paneli login ve rol kontrolu arkasindadir.
- Admin ve employee rolleri ayridir.
- Public registration varsayilan olarak kapalidir.
- Remote ingest endpoint'i bearer token ister.
- Ingest isteklerinde rate limit, JSON boyut limiti ve JPG dogrulamasi vardir.
- Security header middleware'i temel tarayici korumalarini ekler.
- Session cookie ayarlari production icin guvenli olacak sekilde hazirlanmistir.
- Audit log ile kritik islemler takip edilebilir.
- Composer ve npm audit kontrolleri deployment oncesi calistirilir.
- Backup dosyalari `storage/app/backups` altinda tutulur ve git'e eklenmez.

Production sertlestirme listesi: [docs/SECURITY_HARDENING.md](docs/SECURITY_HARDENING.md)

## Backup ve Geri Donus

Yedek almak:

```bash
php artisan backup:run --keep=14
```

Windows/XAMPP:

```powershell
C:\xampp\php\php.exe artisan backup:run --keep=14
```

Yedekleri listelemek:

```bash
php artisan backup:list
```

Secili yedege donmek:

```bash
php artisan backup:restore --backup=20260624_225720 --force
```

Sadece database geri alinacaksa:

```bash
php artisan backup:restore --backup=20260624_225720 --database-only --force
```

Google Drive gibi uzak hedeflere kopyalama icin `rclone` kullanilir:

```env
BACKUP_REMOTE_ENABLED=true
BACKUP_RCLONE_BINARY=rclone
BACKUP_REMOTE_DESTINATION=gdrive:OtoKantarBackups
BACKUP_REMOTE_SYNC_AFTER_RUN=true
```

Tek seferlik uzak kopya:

```bash
php artisan backup:sync latest
```

## Roller

| Rol | Yetki |
| --- | --- |
| admin | Dashboard, canli panel, kullanici yonetimi, audit log, CSV export |
| employee | Canli panel ve operasyon ekranlari |

Production ortaminda ilk admin kullanici su env degerleri ile hazirlanabilir:

```env
ENSURE_ADMIN=true
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=strong-random-password
ADMIN_NAME=Admin
```

## Canli Veri Akisi

Saha uygulamasi canli durum ve gecis olaylarini su endpoint'e gonderir:

```http
POST /api/live-ingest
Authorization: Bearer <LIVE_INGEST_API_TOKEN>
```

Ornek payload:

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

GIRIS ve CIKIS olaylarinda sistem:

- `canli_durum.json` dosyasini gunceller.
- Gecerli JPG varsa `canli_kare.jpg` dosyasini gunceller.
- `gecis_gecmisi.jsonl` icine olay gecmisi yazar.
- Legacy MySQL uygunsa `araclar/gecisler` tablolarina yazar.
- Merkezi `vehicle_passes` tablosuna kayit acar veya mevcut kaydi gunceller.

## Teknoloji

- PHP 8.2+
- Laravel 12
- Laravel Breeze
- MySQL/MariaDB, PostgreSQL veya SQLite
- Vite
- Tailwind CSS
- Docker
- PHPUnit
- Python/OpenCV tabanli legacy saha entegrasyonu

## Lokal Kurulum

Gereksinimler:

- PHP 8.2 veya uzeri
- Composer
- Node.js 22 veya guncel LTS
- MySQL/MariaDB, PostgreSQL veya SQLite

Kurulum:

```bash
composer install
npm install
cp .env.example .env
php artisan key:generate
php artisan migrate
npm run build
php artisan serve
```

Windows/XAMPP kullanirken PHP komutu:

```powershell
C:\xampp\php\php.exe artisan test
```

Vite development server:

```bash
npm run dev
```

Docker ile lokal calistirma:

```bash
docker compose up --build
```

Uygulama varsayilan olarak `http://localhost:8080` uzerinden acilir. Detayli Docker rehberi: [docs/LOCAL_DOCKER.md](docs/LOCAL_DOCKER.md)

## Production Ortam Degiskenleri

Temel production ayarlari:

```env
APP_ENV=production
APP_DEBUG=false
APP_URL=https://example.com

ALLOW_PUBLIC_REGISTRATION=false
ENSURE_ADMIN=true
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=strong-random-password

SESSION_ENCRYPT=true
SESSION_SECURE_COOKIE=true
SESSION_HTTP_ONLY=true
SESSION_SAME_SITE=lax

LEGACY_RUNTIME_PATH=/var/www/html/storage/app/legacy_python
LIVE_INGEST_API_TOKEN=long-random-token

BACKUP_PATH=/var/www/html/storage/app/backups
BACKUP_KEEP=14
```

Ornek production dosyasi: [.env.production.example](.env.production.example)

## Test ve Kalite Kontrol

Testleri calistirma:

```bash
php artisan test
```

Guvenlik auditleri:

```bash
php composer.phar audit --format=plain
npm audit --audit-level=low
```

Frontend production build:

```bash
npm run build
```

Mevcut test kapsami auth, kullanici yonetimi, admin akislari, audit log, canli panel erisimi, remote ingest, CSV export, VehiclePass okuma, backup ve restore akisini kapsar.

## Deployment

Bu proje Docker ile production'a cikmaya hazirdir.

Render:

- `render.yaml` Docker runtime kullanir.
- Health check path: `/up`
- Managed database env degerleri Render uzerinden baglanir.

DigitalOcean / VPS:

- Ornek app spec: [deploy/digitalocean-app.yaml.example](deploy/digitalocean-app.yaml.example)
- Detayli rehber: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)
- Guvenlik checklist: [docs/SECURITY_HARDENING.md](docs/SECURITY_HARDENING.md)

Docker imaji production baslangicinda:

- storage/bootstrap cache dizinlerini hazirlar.
- `APP_KEY_BASE64` varsa `APP_KEY` uretir.
- Config cache temizler.
- `RUN_MIGRATIONS=true` ise `migrate --force` calistirir.
- `ENSURE_ADMIN=true` ise admin kullanicisini hazirlar.
- Config, route ve view cache olusturur.
- Apache baslatir.

Health endpoint:

```http
GET /up
```

## Faydalı Artisan Komutlari

```bash
php artisan otokantar:ensure-admin
php artisan vehicle-passes:latest --limit=10
php artisan vehicle-passes:verify
php artisan backup:run --keep=14
php artisan backup:list
php artisan backup:restore --backup=latest --force
php artisan backup:sync latest
```

## Proje Yapisi

```text
app/
  Http/Controllers/
  Http/Middleware/
  Models/
  Services/
config/
database/
deploy/
docs/
legacy/
legacy_python/
resources/
routes/
tests/
Dockerfile
docker-compose.yml
render.yaml
```

## Kisa Yol Haritasi

1. Production VPS/domain gecisi
2. Google Drive veya S3 uzak backup senkronizasyonu
3. Restore tatbikati ve backup alarm sistemi
4. Dashboard raporlarinin zenginlestirilmesi
5. Musteri demo akisi ve marka sunum dosyalari

## Lisans

Bu repo su an ozel urun gelistirme projesi olarak ele alinmaktadir. Lisans ve ticari kullanim kosullari urunlesme asamasinda ayrica netlestirilmelidir.
