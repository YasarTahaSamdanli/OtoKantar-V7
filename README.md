## OtoKantar V7

OtoKantar V7, **canlı kantar paneli** ve **admin kullanıcı yönetimi** içeren Laravel 12 tabanlı uygulamadır. Proje, eski (legacy) sistemde kullanılan bazı URL'leri yeni Laravel endpoint'lerine yönlendirerek kademeli geçişi destekler.

### Gereksinimler

- PHP **8.2+**
- Composer
- Node.js + npm
- MySQL/MariaDB

### Kurulum (tek komut)

Bu repo `composer.json` içinde hazır script'lerle gelir:

```bash
composer setup
```

Bu script sırasıyla: `composer install`, `.env` oluşturma, `key:generate`, migrate, `npm install` ve `npm run build` çalıştırır.

### Geliştirme modunda çalıştırma

```bash
composer dev
```

Bu komut aynı anda Laravel server, queue listener, log izleme ve Vite dev server'ı başlatır.

### Legacy veri kaynağı (canlı panel)

Canlı panel endpoint'leri `DB::connection('legacy')` ile legacy MySQL bağlantısını kullanır. `.env` içinde şu değişkenleri ayarlayın:

- `LEGACY_DB_HOST`
- `LEGACY_DB_PORT`
- `LEGACY_DB_DATABASE`
- `LEGACY_DB_USERNAME`
- `LEGACY_DB_PASSWORD`

### Legacy kaynak kod / yedekler

- `legacy_python/`: Eski Python servis kodları (web tarafıyla karışmasın diye ayrı tutulur).
- `legacy_backup/`: Eski PHP entrypoint'leri ve çeşitli legacy çıktılar (referans/yedek amaçlı).

### Legacy Python canlı sync

Render canlı panelinin anlık kare/kilo verisi alması için Python uygulamasında remote sync açık olmalıdır.

1. Render servisindeki `LIVE_INGEST_API_TOKEN` değerini ayarlayın.
2. `legacy_python/config.example.json` dosyasını `legacy_python/config.json` olarak kopyalayın.
3. `REMOTE_SYNC_TOKEN` değerini Render'daki `LIVE_INGEST_API_TOKEN` ile aynı yapın.
4. Python uygulamasını şu komutla başlatın:

```bash
python legacy_python/Otokantar.py
```

Başlangıç logunda `Remote sync aktif: https://otokantar-v7.onrender.com/api/live-ingest` görünmelidir.

### URL'ler

- **Canlı panel (UI)**: `/canli` (login gerekir)
- **Canlı API**: `/canli/api?action=panel|durum&limit=40`
- **CSV indir**: `/canli/csv`
- **Anlık kare**: `/canli/kare`

Legacy yönlendirmeleri `routes/web.php` içinde tanımlıdır:

- `/api_canli.php` → `/canli/api`
- `/canli_kare.jpg` → `/canli/kare`
- `/index.php` → `/canli`

### Admin kullanıcı

Admin middleware `role:admin` ile korunur (örn. `/admin/users`).

Varsayılan admin seed'i:

```bash
php artisan db:seed --class=Database\\Seeders\\AdminUserSeeder
```

Seed varsayılan olarak şu kullanıcıyı oluşturur/günceller:

- Email: `admin@example.com`
- Şifre: `ChangeMe123!`

> Üretimde mutlaka değiştirin.
