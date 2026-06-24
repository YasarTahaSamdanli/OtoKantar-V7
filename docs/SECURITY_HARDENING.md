# OtoKantar Security Hardening

Bu liste production domain, VPS ve database ortamına çıkmadan önce uygulanacak güvenlik tabanıdır.

## Uygulama

- `APP_ENV=production`, `APP_DEBUG=false` ve gerçek `APP_URL=https://...` kullan.
- `.env` dosyasını git'e ekleme; secret değerlerini yalnızca hosting/VPS secret manager veya sunucu env içinde tut.
- `APP_KEY`, `LIVE_INGEST_API_TOKEN`, `ADMIN_PASSWORD`, database parolaları ve mail/API tokenlarını her ortam için ayrı üret.
- `ALLOW_PUBLIC_REGISTRATION=false` tut; kullanıcıları yalnızca admin panelinden oluştur.
- `SESSION_ENCRYPT=true`, `SESSION_SECURE_COOKIE=true`, `SESSION_HTTP_ONLY=true`, `SESSION_SAME_SITE=lax` kullan.
- Production deploy sonrası `php artisan config:cache`, `route:cache`, `view:cache` çalıştır.
- `composer audit` ve `npm audit --audit-level=low` sonuçları temiz olmadan deploy etme.

## Domain ve TLS

- Domain'i yalnızca HTTPS üzerinden yayınla; HTTP trafiğini 301 ile HTTPS'e yönlendir.
- TLS sertifikasını otomatik yenileyecek şekilde ayarla.
- HSTS'yi domain doğrulandıktan sonra açık tut: `SECURITY_HSTS_ENABLED=true`.
- `SECURITY_HSTS_PRELOAD=true` değerini yalnızca tüm subdomain'lerin kalıcı HTTPS kullanacağından eminsen aç.

## VPS

- SSH root login'i kapat, yalnızca anahtar tabanlı login kullan.
- Sunucu firewall'unda dışarıya sadece `80`, `443` ve gerekliyse kısıtlı IP'den `22` aç.
- Database portunu internete açma; uygulama ile aynı private network veya localhost üzerinden bağlan.
- Otomatik güvenlik güncellemelerini aç ve sunucuda gereksiz servisleri kapat.
- Web root olarak Laravel proje kökünü değil yalnızca `public/` dizinini yayınla.
- `storage/`, `.env`, `vendor/`, `database/`, `legacy_python/` ve backup dosyalarını public web erişimine kapalı tut.

## Database

- Uygulama için root database kullanıcısı kullanma; sadece gerekli database'e yetkili ayrı kullanıcı aç.
- Production database'de TLS/SSL bağlantısını zorunlu tut.
- Günlük otomatik yedek, haftalık ayrı lokasyona yedek ve en az aylık restore testi yap.
- Lokal/VPS yedek almak için `php artisan backup:run` kullan. XAMPP'te gerekirse `BACKUP_MYSQLDUMP_BINARY=C:\xampp\mysql\bin\mysqldump.exe` ayarla.
- Yedekleri görmek için `php artisan backup:list` kullan.
- Seçili yedeğe dönmek için `php artisan backup:restore --backup=20260624_225720 --force` kullan. Yalnızca database geri alınacaksa `--database-only` ekle.
- VPS'te cron ile her gece çalıştır: `cd /var/www/html && php artisan backup:run --keep=14`.
- Backup dosyalarını şifrele ve uygulama sunucusundan ayrı bir yerde sakla.
- Google Drive veya benzeri uzak depoya kopya almak için `rclone` kur, `rclone config` ile bir remote oluştur ve `.env` içinde `BACKUP_REMOTE_ENABLED=true`, `BACKUP_REMOTE_DESTINATION=gdrive:OtoKantarBackups` ayarla.
- Tek seferlik uzak kopya için `php artisan backup:sync latest`; her backup sonrası otomatik uzak kopya için `BACKUP_REMOTE_SYNC_AFTER_RUN=true`.
- Migration çalıştırmadan önce snapshot/backup al.
- Kritik tablolar için audit log ve düzenli integrity kontrolü planla.

## Operasyon

- Deploy öncesi: testler, auditler, build ve migration dry-run kontrolü.
- Deploy sonrası: `/up`, login, canlı panel, ingest, CSV export ve admin kullanıcı akışını kontrol et.
- Şüpheli durumda önce token/parola rotasyonu yap, sonra log ve audit kayıtlarını incele.
