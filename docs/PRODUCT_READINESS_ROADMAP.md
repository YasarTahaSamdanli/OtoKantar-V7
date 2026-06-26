# OtoKantar Product Readiness Roadmap

Bu dosya OtoKantar'in coklu musteriye kurulacak ticari urun olarak sertlestirilmesi icin uygulanabilir oncelik listesidir.

## P0 - Saha Dayanikliligi

- Ingest islerini queue job haline getir.
- Her olay icin kalici `event_id` ve database unique garanti kullan.
- Python saha uygulamasinda offline kuyruk temizleme, maksimum kuyruk boyutu ve support export ekle.
- `otokantar:health-check` ciktisini n8n veya merkezi monitoringe bagla.

## P1 - Ticari Guvenlik

- Imzali lisans dosyasi ve aktivasyon kaydi ekle.
- Musteri, cihaz, lisans bitis tarihi ve modullerini lisans payload'inda dogrula.
- Her musteri icin ayri ingest token ve token rotasyon komutu kullan.
- Backup paketlerini sifrele ve restore tatbikatini standart kurulum prosedurune ekle.

## P2 - Operasyon ve Bakim

- Tek komut kurulum/upgrade script'i hazirla.
- Support bundle komutu ekle: health raporu, son loglar, env anahtar isimleri, queue durumu, backup manifestleri.
- Docker servislerine production kaynak limitleri, restart policy ve log rotation ekle.
- Versiyonlama, migration dry-run ve rollback prosedurunu dokumante et.

## P3 - Kod Kalitesi

- `LiveIngestController` icindeki is kurallarini servis/job katmanina bol.
- `CanliDataService` icindeki fallback, CSV, JSONL ve VehiclePass okuma sorumluluklarini ayir.
- Legacy Python klasorunu net paket/servis sinirina tasiyip aktif olmayan dosyalari kurulum paketinden ayir.
- Kritik akislara concurrency ve yuk testi ekle.
