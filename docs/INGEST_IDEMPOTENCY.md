# Ingest Idempotency Standard

OtoKantar gecis olaylari idempotent islenir. Ayni olay HTTP retry, queue retry veya duplicate POST ile tekrar geldiyse yeni gecis kaydi uretilmemelidir.

## Event ID Formati

Her GIRIS/CIKIS olayi icin onerilen kalici kimlik:

```text
otokantar:v1:{PLAKA}:{YON}:{YYYYMMDDHHMMSS}
```

Ornek:

```text
otokantar:v1:34ABC123:GIRIS:20260608120000
```

- `PLAKA`: normalize edilmis, buyuk harfli plaka.
- `YON`: `GIRIS` veya `CIKIS`.
- Zaman: gecis olayinin karar verilen tarihi ve saati.

## Laravel Davranisi

- `event_id` varsa ana idempotency anahtari olarak kullanilir.
- Eski istemciler `event_id` gondermezse Laravel ayni standartta canonical `event_id` uretir.
- `legacy_pass_key` eski payload'lar icin ikincil idempotency anahtari olarak korunur.
- `event_id` ve `legacy_pass_key` database seviyesinde unique korunur.
- Duplicate olaylarda `vehicle_passes` ve `gecis_gecmisi.jsonl` cogalmaz.

## Python Davranisi

Python saha uygulamasi gecis karesi gonderirken ayni formatta `event_id` uretir. Retry veya network kopmasi ayni gecis icin yeni kimlik uretmemelidir.

## Temp Image Temizligi

HTTP katmani gorseli queue payload'ina gommez; gecici dosya olarak yazar. Job basarili, duplicate veya exception ile bitse de gecici dosyayi siler.
