# OtoKantar V7

OtoKantar V7; plaka tanıma, kantar otomasyonu ve canlı dashboard içeren bir araç tartım sistemidir.
Python servisi plaka/kantar akışını yönetir, verileri MySQL'e yazar. Web arayüzü (`index.php`) canlı durumu `api_canli.php` üzerinden izler.

## Ozellikler

- YOLOv8 + OCR tabanlı plaka tespiti
- RS232 kantar okuma (Ermet / Tolpa protokol desteği)
- MySQL tabanlı geçiş kayıtları (`araclar`, `gecisler`)
- PHP dashboard (`index.php`) ve canlı API (`api_canli.php`)
- CSV ve JSON yedek/uyumluluk akışı (`kantar_raporu.csv`, `canli_durum.json`)

## Proje Yapisi

```text
OtoKantar_V7/
├── otokantar_app/          # Python uygulama kodu
├── index.php               # Dashboard UI
├── api_canli.php           # Dashboard veri API'si (MySQL + fallback)
├── config.json             # Runtime konfigürasyon
├── requirements.txt        # Python bağımlılıkları
├── captures/               # Araç görüntüleri (runtime output)
└── canli_durum.json        # Canlı durum (runtime output)
```

## Gereksinimler

- Python 3.10+
- XAMPP (Apache + MySQL)
- MySQL veritabanı: `otokantar`

## Kurulum

1) Bağımlılıkları yükleyin:

```bash
pip install -r requirements.txt
```

2) MySQL'i hazırlayın (XAMPP):

- Host: `localhost`
- User: `root`
- Password: ``
- Database: `otokantar`

3) Uygulamayı çalıştırın:

```bash
python -m otokantar_app.main
```

4) Dashboard:

```text
http://localhost/OtoKantar_V7/index.php
```

## Notlar

- `api_canli.php` önce `canli_durum.json` okur, yoksa DB'den fallback durum üretir.
- Kayıtlardaki ağırlık/net değerleri MySQL'de yoksa JSON/CSV uyumluluk katmanından eşleştirilir.
- Runtime çıktıları (`captures`, `canli_durum.json`, `kantar_raporu.csv`, loglar) `.gitignore` ile dışarıda tutulur.

## Opsiyonel

Windows'ta yazıcı desteği kullanılacaksa:

```bash
pip install pywin32
```
