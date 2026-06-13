# 🚛 OtoKantar V7

## Yapay Zeka Destekli Araç Kantar Otomasyon Sistemi

OtoKantar V7, araç kantar operasyonlarını tamamen otomatik hale getirmek amacıyla geliştirilmiş, yapay zeka destekli plaka tanıma ve tartım yönetim sistemidir.

Sistem; kameradan gelen görüntüler üzerinden araçları algılar, plakaları otomatik olarak okur, kantar verilerini toplar, giriş-çıkış kayıtlarını oluşturur, fiş basar ve tüm süreci merkezi bir yönetim paneli üzerinden takip etmenizi sağlar.

---

# 🎯 Amaç

Manuel veri girişini ortadan kaldırarak:

- İşlem sürelerini azaltmak
- Operatör hatalarını önlemek
- Tartım süreçlerini otomatikleştirmek
- Araç hareketlerini kayıt altına almak
- Güvenilir raporlama sağlamak

---

# ✨ Özellikler

## 🚘 Otomatik Plaka Tanıma

- Yapay zeka destekli araç tespiti
- Gerçek zamanlı plaka algılama
- OCR ile plaka okuma
- Akıllı doğrulama sistemi
- Gürültülü görüntülerde hata toleransı
- Bilinen araç veritabanı desteği

---

## ⚖️ Otomatik Tartım Sistemi

- Giriş tartımı
- Çıkış tartımı
- Net ağırlık hesaplama
- Açık seans yönetimi
- Tekrar kayıt koruması
- Otomatik işlem kilitleme

---

## 📷 Kamera Entegrasyonu

- USB Kamera
- IP Kamera
- RTSP Akış Desteği
- Canlı görüntü işleme
- Snapshot kaydetme

---

## 🧠 Yapay Zeka Motoru

Sistem içerisinde:

- Araç tespiti
- Plaka tespiti
- OCR işlemleri
- Takip (Tracking)
- Doğrulama algoritmaları

birlikte çalışmaktadır.

---

## 🖨️ Fiş Yazdırma

- Otomatik fiş oluşturma
- Giriş fişi
- Çıkış fişi
- Net ağırlık bilgileri
- Yazıcı entegrasyonu

---

## 📊 Yönetim Paneli

Dashboard üzerinden:

- Canlı sistem takibi
- Güncel araçlar
- Tartım geçmişi
- Kara liste yönetimi
- Sistem durumu
- İstatistikler

izlenebilir.

---

## 🔒 Güvenlik ve Kararlılık

- Çoklu iş parçacığı (Multi Thread)
- Thread-safe mimari
- Otomatik hata yakalama
- Güvenli dosya işlemleri
- Graceful Shutdown
- Log kayıt sistemi

---

# 🏗️ Sistem Mimarisi

```text
Kamera
   │
   ▼
Araç Tespiti
   │
   ▼
Plaka Tespiti
   │
   ▼
OCR Motoru
   │
   ▼
Doğrulama Sistemi
   │
   ▼
Kantar Verisi
   │
   ▼
Kayıt Motoru
   │
   ▼
Fiş Yazdırma
   │
   ▼
Dashboard + API
```

---

# 📂 Proje Yapısı

```text
OtoKantar-V7
│
├── otokantar_app
│   ├── api
│   ├── core
│   ├── db
│   ├── donanim
│   ├── utils
│   ├── models.py
│   ├── logger.py
│   ├── config.py
│   └── main.py
│
├── dashboard.html
├── dashboard_server.py
└── Otokantar.py
```

---

# 🧩 Ana Bileşenler

## Core

Sistemin yapay zeka ve karar verme katmanı.

- AI Motoru
- OCR Worker
- Takip Sistemi
- Doğrulama Motoru

---

## Donanım Katmanı

Fiziksel cihazlarla haberleşir.

- Kantar Okuyucu
- Yazıcı Kontrolü

---

## Veritabanı Katmanı

- Tartım kayıtları
- Araç bilgileri
- Kara liste kayıtları
- Sistem geçmişi

---

## API Katmanı

FastAPI tabanlı servisler.

- Dashboard veri servisi
- Sistem durumu
- Araç sorguları
- Yönetim işlemleri

---

# 📈 Kullanım Alanları

- Hafriyat Sahaları
- Maden İşletmeleri
- Geri Dönüşüm Tesisleri
- Lojistik Merkezleri
- Fabrikalar
- Depolar
- Tarım Ürün Alım Noktaları
- Organize Sanayi Bölgeleri

---

# 🚀 Avantajlar

### Daha Hızlı İşlem

Araç geçişleri manuel girişe ihtiyaç duymadan tamamlanır.

### Daha Az Hata

Operatör kaynaklı veri giriş hataları minimize edilir.

### Tam Kayıt Takibi

Tüm giriş ve çıkış işlemleri kayıt altına alınır.

### Düşük İş Gücü Maliyeti

Tekrarlayan işlemler otomatikleştirilir.

### Kolay Raporlama

Geçmiş işlemler ve araç hareketleri kolayca incelenebilir.

---

# 🛠️ Kullanılan Teknolojiler

- Python
- FastAPI
- OpenCV
- SQLite
- OCR Teknolojileri
- REST API
- Multi-threading
- Yapay Zeka Destekli Görüntü İşleme

---

# 📌 Proje Durumu

OtoKantar V7 aktif olarak geliştirilen, gerçek saha kullanımı için tasarlanmış profesyonel bir araç kantar otomasyon çözümüdür.

---

© OtoKantar V7
Akıllı Kantar Otomasyon Sistemi
