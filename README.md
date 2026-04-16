# 🚛 OtoKantar V11 - Akıllı Plaka Tanıma ve Kantar Otomasyonu

###  Görseller
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

OtoKantar V11, araç kantarları için geliştirilmiş, yüksek performanslı ve asenkron çalışan bir plaka tanıma ve otomasyon sistemidir.  
Eski usul manuel veri girişi süreçlerini tamamen ortadan kaldırarak; kantar ağırlığını anlık okur, YOLOv8 ve EasyOCR ile plakayı tespit eder ve verileri (fotoğraflarıyla birlikte) SQLite, CSV ve canlı bir web paneline (Dashboard) aktarır.

---

## ✨ Öne Çıkan Özellikler

- **Asenkron OCR Motoru:** Ana kamera akışını (YOLO tespitlerini) dondurmadan arka planda çalışan ayrılmış OCR thread'i.
- **Canlı Web Dashboard:** `dashboard.html` üzerinden çalışan FastAPI destekli canlı izleme ekranı.  
  Kantar ağırlığı, son geçen araçlar ve canlı kamera karesi anlık olarak ofis tarayıcısından izlenebilir.
- **Dinamik Buffer ve Debounce:** Ağırlık tam sabitlenmeden veya araç kantardan inmeden (zıplama koruması) eksik/hatalı kayıt yapılmasını engelleyen seans guard sistemi.
- **Donanım Soyutlaması:** Ermet ve Tolpa protokolleri hazır. Yeni nesil RS232 kantarlar için kolayca genişletilebilir mimari.
- **Gece Modu & Dinamik ROI:** Farların kamerayı kör etmesini engelleyen `%95 persentil` parlaklık kontrolü ve Bilateral filtreleme.
- **Kendi Kendini Temizleme:** Ayarlanan günden eski logları ve plaka fotoğraflarını (`captures/`) otomatik silen bakım modülü.
- **Fiş Yazdırma Desteği:** `win32print` veya `escpos` üzerinden tartım fişi yazdırma özelliği.

---

## 📂 Proje Yapısı

Projeyi kurduğunuz dizin yapısı şu şekilde görünmelidir:

```bash
OtoKantar_V11/
├── otokantar.py # Ana Python uygulaması (Sistemin kalbi)
├── config.json # Dışarıdan yönetilebilir ayar dosyası
├── dashboard.html # FastAPI tarafından sunulan ofis izleme arayüzü
├── kantar_raporu.csv # (Otomatik oluşur) Excel için yedek geçiş raporu
├── otokantar.log # (Otomatik oluşur) Sistem logları
├── otokantar.db # (Otomatik oluşur) SQLite veritabanı
├── captures/ # (Otomatik oluşur) Başarılı geçişlerin fotoğrafları
└── models/ # (Otomatik oluşur) YOLOv8 plaka modeli (.pt)
```

---

## ⚙️ Kurulum ve Gereksinimler

Sistem Python 3.9 veya daha üzeri bir sürüm gerektirir.

### 1. Gerekli kütüphaneleri yükleyin:

```bash
pip install opencv-python ultralytics easyocr pyserial fastapi uvicorn pydantic numpy torch
```

Sadece Windows kullanıcıları için (fiş yazdırma özelliği kullanılacaksa):

```bash
pip install pywin32
```

2. Modelin İndirilmesi

YOLOv8 plaka modeli (license_plate_detector.pt) ilk çalıştırmada otomatik olarak internetten indirilip models/ klasörüne kaydedilecektir.

🚀 Kullanım ve Başlatma
Adım 1: Donanım Bağlantıları
Kantar indikatöründen gelen RS232 kablosunun bilgisayara (COM portu) bağlı olduğundan emin olun.
IP Kamera RTSP linki veya USB kameranın bağlı olduğundan emin olun.
Adım 2: Konfigürasyon (config.json)

Sistemi çalıştırmadan önce config.json dosyasını düzenleyin.

Önemli ayarlar:
```bash
"KANTAR_PORT": "COM3",
"KAMERA_INDEX": 0,
"KARA_LISTE": ["34ABC123", "06TEST99"]
```
Adım 3: Sistemi Başlatma

```bash
python otokantar.py
```

🖥️ Canlı Takip Paneli (Dashboard)

Sistem çalıştığı anda arkada FastAPI sunucusu başlatılır.

Erişim:
```bash
http://localhost:8000
```

Ağ Üzerinden Erişim:

Eğer bilgisayar IP’si:

```bash
192.168.1.50
```

ise aynı ağdaki cihazlardan:

```bash
http://192.168.1.50:8000
```

adresine girerek sistemi izleyebilirsiniz.

🛠️ Sorun Giderme (Troubleshooting)
❌ [KANTAR UYARI] Port açılamadı (COMX)


❌FileNotFoundError

Çözüm:

```bash
Kantar kablosunu kontrol edin
Aygıt Yöneticisi → COM portunu kontrol edin
config.json içinden portu güncelleyin
```

❌ Kamera açılamadı (Deneme 5/5)

Çözüm:

```bash
Kameranın bağlı olduğundan emin olun
KAMERA_INDEX değerini değiştirin (0, 1, 2...)
```

❌ EasyOCR CUDA Hatası / Yavaş Çalışma

Çözüm:
```bash
NVIDIA GPU yoksa:
"OCR_GPU": false
```
