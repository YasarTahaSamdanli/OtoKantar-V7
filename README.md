# OtoKantar V7 — Kullanım Kılavuzu

## Kurulum

```bash
pip install -r requirements.txt
python otokantar.py
```

## Dosyalar

| Dosya | Açıklama |
|---|---|
| `otokantar.py` | Ana Python sistemi |
| `dashboard.html` | Canlı izleme arayüzü (tarayıcıda aç) |
| `kantar_raporu.csv` | Otomatik oluşan kayıt dosyası |
| `canli_durum.json` | GUI için anlık durum verisi |
| `otokantar.log` | Sistem log dosyası |
| `models/` | YOLO model ağırlıkları (otomatik indirilir) |

## Yapılandırma

`otokantar.py` içindeki `CONFIG` sözlüğünü düzenleyin:

```python
CONFIG = {
    "KAMERA_INDEX": 0,       # Farklı kamera için 1, 2...
    "ESIK_DEGERI": 4,        # Kaç kez görülünce kaydet
    "BEKLEME_SURESI_SONRA": 12.0,  # Kayıt sonrası bekleme (sn)
    "OCR_GPU": False,        # GPU varsa True
    "OCR_KARE_ATLAMA": 3,    # Her N karede bir OCR
    ...
}
```

## V6'dan Farklılıklar

- **Mimari**: Tek döngü → 5 bağımsız sınıf (test edilebilir)
- **Hata düzeltmesi**: `O↔0`, `I↔1`, `S↔5`, `B↔8` bağlama göre
- **Bekleme hatası**: V6'da bekleme doğrulama sırasında uygulanıyordu; V7'de sadece kayıt sonrasında
- **CLAHE**: Gece/düşük ışık desteği eklendi
- **Kamera**: Bağlantı kopunca otomatik yeniden bağlanma
- **Loglama**: `print()` yerine `logging` modülü (dosya + ekran)
- **Bellek**: Eski doğrulama durumları periyodik temizleniyor
- **CSV**: Başlık satırı + tip/güven/operatör alanları eklendi
