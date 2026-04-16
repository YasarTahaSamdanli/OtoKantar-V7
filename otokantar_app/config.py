import json
import re
from pathlib import Path


_CONFIG_VARSAYILAN = {
    "KAMERA_INDEX": 0,
    "KAMERA_YENIDEN_BAGLANTI_DENEMESI": 5,
    "KAMERA_BEKLEME_SURESI": 3.0,
    "YOLO_CONF": 0.25,
    "MIN_OCR_CONF": 0.35,
    "PLAKA_MIN_EN": 10,
    "PLAKA_MIN_BOY": 10,
    "ASPECT_RATIO_MIN": 2.0,
    "ASPECT_RATIO_MAX": 6.5,
    "OCR_GPU": False,
    "OCR_DILLER": ["tr", "en"],
    "OCR_IZIN_LISTESI": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    "OCR_KARE_ATLAMA": 3,
    "MORPH_KAPAT": True,
    "MORPH_KERNEL": (3, 3),
    "ALT_KIRP_ORAN": 0.15,
    "BOYUTLANDIRMA_KATSAYI": 2,
    "GAMMA_PARLAKLIK_ESIK": 190.0,
    "GAMMA_US": 1.65,
    "CLAHE_CLIP": 3.0,
    "CLAHE_GRID": (8, 8),
    "BILATERAL_ESIK": 210,
    "BILATERAL_D": 7,
    "BILATERAL_SIGMA_COLOR": 50,
    "BILATERAL_SIGMA_SPACE": 50,
    "PERSPEKTIF_CIKTI_EN": 520,
    "PERSPEKTIF_CIKTI_BOY": 110,
    "CANNY_ESIK1": 50,
    "CANNY_ESIK2": 150,
    "PERSPEKTIF_MIN_ALAN_ORAN": 0.08,
    "PERSPEKTIF_EPSILON_CARPAN": 0.02,
    "ESIK_DEGERI": 4,
    "OYLAMA_MIN_TOPLAM_GUVEN": 2.5,
    "BEKLEME_SURESI_SONRA": 120.0,
    "CSV_DOSYA": "kantar_raporu.csv",
    "LOG_DOSYA": "otokantar.log",
    "JSON_CANLI": "canli_durum.json",
    "DB_DOSYA": "otokantar.db",
    "CANLI_KARE_DOSYA": "canli_kare.jpg",
    "CANLI_KARE_ARALIK": 5,
    "KARE_KUYRUK_BOYUTU": 2,
    "OCR_WORKER_KUYRUK": 4,
    "PLATE_WEIGHTS_URL": (
        "https://raw.githubusercontent.com/Muhammad-Zeerak-Khan/"
        "Automatic-License-Plate-Recognition-using-YOLOv8/main/"
        "license_plate_detector.pt"
    ),
    "MODELS_DIR": "models",
    "KARA_LISTE": ["06ABC123", "34YASAR01"],
    "KANTAR_PORT": "COM1",
    "KANTAR_BAUD": 9600,
    "KANTAR_PROTOKOL": "ermet",
    "SIMULASYON_MODU": True,
    "MIN_KILIT_AGIRLIK": 2000,
    "KANTAR_ROI_NORM": (0.0, 0.185, 1.0, 1.0),
    "PLAKA_BUFFER_TTL": 120.0,
    "SEANS_SIFIR_BEKLEME": 3.0,
    "YAZICI_BACKEND": "win32",
    "YAZICI_ADI": "",
    "ESCPOS_USB_VENDOR": 0x04B8,
    "ESCPOS_USB_PRODUCT": 0x0202,
    "LOG_MAX_BYTES": 5 * 1024 * 1024,
    "LOG_BACKUP_COUNT": 7,
    "CAPTURES_RETENTION_DAYS": 30,
    "FASTAPI_HOST": "0.0.0.0",
    "FASTAPI_PORT": 8000,
}

_TUPLE_ANAHTARLAR = {"MORPH_KERNEL", "CLAHE_GRID", "KANTAR_ROI_NORM"}


def _config_yukle(dosya: str = "config.json") -> dict:
    cfg = dict(_CONFIG_VARSAYILAN)
    yol = Path(dosya)
    if yol.is_file():
        try:
            with open(yol, encoding="utf-8") as f:
                dis = json.load(f)
            dis = {k: v for k, v in dis.items() if not k.startswith("_")}
            cfg.update(dis)
            print(f"[CONFIG] '{dosya}' dosyasından yüklendi.")
        except Exception as e:
            print(f"[CONFIG UYARI] '{dosya}' okunamadı ({e}), varsayılanlar kullanılıyor.")
    else:
        print(f"[CONFIG] '{dosya}' bulunamadı, varsayılanlar kullanılıyor.")

    for anahtar in _TUPLE_ANAHTARLAR:
        if anahtar in cfg and isinstance(cfg[anahtar], list):
            cfg[anahtar] = tuple(cfg[anahtar])
    return cfg


CONFIG = _config_yukle()

PLAKA_REGEX = re.compile(
    r"(0[1-9]|[1-7][0-9]|8[0-1])"
    r"([A-Z]{1,3})"
    r"([0-9]{2,4})"
)

_HARF_DUZELTME = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z"}
_RAKAM_DUZELTME = {"O": "0", "I": "1", "S": "5", "B": "8", "Z": "2"}
