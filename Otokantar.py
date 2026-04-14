"""
OtoKantar V11 — Üretim Kalitesi Plaka Tanıma ve Kantar Otomasyon Sistemi
=========================================================================
Mimari  : KameraUretici(thread) → KareKuyrugu → _kare_isle
          → PlakaTespitci(YOLO) → OcrWorker(thread) → DogrulamaMotoru
          → PlakaBuffer → KantarKaydedici(DbWriter thread) → FisYazdirici
Donanım : Ermet Tartı (RS232/COM) + Epson Yazıcı (win32print / escpos / file)

V11 İyileştirmeleri
-------------------
1. PlakaBuffer (TTL=120sn)
   Kantar eşiği aşıldığında okunan en yüksek güvenli plakayı buffer'da tutar.
   Ağırlık stabilize olunca plaka kameradan çıkmış olsa bile kayıt yapılır.

2. Seans Sıfırlama Guard (3sn)
   Ağırlık eşik altına düştüğünde anında sıfırlanmaz; 3sn kesintisiz eşik
   altında kalması şartı aranır. Araç kantardan inerken zıplama koruması sağlar.

3. Dinamik ROI & Gece Modu
   KANTAR_ROI_NORM normalize (0–1) oranları kullanır; kamera çözünürlüğünden
   bağımsızdır. _gamma_bgr %95 persentil parlaklık kullanır; lokal far yansıması
   tüm görüntüyü etkilemez. Yüksek parlaklıkta bilateral filter eklenir.

4. DB Retry & Yazıcı İzolasyonu
   SQLite yazma thread'i exponential backoff (0.5/1.0/2.0sn, 3 deneme) ile
   geçici "database is locked" hatalarında pes etmez. Yazıcı (win32/escpos)
   thread başlatma dahil tüm hataları ana thread'den tamamen izole eder;
   30sn timeout izleme thread'i ile uzun süreli bloklar loglanır.

Diğer Özellikler
----------------
- Debounce  : Son 3sn ±20kg'dan az değişen ağırlık "sabit" sayılır.
- Güven     : 0.6×OCR + 0.4×YOLO ağırlıklı skor.
- Snapshot  : Başarılı kayıtta captures/ klasörüne görüntü kaydedilir.
- Sesli     : winsound.Beep (Windows); kara listede farklı ton.
- SQLite    : WAL modu + wal_autocheckpoint=100 + ayrı DB-writer thread.
- FastAPI   : /, /api/canli-durum, /canli_kare.jpg, /api/durum, /api/son-kayitlar, /api/kara-liste.
- Tracker   : IoU öncelikli + centroid yedek çoklu plaka takibi.
"""

# ─────────────────────────────────────────────────────────────────────────────
# STANDART KÜTÜPHANELER
# ─────────────────────────────────────────────────────────────────────────────
import csv
import json
import logging
import os
import platform
import queue
import re
import sqlite3
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# ÜÇÜNCÜ TARAF KÜTÜPHANELER
# ─────────────────────────────────────────────────────────────────────────────
import cv2
import easyocr
import numpy as np
import serial                        # pip install pyserial
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# PLATFORM'A ÖZGÜ İMPORTLAR
# ─────────────────────────────────────────────────────────────────────────────
if platform.system() == "Windows":
    import winsound
    _WINSOUND_OK = True
    try:
        import win32api
        import win32print
        _WIN32PRINT_OK = True
    except ImportError:
        _WIN32PRINT_OK = False
else:
    _WINSOUND_OK   = False
    _WIN32PRINT_OK = False

try:
    import escpos.printer as escpos_printer
    _ESCPOS_OK = True
except ImportError:
    _ESCPOS_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# YAPILANDIRMA
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # ── Kamera ───────────────────────────────────────────────────────────────
    "KAMERA_INDEX"                  : 0,
    "KAMERA_YENIDEN_BAGLANTI_DENEMESI": 5,
    "KAMERA_BEKLEME_SURESI"         : 3.0,

    # ── Tespit ───────────────────────────────────────────────────────────────
    "YOLO_CONF"                     : 0.25,
    "MIN_OCR_CONF"                  : 0.35,
    "PLAKA_MIN_EN"                  : 10,
    "PLAKA_MIN_BOY"                 : 10,
    "ASPECT_RATIO_MIN"              : 2.0,
    "ASPECT_RATIO_MAX"              : 6.5,

    # ── OCR ──────────────────────────────────────────────────────────────────
    "OCR_GPU"                       : False,
    "OCR_DILLER"                    : ["tr", "en"],
    "OCR_IZIN_LISTESI"              : "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    "OCR_KARE_ATLAMA"               : 3,
    "MORPH_KAPAT"                   : True,
    "MORPH_KERNEL"                  : (3, 3),
    "ALT_KIRP_ORAN"                 : 0.15,
    "BOYUTLANDIRMA_KATSAYI"         : 2,

    # ── Gece Modu / Gamma (V11: %95 persentil + bilateral) ───────────────────
    "GAMMA_PARLAKLIK_ESIK"          : 190.0,   # %95 persentil üstüyse gamma uygula
    "GAMMA_US"                      : 1.65,
    "CLAHE_CLIP"                    : 3.0,     # Gündüz:2.0 | Gece:3.0–4.0
    "CLAHE_GRID"                    : (8, 8),
    "BILATERAL_ESIK"                : 210,     # %95 persentil bu değeri aşarsa bilateral filter
    "BILATERAL_D"                   : 7,
    "BILATERAL_SIGMA_COLOR"         : 50,
    "BILATERAL_SIGMA_SPACE"         : 50,

    # ── Perspektif ───────────────────────────────────────────────────────────
    "PERSPEKTIF_CIKTI_EN"           : 520,
    "PERSPEKTIF_CIKTI_BOY"          : 110,
    "CANNY_ESIK1"                   : 50,
    "CANNY_ESIK2"                   : 150,
    "PERSPEKTIF_MIN_ALAN_ORAN"      : 0.08,
    "PERSPEKTIF_EPSILON_CARPAN"     : 0.02,

    # ── Doğrulama (oylama) ───────────────────────────────────────────────────
    "ESIK_DEGERI"                   : 4,
    "OYLAMA_MIN_TOPLAM_GUVEN"       : 2.5,
    "BEKLEME_SURESI_SONRA"          : 120.0,   # Kayıt sonrası spam engeli (sn)

    # ── Çıktı ────────────────────────────────────────────────────────────────
    "CSV_DOSYA"                     : "kantar_raporu.csv",
    "LOG_DOSYA"                     : "otokantar.log",
    "JSON_CANLI"                    : "canli_durum.json",
    "DB_DOSYA"                      : "otokantar.db",
    "CANLI_KARE_DOSYA"              : "canli_kare.jpg",
    "CANLI_KARE_ARALIK"             : 5,

    # ── Producer–Consumer kuyruğu ─────────────────────────────────────────────
    "KARE_KUYRUK_BOYUTU"            : 2,
    "OCR_WORKER_KUYRUK"             : 4,

    # ── Model ────────────────────────────────────────────────────────────────
    "PLATE_WEIGHTS_URL"             : (
        "https://raw.githubusercontent.com/Muhammad-Zeerak-Khan/"
        "Automatic-License-Plate-Recognition-using-YOLOv8/main/"
        "license_plate_detector.pt"
    ),
    "MODELS_DIR"                    : "models",

    # ── Kara Liste ────────────────────────────────────────────────────────────
    "KARA_LISTE"                    : ["06ABC123", "34YASAR01"],

    # ── Ermet Tartı RS232 ─────────────────────────────────────────────────────
    "KANTAR_PORT"                   : "COM1",
    "KANTAR_BAUD"                   : 9600,
    "MIN_KILIT_AGIRLIK"             : 2000,     # kg — bu eşiği aşarsa kantar "dolu"

    # ── Kantar ROI — Normalize Oranlar (0.0 – 1.0) (V11: dinamik) ────────────
    # (sol, üst, sağ, alt) — tam ekran = (0.0, 0.0, 1.0, 1.0)
    "KANTAR_ROI_NORM"               : (0.0, 0.185, 1.0, 1.0),

    # ── V11: Plaka Buffer ─────────────────────────────────────────────────────
    "PLAKA_BUFFER_TTL"              : 120.0,   # sn — buffer geçerlilik süresi

    # ── V11: Seans Sıfırlama Guard ────────────────────────────────────────────
    "SEANS_SIFIR_BEKLEME"           : 3.0,     # sn — bu süre eşik altında kalınca sıfırla

    # ── Yazıcı ───────────────────────────────────────────────────────────────
    # "win32" | "escpos" | "file"
    "YAZICI_BACKEND"                : "win32",
    "YAZICI_ADI"                    : "",
    "ESCPOS_USB_VENDOR"             : 0x04B8,
    "ESCPOS_USB_PRODUCT"            : 0x0202,
}

# Türk plaka regex: 01-81 il kodu + 1-3 harf + 2-4 rakam
PLAKA_REGEX = re.compile(
    r"(0[1-9]|[1-7][0-9]|8[0-1])"
    r"([A-Z]{1,3})"
    r"([0-9]{2,4})"
)

_HARF_DUZELTME  = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z"}
_RAKAM_DUZELTME = {"O": "0", "I": "1", "S": "5", "B": "8", "Z": "2"}


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
def _logger_kur(log_dosya: str) -> logging.Logger:
    logger = logging.getLogger("OtoKantar")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_dosya, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


log = _logger_kur(CONFIG["LOG_DOSYA"])


# ─────────────────────────────────────────────────────────────────────────────
# VERİ YAPILARI
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PlakaKayit:
    plaka   : str
    tarih   : str
    saat    : str
    guven   : float
    agirlik : float = 0.0
    tip     : str   = "GIRIS"
    operator: str   = "AUTO"


@dataclass
class TespitSonucu:
    bbox     : tuple
    ham_metin: str
    plaka    : Optional[str]
    guven    : float
    gecerli  : bool = False


@dataclass
class DogrulamaDurumu:
    oylar       : dict  = field(default_factory=dict)
    hane        : dict  = field(default_factory=dict)
    okuma_sayisi: int   = 0
    son_gorulme : float = field(default_factory=time.time)
    son_kayit   : float = 0.0


@dataclass
class PlakaBuffer:
    """
    V11 — Plaka/Ağırlık ayrışmasını çözen buffer.
    Kantar eşiği aşıldığında okunan en yüksek güvenli plaka burada tutulur.
    Ağırlık debounce tamamlanınca kamera boş olsa bile bu buffer kullanılır.
    """
    plaka    : str
    guven    : float
    yolo_conf: float
    zaman    : float = field(default_factory=time.time)

    def suresi_doldu_mu(self, ttl: float) -> bool:
        return (time.time() - self.zaman) > ttl

    def guncelle_eger_daha_iyi(self, plaka: str, guven: float, yolo_conf: float) -> bool:
        """Gelen plaka daha yüksek güven skoruna sahipse buffer'ı güncelle."""
        yeni_skor = 0.6 * guven + 0.4 * yolo_conf
        eski_skor = 0.6 * self.guven + 0.4 * self.yolo_conf
        if yeni_skor > eski_skor:
            self.plaka     = plaka
            self.guven     = guven
            self.yolo_conf = yolo_conf
            self.zaman     = time.time()
            return True
        return False


@dataclass
class OcrGorevi:
    roi_bgr  : np.ndarray
    arac_id  : int
    yolo_conf: float
    bbox     : tuple


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="OtoKantar V11 API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
sistem_referansi = None
_PROJE_KOKU = Path(__file__).resolve().parent


def _canli_json_dosyadan_oku() -> dict:
    yol = _PROJE_KOKU / CONFIG["JSON_CANLI"]
    if not yol.is_file():
        return {"son_guncelleme": None, "son_kayit": None, "son_10": []}
    try:
        with open(yol, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"son_guncelleme": None, "son_kayit": None, "son_10": []}


@app.get("/")
def dashboard_sayfa():
    """Ofis tarayıcısı: http://127.0.0.1:8000/"""
    html_yol = _PROJE_KOKU / "dashboard.html"
    if not html_yol.is_file():
        raise HTTPException(404, "dashboard.html bulunamadı.")
    return FileResponse(html_yol, media_type="text/html; charset=utf-8")


@app.get("/api/canli-durum")
def api_canli_durum():
    """
    dashboard.html için birleşik durum: son kayıtlar + anlık kantar kg (+ dosya yedeği).
    """
    veri = _canli_json_dosyadan_oku()
    if sistem_referansi is not None:
        ag, st = sistem_referansi.kantar_okuyucu.veri
        veri["kantar_kg"] = round(ag, 1)
        veri["kantar_sabit"] = st
        veri["plaka_buffer"] = (
            sistem_referansi._plaka_buffer.plaka
            if sistem_referansi._plaka_buffer else None
        )
        veri["seans_kilitli"] = sistem_referansi.seans_kilitli_mi
        kk = sistem_referansi.kaydedici.son_kayitlar
        if kk:
            veri["son_kayit"] = asdict(kk[-1])
            veri["son_10"] = [asdict(k) for k in kk[-10:]]
        veri["son_guncelleme"] = datetime.now().isoformat()
    else:
        veri.setdefault("kantar_kg", None)
        veri.setdefault("kantar_sabit", None)
        veri.setdefault("plaka_buffer", None)
        veri.setdefault("seans_kilitli", None)
    return veri


@app.get("/canli_kare.jpg")
def api_canli_kare_dosyasi():
    """Son OCR/kamera karesi (JPEG); tarayıcıda periyodik yenileme için no-store."""
    yol = _PROJE_KOKU / CONFIG["CANLI_KARE_DOSYA"]
    if not yol.is_file():
        raise HTTPException(404, "Canlı kare henüz yok.")
    return FileResponse(
        yol,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/api/log-son")
def api_log_son():
    """dashboard.html sistem logu için son satırlar."""
    yol = _PROJE_KOKU / CONFIG["LOG_DOSYA"]
    if not yol.is_file():
        return {"satirlar": []}
    try:
        text = yol.read_text(encoding="utf-8", errors="ignore")
        tum = [s for s in text.splitlines() if s.strip()]
        return {"satirlar": tum[-8:]}
    except Exception:
        return {"satirlar": []}


@app.get("/api/durum")
def api_durum():
    if sistem_referansi is None:
        return {"hata": "Sistem henüz başlatılmadı"}
    return {
        "kantar_kg"     : sistem_referansi.kantar_okuyucu.agirlik,
        "sabit"         : sistem_referansi.kantar_okuyucu.sabit,
        "seans_kilitli" : sistem_referansi.seans_kilitli_mi,
        "plaka_buffer"  : (
            sistem_referansi._plaka_buffer.plaka
            if sistem_referansi._plaka_buffer else None
        ),
    }


@app.get("/api/son-kayitlar")
def api_son_kayitlar():
    if sistem_referansi is None:
        return []
    return [asdict(k) for k in sistem_referansi.kaydedici.son_kayitlar]


class KaraListeEkleIstek(BaseModel):
    plaka: str


@app.get("/api/kara-liste")
def api_kara_liste_listele():
    return {"kara_liste": list(CONFIG.get("KARA_LISTE", []))}


@app.post("/api/kara-liste")
def api_kara_liste_ekle(istek: KaraListeEkleIstek):
    plaka = istek.plaka.strip().upper()
    if not PLAKA_REGEX.fullmatch(plaka):
        raise HTTPException(422, f"Geçersiz Türk plaka formatı: '{plaka}'")
    kara_liste: list = CONFIG.setdefault("KARA_LISTE", [])
    if plaka in kara_liste:
        raise HTTPException(409, f"'{plaka}' zaten kara listede.")
    kara_liste.append(plaka)
    log.info("Kara listeye eklendi (API): %s", plaka)
    return {"eklendi": plaka, "kara_liste": list(kara_liste)}


@app.delete("/api/kara-liste/{plaka}")
def api_kara_liste_sil(plaka: str):
    plaka = plaka.strip().upper()
    kara_liste: list = CONFIG.get("KARA_LISTE", [])
    if plaka not in kara_liste:
        raise HTTPException(404, f"'{plaka}' kara listede bulunamadı.")
    kara_liste.remove(plaka)
    log.info("Kara listeden silindi (API): %s", plaka)
    return {"silindi": plaka, "kara_liste": list(kara_liste)}


# ─────────────────────────────────────────────────────────────────────────────
# KANTAR OKUYUCU (RS232 / Ermet)
# ─────────────────────────────────────────────────────────────────────────────
class KantarOkuyucu(threading.Thread):
    """
    Ermet tartıyı RS232 üzerinden sürekli dinleyen daemon thread.
    Debounce: son 3sn ±20kg'dan az değişen ağırlık "sabit" sayılır.
    Port açılamazsa çökmez; YENIDEN_BAGLANTI_BEKLEME sn sonra tekrar dener.
    """
    YENIDEN_BAGLANTI_BEKLEME = 5.0
    DEBOUNCE_SURE            = 3.0
    DEBOUNCE_TOLERANS        = 20.0

    def __init__(self):
        super().__init__(name="KantarOkuyucu", daemon=True)
        self.guncel_agirlik  : float = 0.0
        self.agirlik_sabit_mi: bool  = False
        self._kilit          = threading.Lock()
        self._dur            = threading.Event()
        self._debounce_ref   : float = 0.0
        self._debounce_bas   : float = 0.0

    @staticmethod
    def _satiri_cozumle(satir: str) -> Optional[float]:
        """
        Ham ASCII satırından "sayı + kg/k" desenini çıkarır.
        'ERR 1234' gibi hata kodları kilo olarak yorumlanmaz.
        Örnek: '  +0002345 kg' → 2345.0
        """
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:kg|k)", satir.lower())
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
        return None

    def _debounce_guncelle(self, yeni: float) -> None:
        simdi = time.time()
        if abs(yeni - self._debounce_ref) > self.DEBOUNCE_TOLERANS:
            self._debounce_ref = yeni
            self._debounce_bas = simdi
            sabit = False
        else:
            sabit = (simdi - self._debounce_bas) >= self.DEBOUNCE_SURE
        with self._kilit:
            self.guncel_agirlik   = yeni
            self.agirlik_sabit_mi = sabit

    def run(self):
        log.info(
            "KantarOkuyucu başlatıldı → port=%s baud=%s debounce=%.0fsn/±%.0fkg",
            CONFIG["KANTAR_PORT"], CONFIG["KANTAR_BAUD"],
            self.DEBOUNCE_SURE, self.DEBOUNCE_TOLERANS,
        )
        while not self._dur.is_set():
            port = None
            try:
                port = serial.Serial(
                    port    =CONFIG["KANTAR_PORT"],
                    baudrate=int(CONFIG["KANTAR_BAUD"]),
                    bytesize=serial.EIGHTBITS,
                    parity  =serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    timeout =1.0,
                )
                log.info("Kantar portu açıldı: %s", CONFIG["KANTAR_PORT"])
                self._debounce_ref = 0.0
                self._debounce_bas = time.time()

                while not self._dur.is_set():
                    try:
                        ham = port.readline()
                    except serial.SerialException as e:
                        log.warning("Kantar okuma hatası: %s — yeniden bağlanılıyor.", e)
                        break
                    if not ham:
                        continue
                    try:
                        satir = ham.decode("ascii", errors="ignore")
                    except Exception:
                        continue
                    deger = self._satiri_cozumle(satir)
                    if deger is not None:
                        self._debounce_guncelle(deger)

            except serial.SerialException as e:
                print(
                    f"[KANTAR UYARI] Port açılamadı ({CONFIG['KANTAR_PORT']}): {e} "
                    f"— {self.YENIDEN_BAGLANTI_BEKLEME:.0f}sn sonra tekrar."
                )
                log.warning("Kantar bağlantı hatası: %s", e)
            finally:
                if port and port.is_open:
                    try:
                        port.close()
                    except Exception:
                        pass

            if not self._dur.is_set():
                with self._kilit:
                    self.agirlik_sabit_mi = False
                time.sleep(self.YENIDEN_BAGLANTI_BEKLEME)

        log.info("KantarOkuyucu durduruldu.")

    def durdur(self):
        self._dur.set()

    @property
    def agirlik(self) -> float:
        with self._kilit:
            return self.guncel_agirlik

    @property
    def sabit(self) -> bool:
        with self._kilit:
            return self.agirlik_sabit_mi

    @property
    def veri(self) -> tuple:
        """Thread-safe atomik çift okuma: (agirlik, sabit)"""
        with self._kilit:
            return (self.guncel_agirlik, self.agirlik_sabit_mi)


# ─────────────────────────────────────────────────────────────────────────────
# FİŞ YAZICI — Çok-Backend (win32print / escpos / file)
# ─────────────────────────────────────────────────────────────────────────────
class FisYazdirici:
    """
    V11: Yazıcı izolasyonu — tüm backend'ler ayrı daemon thread'de çalışır.
    win32print için 30sn timeout izleme thread'i; ana döngü asla bloklanmaz.
    """
    FIS_DOSYA = "kantar_fisi.txt"
    GENISLIK  = 42

    _ESC_INIT     = b"\x1b\x40"
    _ESC_BOLD_ON  = b"\x1b\x45\x01"
    _ESC_BOLD_OFF = b"\x1b\x45\x00"
    _ESC_CENTER   = b"\x1b\x61\x01"
    _ESC_LEFT     = b"\x1b\x61\x00"
    _ESC_FEED     = b"\x1b\x64\x04"
    _ESC_CUT      = b"\x1d\x56\x41\x00"

    def yazdir(self, kayit: PlakaKayit) -> None:
        backend = str(CONFIG.get("YAZICI_BACKEND", "file")).lower()
        if backend == "win32" and not _WIN32PRINT_OK:
            log.warning("win32print yok — 'file' moduna düşürüldü.")
            backend = "file"
        if backend == "escpos" and not _ESCPOS_OK:
            log.warning("python-escpos yok — 'file' moduna düşürüldü.")
            backend = "file"

        try:
            icerik = self._fis_metin_olustur(kayit)
            self._dosyaya_yaz(icerik)

            if backend == "win32":
                self._win32_gonder_async(kayit)
            elif backend == "escpos":
                self._escpos_gonder_async(kayit)
            else:
                log.info("Yazıcı backend=file; fiş '%s' dosyasına kaydedildi.", self.FIS_DOSYA)

        except PermissionError as e:
            log.error("Fiş dosyası yazılamadı — izin hatası: %s", e)
        except OSError as e:
            log.error("Fiş dosyası yazılamadı — I/O hatası (errno %s): %s", e.errno, e.strerror)
        except Exception as e:
            log.error("FisYazdirici beklenmedik hata (%s): %s — devam ediyor.", type(e).__name__, e)

    def _fis_metin_olustur(self, kayit: PlakaKayit) -> str:
        sep  = "=" * self.GENISLIK
        dash = "-" * self.GENISLIK
        return "\n".join([
            sep,
            "         BRİKET FABRİKASI KANTAR FİŞİ",
            sep,
            f"  Tarih    : {kayit.tarih}",
            f"  Saat     : {kayit.saat}",
            dash,
            f"  Plaka    : {kayit.plaka}",
            f"  Ağırlık  : {kayit.agirlik:.1f} kg",
            f"  Tip      : {kayit.tip}",
            f"  Operatör : {kayit.operator}",
            dash,
            f"  Güven    : %{kayit.guven * 100:.1f}",
            sep,
            "        Teşekkür Ederiz — İyi Yolculuklar",
            sep,
            "",
        ])

    def _dosyaya_yaz(self, icerik: str) -> None:
        with open(self.FIS_DOSYA, "w", encoding="utf-8") as f:
            f.write(icerik)
        log.debug("Fiş dosyaya yazıldı: %s", self.FIS_DOSYA)

    def _escpos_ham_olustur(self, kayit: PlakaKayit) -> bytes:
        enc  = "cp857"
        sep  = ("=" * self.GENISLIK + "\n").encode(enc, errors="replace")
        dash = ("-" * self.GENISLIK + "\n").encode(enc, errors="replace")

        def satir(m: str) -> bytes:
            return (m + "\n").encode(enc, errors="replace")

        return (
            self._ESC_INIT
            + self._ESC_CENTER + self._ESC_BOLD_ON
            + satir("BRİKET FABRİKASI KANTAR FİŞİ")
            + self._ESC_BOLD_OFF + self._ESC_LEFT
            + sep
            + satir(f"  Tarih    : {kayit.tarih}")
            + satir(f"  Saat     : {kayit.saat}")
            + dash
            + self._ESC_BOLD_ON
            + satir(f"  Plaka    : {kayit.plaka}")
            + satir(f"  Ağırlık  : {kayit.agirlik:.1f} kg")
            + self._ESC_BOLD_OFF
            + satir(f"  Tip      : {kayit.tip}")
            + satir(f"  Operatör : {kayit.operator}")
            + dash
            + satir(f"  Güven    : %{kayit.guven * 100:.1f}")
            + sep
            + self._ESC_CENTER
            + satir("Teşekkür Ederiz — İyi Yolculuklar")
            + self._ESC_FEED
            + self._ESC_CUT
        )

    def _win32_gonder_async(self, kayit: PlakaKayit) -> None:
        """
        V11: Daemon thread + 30sn timeout izleme.
        Thread başlatma hatası dahil tüm istisnalar çağıran thread'e sızmaz.
        """
        ham_veri   = self._escpos_ham_olustur(kayit)
        yazici_adi = str(CONFIG.get("YAZICI_ADI", "")).strip()

        def _gonder():
            try:
                hedef  = yazici_adi if yazici_adi else win32print.GetDefaultPrinter()
                handle = win32print.OpenPrinter(hedef)
                try:
                    win32print.StartDocPrinter(handle, 1, ("KantarFisi", None, "RAW"))
                    try:
                        win32print.StartPagePrinter(handle)
                        win32print.WritePrinter(handle, ham_veri)
                        win32print.EndPagePrinter(handle)
                    finally:
                        win32print.EndDocPrinter(handle)
                finally:
                    win32print.ClosePrinter(handle)
                log.info("Fiş win32print ile gönderildi → '%s'", hedef)
            except FileNotFoundError:
                log.error("win32print: yazıcı bulunamadı → '%s'", yazici_adi or "(varsayılan)")
            except OSError as e:
                log.error("win32print OSError (errno %s): %s", e.errno, e.strerror)
            except Exception as e:
                log.error("win32print beklenmedik hata (%s): %s", type(e).__name__, e)

        try:
            t = threading.Thread(target=_gonder, name="FisGonderici-win32", daemon=True)
            t.start()
            # 30sn timeout izleyici — ana thread'i bloklamaz
            def _izle():
                t.join(30)
                if t.is_alive():
                    log.warning("FisGonderici-win32: 30sn timeout aşıldı.")
            threading.Thread(target=_izle, daemon=True).start()
        except Exception as e:
            log.error("win32 thread başlatılamadı (%s): %s", type(e).__name__, e)

    def _escpos_gonder_async(self, kayit: PlakaKayit) -> None:
        ham_veri = self._escpos_ham_olustur(kayit)
        vendor   = int(CONFIG.get("ESCPOS_USB_VENDOR",  0x04B8))
        product  = int(CONFIG.get("ESCPOS_USB_PRODUCT", 0x0202))

        def _gonder():
            try:
                p = escpos_printer.Usb(vendor, product)
                p._raw(ham_veri)
                log.info("Fiş escpos ile gönderildi → USB %04X:%04X", vendor, product)
            except Exception as e:
                log.error("escpos hata (%s): %s", type(e).__name__, e)

        try:
            threading.Thread(target=_gonder, name="FisGonderici-escpos", daemon=True).start()
        except Exception as e:
            log.error("escpos thread başlatılamadı (%s): %s", type(e).__name__, e)


# ─────────────────────────────────────────────────────────────────────────────
# MODÜL 1: PLAKA TESPİTCİ (YOLOv8)
# ─────────────────────────────────────────────────────────────────────────────
class PlakaTespitci:
    _PLAKA_ETIKETLERI = {"plate", "license_plate", "licence_plate", "plaka", "number_plate"}

    def __init__(self, weights_url: str, models_dir: str, conf: float, gpu: bool):
        self.conf = conf
        weights_path = self._model_indir(weights_url, models_dir)
        self.model   = YOLO(weights_path)
        if gpu and torch.cuda.is_available():
            self.model.to("cuda")
            log.info("YOLO → CUDA")
        else:
            log.info("YOLO → CPU")

    @staticmethod
    def _model_indir(url: str, models_dir: str) -> str:
        path = Path(models_dir) / "license_plate_detector.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            log.info("Plaka modeli indiriliyor: %s", url)
            urllib.request.urlretrieve(url, path)
            log.info("Model kaydedildi: %s", path)
        return str(path)

    def plakalari_bul(self, bgr) -> list:
        """
        Dönüş: [(x1, y1, x2, y2, conf), ...] conf azalan sırada.
        En-boy oranı filtresi uygulanır.
        """
        sonuclar = self.model(bgr, verbose=False)[0]
        if sonuclar.boxes is None or len(sonuclar.boxes) == 0:
            return []

        names  = sonuclar.names or {}
        cikti  = []

        for b in sonuclar.boxes:
            conf = float(b.conf[0])
            if conf < self.conf:
                continue
            cls_id   = int(b.cls[0])
            cls_name = names.get(cls_id, "")
            if not (cls_name.lower() in self._PLAKA_ETIKETLERI or cls_id == 0):
                continue
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            w  = max(1.0, x2 - x1)
            h  = max(1.0, y2 - y1)
            ar = w / h
            if not (CONFIG["ASPECT_RATIO_MIN"] <= ar <= CONFIG["ASPECT_RATIO_MAX"]):
                continue
            cikti.append((x1, y1, x2, y2, conf))

        cikti.sort(key=lambda t: t[4], reverse=True)
        return cikti


# ─────────────────────────────────────────────────────────────────────────────
# MODÜL 2: PLAKA ÇÖZÜCÜ (OCR + Regex)
# ─────────────────────────────────────────────────────────────────────────────
class PlakaCozucu:
    def __init__(self, diller: list, gpu: bool, min_conf: float):
        log.info("EasyOCR yükleniyor...")
        self.reader   = easyocr.Reader(diller, gpu=gpu)
        self.min_conf = min_conf
        self._clahe   = cv2.createCLAHE(
            clipLimit    =CONFIG["CLAHE_CLIP"],
            tileGridSize =CONFIG["CLAHE_GRID"],
        )
        log.info("EasyOCR hazır.")

    # ── Ön işleme ─────────────────────────────────────────────────────────────

    @staticmethod
    def _sirala_dort_kose(pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
        s   = pts.sum(axis=1)
        d   = np.diff(pts, axis=1).flatten()
        return np.array([
            pts[np.argmin(s)],
            pts[np.argmin(d)],
            pts[np.argmax(s)],
            pts[np.argmax(d)],
        ], dtype=np.float32)

    def _gamma_bgr(self, bgr: np.ndarray) -> np.ndarray:
        """
        V11: %95 persentil parlaklık kullanır.
        Lokal far yansıması (küçük parlak bölge) ortalamayı yanıltmaz;
        görüntünün %95'i belirtilen eşiğin üstündeyse gamma uygulanır.
        """
        gri = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        p95 = float(np.percentile(gri, 95))
        if p95 <= float(CONFIG["GAMMA_PARLAKLIK_ESIK"]):
            return bgr
        us    = float(CONFIG["GAMMA_US"])
        table = (
            (np.arange(256, dtype=np.float64) / 255.0) ** us * 255.0
        ).clip(0, 255).astype(np.uint8)
        return cv2.LUT(bgr, cv2.merge([table, table, table]))

    def _dortgen_kose_bul(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        h, w = bgr.shape[:2]
        if w < 8 or h < 8:
            return None
        gri   = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gri, (5, 5), 0)
        kenar = cv2.Canny(blur, int(CONFIG["CANNY_ESIK1"]), int(CONFIG["CANNY_ESIK2"]))
        k     = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kenar = cv2.morphologyEx(kenar, cv2.MORPH_CLOSE, k, iterations=1)
        konturlar, _ = cv2.findContours(kenar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not konturlar:
            return None
        min_alan = float(CONFIG["PERSPEKTIF_MIN_ALAN_ORAN"]) * float(w * h)
        eps_c    = float(CONFIG["PERSPEKTIF_EPSILON_CARPAN"])
        for cnt in sorted(konturlar, key=cv2.contourArea, reverse=True)[:20]:
            alan = cv2.contourArea(cnt)
            if alan < min_alan:
                continue
            peri = cv2.arcLength(cnt, True)
            if peri < 1e-6:
                continue
            yaklasik = cv2.approxPolyDP(cnt, eps_c * peri, True)
            if len(yaklasik) != 4 or not cv2.isContourConvex(yaklasik):
                continue
            kose = yaklasik.reshape(4, 2).astype(np.float32)
            if np.any(kose[:, 0] < -0.5) or np.any(kose[:, 0] > w - 0.5):
                continue
            if np.any(kose[:, 1] < -0.5) or np.any(kose[:, 1] > h - 0.5):
                continue
            return self._sirala_dort_kose(kose)
        return None

    def _perspektif_duzelt(self, bgr: np.ndarray) -> np.ndarray:
        h, w  = bgr.shape[:2]
        out_w = int(CONFIG["PERSPEKTIF_CIKTI_EN"])
        out_h = int(CONFIG["PERSPEKTIF_CIKTI_BOY"])
        kose  = self._dortgen_kose_bul(bgr)
        if kose is None:
            src = np.array([
                [0.0, 0.0], [float(w), 0.0],
                [float(w), float(h)], [0.0, float(h)],
            ], dtype=np.float32)
        else:
            src = kose
        dst = np.array([
            [0.0, 0.0], [float(out_w), 0.0],
            [float(out_w), float(out_h)], [0.0, float(out_h)],
        ], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(bgr, M, (out_w, out_h), flags=cv2.INTER_CUBIC)

    def _roi_hazirla(self, bgr) -> Optional[np.ndarray]:
        """
        V11: Gece far yansıması için bilateral filter eklendi.
        %95 persentil parlaklık CONFIG["BILATERAL_ESIK"]'i aşarsa uygulanır.
        """
        if bgr is None or bgr.size == 0:
            return None

        h, w = bgr.shape[:2]
        if h >= 30:
            oran     = min(CONFIG["ALT_KIRP_ORAN"], 0.49)
            koru_min = 0.80
            efektif  = min(oran, 1.0 - koru_min)
            bgr      = bgr[: max(1, int(h * (1.0 - efektif))), :w]

        bgr = self._gamma_bgr(bgr)
        bgr = cv2.resize(
            bgr, None,
            fx=CONFIG["BOYUTLANDIRMA_KATSAYI"],
            fy=CONFIG["BOYUTLANDIRMA_KATSAYI"],
            interpolation=cv2.INTER_CUBIC,
        )
        bgr = self._perspektif_duzelt(bgr)
        gri = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # V11: Yüksek parlaklıkta bilateral filter (kenarları korur, lekeleri yumuşatır)
        p95_ici = float(np.percentile(gri, 95))
        if p95_ici > int(CONFIG.get("BILATERAL_ESIK", 210)):
            gri = cv2.bilateralFilter(
                gri,
                d          =int(CONFIG.get("BILATERAL_D", 7)),
                sigmaColor =float(CONFIG.get("BILATERAL_SIGMA_COLOR", 50)),
                sigmaSpace =float(CONFIG.get("BILATERAL_SIGMA_SPACE", 50)),
            )

        gri = self._clahe.apply(gri)
        blur = cv2.GaussianBlur(gri, (3, 3), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if CONFIG["MORPH_KAPAT"]:
            k      = cv2.getStructuringElement(cv2.MORPH_RECT, CONFIG["MORPH_KERNEL"])
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=1)

        return binary

    # ── OCR ───────────────────────────────────────────────────────────────────

    def _ocr_oku(self, binary: np.ndarray) -> tuple:
        sonuclar = self.reader.readtext(
            binary,
            allowlist =CONFIG["OCR_IZIN_LISTESI"],
            paragraph =False,
        )
        if not sonuclar:
            return "", 0.0

        def sol_x(item):
            try:
                return min(pt[0] for pt in item[0])
            except Exception:
                return 0

        parcalar, confs = [], []
        for (bbox, metin, conf) in sorted(sonuclar, key=sol_x):
            if conf is None or conf < self.min_conf:
                continue
            temiz = re.sub(r"[^A-Z0-9]", "", str(metin).upper())
            if temiz:
                parcalar.append(temiz)
                confs.append(float(conf))

        if not parcalar:
            return "", 0.0
        return "".join(parcalar), sum(confs) / max(1, len(confs))

    @staticmethod
    def _plaka_duzelt(il: str, harf: str, rakam: str) -> str:
        il_d    = "".join(_RAKAM_DUZELTME.get(c, c) for c in il)
        harf_d  = "".join(_HARF_DUZELTME.get(c, c) for c in harf)
        rakam_d = "".join(_RAKAM_DUZELTME.get(c, c) for c in rakam)
        return il_d + harf_d + rakam_d

    def coz(self, bgr_roi) -> TespitSonucu:
        binary = self._roi_hazirla(bgr_roi)
        if binary is None:
            return TespitSonucu(bbox=(), ham_metin="", plaka=None, guven=0.0)

        ham, ocr_conf = self._ocr_oku(binary)
        eslesen = PLAKA_REGEX.search(ham)
        if eslesen:
            il, harf, rakam = eslesen.group(1), eslesen.group(2), eslesen.group(3)
            plaka = self._plaka_duzelt(il, harf, rakam)
            if 7 <= len(plaka) <= 9:
                return TespitSonucu(
                    bbox=(), ham_metin=ham, plaka=plaka,
                    guven=float(ocr_conf), gecerli=True,
                )
        return TespitSonucu(bbox=(), ham_metin=ham, plaka=None, guven=0.0)

    def roi_hazirla_debug(self, bgr_roi) -> Optional[np.ndarray]:
        return self._roi_hazirla(bgr_roi)


# ─────────────────────────────────────────────────────────────────────────────
# MODÜL 3: OCR WORKER THREAD
# ─────────────────────────────────────────────────────────────────────────────
class OcrWorker(threading.Thread):
    """
    EasyOCR'ı ana döngüden ayıran asenkron worker.
    Ana döngü YOLO'yu çalıştırır, ROI'yi kuyruğa bırakır, bir sonraki kareye geçer.
    OCR sonucu hazır olunca çıkış kuyruğundan okunur; hiçbir kare bloklanmaz.
    """
    def __init__(self, cozucu: PlakaCozucu, kuyruk_boyutu: int = 4):
        super().__init__(name="OcrWorker", daemon=True)
        self._cozucu        = cozucu
        self._giris_kuyrugu : queue.Queue = queue.Queue(maxsize=kuyruk_boyutu)
        self._cikis_kuyrugu : queue.Queue = queue.Queue(maxsize=kuyruk_boyutu * 2)
        self._dur           = threading.Event()

    def gorevi_gonder(self, gorev: OcrGorevi) -> bool:
        try:
            self._giris_kuyrugu.put_nowait(gorev)
            return True
        except queue.Full:
            return False

    def sonuclari_topla(self) -> list:
        sonuclar = []
        while True:
            try:
                sonuclar.append(self._cikis_kuyrugu.get_nowait())
            except queue.Empty:
                break
        return sonuclar

    def durdur(self) -> None:
        self._dur.set()
        try:
            self._giris_kuyrugu.put_nowait(None)
        except queue.Full:
            pass

    def run(self) -> None:
        log.info("OcrWorker başlatıldı.")
        while not self._dur.is_set():
            try:
                gorev = self._giris_kuyrugu.get(timeout=1.0)
            except queue.Empty:
                continue
            if gorev is None:
                break
            try:
                sonuc = self._cozucu.coz(gorev.roi_bgr)
                self._cikis_kuyrugu.put((gorev.arac_id, sonuc, gorev.yolo_conf, gorev.bbox))
            except Exception as e:
                log.error("OcrWorker işleme hatası (%s): %s", type(e).__name__, e)
        log.info("OcrWorker durduruldu.")


# ─────────────────────────────────────────────────────────────────────────────
# MODÜL 4: CENTROID TRACKER (IoU + Centroid hibrit)
# ─────────────────────────────────────────────────────────────────────────────
class CentroidTracker:
    def __init__(
        self,
        max_distance   : float = 60.0,
        max_age_s      : float = 10.0,
        iou_esik       : float = 0.2,
        gorunmezlik_max: float = 2.0,
    ):
        self.max_distance       = float(max_distance)
        self.max_age_s          = float(max_age_s)
        self.iou_esik           = float(iou_esik)
        self._purge_after_s     = min(float(max_age_s), float(gorunmezlik_max))
        self._next_id           = 1
        self._tracks            : dict = {}

    @staticmethod
    def _centroid(bbox: tuple) -> tuple:
        x1, y1, x2, y2 = bbox
        return ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)

    @staticmethod
    def _dist(a: tuple, b: tuple) -> float:
        dx, dy = a[0] - b[0], a[1] - b[1]
        return (dx * dx + dy * dy) ** 0.5

    @staticmethod
    def _iou(a: tuple, b: tuple) -> float:
        ax1, ay1, ax2, ay2 = (float(v) for v in a)
        bx1, by1, bx2, by2 = (float(v) for v in b)
        ix1   = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2   = min(ax2, bx2); iy2 = min(ay2, by2)
        iw    = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        aa    = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        ab    = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = aa + ab - inter
        return inter / union if union > 0.0 else 0.0

    def _purge(self, now: float) -> list:
        stale = [
            tid for tid, tr in self._tracks.items()
            if now - float(tr["last_matched"]) > self._purge_after_s
        ]
        for tid in stale:
            del self._tracks[tid]
        return stale

    def purge_expired(self) -> list:
        return self._purge(time.time())

    def is_empty(self) -> bool:
        return len(self._tracks) == 0

    def sifirla(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def assign_id(self, bbox: tuple) -> int:
        now    = time.time()
        self._purge(now)
        bbox_f = tuple(float(v) for v in bbox)
        c_new  = self._centroid(bbox_f)

        best_iou_id, best_iou = None, self.iou_esik
        for tid, tr in self._tracks.items():
            iou = self._iou(bbox_f, tr["bbox"])
            if iou > best_iou:
                best_iou    = iou
                best_iou_id = tid

        if best_iou_id is not None:
            chosen = best_iou_id
        else:
            best_c_id, best_d = None, None
            for tid, tr in self._tracks.items():
                d = self._dist(c_new, self._centroid(tr["bbox"]))
                if d <= self.max_distance and (best_d is None or d < best_d):
                    best_d, best_c_id = d, tid
            chosen = best_c_id if best_c_id is not None else self._next_id
            if best_c_id is None:
                self._next_id += 1

        self._tracks[chosen] = {"bbox": bbox_f, "last_matched": now}
        return chosen


# ─────────────────────────────────────────────────────────────────────────────
# MODÜL 5: DOĞRULAMA MOTORU
# ─────────────────────────────────────────────────────────────────────────────
class DogrulamaMotoru:
    def __init__(self, esik: int, min_toplam_guven: float, kayit_sonrasi_bekleme: float):
        self.esik              = int(esik)
        self.min_toplam_guven  = float(min_toplam_guven)
        self.bekleme           = float(kayit_sonrasi_bekleme)
        self._durum            : dict = {}

    def isle(self, arac_id: int, plaka: str, guven: float) -> tuple:
        su_an = time.time()
        d     = self._durum.get(arac_id)
        if d is None:
            d = DogrulamaDurumu()
            self._durum[arac_id] = d

        if su_an - d.son_kayit < self.bekleme:
            return (False, None, 0.0, 0)
        if su_an - d.son_gorulme > 5.0:
            d.oylar.clear(); d.hane.clear(); d.okuma_sayisi = 0

        d.son_gorulme  = su_an
        d.okuma_sayisi += 1
        g              = float(guven)
        d.oylar[plaka] = d.oylar.get(plaka, 0.0) + g
        d.hane[plaka]  = d.hane.get(plaka, 0) + 1

        en_iyi = max(d.oylar.values()) if d.oylar else 0.0
        tamam  = (d.okuma_sayisi >= self.esik) or (en_iyi >= self.min_toplam_guven)
        if not tamam:
            return (False, None, 0.0, 0)

        final        = max(d.oylar, key=lambda k: (d.oylar[k], d.hane.get(k, 0), k))
        kazan_toplam = float(d.oylar[final])
        kazan_n      = int(d.hane[final])
        d.son_kayit  = su_an
        d.oylar.clear(); d.hane.clear(); d.okuma_sayisi = 0
        return (True, final, kazan_toplam, kazan_n)

    def durum_ozeti(self, arac_id: int) -> tuple:
        d = self._durum.get(arac_id)
        if d is None or not d.oylar:
            return ("", 0, 0.0, self.esik, self.min_toplam_guven)
        lider = max(d.oylar, key=lambda k: (d.oylar[k], d.hane.get(k, 0), k))
        return (lider, d.okuma_sayisi, d.oylar[lider], self.esik, self.min_toplam_guven)

    def temizle_eski(self, yasam_suresi: float = 30.0):
        su_an = time.time()
        for aid in [aid for aid, d in self._durum.items() if su_an - d.son_gorulme > yasam_suresi]:
            del self._durum[aid]

    def sifirla(self) -> None:
        self._durum.clear()


# ─────────────────────────────────────────────────────────────────────────────
# MODÜL 6: KANTAR KAYDEDİCİ (CSV + SQLite WAL + JSON)
# ─────────────────────────────────────────────────────────────────────────────
class KantarKaydedici:
    """
    V11: DB Writer thread'i exponential backoff retry ile güçlendirildi.
    Geçici 'database is locked' hatalarında 0.5/1.0/2.0sn bekleme ile 3 deneme.
    Tüm denemeler başarısız olursa kayıt loglanır ve atlanır; sistem durmaz.
    """
    _RETRY_GECIKME = [0.5, 1.0, 2.0]

    def __init__(
        self,
        csv_dosya    : str,
        json_dosya   : str,
        db_dosya     : str,
        fis_yazdirici: Optional[FisYazdirici] = None,
    ):
        self.csv_dosya      = csv_dosya
        self.json_dosya     = json_dosya
        self.db_dosya       = db_dosya
        self.fis_yazdirici  = fis_yazdirici
        self.son_kayitlar   : list = []
        self._kilit         = threading.Lock()
        self._csv_aktif     = True

        self._db_kuyrugu    : queue.Queue = queue.Queue()
        self._db_dur        = threading.Event()
        self._db_thread     = threading.Thread(
            target=self._db_writer_loop,
            name  ="DbWriter",
            daemon=True,
        )
        self._db_kur_sema()
        self._csv_baslik_yaz()
        self._db_thread.start()
        log.info("DbWriter thread başlatıldı (WAL + retry aktif).")

    def _db_kur_sema(self):
        con = sqlite3.connect(self.db_dosya, timeout=10)
        try:
            con.execute("PRAGMA journal_mode=WAL;")
            con.execute("PRAGMA synchronous=NORMAL;")
            con.execute("PRAGMA wal_autocheckpoint=100;")
            con.execute("""
                CREATE TABLE IF NOT EXISTS kayitli_araclar (
                    plaka TEXT PRIMARY KEY,
                    ilk_kayit_tarihi TEXT
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS gecis_raporlari (
                    id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    plaka   TEXT,
                    tarih   TEXT,
                    saat    TEXT,
                    guven   REAL,
                    agirlik REAL DEFAULT 0.0
                )
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_plaka ON kayitli_araclar(plaka)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_gecis_plaka_tarih ON gecis_raporlari(plaka, tarih)")
            con.commit()
            log.debug("SQLite WAL şema hazır: %s", self.db_dosya)
        finally:
            con.close()

    def _db_writer_loop(self) -> None:
        """
        V11: Exponential backoff retry.
        sqlite3.OperationalError → 3 deneme × artan bekleme.
        sqlite3.IntegrityError   → kasıtlı atlama (PRIMARY KEY çakışması).
        """
        def _yaz(con: sqlite3.Connection, kayit: PlakaKayit) -> None:
            cur = con.cursor()
            cur.execute(
                "SELECT 1 FROM kayitli_araclar WHERE plaka = ? LIMIT 1",
                (kayit.plaka,),
            )
            if cur.fetchone() is None:
                cur.execute(
                    "INSERT INTO kayitli_araclar (plaka, ilk_kayit_tarihi) VALUES (?, ?)",
                    (kayit.plaka, f"{kayit.tarih} {kayit.saat}"),
                )
            cur.execute(
                "INSERT INTO gecis_raporlari (plaka, tarih, saat, guven, agirlik) "
                "VALUES (?, ?, ?, ?, ?)",
                (kayit.plaka, kayit.tarih, kayit.saat,
                 float(kayit.guven), float(kayit.agirlik)),
            )
            con.commit()

        con = sqlite3.connect(self.db_dosya, timeout=30)
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        con.execute("PRAGMA wal_autocheckpoint=100;")
        log.debug("DbWriter bağlantısı açıldı.")

        try:
            while not self._db_dur.is_set():
                try:
                    kayit = self._db_kuyrugu.get(timeout=1.0)
                except queue.Empty:
                    continue
                if kayit is None:
                    break

                basarili = False
                for deneme, bekleme in enumerate(self._RETRY_GECIKME, start=1):
                    try:
                        _yaz(con, kayit)
                        log.debug("DbWriter: kayıt yazıldı → %s (deneme %d)", kayit.plaka, deneme)
                        basarili = True
                        break
                    except sqlite3.OperationalError as e:
                        log.warning(
                            "DbWriter geçici SQLite hatası (deneme %d/%d): %s — %.1fsn bekle.",
                            deneme, len(self._RETRY_GECIKME), e, bekleme,
                        )
                        try:
                            con.rollback()
                        except Exception:
                            pass
                        time.sleep(bekleme)
                    except sqlite3.IntegrityError as e:
                        log.warning("DbWriter IntegrityError (atlandı): %s → %s", e, kayit.plaka)
                        basarili = True
                        break
                    except Exception as e:
                        log.error("DbWriter beklenmedik hata (%s): %s", type(e).__name__, e)
                        break

                if not basarili:
                    log.error(
                        "DbWriter: %d deneme sonrası BAŞARISIZ → %s %.1fkg",
                        len(self._RETRY_GECIKME), kayit.plaka, kayit.agirlik,
                    )
        finally:
            con.close()
            log.debug("DbWriter bağlantısı kapatıldı.")

    def kapat(self) -> None:
        self._db_dur.set()
        self._db_kuyrugu.put(None)
        self._db_thread.join(timeout=5.0)

    def _csv_baslik_yaz(self):
        dosya_var = Path(self.csv_dosya).exists()
        try:
            with open(self.csv_dosya, mode="a", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f, delimiter=";")
                if not dosya_var:
                    w.writerow(["Tarih", "Saat", "Plaka", "Agirlik(kg)", "Tip", "Guven", "Operator"])
        except PermissionError as e:
            self._csv_aktif = False
            log.warning("CSV erişilemedi, devre dışı: %s", e)

    def gecis_kaydet(self, kayit: PlakaKayit):
        with self._kilit:
            try:
                self._db_kuyrugu.put_nowait(kayit)
            except queue.Full:
                log.error("DbWriter kuyruğu dolu — kayıt %s geçici atlandı!", kayit.plaka)

            if self._csv_aktif:
                try:
                    with open(self.csv_dosya, mode="a", newline="", encoding="utf-8-sig") as f:
                        w = csv.writer(f, delimiter=";")
                        w.writerow([
                            kayit.tarih, kayit.saat, kayit.plaka,
                            f"{kayit.agirlik:.1f}",
                            kayit.tip, f"{kayit.guven:.2f}", kayit.operator,
                        ])
                except PermissionError as e:
                    self._csv_aktif = False
                    log.warning("CSV yazılamadı: %s", e)

            self.son_kayitlar.append(kayit)
            if len(self.son_kayitlar) > 50:
                self.son_kayitlar = self.son_kayitlar[-50:]

            self._json_guncelle(kayit)
            log.info(
                "KAYIT: %s | %s | %.1fkg | %s %s",
                kayit.plaka, kayit.tip, kayit.agirlik, kayit.tarih, kayit.saat,
            )

            if self.fis_yazdirici is not None:
                self.fis_yazdirici.yazdir(kayit)

    # Geriye uyumluluk
    def kaydet(self, kayit: PlakaKayit):
        self.gecis_kaydet(kayit)

    def _json_guncelle(self, son_kayit: PlakaKayit):
        try:
            with open(self.json_dosya, "w", encoding="utf-8") as f:
                json.dump({
                    "son_guncelleme": datetime.now().isoformat(),
                    "son_kayit"     : asdict(son_kayit),
                    "son_10"        : [asdict(k) for k in self.son_kayitlar[-10:]],
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.warning("JSON güncellenemedi: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# MODÜL 7: EKRAN ÇİZİCİ
# ─────────────────────────────────────────────────────────────────────────────
class EkranCizici:
    @staticmethod
    def plaka_kutusu(
        kare, bbox, plaka, sayac, esik,
        arac_id     : Optional[int] = None,
        kara_liste  : bool = False,
        renk        = (255, 0, 255),
        kalin       : int = 2,
        oy_lider    : str = "",
        oy_lider_toplam: float = 0.0,
        oy_min_guven: float = 0.0,
    ):
        x1, y1, x2, y2 = bbox
        if kara_liste:
            renk  = (0, 0, 255)
            kalin = max(kalin, 3)
        cv2.rectangle(kare, (x1, y1), (x2, y2), renk, kalin)
        oy_parca = ""
        if oy_lider and (oy_lider_toplam > 0 or sayac > 0):
            oy_parca = f" | oy:{oy_lider}={oy_lider_toplam:.2f}>={oy_min_guven:.1f}"
        label = (
            f"ID:{arac_id} - {plaka}  ({sayac}/{esik}){oy_parca}"
            if arac_id is not None
            else f"{plaka}  ({sayac}/{esik}){oy_parca}"
        )
        cv2.putText(kare, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, renk, 2)

    @staticmethod
    def basarili_kayit(kare, bbox, plaka):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(kare, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.putText(
            kare, f"KAYDEDILDI: {plaka}",
            (x1, y1 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )

    @staticmethod
    def fps_ve_bilgi(kare, fps: float, toplam_kayit: int, yakalama_fps: Optional[float] = None):
        metin = (
            f"Kamera FPS: {yakalama_fps:.1f}  |  İşlem FPS: {fps:.1f}  |  Kayıt: {toplam_kayit}"
            if yakalama_fps is not None
            else f"FPS: {fps:.1f}  |  Toplam Kayıt: {toplam_kayit}"
        )
        cv2.putText(kare, metin, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)


# ─────────────────────────────────────────────────────────────────────────────
# ANA SİSTEM: OtoKantar V11
# ─────────────────────────────────────────────────────────────────────────────
class OtoKantar:
    def __init__(self):
        log.info("=" * 60)
        log.info("OtoKantar V11 başlatılıyor...")

        self.tespitci = PlakaTespitci(
            weights_url=CONFIG["PLATE_WEIGHTS_URL"],
            models_dir =CONFIG["MODELS_DIR"],
            conf       =CONFIG["YOLO_CONF"],
            gpu        =CONFIG["OCR_GPU"],
        )
        self.cozucu = PlakaCozucu(
            diller  =CONFIG["OCR_DILLER"],
            gpu     =CONFIG["OCR_GPU"],
            min_conf=CONFIG["MIN_OCR_CONF"],
        )
        self.dogrulama = DogrulamaMotoru(
            esik               =CONFIG["ESIK_DEGERI"],
            min_toplam_guven   =CONFIG["OYLAMA_MIN_TOPLAM_GUVEN"],
            kayit_sonrasi_bekleme=CONFIG["BEKLEME_SURESI_SONRA"],
        )
        self.kantar_okuyucu = KantarOkuyucu()
        self.fis_yazdirici  = FisYazdirici()
        self.kaydedici = KantarKaydedici(
            csv_dosya    =CONFIG["CSV_DOSYA"],
            json_dosya   =CONFIG["JSON_CANLI"],
            db_dosya     =CONFIG["DB_DOSYA"],
            fis_yazdirici=self.fis_yazdirici,
        )
        self.cizici  = EkranCizici()
        self.tracker = CentroidTracker()
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=False
        )
        self.plaka_hafizasi: dict = {}

        # ── V11: PlakaBuffer ──────────────────────────────────────────────────
        # Kantar eşiği aşıldığında okunan en yüksek güvenli plaka burada tutulur.
        # Ağırlık sabitlenince kamera boş olsa bile bu buffer kullanılarak kayıt yapılır.
        self._plaka_buffer: Optional[PlakaBuffer] = None

        # ── Seans Kilidi ──────────────────────────────────────────────────────
        self._kantar_seans_kilitli: bool = False

        # ── V11: Seans Sıfırlama Guard ────────────────────────────────────────
        # Ağırlık eşik altına düştüğünde bu zaman damgası set edilir.
        # CONFIG["SEANS_SIFIR_BEKLEME"] sn kesintisiz altında kalmazsa sıfırlanmaz.
        self._seans_sifir_baslangic: Optional[float] = None

        # ── V11: Dinamik ROI ─────────────────────────────────────────────────
        # KANTAR_ROI_NORM normalize → piksel koordinatı (ilk karede hesaplanır, önbelleklenir)
        self._kantar_roi_piksel: Optional[tuple] = None

        self._kare_sayaci         = 0
        self._fps_onceki          = time.time()
        self._fps                 = 0.0
        self._yakalama_fps_onceki = time.time()
        self._yakalama_fps        = 0.0
        self._aktif_bbox          = None
        self._canli_kare_sayac    = 0
        self._son_plaka_listesi   : list = []

    @property
    def seans_kilitli_mi(self) -> bool:
        return self._kantar_seans_kilitli

    # ── V11: Dinamik ROI hesaplama ────────────────────────────────────────────

    def _roi_piksel_guncelle(self, kare_h: int, kare_w: int) -> tuple:
        """
        KANTAR_ROI_NORM (0.0–1.0) → mevcut çözünürlükte piksel koordinatı.
        Sonuç önbelleklenir; kamera çözünürlüğü değişirse seans sıfırında temizlenir.
        """
        if self._kantar_roi_piksel is not None:
            return self._kantar_roi_piksel
        l, t, r, b = CONFIG["KANTAR_ROI_NORM"]
        roi = (
            int(l * kare_w),
            int(t * kare_h),
            int(r * kare_w),
            int(b * kare_h),
        )
        self._kantar_roi_piksel = roi
        log.info(
            "Dinamik ROI hesaplandı: norm=%s → piksel=%s (%dx%d)",
            CONFIG["KANTAR_ROI_NORM"], roi, kare_w, kare_h,
        )
        return roi

    # ── V11: Seans sıfırlama (Guard ile) ─────────────────────────────────────

    def _seans_temizle(self) -> None:
        """Guard süresi tamamlandığında çağrılır. Tüm hafızaları sıfırlar."""
        self._kantar_seans_kilitli  = False
        self._seans_sifir_baslangic = None
        self._plaka_buffer          = None    # Buffer sıfırla
        self._kantar_roi_piksel     = None    # ROI önbelleği temizle (çözünürlük değişebilir)
        self.plaka_hafizasi.clear()
        self.dogrulama.sifirla()
        self.tracker.sifirla()
        log.info("Seans sıfırlandı — guard tamamlandı, tüm hafızalar temizlendi.")

    # ── Kamera ───────────────────────────────────────────────────────────────

    def _kamera_ac(self) -> cv2.VideoCapture:
        for deneme in range(CONFIG["KAMERA_YENIDEN_BAGLANTI_DENEMESI"]):
            kamera = cv2.VideoCapture(CONFIG["KAMERA_INDEX"])
            if kamera.isOpened():
                log.info("Kamera bağlantısı kuruldu.")
                return kamera
            log.warning("Kamera açılamadı. Deneme %d/%d",
                        deneme + 1, CONFIG["KAMERA_YENIDEN_BAGLANTI_DENEMESI"])
            time.sleep(CONFIG["KAMERA_BEKLEME_SURESI"])
        raise RuntimeError("Kamera açılamadı. Cihazı kontrol edin.")

    def _fps_guncelle(self):
        su_an = time.time()
        gecen = su_an - self._fps_onceki
        if gecen > 0:
            self._fps = 1.0 / gecen
        self._fps_onceki = su_an

    def _yakalama_fps_guncelle(self):
        su_an = time.time()
        gecen = su_an - self._yakalama_fps_onceki
        if gecen > 0:
            self._yakalama_fps = 1.0 / gecen
        self._yakalama_fps_onceki = su_an

    @staticmethod
    def _kuyruga_kare_koy(q: queue.Queue, kare: np.ndarray) -> None:
        yuk = kare.copy()
        try:
            q.put_nowait(yuk)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(yuk)
            except queue.Full:
                pass

    def _kamera_uretici(self, kare_kuyrugu: queue.Queue, dur: threading.Event) -> None:
        kamera = self._kamera_ac()
        try:
            while not dur.is_set():
                ret, kare = kamera.read()
                if not ret:
                    log.warning("Kare okunamadı — yeniden bağlanılıyor...")
                    kamera.release()
                    try:
                        kamera = self._kamera_ac()
                    except RuntimeError as e:
                        log.error(str(e))
                        break
                    continue
                self._yakalama_fps_guncelle()
                self._kuyruga_kare_koy(kare_kuyrugu, kare)
        finally:
            kamera.release()
            log.debug("Kamera (üretici) serbest bırakıldı.")

    def _canli_kare_yaz(self, kare: np.ndarray):
        self._canli_kare_sayac += 1
        if self._canli_kare_sayac % max(1, int(CONFIG["CANLI_KARE_ARALIK"])) != 0:
            return
        try:
            cv2.imwrite(CONFIG["CANLI_KARE_DOSYA"], kare)
        except Exception as e:
            log.warning("Canlı kare yazılamadı: %s", e)

    # ── ANA KARE İŞLEYİCİ ─────────────────────────────────────────────────────

    def _kare_isle(self, kare: np.ndarray):
        self._kare_sayaci += 1
        self._fps_guncelle()

        kare_h, kare_w = kare.shape[:2]
        simdi          = time.time()

        # ── V11 Değişiklik 3: Dinamik ROI ─────────────────────────────────────
        roi_x1, roi_y1, roi_x2, roi_y2 = self._roi_piksel_guncelle(kare_h, kare_w)

        # ── Kantar atomik okuma ────────────────────────────────────────────────
        guncel_kg, agirlik_sabit = self.kantar_okuyucu.veri
        esik_kg     = float(CONFIG["MIN_KILIT_AGIRLIK"])
        kantar_dolu = guncel_kg >= esik_kg

        # ── V11 Değişiklik 2: Seans Sıfırlama Guard ───────────────────────────
        # Ağırlık eşik altına düştüğünde sayacı başlat;
        # CONFIG["SEANS_SIFIR_BEKLEME"] sn kesintisiz altında kalmazsa sıfırlanmaz.
        if not kantar_dolu and self._kantar_seans_kilitli:
            if self._seans_sifir_baslangic is None:
                # İlk kez eşik altına düştü — guard başlat
                self._seans_sifir_baslangic = simdi
                log.debug(
                    "Seans sıfır guard başladı — %.1fsn beklenecek (%.0fkg < %.0fkg)",
                    CONFIG["SEANS_SIFIR_BEKLEME"], guncel_kg, esik_kg,
                )
            else:
                # Guard devam ediyor — süre doldu mu?
                gecen = simdi - self._seans_sifir_baslangic
                if gecen >= float(CONFIG["SEANS_SIFIR_BEKLEME"]):
                    self._seans_temizle()
                else:
                    # Henüz bekleme bitmedi — ekranda göster
                    kalan = float(CONFIG["SEANS_SIFIR_BEKLEME"]) - gecen
                    cv2.putText(
                        kare,
                        f"ARAÇ İNİYOR — Guard: {kalan:.1f}sn kaldı",
                        (10, 106),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1,
                    )
        elif kantar_dolu and self._seans_sifir_baslangic is not None:
            # Ağırlık tekrar yükseldi — guard'ı iptal et (zıplama koruması devreye girdi)
            log.debug(
                "Seans sıfır guard iptal — ağırlık tekrar eşiğin üstüne çıktı (%.0fkg)",
                guncel_kg,
            )
            self._seans_sifir_baslangic = None

        # ── V11 Değişiklik 1: PlakaBuffer güncelleme ──────────────────────────
        # Kantar eşiği altındaysa buffer'ı sıfırla veya TTL kontrolü yap
        if not kantar_dolu:
            if self._plaka_buffer is not None:
                if self._plaka_buffer.suresi_doldu_mu(float(CONFIG["PLAKA_BUFFER_TTL"])):
                    log.debug("PlakaBuffer TTL doldu, temizlendi.")
                    self._plaka_buffer = None

        # ── Ekran: kantar durumu ───────────────────────────────────────────────
        if not kantar_dolu and self.tracker.is_empty():
            self._aktif_bbox = None
            cv2.putText(
                kare,
                f"DURUM: KANTAR BOŞ ({guncel_kg:.0f} kg)",
                (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
            )
            return

        if kantar_dolu:
            if agirlik_sabit:
                cv2.putText(
                    kare,
                    f"KANTARDA ARAÇ VAR: {guncel_kg:.0f} kg  [SABİT]",
                    (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2,
                )
            else:
                cv2.putText(
                    kare,
                    f"AĞIRLIK BEKLENİYOR... {guncel_kg:.0f} kg",
                    (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2,
                )

            # V11: PlakaBuffer doluysa ve ağırlık sabitlenmişse — kayıt zamanı
            if (
                agirlik_sabit
                and self._plaka_buffer is not None
                and not self._kantar_seans_kilitli
                and not self._plaka_buffer.suresi_doldu_mu(float(CONFIG["PLAKA_BUFFER_TTL"]))
            ):
                buf          = self._plaka_buffer
                bugun_tarih  = datetime.now().strftime("%Y-%m-%d")
                bugun_saat   = datetime.now().strftime("%H:%M:%S")
                final_conf   = 0.6 * buf.guven + 0.4 * buf.yolo_conf
                kara_listede = buf.plaka in CONFIG.get("KARA_LISTE", [])

                kayit = PlakaKayit(
                    plaka   =buf.plaka,
                    tarih   =bugun_tarih,
                    saat    =bugun_saat,
                    guven   =final_conf,
                    agirlik =guncel_kg,
                    tip     =("KARA LİSTE" if kara_listede else "GIRIS"),
                )
                self.kaydedici.gecis_kaydet(kayit)
                self._kantar_seans_kilitli = True
                self._plaka_buffer         = None   # Kullanıldı, temizle

                log.info(
                    "BUFFER'DAN KAYIT: plaka=%s ağırlık=%.1fkg güven=%.2f",
                    buf.plaka, guncel_kg, final_conf,
                )

                # Snapshot
                try:
                    Path("captures").mkdir(exist_ok=True)
                    snap = f"captures/{buf.plaka}_{bugun_tarih}_{bugun_saat.replace(':', '-')}.jpg"
                    cv2.imwrite(snap, kare)
                    log.info("Snapshot (buffer): %s", snap)
                except Exception as e:
                    log.warning("Snapshot yazılamadı: %s", e)

                # Sesli uyarı
                if _WINSOUND_OK:
                    try:
                        winsound.Beep(500 if kara_listede else 1000, 1000 if kara_listede else 300)
                    except Exception:
                        pass

        # Seans kilitliyse yalnızca ekran güncelle
        if self._kantar_seans_kilitli:
            cv2.putText(
                kare,
                "SEANS KİLİTLİ — ARAÇ ÇIKIŞI BEKLENİYOR",
                (10, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 80, 255), 2,
            )
            return

        # ── YOLO (çift karelerde çalışır) ─────────────────────────────────────
        if self._kare_sayaci % 2 == 0:
            plaka_listesi          = self.tespitci.plakalari_bul(kare)
            self._son_plaka_listesi = plaka_listesi
        else:
            plaka_listesi = self._son_plaka_listesi

        if not plaka_listesi:
            self._aktif_bbox = None
            return

        for silinen_id in self.tracker.purge_expired():
            self.plaka_hafizasi.pop(silinen_id, None)

        self._aktif_bbox         = None
        ocr_calis                = self._kare_sayaci % CONFIG["OCR_KARE_ATLAMA"] == 0
        binary_debug_son         = None

        for x1, y1, x2, y2, yolo_conf in plaka_listesi:
            ix1 = max(0, int(x1)); iy1 = max(0, int(y1))
            ix2 = min(kare_w, int(x2)); iy2 = min(kare_h, int(y2))
            if ix2 <= ix1 or iy2 <= iy1:
                continue

            # ── V11 Değişiklik 3: Dinamik ROI filtresi ────────────────────────
            cx = (ix1 + ix2) / 2.0
            cy = (iy1 + iy2) / 2.0
            if not (roi_x1 <= cx <= roi_x2 and roi_y1 <= cy <= roi_y2):
                continue

            bbox    = (ix1, iy1, ix2, iy2)
            arac_id = self.tracker.assign_id(bbox)
            if self._aktif_bbox is None:
                self._aktif_bbox = bbox

            cv2.rectangle(kare, (ix1, iy1), (ix2, iy2), (128, 0, 128), 1)

            # Önbellekten oku
            if arac_id in self.plaka_hafizasi:
                plaka_cached = self.plaka_hafizasi[arac_id]
                kara_listede = plaka_cached in CONFIG.get("KARA_LISTE", [])
                if kara_listede:
                    log.warning("ALARM: KARA LİSTEDEKİ ARAÇ TESPİT EDİLDİ!")
                self.cizici.basarili_kayit(kare, bbox, plaka_cached)
                cv2.putText(
                    kare,
                    f"ID:{arac_id} - {plaka_cached}",
                    (ix1, max(18, iy1 - 38)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255) if kara_listede else (0, 255, 0), 2,
                )
                continue

            if not ocr_calis:
                continue

            w_roi = ix2 - ix1
            pad   = int(w_roi * 0.10)
            roi   = kare[iy1:iy2, max(0, ix1 - pad):min(kare_w, ix2 + pad)]

            bd = self.cozucu.roi_hazirla_debug(roi)
            if bd is not None:
                binary_debug_son = bd

            sonuc = self.cozucu.coz(roi)

            if not sonuc.gecerli or sonuc.plaka is None:
                continue

            plaka               = sonuc.plaka
            kara_listede_okunan = plaka in CONFIG.get("KARA_LISTE", [])
            if kara_listede_okunan:
                log.warning("ALARM: KARA LİSTEDEKİ ARAÇ TESPİT EDİLDİ! (%s)", plaka)

            kaydet, final_plaka, kazan_toplam, kazan_n = self.dogrulama.isle(
                arac_id, plaka, float(sonuc.guven or 0.0)
            )

            if not kaydet:
                lider, oku_n, lider_oy, esik_c, min_g = self.dogrulama.durum_ozeti(arac_id)
                self.cizici.plaka_kutusu(
                    kare, bbox, plaka, oku_n, esik_c,
                    arac_id=arac_id, kara_liste=kara_listede_okunan,
                    oy_lider=lider, oy_lider_toplam=lider_oy, oy_min_guven=min_g,
                )

            if kaydet and final_plaka is not None:
                kara_listede_final = final_plaka in CONFIG.get("KARA_LISTE", [])
                ort_ocr            = float(kazan_toplam) / max(1, int(kazan_n))
                final_conf         = 0.6 * ort_ocr + 0.4 * float(yolo_conf)
                simdi_dt           = datetime.now()
                bugun_tarih        = simdi_dt.strftime("%Y-%m-%d")

                # ── V11 Değişiklik 1: PlakaBuffer güncelle ────────────────────
                # Kantar dolu ve ağırlık henüz sabit değilse buffer'a yaz
                if kantar_dolu:
                    if self._plaka_buffer is None:
                        self._plaka_buffer = PlakaBuffer(
                            plaka    =final_plaka,
                            guven    =ort_ocr,
                            yolo_conf=float(yolo_conf),
                        )
                        log.debug(
                            "PlakaBuffer oluşturuldu: %s (güven=%.2f)",
                            final_plaka, final_conf,
                        )
                    else:
                        guncellendi = self._plaka_buffer.guncelle_eger_daha_iyi(
                            final_plaka, ort_ocr, float(yolo_conf)
                        )
                        if guncellendi:
                            log.debug(
                                "PlakaBuffer güncellendi: %s (yeni güven=%.2f)",
                                final_plaka, final_conf,
                            )

                # Ağırlık sabit ve seans açıksa — doğrudan kayıt (buffer bypass)
                if agirlik_sabit and not self._kantar_seans_kilitli:
                    kayit = PlakaKayit(
                        plaka   =final_plaka,
                        tarih   =bugun_tarih,
                        saat    =simdi_dt.strftime("%H:%M:%S"),
                        guven   =final_conf,
                        agirlik =guncel_kg,
                        tip     =("KARA LİSTE" if kara_listede_final else "GIRIS"),
                    )
                    self.kaydedici.gecis_kaydet(kayit)
                    self.plaka_hafizasi[arac_id] = final_plaka
                    self._kantar_seans_kilitli   = True
                    self._plaka_buffer           = None   # Kayıt yapıldı, buffer'a gerek yok
                    self.cizici.basarili_kayit(kare, bbox, final_plaka)
                    log.info(
                        "DOĞRUDAN KAYIT: plaka=%s ağırlık=%.1fkg güven=%.2f",
                        final_plaka, guncel_kg, final_conf,
                    )

                    # Snapshot
                    try:
                        Path("captures").mkdir(exist_ok=True)
                        saat_dosya = simdi_dt.strftime("%H-%M-%S")
                        snap = f"captures/{final_plaka}_{bugun_tarih}_{saat_dosya}.jpg"
                        cv2.imwrite(snap, kare)
                        log.info("Snapshot: %s", snap)
                    except Exception as e:
                        log.warning("Snapshot yazılamadı: %s", e)

                    # Sesli uyarı
                    if _WINSOUND_OK:
                        try:
                            winsound.Beep(
                                500 if kara_listede_final else 1000,
                                1000 if kara_listede_final else 300,
                            )
                        except Exception:
                            pass

                elif not agirlik_sabit and kantar_dolu:
                    # Plaka okundu ama ağırlık henüz sabit değil — ekranda göster
                    cv2.putText(
                        kare,
                        f"PLAKA HAZIR ({final_plaka}) — AĞIRLIK SABİTLENİYOR...",
                        (ix1, max(18, iy1 - 38)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 200, 255), 2,
                    )

        if binary_debug_son is not None:
            cv2.imshow("OCR Debug", binary_debug_son)

        # Bellek temizliği (her 300 karede bir)
        if self._kare_sayaci % 300 == 0:
            self.dogrulama.temizle_eski()

    # ── Ana çalıştırıcı ───────────────────────────────────────────────────────

    def calistir(self):
        global sistem_referansi
        sistem_referansi = self

        # FastAPI — daemon thread
        threading.Thread(
            target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000),
            name  ="FastAPIServer",
            daemon=True,
        ).start()
        log.info("FastAPI sunucusu başlatıldı → http://0.0.0.0:8000")

        # Kantar okuyucu — daemon thread
        self.kantar_okuyucu.start()
        log.info("KantarOkuyucu başlatıldı.")

        kare_kuyrugu : queue.Queue = queue.Queue(
            maxsize=max(1, int(CONFIG.get("KARE_KUYRUK_BOYUTU", 2)))
        )
        dur     = threading.Event()
        uretici = threading.Thread(
            target=self._kamera_uretici,
            args  =(kare_kuyrugu, dur),
            name  ="KameraUretici",
            daemon=True,
        )
        uretici.start()
        log.info(
            "Producer–Consumer aktif (kuyruk boyutu=%d). Çıkmak için 'q'.",
            kare_kuyrugu.maxsize,
        )

        try:
            while True:
                try:
                    kare = kare_kuyrugu.get(timeout=0.25)
                except queue.Empty:
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        log.info("Kullanıcı çıkış yaptı.")
                        break
                    continue

                self._kare_isle(kare)
                self.cizici.fps_ve_bilgi(
                    kare,
                    self._fps,
                    len(self.kaydedici.son_kayitlar),
                    yakalama_fps=self._yakalama_fps,
                )
                self._canli_kare_yaz(kare)
                cv2.imshow("OtoKantar V11", kare)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    log.info("Kullanıcı çıkış yaptı.")
                    break

        except KeyboardInterrupt:
            log.info("Ctrl+C ile durduruldu.")
        finally:
            dur.set()
            self.kantar_okuyucu.durdur()
            self.kaydedici.kapat()
            uretici.join(timeout=5.0)
            if uretici.is_alive():
                log.warning("Üretici thread 5sn içinde sonlanmadı.")
            # Kuyruktaki kareleri temizle
            while True:
                try:
                    kare_kuyrugu.get_nowait()
                except queue.Empty:
                    break
            cv2.destroyAllWindows()
            log.info("OtoKantar V11 kapatıldı.")


# ─────────────────────────────────────────────────────────────────────────────
# BAŞLAT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sistem = OtoKantar()
    sistem.calistir()