"""
OtoKantar V7 — Üretim Kalitesi Plaka Tanıma Sistemi
====================================================
Mimari: PlakaTespitci → PlakaCozucu → DogrulamaMotoru → KantarKaydedici → OtoKantar
"""

import cv2
import easyocr
import csv
import os
import re
import time
import logging
import urllib.request
import threading
import json
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

import torch
from ultralytics import YOLO

# ─────────────────────────────────────────────
# YAPILANDIRMA
# ─────────────────────────────────────────────

CONFIG = {
    # Kamera
    "KAMERA_INDEX": 0,
    "KAMERA_YENIDEN_BAGLANTI_DENEMESI": 5,
    "KAMERA_BEKLEME_SURESI": 3.0,

    # Tespit
    "YOLO_CONF": 0.25,
    "MIN_OCR_CONF": 0.35,
    "PLAKA_MIN_EN": 10,
    "PLAKA_MIN_BOY": 10,
    "ASPECT_RATIO_MIN": 2.0,
    "ASPECT_RATIO_MAX": 6.5,

    # OCR
    "OCR_GPU": False,               # GPU varsa True yap
    "OCR_DILLER": ["tr", "en"],
    "OCR_IZIN_LISTESI": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    "OCR_KARE_ATLAMA": 3,           # Her N karede bir OCR çalıştır
    "MORPH_KAPAT": True,
    "MORPH_KERNEL": (3, 3),
    "ALT_KIRP_ORAN": 0.15,
    "BOYUTLANDIRMA_KATSAYI": 2,
    "CLAHE_CLIP": 2.0,
    "CLAHE_GRID": (8, 8),

    # Doğrulama
    "ESIK_DEGERI": 4,
    "BEKLEME_SURESI_SONRA": 12.0,   # KAYIT sonrası bekleme (önceki v6 hatası düzeltildi)

    # Çıktı
    "CSV_DOSYA": "kantar_raporu.csv",
    "LOG_DOSYA": "otokantar.log",
    "JSON_CANLI": "canli_durum.json",   # GUI için canlı durum dosyası

    # Model
    "PLATE_WEIGHTS_URL": (
        "https://raw.githubusercontent.com/Muhammad-Zeerak-Khan/"
        "Automatic-License-Plate-Recognition-using-YOLOv8/main/"
        "license_plate_detector.pt"
    ),
    "MODELS_DIR": "models",
}

# Türk plaka regex: 01-81 il kodu + 1-3 harf + 2-4 rakam
PLAKA_REGEX = re.compile(
    r"(0[1-9]|[1-7][0-9]|8[0-1])"   # il kodu
    r"([A-Z]{1,3})"                    # harf grubu
    r"([0-9]{2,4})"                    # rakam grubu
)

# OCR karakter düzeltme tablosu (harf bölgesi ve rakam bölgesi ayrı)
_HARF_DUZELTME = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z"}
_RAKAM_DUZELTME = {"O": "0", "I": "1", "S": "5", "B": "8", "Z": "2"}


# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

def _logger_kur(log_dosya: str) -> logging.Logger:
    logger = logging.getLogger("OtoKantar")
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


# ─────────────────────────────────────────────
# VERİ YAPILARI
# ─────────────────────────────────────────────

@dataclass
class PlakaKayit:
    plaka: str
    tarih: str
    saat: str
    guven: float
    tip: str = "GIRIS"      # GIRIS veya CIKIS (ileride genişletilebilir)
    operator: str = "AUTO"


@dataclass
class TespitSonucu:
    """Bir karedeki YOLO + OCR sonucu."""
    bbox: tuple                     # (x1, y1, x2, y2)
    ham_metin: str
    plaka: Optional[str]
    guven: float
    gecerli: bool = False


@dataclass
class DogrulamaDurumu:
    """Belirli bir plaka için doğrulama sayacı ve zaman bilgisi."""
    plaka: str
    sayac: int = 0
    son_gorulme: float = field(default_factory=time.time)
    son_kayit: float = 0.0


# ─────────────────────────────────────────────
# MODÜL 1: PLAKA TESPİTCİ (YOLO)
# ─────────────────────────────────────────────

class PlakaTespitci:
    """YOLOv8 ile görüntüden plaka bounding box'ı bulur."""

    _PLAKA_ETIKETLERI = {
        "plate", "license_plate", "licence_plate", "plaka", "number_plate"
    }

    def __init__(self, weights_url: str, models_dir: str, conf: float, gpu: bool):
        self.conf = conf
        weights_path = self._model_indir(weights_url, models_dir)
        self.model = YOLO(weights_path)
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
            log.info("Plaka modeli indiriliyor...")
            urllib.request.urlretrieve(url, path)
            log.info(f"Model kaydedildi: {path}")
        return str(path)

    def en_iyi_plakayi_bul(self, bgr: "cv2.Mat") -> Optional[tuple]:
        """
        En yüksek güven skorlu, geçerli aspect ratio'lu plakayı döndürür.
        Dönüş: (x1, y1, x2, y2, conf) veya None
        """
        sonuclar = self.model(bgr, verbose=False)[0]
        if sonuclar.boxes is None or len(sonuclar.boxes) == 0:
            return None

        names = sonuclar.names or {}
        en_iyi = None

        for b in sonuclar.boxes:
            conf = float(b.conf[0])
            if conf < self.conf:
                continue

            cls_id = int(b.cls[0])
            cls_name = names.get(cls_id, "")

            # Repo modeli genelde tek sınıflı (cls_id==0); etikete göre esneklik
            if not (cls_name.lower() in self._PLAKA_ETIKETLERI or cls_id == 0):
                continue

            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            ar = w / h

            if not (CONFIG["ASPECT_RATIO_MIN"] <= ar <= CONFIG["ASPECT_RATIO_MAX"]):
                continue

            if en_iyi is None or conf > en_iyi[4]:
                en_iyi = (x1, y1, x2, y2, conf)

        return en_iyi


# ─────────────────────────────────────────────
# MODÜL 2: PLAKA ÇÖZÜCÜ (OCR + Regex)
# ─────────────────────────────────────────────

class PlakaCozucu:
    """Plaka ROI görüntüsünden metin okur, düzeltir ve plaka formatını çıkarır."""

    def __init__(self, diller: list, gpu: bool, min_conf: float):
        log.info("EasyOCR yükleniyor...")
        self.reader = easyocr.Reader(diller, gpu=gpu)
        self.min_conf = min_conf
        self._clahe = cv2.createCLAHE(
            clipLimit=CONFIG["CLAHE_CLIP"],
            tileGridSize=CONFIG["CLAHE_GRID"],
        )
        log.info("EasyOCR hazır.")

    # ── Ön işleme ──────────────────────────────

    def _roi_hazirla(self, bgr: "cv2.Mat") -> Optional["cv2.Mat"]:
        """Ham plaka bölgesini OCR için optimize eder."""
        if bgr is None or bgr.size == 0:
            return None

        # Alt reklam alanını kırp
        h, w = bgr.shape[:2]
        if h >= 30:
            oran = min(CONFIG["ALT_KIRP_ORAN"], 0.49)
            koru_min = 0.80
            efektif = min(oran, 1.0 - koru_min)
            bgr = bgr[:max(1, int(h * (1.0 - efektif))), :w]

        # Büyüt
        bgr = cv2.resize(
            bgr,
            None,
            fx=CONFIG["BOYUTLANDIRMA_KATSAYI"],
            fy=CONFIG["BOYUTLANDIRMA_KATSAYI"],
            interpolation=cv2.INTER_CUBIC,
        )

        # Gri + CLAHE + Blur + Otsu
        gri = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gri = self._clahe.apply(gri)
        blur = cv2.GaussianBlur(gri, (3, 3), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morfolojik kapama (harf birleştirme)
        if CONFIG["MORPH_KAPAT"]:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, CONFIG["MORPH_KERNEL"])
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=1)

        return binary

    # ── OCR ────────────────────────────────────

    def _ocr_oku(self, binary: "cv2.Mat") -> str:
        """EasyOCR çalıştırır, parçaları soldan sağa birleştirir."""
        sonuclar = self.reader.readtext(
            binary,
            allowlist=CONFIG["OCR_IZIN_LISTESI"],
            paragraph=False,
        )
        if not sonuclar:
            return ""

        def sol_x(item):
            try:
                return min(pt[0] for pt in item[0])
            except Exception:
                return 0

        parcalar = []
        for (bbox, metin, conf) in sorted(sonuclar, key=sol_x):
            if conf is None or conf < self.min_conf:
                continue
            temiz = re.sub(r"[^A-Z0-9]", "", str(metin).upper())
            if temiz:
                parcalar.append(temiz)

        return "".join(parcalar)

    # ── Karakter Düzeltme ───────────────────────

    @staticmethod
    def _plaka_duzelt(il: str, harf: str, rakam: str) -> str:
        """
        Plaka bölümlerine göre bağlama duyarlı karakter düzeltmesi uygular.
        - il kodu: sadece rakam olmalı
        - harf grubu: sadece harf olmalı
        - rakam grubu: sadece rakam olmalı
        """
        il_d = "".join(_RAKAM_DUZELTME.get(c, c) for c in il)
        harf_d = "".join(_HARF_DUZELTME.get(c, c) for c in harf)
        rakam_d = "".join(_RAKAM_DUZELTME.get(c, c) for c in rakam)
        return il_d + harf_d + rakam_d

    # ── Ana Metot ───────────────────────────────

    def coz(self, bgr_roi: "cv2.Mat") -> TespitSonucu:
        """
        ROI'den plaka metnini çıkarır.
        Döndürür: TespitSonucu (geçerli=True ise plaka alanı dolu)
        """
        binary = self._roi_hazirla(bgr_roi)
        if binary is None:
            return TespitSonucu(bbox=(), ham_metin="", plaka=None, guven=0.0)

        ham = self._ocr_oku(binary)
        eslesen = PLAKA_REGEX.search(ham)

        if eslesen:
            il, harf, rakam = eslesen.group(1), eslesen.group(2), eslesen.group(3)
            plaka = self._plaka_duzelt(il, harf, rakam)
            if 7 <= len(plaka) <= 9:
                return TespitSonucu(
                    bbox=(),
                    ham_metin=ham,
                    plaka=plaka,
                    guven=1.0,
                    gecerli=True,
                )

        return TespitSonucu(bbox=(), ham_metin=ham, plaka=None, guven=0.0)

    def debug_goster(self, binary: Optional["cv2.Mat"]):
        if binary is not None:
            cv2.imshow("OCR Girdi (Debug)", binary)

    def roi_hazirla_debug(self, bgr_roi: "cv2.Mat") -> Optional["cv2.Mat"]:
        return self._roi_hazirla(bgr_roi)


# ─────────────────────────────────────────────
# MODÜL 3: DOĞRULAMA MOTORU
# ─────────────────────────────────────────────

class DogrulamaMotoru:
    """
    Aynı plakayı N kez art arda görmeden kayıt açmaz.
    KAYIT SONRASI bekleme süresi uygular (v6 hatası düzeltildi).
    """

    def __init__(self, esik: int, kayit_sonrasi_bekleme: float):
        self.esik = esik
        self.bekleme = kayit_sonrasi_bekleme
        self._durum: dict[str, DogrulamaDurumu] = {}

    def isle(self, plaka: str) -> bool:
        """
        Plakayı doğrulama kuyruğuna ekler.
        Eşik aşıldıysa True döner (kayıt yapılabilir).
        """
        su_an = time.time()
        d = self._durum.get(plaka)

        if d is None:
            self._durum[plaka] = DogrulamaDurumu(plaka=plaka, sayac=1)
            return False

        # Başka bir plaka baskın geldiyse sıfırla
        if su_an - d.son_gorulme > 5.0:
            d.sayac = 1
            d.son_gorulme = su_an
            return False

        # Kayıt sonrası bekleme süresi dolmadıysa sayma
        if su_an - d.son_kayit < self.bekleme:
            return False

        d.sayac += 1
        d.son_gorulme = su_an

        if d.sayac >= self.esik:
            d.son_kayit = su_an
            d.sayac = 0
            return True

        return False

    def sayac_al(self, plaka: str) -> int:
        d = self._durum.get(plaka)
        return d.sayac if d else 0

    def temizle_eski(self, yasam_suresi: float = 30.0):
        """Uzun süredir görülmeyen plaka durumlarını temizler (bellek yönetimi)."""
        su_an = time.time()
        silincekler = [
            p for p, d in self._durum.items()
            if su_an - d.son_gorulme > yasam_suresi
        ]
        for p in silincekler:
            del self._durum[p]


# ─────────────────────────────────────────────
# MODÜL 4: KANTAR KAYDEDİCİ
# ─────────────────────────────────────────────

class KantarKaydedici:
    """CSV'ye kayıt yazar, JSON durum dosyasını günceller."""

    def __init__(self, csv_dosya: str, json_dosya: str):
        self.csv_dosya = csv_dosya
        self.json_dosya = json_dosya
        self.son_kayitlar: list[PlakaKayit] = []
        self._kilit = threading.Lock()
        self._csv_baslik_yaz()

    def _csv_baslik_yaz(self):
        dosya_var = Path(self.csv_dosya).exists()
        with open(self.csv_dosya, mode="a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=";")
            if not dosya_var:
                w.writerow(["Tarih", "Saat", "Plaka", "Tip", "Guven", "Operator"])

    def kaydet(self, kayit: PlakaKayit):
        with self._kilit:
            with open(self.csv_dosya, mode="a", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f, delimiter=";")
                w.writerow([
                    kayit.tarih, kayit.saat, kayit.plaka,
                    kayit.tip, f"{kayit.guven:.2f}", kayit.operator,
                ])

            self.son_kayitlar.append(kayit)
            if len(self.son_kayitlar) > 50:
                self.son_kayitlar = self.son_kayitlar[-50:]

            self._json_guncelle(kayit)
            log.info(f"KAYIT: {kayit.plaka} | {kayit.tip} | {kayit.tarih} {kayit.saat}")

    def _json_guncelle(self, son_kayit: PlakaKayit):
        """Canlı GUI için JSON dosyasını arka planda günceller."""
        try:
            with open(self.json_dosya, "w", encoding="utf-8") as f:
                json.dump({
                    "son_guncelleme": datetime.now().isoformat(),
                    "son_kayit": asdict(son_kayit),
                    "son_10": [asdict(k) for k in self.son_kayitlar[-10:]],
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.warning(f"JSON güncellenemedi: {e}")


# ─────────────────────────────────────────────
# MODÜL 5: EKRAN ÇIZICI
# ─────────────────────────────────────────────

class EkranCizici:
    """OpenCV üzerine bilgi katmanı çizer."""

    @staticmethod
    def plaka_kutusu(kare, bbox, plaka, sayac, esik, renk=(255, 0, 255), kalin=2):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(kare, (x1, y1), (x2, y2), renk, kalin)
        cv2.putText(
            kare,
            f"{plaka}  ({sayac}/{esik})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, renk, 2,
        )

    @staticmethod
    def basarili_kayit(kare, bbox, plaka):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(kare, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.putText(
            kare,
            f"KAYDEDILDI: {plaka}",
            (x1, y1 - 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )

    @staticmethod
    def fps_ve_bilgi(kare, fps: float, toplam_kayit: int):
        cv2.putText(
            kare,
            f"FPS: {fps:.1f}  |  Toplam Kayit: {toplam_kayit}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1,
        )


# ─────────────────────────────────────────────
# ANA DÖNGÜ: OtoKantar
# ─────────────────────────────────────────────

class OtoKantar:
    def __init__(self):
        log.info("=" * 55)
        log.info("OtoKantar V7 başlatılıyor...")

        self.tespitci = PlakaTespitci(
            weights_url=CONFIG["PLATE_WEIGHTS_URL"],
            models_dir=CONFIG["MODELS_DIR"],
            conf=CONFIG["YOLO_CONF"],
            gpu=CONFIG["OCR_GPU"],
        )
        self.cozucu = PlakaCozucu(
            diller=CONFIG["OCR_DILLER"],
            gpu=CONFIG["OCR_GPU"],
            min_conf=CONFIG["MIN_OCR_CONF"],
        )
        self.dogrulama = DogrulamaMotoru(
            esik=CONFIG["ESIK_DEGERI"],
            kayit_sonrasi_bekleme=CONFIG["BEKLEME_SURESI_SONRA"],
        )
        self.kaydedici = KantarKaydedici(
            csv_dosya=CONFIG["CSV_DOSYA"],
            json_dosya=CONFIG["JSON_CANLI"],
        )
        self.cizici = EkranCizici()

        self._kare_sayaci = 0
        self._fps_onceki = time.time()
        self._fps = 0.0
        self._aktif_bbox = None   # Son başarılı YOLO bbox (OCR atlandığında çiz)

    # ── Kamera yönetimi ─────────────────────────

    def _kamera_ac(self) -> cv2.VideoCapture:
        for deneme in range(CONFIG["KAMERA_YENIDEN_BAGLANTI_DENEMESI"]):
            kamera = cv2.VideoCapture(CONFIG["KAMERA_INDEX"])
            if kamera.isOpened():
                log.info("Kamera bağlantısı kuruldu.")
                return kamera
            log.warning(
                f"Kamera açılamadı. Deneme {deneme+1}/"
                f"{CONFIG['KAMERA_YENIDEN_BAGLANTI_DENEMESI']}"
            )
            time.sleep(CONFIG["KAMERA_BEKLEME_SURESI"])
        raise RuntimeError("Kamera bir türlü açılamadı. Cihazı kontrol edin.")

    # ── FPS hesap ───────────────────────────────

    def _fps_guncelle(self):
        su_an = time.time()
        gecen = su_an - self._fps_onceki
        if gecen > 0:
            self._fps = 1.0 / gecen
        self._fps_onceki = su_an

    # ── Tek kare işle ───────────────────────────

    def _kare_isle(self, kare: "cv2.Mat"):
        self._kare_sayaci += 1
        self._fps_guncelle()

        # YOLO: her karede çalışır (hızlı)
        plaka_det = self.tespitci.en_iyi_plakayi_bul(kare)

        if plaka_det is None:
            self._aktif_bbox = None
            return

        x1, y1, x2, y2, yolo_conf = plaka_det
        ix1 = max(0, int(x1))
        iy1 = max(0, int(y1))
        ix2 = min(kare.shape[1], int(x2))
        iy2 = min(kare.shape[0], int(y2))
        self._aktif_bbox = (ix1, iy1, ix2, iy2)

        # YOLO kutusu çiz (soluk)
        cv2.rectangle(kare, (ix1, iy1), (ix2, iy2), (128, 0, 128), 1)

        # OCR: her N karede bir (ağır işlem)
        if self._kare_sayaci % CONFIG["OCR_KARE_ATLAMA"] != 0:
            return

        # Yatay padding ile ROI kes
        w_roi = ix2 - ix1
        pad = int(w_roi * 0.10)
        roi = kare[iy1:iy2, max(0, ix1 - pad):min(kare.shape[1], ix2 + pad)]

        # Debug: işlenmiş binary göster
        binary_debug = self.cozucu.roi_hazirla_debug(roi)
        if binary_debug is not None:
            cv2.imshow("OCR Debug", binary_debug)

        # Çöz
        sonuc = self.cozucu.coz(roi)

        if not sonuc.gecerli or sonuc.plaka is None:
            return

        plaka = sonuc.plaka
        sayac = self.dogrulama.sayac_al(plaka)

        # Doğrulama + sayaç göster
        self.cizici.plaka_kutusu(
            kare, self._aktif_bbox, plaka,
            sayac, CONFIG["ESIK_DEGERI"],
        )

        # Kayıt kararı
        kaydet = self.dogrulama.isle(plaka)

        if kaydet:
            simdi = datetime.now()
            kayit = PlakaKayit(
                plaka=plaka,
                tarih=simdi.strftime("%Y-%m-%d"),
                saat=simdi.strftime("%H:%M:%S"),
                guven=yolo_conf,
            )
            self.kaydedici.kaydet(kayit)
            self.cizici.basarili_kayit(kare, self._aktif_bbox, plaka)

        # Bellek temizliği (her 300 karede bir)
        if self._kare_sayaci % 300 == 0:
            self.dogrulama.temizle_eski()

    # ── Ana çalıştırıcı ─────────────────────────

    def calistir(self):
        kamera = self._kamera_ac()
        log.info("Döngü başladı. Çıkmak için 'q' tuşuna basın.")

        try:
            while True:
                ret, kare = kamera.read()

                if not ret:
                    log.warning("Kare okunamadı, yeniden bağlanılıyor...")
                    kamera.release()
                    try:
                        kamera = self._kamera_ac()
                    except RuntimeError as e:
                        log.error(str(e))
                        break
                    continue

                self._kare_isle(kare)

                self.cizici.fps_ve_bilgi(
                    kare,
                    self._fps,
                    len(self.kaydedici.son_kayitlar),
                )

                cv2.imshow("OtoKantar V7", kare)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    log.info("Kullanıcı çıkış yaptı.")
                    break

        except KeyboardInterrupt:
            log.info("Ctrl+C ile durduruldu.")
        finally:
            kamera.release()
            cv2.destroyAllWindows()
            log.info("OtoKantar V7 kapatıldı.")


# ─────────────────────────────────────────────
# BAŞLAT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    sistem = OtoKantar()
    sistem.calistir()