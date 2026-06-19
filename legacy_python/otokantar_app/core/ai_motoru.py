"""
ai_motoru.py — Production-Grade Turkish License Plate AI Engine
================================================================
YOLO v8 (detection) + PaddleOCR / EasyOCR (adaptive OCR)
+ OpenCV (preprocessing) + threading (worker pool)

Architecture role: SENSOR only.
  • Detects plate bounding boxes (YOLO)
  • Preprocesses ROI
  • Runs OCR (primary + conditional fallback)
  • Returns raw plate text + confidence

❌ This layer NEVER:
  • Votes / tracks across frames  (→ dogrulama.py)
  • Makes final plate decisions   (→ dogrulama.py)
  • Assigns vehicle IDs           (→ tracker.py)

OCR Strategy (CPU-optimised):
  • Pass 1 — original binary  → primary backend
  • Pass 2 — inverted binary  → primary backend  (only if pass-1 conf < 0.6)
  • Fallback backend           → only if both passes fail or conf < OCR_MIN_CONF
"""

from __future__ import annotations

import os
import queue
import re
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

# Paddle (özellikle Windows + CPU) bazı sürümlerde oneDNN/PIR executor ile çökebiliyor.
# Bu bayraklar, import *öncesi* set edilmezse etkisiz kalır.
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_onednn", "0")
os.environ.setdefault("FLAGS_use_onednn", "0")

try:
    from paddleocr import PaddleOCR
    _PADDLE_AVAILABLE = True
except ImportError:
    _PADDLE_AVAILABLE = False

try:
    import easyocr as _easyocr
    _EASY_AVAILABLE = True
except ImportError:
    _EASY_AVAILABLE = False

from ultralytics import YOLO

from otokantar_app.config import CONFIG, PLAKA_REGEX, _HARF_DUZELTME, _RAKAM_DUZELTME
from otokantar_app.logger import log
from otokantar_app.models import OcrGorevi, TespitSonucu


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------
_CONFIG_DEFAULTS: dict = {
    # --- detection ---
    "ASPECT_RATIO_MIN": 1.8,
    "ASPECT_RATIO_MAX": 6.5,
    # --- preprocessing toggles ---
    "PREP_GAMMA": True,
    "PREP_BILATERAL": False,          # OCR için tehlikeli — kapalı bırak
    "PREP_CLAHE": True,
    "PREP_SHARPEN": False,            # karıncalanmanın baş sorumlusu — kapalı
    "PREP_ADAPTIVE": False,           # içi boş harflerin baş sorumlusu — kapalı; Otsu yeterli
    "PREP_PERSPECTIVE": True,
    "PREP_SUPERRES": False,          # requires Real-ESRGAN weights
    "PREP_DESKEW": True,
    # --- gamma ---
    "GAMMA_PARLAKLIK_ESIK": 100,     # boost when DARK (p95 < threshold)
    "GAMMA_US": 0.6,                 # <1 → brightens dark images
    # --- bilateral ---
    "BILATERAL_ESIK": 210,
    "BILATERAL_D": 7,
    "BILATERAL_SIGMA_COLOR": 50,
    "BILATERAL_SIGMA_SPACE": 50,
    # --- CLAHE ---
    "CLAHE_CLIP": 2.0,
    "CLAHE_GRID": (8, 8),
    # --- sharpen ---
    "SHARPEN_AMOUNT": 1.5,
    "SHARPEN_SIGMA": 1.0,
    # --- adaptive threshold (yalnızca PREP_ADAPTIVE=True ise aktif) ---
    "ADAPTIVE_BLOCK": 51,             # büyük blok → harflerin içi dolar; tek sayı olmalı
    "ADAPTIVE_C": 4,                  # düşük C → daha az boşaltma
    # --- morphology ---
    "MORPH_KAPAT": True,
    "MORPH_KERNEL": (3, 3),
    # --- Canny (perspective corner detection only) ---
    "CANNY_ESIK1": 40,
    "CANNY_ESIK2": 200,
    # --- perspective ---
    "PERSPEKTIF_MIN_ALAN_ORAN": 0.05,
    "PERSPEKTIF_EPSILON_CARPAN": 0.02,
    "PERSPEKTIF_CIKTI_EN": 400,
    "PERSPEKTIF_CIKTI_BOY": 120,
    # --- resize ---
    "BOYUTLANDIRMA_KATSAYI": 2.0,
    # --- crop ---
    "ALT_KIRP_ORAN": 0.15,
    # --- OCR ---
    "OCR_IZIN_LISTESI": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    "OCR_MIN_CONF": 0.35,
    "OCR_FALLBACK_ENABLED": True,
    # ↓ Threshold below which a second OCR pass (inverted) is triggered
    "OCR_IKINCI_GECIS_ESIK": 0.6,
    # --- worker ---
    "WORKER_KUYRUK": 8,
}


def _cfg(key: str):
    """Read from project CONFIG, fall back to local defaults."""
    return CONFIG.get(key, _CONFIG_DEFAULTS.get(key))


# ---------------------------------------------------------------------------
# PlakaTespitci — YOLO-based plate detector
# ---------------------------------------------------------------------------

class PlakaTespitci:
    """Detects license plate bounding boxes using YOLO."""

    _PLAKA_ETIKETLERI = {"plate", "license_plate", "licence_plate", "plaka", "number_plate"}

    def __init__(self, weights_url: str, models_dir: str, conf: float, gpu: bool) -> None:
        self.conf = conf
        weights_path = self._model_indir(weights_url, models_dir)
        self.model = YOLO(weights_path)
        device = "cuda" if (gpu and torch.cuda.is_available()) else "cpu"
        self.model.to(device)
        log.info("YOLO → %s", device.upper())

    @staticmethod
    def _model_indir(url: str, models_dir: str) -> str:
        """Download YOLO weights if not cached."""
        path = Path(models_dir) / "license_plate_detector.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            log.info("Plaka modeli indiriliyor: %s", url)
            urllib.request.urlretrieve(url, path)
            log.info("Model kaydedildi: %s", path)
        return str(path)

    def plakalari_bul(self, bgr: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        """
        Run YOLO inference and return filtered plate bounding boxes.

        Returns:
            List of (x1, y1, x2, y2, conf) sorted by confidence descending.
        """
        sonuclar = self.model(bgr, verbose=False)[0]
        if sonuclar.boxes is None or len(sonuclar.boxes) == 0:
            return []
        names: dict = sonuclar.names or {}
        cikti: list[tuple[float, float, float, float, float]] = []
        for b in sonuclar.boxes:
            conf = float(b.conf[0])
            if conf < self.conf:
                continue
            cls_id = int(b.cls[0])
            cls_name = names.get(cls_id, "")
            if not (cls_name.lower() in self._PLAKA_ETIKETLERI or cls_id == 0):
                continue
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            ar = w / h
            ar_min = float(_cfg("ASPECT_RATIO_MIN"))
            ar_max = float(_cfg("ASPECT_RATIO_MAX"))
            if not (ar_min <= ar <= ar_max):
                continue
            log.debug(
                "bbox tespit edildi bbox=(%.1f,%.1f,%.1f,%.1f) conf=%.3f class=%s aspect=%.2f",
                x1, y1, x2, y2, conf, cls_name or cls_id, ar,
            )
            cikti.append((x1, y1, x2, y2, conf))
        cikti.sort(key=lambda t: t[4], reverse=True)
        log.debug("bbox tespit ozeti adet=%d frame_shape=%s", len(cikti), tuple(bgr.shape[:2]))
        return cikti


# ---------------------------------------------------------------------------
# OCR Backends
# ---------------------------------------------------------------------------

class _OcrBackend:
    """Abstract base for OCR backends."""

    def oku(self, gray: np.ndarray) -> tuple[str, float]:
        raise NotImplementedError


class _PaddleBackend(_OcrBackend):
    """PaddleOCR — primary backend, best accuracy for Turkish plates."""

    def __init__(self, gpu: bool) -> None:
        self._broken = False
        self.calisiyor = False
        kwargs = dict(
            use_textline_orientation=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            lang="en",
            enable_mkldnn=bool(_cfg("PADDLE_ENABLE_MKLDNN")),
        )
        try:
            self._reader = PaddleOCR(**kwargs, show_log=False)
        except ValueError as e:
            if "Unknown argument: show_log" not in str(e):
                raise
            self._reader = PaddleOCR(**kwargs)
        except TypeError:
            self._reader = PaddleOCR(
                use_angle_cls=False,
                lang="en",
            )
        self.calisiyor = self._probe()
        if self.calisiyor:
            log.info("PaddleOCR hazır (gpu=%s)", gpu)
        else:
            log.warning("PaddleOCR bu sistemde inference yapamıyor.")

    def _probe(self) -> bool:
        img = np.ones((32, 100, 3), dtype=np.uint8) * 255
        cv2.putText(img, "AB12", (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        try:
            self._run_ocr(img)
            return True
        except Exception as exc:
            log.warning("PaddleOCR probe başarısız: %s", exc)
            self._broken = True
            return False

    def _run_ocr(self, bgr: np.ndarray):
        if hasattr(self._reader, "predict"):
            return self._reader.predict(bgr)
        return self._reader.ocr(bgr)

    @staticmethod
    def _metin_cikar(sonuc) -> tuple[str, float]:
        if not sonuc:
            return "", 0.0

        page = sonuc[0] if isinstance(sonuc, list) and sonuc else sonuc
        min_conf = float(_cfg("OCR_MIN_CONF"))
        parcalar: list[str] = []
        confs: list[float] = []

        rec_texts = None
        rec_scores = None
        if isinstance(page, dict):
            rec_texts = page.get("rec_texts")
            rec_scores = page.get("rec_scores")
        else:
            try:
                rec_texts = page["rec_texts"]
                rec_scores = page["rec_scores"]
            except (KeyError, TypeError):
                rec_texts = None

        if rec_texts is not None:
            scores = list(rec_scores or [])
            for idx, metin in enumerate(rec_texts):
                conf = float(scores[idx]) if idx < len(scores) else 0.0
                if conf < min_conf:
                    continue
                temiz = re.sub(r"[^A-Z0-9]", "", str(metin).upper())
                if temiz:
                    parcalar.append(temiz)
                    confs.append(conf)
            if parcalar:
                return "".join(parcalar), sum(confs) / len(confs)
            return "", 0.0

        if isinstance(page, list):
            for line in page:
                if not line or len(line) < 2:
                    continue
                metin, conf = line[1]
                if conf < min_conf:
                    continue
                temiz = re.sub(r"[^A-Z0-9]", "", str(metin).upper())
                if temiz:
                    parcalar.append(temiz)
                    confs.append(float(conf))
            if parcalar:
                return "".join(parcalar), sum(confs) / len(confs)
        return "", 0.0

    def oku(self, gray: np.ndarray) -> tuple[str, float]:
        if self._broken:
            raise RuntimeError("PaddleOCR devre dışı (önceki hata nedeniyle).")
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        try:
            sonuc = self._run_ocr(bgr)
        except Exception:
            self._broken = True
            raise
        return self._metin_cikar(sonuc)


class _EasyBackend(_OcrBackend):
    """EasyOCR — fallback backend."""

    def __init__(self, diller: list[str], gpu: bool) -> None:
        self._reader = _easyocr.Reader(diller, gpu=gpu and torch.cuda.is_available())
        log.info("EasyOCR hazır.")

    def oku(self, gray: np.ndarray) -> tuple[str, float]:
        sonuclar = self._reader.readtext(
            gray,
            allowlist=_cfg("OCR_IZIN_LISTESI"),
            paragraph=False,
        )
        if not sonuclar:
            return "", 0.0

        def sol_x(item):
            try:
                return min(pt[0] for pt in item[0])
            except Exception:
                return 0

        parcalar, confs = [], []
        min_conf = float(_cfg("OCR_MIN_CONF"))
        for (_, metin, conf) in sorted(sonuclar, key=sol_x):
            if conf is None or conf < min_conf:
                continue
            temiz = re.sub(r"[^A-Z0-9]", "", str(metin).upper())
            if temiz:
                parcalar.append(temiz)
                confs.append(float(conf))
        if not parcalar:
            return "", 0.0
        return "".join(parcalar), sum(confs) / len(confs)


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _sharpen(img: np.ndarray) -> np.ndarray:
    """Unsharp-mask sharpening."""
    sigma = float(_cfg("SHARPEN_SIGMA"))
    amount = float(_cfg("SHARPEN_AMOUNT"))
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)


def _deskew(gray: np.ndarray) -> np.ndarray:
    """
    Correct slight rotation using Hough line angle estimation.
    Only corrects angles within ±15°. Skipped for very small images.
    """
    h, w = gray.shape[:2]
    # Skip deskew on small crops — HoughLines on tiny images produces noise
    if w < 60 or h < 20:
        return gray
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=max(30, w // 4))
    if lines is None:
        return gray
    angles = []
    for line in lines[:20]:
        theta = float(line[0][1])
        angle = np.degrees(theta) - 90.0
        if abs(angle) < 15:
            angles.append(angle)
    if not angles:
        return gray
    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return gray
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), median_angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _superres_hook(bgr: np.ndarray) -> np.ndarray:
    """
    Super-resolution hook. Drop-in: replace bicubic fallback with Real-ESRGAN
    by setting PREP_SUPERRES=True and injecting a model via CONFIG["SUPERRES_MODEL"].
    """
    sr_model = CONFIG.get("SUPERRES_MODEL")
    if sr_model is not None:
        try:
            return sr_model.upsample(bgr)
        except Exception as exc:
            log.warning("Super-resolution hatası, fallback: %s", exc)
    h, w = bgr.shape[:2]
    return cv2.resize(bgr, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)


# ---------------------------------------------------------------------------
# PlakaCozucu — preprocessing + adaptive OCR
# ---------------------------------------------------------------------------

class PlakaCozucu:
    """
    Preprocessing pipeline + adaptive two-pass OCR.

    OCR flow (CPU-optimised):
      1. Preprocess → binary image
      2. Primary backend reads original binary
      3. If conf < OCR_IKINCI_GECIS_ESIK → primary re-reads inverted binary
      4. If still low → fallback backend reads best variant
      Best result is selected by: valid regex match → highest confidence.

    This class is a SENSOR: it returns raw candidates only.
    All voting, clustering and final decisions are in dogrulama.py.
    """

    def __init__(self, diller: list[str], gpu: bool, min_conf: float) -> None:
        self.min_conf = min_conf
        self._gpu = gpu
        self._diller = diller
        self._fallback_lock = threading.Lock()
        self._clahe = cv2.createCLAHE(
            clipLimit=float(_cfg("CLAHE_CLIP")),
            tileGridSize=tuple(_cfg("CLAHE_GRID")),
        )

        paddle_ok = False
        if _PADDLE_AVAILABLE:
            paddle = _PaddleBackend(gpu)
            if paddle.calisiyor:
                self._primary: _OcrBackend = paddle
                paddle_ok = True

        if not paddle_ok:
            if _EASY_AVAILABLE:
                log.warning("EasyOCR birincil backend olarak kullanılacak.")
                self._primary = _EasyBackend(diller, gpu)
            elif _PADDLE_AVAILABLE:
                raise RuntimeError(
                    "PaddleOCR çalışmıyor ve EasyOCR yüklü değil. "
                    "pip install easyocr ile kurun."
                )
            else:
                raise RuntimeError("Hiçbir OCR backend bulunamadı (PaddleOCR veya EasyOCR gerekli).")

        self.primary_backend_adi = type(self._primary).__name__.lstrip("_").replace("Backend", "")

        self._fallback: Optional[_OcrBackend] = None
        self.fallback_backend_adi = (
            type(self._fallback).__name__.lstrip("_").replace("Backend", "")
            if self._fallback is not None
            else None
        )

    def _ensure_fallback(self) -> Optional[_OcrBackend]:
        if self._fallback is not None:
            return self._fallback
        if not _EASY_AVAILABLE:
            return None
        with self._fallback_lock:
            if self._fallback is not None:
                return self._fallback
            try:
                self._fallback = _EasyBackend(self._diller, self._gpu)
                self.fallback_backend_adi = "Easy"
                log.info("EasyOCR fallback devreye alındı.")
            except Exception as exc:
                log.error("EasyOCR fallback başlatılamadı: %s", exc)
                return None
        return self._fallback

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sirala_dort_kose(pts: np.ndarray) -> np.ndarray:
        """Sort 4 corner points: TL, TR, BR, BL."""
        pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).flatten()
        return np.array(
            [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
            dtype=np.float32,
        )

    def _dortgen_kose_bul(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Find the 4-corner quadrilateral of the plate inside the ROI.
        Falls back to min-area rotated rect if no clean quad found.
        """
        h, w = bgr.shape[:2]
        if w < 8 or h < 8:
            return None
        gri = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gri, (5, 5), 0)
        kenar = cv2.Canny(blur, int(_cfg("CANNY_ESIK1")), int(_cfg("CANNY_ESIK2")))
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kenar = cv2.morphologyEx(kenar, cv2.MORPH_CLOSE, k, iterations=2)
        konturlar, _ = cv2.findContours(kenar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not konturlar:
            return None
        min_alan = float(_cfg("PERSPEKTIF_MIN_ALAN_ORAN")) * float(w * h)
        eps_c = float(_cfg("PERSPEKTIF_EPSILON_CARPAN"))
        for cnt in sorted(konturlar, key=cv2.contourArea, reverse=True)[:20]:
            if cv2.contourArea(cnt) < min_alan:
                continue
            peri = cv2.arcLength(cnt, True)
            if peri < 1e-6:
                continue
            yaklasik = cv2.approxPolyDP(cnt, eps_c * peri, True)
            if len(yaklasik) != 4 or not cv2.isContourConvex(yaklasik):
                continue
            kose = yaklasik.reshape(4, 2).astype(np.float32)
            if (np.any(kose[:, 0] < -0.5) or np.any(kose[:, 0] > w - 0.5)
                    or np.any(kose[:, 1] < -0.5) or np.any(kose[:, 1] > h - 0.5)):
                continue
            return self._sirala_dort_kose(kose)
        # Fallback: minimum-area rotated rect
        all_pts = np.vstack(konturlar)
        rect = cv2.minAreaRect(all_pts)
        box = cv2.boxPoints(rect).astype(np.float32)
        alan = cv2.contourArea(box)
        if alan >= min_alan:
            return self._sirala_dort_kose(box)
        return None

    def _perspektif_duzelt(self, bgr: np.ndarray) -> np.ndarray:
        """Apply perspective correction to a plate ROI."""
        h, w = bgr.shape[:2]
        out_w = int(_cfg("PERSPEKTIF_CIKTI_EN"))
        out_h = int(_cfg("PERSPEKTIF_CIKTI_BOY"))
        kose = self._dortgen_kose_bul(bgr)
        src = kose if kose is not None else np.array(
            [[0.0, 0.0], [float(w), 0.0], [float(w), float(h)], [0.0, float(h)]],
            dtype=np.float32,
        )
        dst = np.array(
            [[0.0, 0.0], [float(out_w), 0.0], [float(out_w), float(out_h)], [0.0, float(out_h)]],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(bgr, M, (out_w, out_h), flags=cv2.INTER_CUBIC)

    # ------------------------------------------------------------------
    # Gamma correction
    # ------------------------------------------------------------------

    def _gamma_bgr(self, bgr: np.ndarray) -> np.ndarray:
        """Boost brightness of DARK images only (p95 < threshold)."""
        if not _cfg("PREP_GAMMA"):
            return bgr
        gri = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        p95 = float(np.percentile(gri, 95))
        esik = float(_cfg("GAMMA_PARLAKLIK_ESIK"))
        if p95 >= esik:          # already bright, skip
            return bgr
        us = float(_cfg("GAMMA_US"))
        table = (
            (np.arange(256, dtype=np.float64) / 255.0) ** us * 255.0
        ).clip(0, 255).astype(np.uint8)
        # Stack once — cv2.LUT on BGR needs a (256,1,3) LUT
        lut = np.stack([table, table, table], axis=-1).reshape(256, 1, 3)
        return cv2.LUT(bgr, lut)

    # ------------------------------------------------------------------
    # ROI preprocessing pipeline
    # ------------------------------------------------------------------

    def _roi_hazirla(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Modular preprocessing pipeline. Returns a binary (thresholded)
        grayscale image ready for OCR, or None on failure.

        Pipeline:
          1.  Bottom-strip crop   (kirlilik / cıvata gölgelerini at)
          2.  2× resize
          3.  Gamma düzeltme      (yalnızca karanlık görüntülerde)
          4.  Perspektif düzeltme (opsiyonel)
          5.  Super-resolution    (opsiyonel, varsayılan kapalı)
          6.  Gri tonlamaya çevir
          7.  Bilateral filtre    (opsiyonel, varsayılan KAPALI — OCR için riskli)
          8.  Deskew              (opsiyonel)
          9.  CLAHE
          10. Gaussian blur       (hafif yumuşatma)
          11. Keskinleştirme      (opsiyonel, EŞIKLEMEDEN ÖNCE ve hafif — PREP_SHARPEN=True ise)
          12. Otsu eşikleme       (birincil; dolgu harfler için en güvenilir)
              └─ Adaptif eşikleme (opsiyonel, PREP_ADAPTIVE=True ise; ADAPTIVE_BLOCK≥35 kullan)
          13. Median blur (3×3)   (tuz-biber karıncalanmasını yok et)
          14. Morfolojik kapama   (küçük boşlukları doldur)

        Tasarım kararları:
          • PREP_ADAPTIVE=False  → Otsu tek başına; içi dolu, temiz harfler.
          • PREP_SHARPEN=False   → Keskinleştirme gürültüyü büyütür; kapalı bırak.
          • PREP_BILATERAL=False → Bilateral OCR için tehlikelidir; kapalı bırak.
          • medianBlur(3)        → Eşikleme sonrası karıncalanmayı temizler.
        """
        if bgr is None or bgr.size == 0:
            return None
        h, w = bgr.shape[:2]

        # 1. Bottom-strip crop
        if h >= 30:
            oran = min(float(_cfg("ALT_KIRP_ORAN")), 0.49)
            efektif = min(oran, 1.0 - 0.80)
            bgr = bgr[: max(1, int(h * (1.0 - efektif))), :w]

        # 2. Resize
        katsayi = float(_cfg("BOYUTLANDIRMA_KATSAYI"))
        bgr = cv2.resize(bgr, None, fx=katsayi, fy=katsayi, interpolation=cv2.INTER_CUBIC)

        # 3. Gamma
        bgr = self._gamma_bgr(bgr)

        # 4. Perspective
        if _cfg("PREP_PERSPECTIVE"):
            bgr = self._perspektif_duzelt(bgr)

        # 5. Super-resolution (off by default)
        if _cfg("PREP_SUPERRES"):
            bgr = _superres_hook(bgr)

        # 6. Gray
        gri = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 7. Bilateral filtre — yalnızca parlak/yansımalı plakalarda; OCR için risklidir.
        #    Varsayılan: PREP_BILATERAL=False.  Açmak istersen BILATERAL_ESIK'i dikkatlice ayarla.
        if _cfg("PREP_BILATERAL"):
            p95 = float(np.percentile(gri, 95))
            if p95 > int(_cfg("BILATERAL_ESIK")):
                gri = cv2.bilateralFilter(
                    gri,
                    d=int(_cfg("BILATERAL_D")),
                    sigmaColor=float(_cfg("BILATERAL_SIGMA_COLOR")),
                    sigmaSpace=float(_cfg("BILATERAL_SIGMA_SPACE")),
                )

        # 8. Deskew
        if _cfg("PREP_DESKEW"):
            gri = _deskew(gri)

        # 9. CLAHE
        if _cfg("PREP_CLAHE"):
            gri = self._clahe.apply(gri)

        # 10. Gaussian blur — hafif yumuşatma; eşikleme öncesi gürültüyü azaltır
        blur = cv2.GaussianBlur(gri, (3, 3), 0)

        # 11. Keskinleştirme — eşiklemeden ÖNCE ve hafifçe uygula.
        #     NOT: Bu adım gürültü içeren görüntülerde karıncalanmayı artırır.
        #     Varsayılan: PREP_SHARPEN=False.  Açmak istersen SHARPEN_AMOUNT'u düşür (≤1.0).
        if _cfg("PREP_SHARPEN"):
            blur = _sharpen(blur)

        # 12. Eşikleme — birincil yöntem Otsu; harflerin içi dolu, temiz çıktı verir.
        #     Adaptif eşikleme (PREP_ADAPTIVE=True) yalnızca çok düzensiz aydınlatmada dene.
        #     Açmak istersen ADAPTIVE_BLOCK≥35 (tek sayı) ve ADAPTIVE_C≤4 kullan.
        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = otsu
        if _cfg("PREP_ADAPTIVE"):
            adaptive = cv2.adaptiveThreshold(
                blur,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                int(_cfg("ADAPTIVE_BLOCK")),
                int(_cfg("ADAPTIVE_C")),
            )
            # Daha fazla ön plan detayı içeren versiyonu seç
            binary = adaptive if adaptive.mean() > otsu.mean() else otsu

        # 13. Median blur (3×3) — tuz-biber karıncalanmasını yok et; eşikleme hemen sonrası
        binary = cv2.medianBlur(binary, 3)

        # 14. Morfolojik kapama — kalan küçük boşlukları doldur
        if _cfg("MORPH_KAPAT"):
            k = cv2.getStructuringElement(cv2.MORPH_RECT, tuple(_cfg("MORPH_KERNEL")))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=1)

        return binary

    # ------------------------------------------------------------------
    # Adaptive two-pass OCR
    # ------------------------------------------------------------------

    def _plaka_adayi_cikar(self, metin: str) -> tuple[Optional[str], int, int]:
        """
        Return a valid corrected Turkish plate candidate from OCR text.

        PLAKA_REGEX is intentionally a search because OCR often adds junk at
        the edges (e.g. N06ABC123). The caller still gets the amount of junk so
        scoring can prefer cleaner full-plate readings.
        """
        ham = (metin or "").upper()
        for eslesen in PLAKA_REGEX.finditer(ham):
            il, harf, rakam = eslesen.group(1), eslesen.group(2), eslesen.group(3)
            plaka = self._plaka_duzelt(il, harf, rakam)
            if self._uzunluk_gecerli(plaka):
                sol_junk = eslesen.start()
                sag_junk = len(ham) - eslesen.end()
                return plaka, sol_junk, sag_junk
        return None, 0, 0

    def _ocr_calistir(self, binary: np.ndarray) -> tuple[str, float]:
        """
        Adaptive two-pass OCR — CPU-optimised.

        Pass 1: primary backend on original binary.
        Pass 2: primary backend on inverted binary (only if pass-1 conf < threshold).
        Fallback: secondary backend if passes are insufficient.

        Eliminates 'edge' variant (useless for OCR) and unconditional
        multi-pass (was 6-9 calls per plate → now 1-3 max).
        """
        ikinci_esik = float(_cfg("OCR_IKINCI_GECIS_ESIK"))
        min_conf = float(_cfg("OCR_MIN_CONF"))
        metin2, conf2 = "", 0.0
        fb_metin, fb_conf = "", 0.0

        # --- Pass 1: original ---
        try:
            metin1, conf1 = self._primary.oku(binary)
            primary_ok_1 = True
        except Exception as e:
            log.warning("Birincil OCR çalıştırılamadı; fallback denenecek. (%s)", e)
            metin1, conf1 = "", 0.0
            primary_ok_1 = False
        log.debug(
            "OCR_PASS1 backend=%s text=%r confidence=%.3f",
            self.primary_backend_adi,
            metin1,
            float(conf1),
        )

        # Early exit: high-confidence valid plate on first pass
        if conf1 >= ikinci_esik and self._plaka_adayi_cikar(metin1)[0]:
            log.debug(
                "OCR_ADAY pass1=%r conf1=%.3f pass2=%r conf2=%.3f fallback=%r fallback_conf=%.3f",
                metin1,
                float(conf1),
                metin2,
                float(conf2),
                fb_metin,
                float(fb_conf),
            )
            log.debug(
                "OCR_SECILEN text=%r confidence=%.3f regex_match=%s",
                metin1,
                float(conf1),
                True,
            )
            return metin1, conf1

        # --- Pass 2: inverted (only when needed) ---
        inverted = cv2.bitwise_not(binary)
        if primary_ok_1:
            try:
                metin2, conf2 = self._primary.oku(inverted)
            except Exception as e:
                log.warning("Birincil OCR (inverted) çalıştırılamadı; pass-1 sonucu kullanılacak. (%s)", e)
                metin2, conf2 = "", 0.0
        else:
            metin2, conf2 = "", 0.0
        log.debug(
            "OCR_PASS2 backend=%s text=%r confidence=%.3f",
            self.primary_backend_adi,
            metin2,
            float(conf2),
        )

        # Pick the better of the two primary passes
        def _skor(m: str, c: float) -> float:
            plaka, sol_junk, sag_junk = self._plaka_adayi_cikar(m)
            if plaka:
                # Prefer readings that produce a real plate after correction.
                # Edge junk is tolerated but penalized so "06ABC123" beats
                # "N06ABC123" at similar confidence.
                return c + 2.0 - ((sol_junk + sag_junk) * 0.15)
            # A raw regex fragment that cannot become a valid plate is weak
            # evidence only; examples like "6644Y895" used to win here.
            if PLAKA_REGEX.search(m or ""):
                return c + 0.15
            # If neither pass is valid, avoid letting tiny high-confidence
            # fragments such as "06" dominate longer near-complete readings.
            return c + min(len(m or ""), 9) * 0.02

        if _skor(metin2, conf2) > _skor(metin1, conf1):
            best_metin, best_conf = metin2, conf2
        else:
            best_metin, best_conf = metin1, conf1

        # If primary backend couldn't run at all, go straight to fallback (if present).
        if not primary_ok_1:
            fb = self._ensure_fallback()
            if fb is not None:
                fb1_m, fb1_c = fb.oku(binary)
                fb2_m, fb2_c = fb.oku(inverted)
                if _skor(fb2_m, fb2_c) > _skor(fb1_m, fb1_c):
                    fb_metin, fb_conf = fb2_m, fb2_c
                else:
                    fb_metin, fb_conf = fb1_m, fb1_c
                log.debug(
                    "OCR_FALLBACK backend=%s text=%r confidence=%.3f",
                    self.fallback_backend_adi,
                    fb_metin,
                    float(fb_conf),
                )
                log.debug(
                    "OCR_ADAY pass1=%r conf1=%.3f pass2=%r conf2=%.3f fallback=%r fallback_conf=%.3f",
                    metin1,
                    float(conf1),
                    metin2,
                    float(conf2),
                    fb_metin,
                    float(fb_conf),
                )
                log.debug(
                    "OCR_SECILEN text=%r confidence=%.3f regex_match=%s",
                    fb_metin,
                    float(fb_conf),
                    bool(PLAKA_REGEX.search(fb_metin)),
                )
                return fb_metin, fb_conf

        # --- Fallback backend (only if primary result is still insufficient) ---
        if best_conf < min_conf:
            fb = self._ensure_fallback()
            if fb is not None:
                fb_metin, fb_conf = fb.oku(
                    binary if conf1 >= conf2 else inverted
                )
                log.debug(
                    "OCR_FALLBACK backend=%s text=%r confidence=%.3f",
                    self.fallback_backend_adi,
                    fb_metin,
                    float(fb_conf),
                )
                if _skor(fb_metin, fb_conf) > _skor(best_metin, best_conf):
                    best_metin, best_conf = fb_metin, fb_conf

        log.debug(
            "OCR_ADAY pass1=%r conf1=%.3f pass2=%r conf2=%.3f fallback=%r fallback_conf=%.3f",
            metin1,
            float(conf1),
            metin2,
            float(conf2),
            fb_metin,
            float(fb_conf),
        )
        log.debug(
            "OCR_SECILEN text=%r confidence=%.3f regex_match=%s",
            best_metin,
            float(best_conf),
            bool(PLAKA_REGEX.search(best_metin)),
        )
        return best_metin, best_conf

    # ------------------------------------------------------------------
    # Post-processing / correction
    # ------------------------------------------------------------------

    @staticmethod
    def _plaka_duzelt(il: str, harf: str, rakam: str) -> str:
        """Apply OCR confusion correction maps to each plate segment."""
        il_d = "".join(_RAKAM_DUZELTME.get(c, c) for c in il)
        harf_d = "".join(_HARF_DUZELTME.get(c, c) for c in harf)
        rakam_d = "".join(_RAKAM_DUZELTME.get(c, c) for c in rakam)
        return il_d + harf_d + rakam_d

    @staticmethod
    def _uzunluk_gecerli(plaka: str) -> bool:
        """Turkish plates are 7–9 characters (e.g. 06ABC123 or 34AB1234)."""
        return 7 <= len(plaka) <= 9

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def coz(self, bgr_roi: np.ndarray, bbox: tuple = ()) -> TespitSonucu:
        """
        Full pipeline: preprocess → adaptive OCR → validate → correct.

        Returns:
            TespitSonucu with raw plate text and confidence.
            Decision (accept/reject) is made downstream in dogrulama.py.
        """
        if bgr_roi is not None and bgr_roi.size > 0:
            crop_h, crop_w = bgr_roi.shape[:2]
        else:
            crop_h, crop_w = 0, 0
        log.debug(
            "OCR_CROP bbox=%s crop_width=%d crop_height=%d",
            bbox,
            crop_w,
            crop_h,
        )
        binary = self._roi_hazirla(bgr_roi)
        if binary is None:
            log.debug(
                "OCR_FINAL ham_metin= duzeltilmis_plaka= guven=0.000 gecerli=False",
            )
            return TespitSonucu(bbox=(), ham_metin="", plaka=None, guven=0.0)

        prep_h, prep_w = binary.shape[:2]
        log.debug(
            "OCR_PREP binary_width=%d binary_height=%d",
            prep_w,
            prep_h,
        )

        ham, ocr_conf = self._ocr_calistir(binary)
        eslesen = PLAKA_REGEX.search(ham)
        if eslesen:
            il, harf, rakam = eslesen.group(1), eslesen.group(2), eslesen.group(3)
            plaka = self._plaka_duzelt(il, harf, rakam)
            if self._uzunluk_gecerli(plaka):
                log.debug(
                    "OCR_FINAL ham_metin=%r duzeltilmis_plaka=%s guven=%.3f gecerli=True",
                    ham,
                    plaka,
                    float(ocr_conf),
                )
                return TespitSonucu(
                    bbox=(),
                    ham_metin=ham,
                    plaka=plaka,
                    guven=float(ocr_conf),
                    gecerli=True,
                )
        log.debug(
            "OCR_FINAL ham_metin=%r duzeltilmis_plaka= guven=%.3f gecerli=False",
            ham,
            float(ocr_conf),
        )
        return TespitSonucu(bbox=(), ham_metin=ham, plaka=None, guven=0.0)

    def coz_batch(self, roi_listesi: list[np.ndarray]) -> list[TespitSonucu]:
        """Process multiple ROIs sequentially."""
        return [self.coz(roi) for roi in roi_listesi]

    def roi_hazirla_debug(self, bgr_roi: np.ndarray) -> Optional[np.ndarray]:
        """Return preprocessed binary image for debug visualisation."""
        return self._roi_hazirla(bgr_roi)


# ---------------------------------------------------------------------------
# OcrWorker — threaded consumer (SENSOR only, no decision logic)
# ---------------------------------------------------------------------------

class OcrWorker(threading.Thread):
    """
    Background daemon thread that consumes OcrGorevi tasks,
    runs the full PlakaCozucu pipeline, and pushes raw results
    to an output queue.

    ❌ NO temporal smoothing here — that is dogrulama.py's job.
    """

    def __init__(
        self,
        cozucu: PlakaCozucu,
        kuyruk_boyutu: int = 0,
    ) -> None:
        super().__init__(name="OcrWorker", daemon=True)
        boyut = kuyruk_boyutu or int(_cfg("WORKER_KUYRUK"))
        self._cozucu = cozucu
        self._giris_kuyrugu: queue.Queue[Optional[OcrGorevi]] = queue.Queue(maxsize=boyut)
        self._cikis_kuyrugu: queue.Queue = queue.Queue(maxsize=0)
        self._dur = threading.Event()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def gorevi_gonder(self, gorev: OcrGorevi) -> bool:
        """Enqueue a task. Returns False if the queue is full (back-pressure)."""
        try:
            self._giris_kuyrugu.put_nowait(gorev)
            log.debug(
                "OCR worker queue boyutu=%d/%d arac_id=%s bbox=%s crop_shape=%s",
                self._giris_kuyrugu.qsize(),
                self._giris_kuyrugu.maxsize,
                gorev.arac_id,
                gorev.bbox,
                tuple(gorev.roi_bgr.shape[:2]),
            )
            return True
        except queue.Full:
            log.debug(
                "OCR worker queue overflow boyutu=%d/%d arac_id=%s bbox=%s",
                self._giris_kuyrugu.qsize(),
                self._giris_kuyrugu.maxsize,
                gorev.arac_id,
                gorev.bbox,
            )
            return False

    def gorevi_gonder_bekle(self, gorev: OcrGorevi, timeout: float = 0.1) -> bool:
        """Blocking enqueue with timeout. More reliable under load."""
        try:
            self._giris_kuyrugu.put(gorev, timeout=timeout)
            return True
        except queue.Full:
            return False

    def sonuclari_topla(self) -> list:
        """Drain and return all available results from the output queue."""
        sonuclar = []
        while True:
            try:
                sonuclar.append(self._cikis_kuyrugu.get_nowait())
            except queue.Empty:
                break
        return sonuclar

    def durdur(self) -> None:
        """Signal the worker to stop gracefully."""
        log.debug("thread stop istendi thread=%s", self.name)
        self._dur.set()
        try:
            self._giris_kuyrugu.put_nowait(None)
        except queue.Full:
            pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _isle(self, gorev: OcrGorevi) -> None:
        """Process one task and push raw result to the output queue."""
        log.debug(
            "OCR isle basladi arac_id=%s bbox=%s crop_shape=%s yolo_conf=%.3f",
            gorev.arac_id,
            gorev.bbox,
            tuple(gorev.roi_bgr.shape[:2]),
            float(gorev.yolo_conf),
        )
        sonuc = self._cozucu.coz(gorev.roi_bgr, gorev.bbox)
        log.debug(
            "OCR isle bitti arac_id=%s ham=%r plaka=%s guven=%.3f gecerli=%s",
            gorev.arac_id,
            sonuc.ham_metin,
            sonuc.plaka,
            float(sonuc.guven or 0.0),
            sonuc.gecerli,
        )
        self._cikis_kuyrugu.put((gorev.arac_id, sonuc, gorev.yolo_conf, gorev.bbox))

    def run(self) -> None:
        log.info("OcrWorker başlatıldı.")
        log.debug(
            "thread start thread=%s giris_max=%d cikis_max=%d",
            self.name,
            self._giris_kuyrugu.maxsize,
            self._cikis_kuyrugu.maxsize,
        )
        while not self._dur.is_set():
            try:
                gorev = self._giris_kuyrugu.get(timeout=1.0)
            except queue.Empty:
                continue
            if gorev is None:
                break
            try:
                self._isle(gorev)
            except Exception as exc:
                log.error("OcrWorker işleme hatası (%s): %s", type(exc).__name__, exc)
        log.info("OcrWorker durduruldu.")


# ---------------------------------------------------------------------------
# OcrWorkerPool — multiple parallel workers
# ---------------------------------------------------------------------------

class OcrWorkerPool:
    """
    Manages N OcrWorker threads for parallel processing.

    Tasks are dispatched round-robin. Collect results from all workers
    via ``sonuclari_topla``.

    Example::

        pool = OcrWorkerPool(cozucu, n_workers=2)
        pool.baslat()
        pool.gorevi_gonder(gorev)
        ...
        sonuclar = pool.sonuclari_topla()
        pool.durdur()
    """

    def __init__(
        self,
        cozucu: PlakaCozucu,
        n_workers: int = 2,
        kuyruk_boyutu: int = 0,
    ) -> None:
        self._workers: list[OcrWorker] = [
            OcrWorker(cozucu, kuyruk_boyutu=kuyruk_boyutu)
            for _ in range(max(1, n_workers))
        ]
        self._idx = 0
        self._kilit = threading.Lock()

    def baslat(self) -> None:
        for w in self._workers:
            w.start()

    def start(self) -> None:
        self.baslat()

    def gorevi_gonder(self, gorev: OcrGorevi) -> bool:
        with self._kilit:
            baslangic = self._idx
            self._idx += 1
        for offset in range(len(self._workers)):
            worker = self._workers[(baslangic + offset) % len(self._workers)]
            if worker.gorevi_gonder(gorev):
                return True
        return False

    def gorevi_gonder_bekle(self, gorev: OcrGorevi, timeout: float = 0.1) -> bool:
        with self._kilit:
            baslangic = self._idx
            self._idx += 1
        deneme_timeout = max(0.0, float(timeout)) / max(1, len(self._workers))
        for offset in range(len(self._workers)):
            worker = self._workers[(baslangic + offset) % len(self._workers)]
            if worker.gorevi_gonder_bekle(gorev, timeout=deneme_timeout):
                return True
        return False

    def sonuclari_topla(self) -> list:
        sonuclar = []
        for w in self._workers:
            sonuclar.extend(w.sonuclari_topla())
        return sonuclar

    def durdur(self) -> None:
        for w in self._workers:
            w.durdur()

    def join(self, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + max(0.0, float(timeout))
        for w in self._workers:
            kalan = max(0.0, deadline - time.monotonic())
            w.join(timeout=kalan)

    def is_alive(self) -> bool:
        return any(w.is_alive() for w in self._workers)
