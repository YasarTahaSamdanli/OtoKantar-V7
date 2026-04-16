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

import queue
import re
import threading
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

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
    "PREP_BILATERAL": True,
    "PREP_CLAHE": True,
    "PREP_SHARPEN": True,
    "PREP_ADAPTIVE": True,
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
    # --- adaptive threshold ---
    "ADAPTIVE_BLOCK": 15,
    "ADAPTIVE_C": 8,
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
            cikti.append((x1, y1, x2, y2, conf))
        cikti.sort(key=lambda t: t[4], reverse=True)
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
        self._reader = PaddleOCR(
            use_angle_cls=False,
            lang="en",
            use_gpu=gpu and torch.cuda.is_available(),
            show_log=False,
        )
        log.info("PaddleOCR hazır (gpu=%s)", gpu)

    def oku(self, gray: np.ndarray) -> tuple[str, float]:
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        sonuc = self._reader.ocr(bgr, cls=False)
        if not sonuc or not sonuc[0]:
            return "", 0.0
        parcalar, confs = [], []
        min_conf = float(_cfg("OCR_MIN_CONF"))
        for line in sonuc[0]:
            metin, conf = line[1]
            if conf < min_conf:
                continue
            temiz = re.sub(r"[^A-Z0-9]", "", str(metin).upper())
            if temiz:
                parcalar.append(temiz)
                confs.append(float(conf))
        if not parcalar:
            return "", 0.0
        return "".join(parcalar), sum(confs) / len(confs)


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
        self._clahe = cv2.createCLAHE(
            clipLimit=float(_cfg("CLAHE_CLIP")),
            tileGridSize=tuple(_cfg("CLAHE_GRID")),
        )

        # Build primary and fallback backends
        # Primary: PaddleOCR (best accuracy)
        # Fallback: EasyOCR (only triggered when primary is insufficient)
        if _PADDLE_AVAILABLE:
            self._primary: _OcrBackend = _PaddleBackend(gpu)
        elif _EASY_AVAILABLE:
            self._primary = _EasyBackend(diller, gpu)
        else:
            raise RuntimeError("Hiçbir OCR backend bulunamadı (PaddleOCR veya EasyOCR gerekli).")

        self._fallback: Optional[_OcrBackend] = None
        if _PADDLE_AVAILABLE and _EASY_AVAILABLE:
            self._fallback = _EasyBackend(diller, gpu)

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
          1.  Bottom-strip crop (remove dirt / bolt shadows)
          2.  2× resize
          3.  Gamma correction  (dark images only)
          4.  Perspective correction  (optional)
          5.  Super-resolution hook  (optional, off by default)
          6.  Convert to gray
          7.  Bilateral filter  (bright/reflective plates)
          8.  Deskew  (optional)
          9.  CLAHE
          10. Gaussian blur
          11. Sharpening  (unsharp mask)
          12. Dual thresholding  (Otsu vs Adaptive — best chosen)
          13. Morphological closing
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

        # 7. Bilateral filter (only on bright/reflective plates)
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

        # 10. Gaussian blur
        blur = cv2.GaussianBlur(gri, (3, 3), 0)

        # 11. Sharpen
        if _cfg("PREP_SHARPEN"):
            blur = _sharpen(blur)

        # 12. Dual thresholding — pick version with more foreground detail
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
            binary = adaptive if adaptive.mean() > otsu.mean() else otsu

        # 13. Morphological closing
        if _cfg("MORPH_KAPAT"):
            k = cv2.getStructuringElement(cv2.MORPH_RECT, tuple(_cfg("MORPH_KERNEL")))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=1)

        return binary

    # ------------------------------------------------------------------
    # Adaptive two-pass OCR
    # ------------------------------------------------------------------

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

        # --- Pass 1: original ---
        metin1, conf1 = self._primary.oku(binary)

        # Early exit: high-confidence valid plate on first pass
        if conf1 >= ikinci_esik and PLAKA_REGEX.search(metin1):
            return metin1, conf1

        # --- Pass 2: inverted (only when needed) ---
        inverted = cv2.bitwise_not(binary)
        metin2, conf2 = self._primary.oku(inverted)

        # Pick the better of the two primary passes
        def _skor(m: str, c: float) -> float:
            return c + (1.0 if PLAKA_REGEX.search(m) else 0.0)

        if _skor(metin2, conf2) > _skor(metin1, conf1):
            best_metin, best_conf = metin2, conf2
        else:
            best_metin, best_conf = metin1, conf1

        # --- Fallback backend (only if primary result is still insufficient) ---
        if self._fallback is not None and best_conf < min_conf:
            fb_metin, fb_conf = self._fallback.oku(
                binary if conf1 >= conf2 else inverted
            )
            if _skor(fb_metin, fb_conf) > _skor(best_metin, best_conf):
                best_metin, best_conf = fb_metin, fb_conf

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

    def coz(self, bgr_roi: np.ndarray) -> TespitSonucu:
        """
        Full pipeline: preprocess → adaptive OCR → validate → correct.

        Returns:
            TespitSonucu with raw plate text and confidence.
            Decision (accept/reject) is made downstream in dogrulama.py.
        """
        binary = self._roi_hazirla(bgr_roi)
        if binary is None:
            return TespitSonucu(bbox=(), ham_metin="", plaka=None, guven=0.0)

        ham, ocr_conf = self._ocr_calistir(binary)
        eslesen = PLAKA_REGEX.search(ham)
        if eslesen:
            il, harf, rakam = eslesen.group(1), eslesen.group(2), eslesen.group(3)
            plaka = self._plaka_duzelt(il, harf, rakam)
            if self._uzunluk_gecerli(plaka):
                return TespitSonucu(
                    bbox=(),
                    ham_metin=ham,
                    plaka=plaka,
                    guven=float(ocr_conf),
                    gecerli=True,
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
        self._cikis_kuyrugu: queue.Queue = queue.Queue(maxsize=boyut * 2)
        self._dur = threading.Event()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def gorevi_gonder(self, gorev: OcrGorevi) -> bool:
        """Enqueue a task. Returns False if the queue is full (back-pressure)."""
        try:
            self._giris_kuyrugu.put_nowait(gorev)
            return True
        except queue.Full:
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
        sonuc = self._cozucu.coz(gorev.roi_bgr)
        try:
            self._cikis_kuyrugu.put_nowait((gorev.arac_id, sonuc, gorev.yolo_conf, gorev.bbox))
        except queue.Full:
            log.warning("OcrWorker çıkış kuyruğu dolu, sonuç atıldı.")

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

    def gorevi_gonder(self, gorev: OcrGorevi) -> bool:
        with self._kilit:
            worker = self._workers[self._idx % len(self._workers)]
            self._idx += 1
        return worker.gorevi_gonder(gorev)

    def sonuclari_topla(self) -> list:
        sonuclar = []
        for w in self._workers:
            sonuclar.extend(w.sonuclari_topla())
        return sonuclar

    def durdur(self) -> None:
        for w in self._workers:
            w.durdur()
        for w in self._workers:
            w.join(timeout=5.0)