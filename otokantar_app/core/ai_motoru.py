import queue
import re
import threading
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import easyocr
import numpy as np
import torch
from ultralytics import YOLO

from otokantar_app.config import CONFIG, PLAKA_REGEX, _HARF_DUZELTME, _RAKAM_DUZELTME
from otokantar_app.logger import log
from otokantar_app.models import OcrGorevi, TespitSonucu


class PlakaTespitci:
    _PLAKA_ETIKETLERI = {"plate", "license_plate", "licence_plate", "plaka", "number_plate"}

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
            log.info("Plaka modeli indiriliyor: %s", url)
            urllib.request.urlretrieve(url, path)
            log.info("Model kaydedildi: %s", path)
        return str(path)

    def plakalari_bul(self, bgr) -> list:
        sonuclar = self.model(bgr, verbose=False)[0]
        if sonuclar.boxes is None or len(sonuclar.boxes) == 0:
            return []
        names = sonuclar.names or {}
        cikti = []
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
            if not (CONFIG["ASPECT_RATIO_MIN"] <= ar <= CONFIG["ASPECT_RATIO_MAX"]):
                continue
            cikti.append((x1, y1, x2, y2, conf))
        cikti.sort(key=lambda t: t[4], reverse=True)
        return cikti


class PlakaCozucu:
    def __init__(self, diller: list, gpu: bool, min_conf: float):
        log.info("EasyOCR yükleniyor...")
        self.reader = easyocr.Reader(diller, gpu=gpu)
        self.min_conf = min_conf
        self._clahe = cv2.createCLAHE(
            clipLimit=CONFIG["CLAHE_CLIP"],
            tileGridSize=CONFIG["CLAHE_GRID"],
        )
        log.info("EasyOCR hazır.")

    @staticmethod
    def _sirala_dort_kose(pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).flatten()
        return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)

    def _gamma_bgr(self, bgr: np.ndarray) -> np.ndarray:
        gri = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        p95 = float(np.percentile(gri, 95))
        if p95 <= float(CONFIG["GAMMA_PARLAKLIK_ESIK"]):
            return bgr
        us = float(CONFIG["GAMMA_US"])
        table = ((np.arange(256, dtype=np.float64) / 255.0) ** us * 255.0).clip(0, 255).astype(np.uint8)
        return cv2.LUT(bgr, cv2.merge([table, table, table]))

    def _dortgen_kose_bul(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        h, w = bgr.shape[:2]
        if w < 8 or h < 8:
            return None
        gri = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gri, (5, 5), 0)
        kenar = cv2.Canny(blur, int(CONFIG["CANNY_ESIK1"]), int(CONFIG["CANNY_ESIK2"]))
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kenar = cv2.morphologyEx(kenar, cv2.MORPH_CLOSE, k, iterations=1)
        konturlar, _ = cv2.findContours(kenar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not konturlar:
            return None
        min_alan = float(CONFIG["PERSPEKTIF_MIN_ALAN_ORAN"]) * float(w * h)
        eps_c = float(CONFIG["PERSPEKTIF_EPSILON_CARPAN"])
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
        h, w = bgr.shape[:2]
        out_w = int(CONFIG["PERSPEKTIF_CIKTI_EN"])
        out_h = int(CONFIG["PERSPEKTIF_CIKTI_BOY"])
        kose = self._dortgen_kose_bul(bgr)
        if kose is None:
            src = np.array([[0.0, 0.0], [float(w), 0.0], [float(w), float(h)], [0.0, float(h)]], dtype=np.float32)
        else:
            src = kose
        dst = np.array([[0.0, 0.0], [float(out_w), 0.0], [float(out_w), float(out_h)], [0.0, float(out_h)]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(bgr, M, (out_w, out_h), flags=cv2.INTER_CUBIC)

    def _roi_hazirla(self, bgr) -> Optional[np.ndarray]:
        if bgr is None or bgr.size == 0:
            return None
        h, w = bgr.shape[:2]
        if h >= 30:
            oran = min(CONFIG["ALT_KIRP_ORAN"], 0.49)
            koru_min = 0.80
            efektif = min(oran, 1.0 - koru_min)
            bgr = bgr[: max(1, int(h * (1.0 - efektif))), :w]
        bgr = self._gamma_bgr(bgr)
        bgr = cv2.resize(
            bgr, None,
            fx=CONFIG["BOYUTLANDIRMA_KATSAYI"],
            fy=CONFIG["BOYUTLANDIRMA_KATSAYI"],
            interpolation=cv2.INTER_CUBIC,
        )
        bgr = self._perspektif_duzelt(bgr)
        gri = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        p95_ici = float(np.percentile(gri, 95))
        if p95_ici > int(CONFIG.get("BILATERAL_ESIK", 210)):
            gri = cv2.bilateralFilter(
                gri,
                d=int(CONFIG.get("BILATERAL_D", 7)),
                sigmaColor=float(CONFIG.get("BILATERAL_SIGMA_COLOR", 50)),
                sigmaSpace=float(CONFIG.get("BILATERAL_SIGMA_SPACE", 50)),
            )
        gri = self._clahe.apply(gri)
        blur = cv2.GaussianBlur(gri, (3, 3), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if CONFIG["MORPH_KAPAT"]:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, CONFIG["MORPH_KERNEL"])
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=1)
        return binary

    def _ocr_oku(self, binary: np.ndarray) -> tuple:
        sonuclar = self.reader.readtext(binary, allowlist=CONFIG["OCR_IZIN_LISTESI"], paragraph=False)
        if not sonuclar:
            return "", 0.0

        def sol_x(item):
            try:
                return min(pt[0] for pt in item[0])
            except Exception:
                return 0

        parcalar, confs = [], []
        for (_, metin, conf) in sorted(sonuclar, key=sol_x):
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
        il_d = "".join(_RAKAM_DUZELTME.get(c, c) for c in il)
        harf_d = "".join(_HARF_DUZELTME.get(c, c) for c in harf)
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
                return TespitSonucu(bbox=(), ham_metin=ham, plaka=plaka, guven=float(ocr_conf), gecerli=True)
        return TespitSonucu(bbox=(), ham_metin=ham, plaka=None, guven=0.0)

    def roi_hazirla_debug(self, bgr_roi) -> Optional[np.ndarray]:
        return self._roi_hazirla(bgr_roi)


class OcrWorker(threading.Thread):
    def __init__(self, cozucu: PlakaCozucu, kuyruk_boyutu: int = 4):
        super().__init__(name="OcrWorker", daemon=True)
        self._cozucu = cozucu
        self._giris_kuyrugu: queue.Queue = queue.Queue(maxsize=kuyruk_boyutu)
        self._cikis_kuyrugu: queue.Queue = queue.Queue(maxsize=kuyruk_boyutu * 2)
        self._dur = threading.Event()

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
