from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from otokantar_app.config import CONFIG
from otokantar_app.core.ai_motoru import PlakaCozucu, PlakaTespitci
from otokantar_app.core.dogrulama import DogrulamaMotoru


def _iter_gorseller(girdi: Path) -> Iterable[Path]:
    if girdi.is_file():
        yield girdi
        return

    if not girdi.is_dir():
        return

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    for p in sorted(girdi.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _roi_kirp(bgr: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    x1i = max(0, min(w - 1, int(round(x1))))
    y1i = max(0, min(h - 1, int(round(y1))))
    x2i = max(0, min(w, int(round(x2))))
    y2i = max(0, min(h, int(round(y2))))
    if x2i <= x1i or y2i <= y1i:
        return bgr
    return bgr[y1i:y2i, x1i:x2i]


def _fmt_bbox(b: tuple[float, float, float, float]) -> str:
    x1, y1, x2, y2 = b
    return f"({x1:.1f},{y1:.1f})-({x2:.1f},{y2:.1f})"


def _gorsel_isle(
    img_path: Path,
    tespitci: PlakaTespitci,
    cozucu: PlakaCozucu,
    dogrulama: DogrulamaMotoru,
    debug: bool,
) -> None:
    print("\n" + "=" * 80)
    print(f"[GIRDI] {img_path}")

    bgr = cv2.imread(str(img_path))
    if bgr is None:
        print("[HATA] Görsel okunamadı.")
        return

    # 1) YOLO Tespiti
    bboxes = tespitci.plakalari_bul(bgr)
    if not bboxes:
        h, w = bgr.shape[:2]
        bboxes = [(0.0, 0.0, float(w), float(h), 1.0)]
        print("[YOLO] Plaka bulunamadı → tüm görsel ROI kabul edildi (manuel devam).")
    else:
        print(f"[YOLO] {len(bboxes)} aday bbox bulundu.")

    roiler: list[tuple[np.ndarray, tuple[float, float, float, float], float]] = []
    for i, (x1, y1, x2, y2, conf) in enumerate(bboxes, start=1):
        bbox = (x1, y1, x2, y2)
        roi = _roi_kirp(bgr, bbox)
        roiler.append((roi, bbox, float(conf)))
        print(f"  - bbox#{i} conf={conf:.3f} bbox={_fmt_bbox(bbox)} roi_shape={roi.shape}")

    # 2) OCR Testi
    print("\n[OCR]")
    ocr_sonuclari = []
    for i, (roi, bbox, yolo_conf) in enumerate(roiler, start=1):
        if debug:
            bw = cozucu.roi_hazirla_debug(roi)
            if bw is not None:
                out = img_path.parent / f"{img_path.stem}__roi{i:02d}_bw.png"
                try:
                    cv2.imwrite(str(out), bw)
                    print(f"  - debug ROI kaydedildi: {out.name}")
                except Exception as exc:
                    print(f"  - debug ROI kaydedilemedi: {type(exc).__name__}: {exc}")

        sonuc = cozucu.coz(roi)
        ham = (sonuc.ham_metin or "").strip()
        normalize = dogrulama._normalize(sonuc.plaka or ham)  # noqa: SLF001 (project-local usage)
        guven = float(getattr(sonuc, "guven", 0.0) or 0.0)
        print(
            f"  - roi#{i:02d} yolo={yolo_conf:.3f} bbox={_fmt_bbox(bbox)} | "
            f"ham='{ham}' normalize='{normalize}' guven={guven:.3f}"
        )
        ocr_sonuclari.append((ham, normalize, guven))

    # 3) Doğrulama Simülasyonu
    secili = next((n for (_, n, g) in ocr_sonuclari if dogrulama._tr_plaka_gecerli_mi(n) and g > 0), None)  # noqa: SLF001
    if not secili:
        print("\n[DOGRULAMA] Geçerli plaka bulunamadı → simülasyon atlandı.")
        return

    base_guven = next((g for (_, n, g) in ocr_sonuclari if n == secili), 0.0)
    print(f"\n[DOGRULAMA] Simülasyon plakası: {secili} (base_guven={base_guven:.3f})")

    arac_id = 1
    for k in range(1, 8):
        noise = float(np.random.uniform(-0.1, 0.1))
        guven_noisy = float(np.clip(base_guven + noise, 0.0, 1.0))
        onay, lider, toplam_guven, toplam_hane = dogrulama.isle(arac_id, secili, guven_noisy)
        print(
            f"  - kare#{k} guven={guven_noisy:.3f} (noise={noise:+.3f}) → "
            f"onay={bool(onay)} lider={lider} toplam_guven={toplam_guven:.3f} toplam_hane={toplam_hane}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Canlı kamera olmadan plaka tespiti/OCR/doğrulama test betiği.",
    )
    parser.add_argument(
        "girdi",
        help="Tek görsel dosyası veya klasör yolu (örn: test.jpg / ./ornekler/)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="ROI siyah-beyaz çıktısını PNG olarak aynı dizine kaydet.",
    )
    args = parser.parse_args()

    girdi = Path(args.girdi)
    if not girdi.exists():
        print(f"[HATA] Yol bulunamadı: {girdi}")
        return 2

    # Engine init (project config)
    tespitci = PlakaTespitci(
        weights_url=str(CONFIG.get("PLATE_WEIGHTS_URL")),
        models_dir=str(CONFIG.get("MODELS_DIR", "models")),
        conf=float(CONFIG.get("YOLO_CONF", 0.25)),
        gpu=bool(CONFIG.get("OCR_GPU", False)),
    )
    cozucu = PlakaCozucu(
        diller=list(CONFIG.get("OCR_DILLER", ["tr", "en"])),
        gpu=bool(CONFIG.get("OCR_GPU", False)),
        min_conf=float(CONFIG.get("MIN_OCR_CONF", 0.35)),
    )
    dogrulama = DogrulamaMotoru(
        esik=int(CONFIG.get("ESIK_DEGERI", 4)),
        min_toplam_guven=float(CONFIG.get("OYLAMA_MIN_TOPLAM_GUVEN", 2.5)),
        kayit_sonrasi_bekleme=float(CONFIG.get("BEKLEME_SURESI_SONRA", 120.0)),
    )
    dogrulama.bilinen_plakalar = dogrulama.hazirla_bilinen_plakalar(CONFIG.get("KARA_LISTE", []))

    paths = list(_iter_gorseller(girdi))
    if not paths:
        print("[HATA] İşlenecek görsel bulunamadı.")
        return 3

    for p in paths:
        _gorsel_isle(
            img_path=p,
            tespitci=tespitci,
            cozucu=cozucu,
            dogrulama=dogrulama,
            debug=bool(args.debug),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

