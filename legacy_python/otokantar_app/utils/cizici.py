from typing import Optional

import cv2


class EkranCizici:
    @staticmethod
    def plaka_kutusu(
        kare, bbox, plaka, sayac, esik,
        arac_id: Optional[int] = None,
        kara_liste: bool = False,
        renk=(255, 0, 255),
        kalin: int = 2,
        oy_lider: str = "",
        oy_lider_toplam: float = 0.0,
        oy_min_guven: float = 0.0,
    ):
        x1, y1, x2, y2 = bbox
        if kara_liste:
            renk = (0, 0, 255)
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
    def giris_yapildi(kare, bbox: tuple, plaka: str) -> None:
        x1, y1, x2, y2 = bbox
        renk = (0, 200, 0)
        cv2.rectangle(kare, (x1, y1), (x2, y2), renk, 4)
        cv2.putText(kare, f"GIRIS YAPILDI: {plaka}", (x1, y1 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.85, renk, 2)

    @staticmethod
    def cikis_yapildi(kare, bbox: tuple, plaka: str, net_kg: float) -> None:
        x1, y1, x2, y2 = bbox
        renk = (0, 220, 255)
        cv2.rectangle(kare, (x1, y1), (x2, y2), renk, 4)
        cv2.putText(kare, f"CIKIS YAPILDI - NET: {net_kg:.1f} kg  [{plaka}]", (x1, y1 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.85, renk, 2)

    @staticmethod
    def basarili_kayit(kare, bbox, plaka):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(kare, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.putText(kare, f"KAYDEDILDI: {plaka}", (x1, y1 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    @staticmethod
    def fps_ve_bilgi(kare, fps: float, toplam_kayit: int, yakalama_fps: Optional[float] = None):
        metin = (
            f"Kamera FPS: {yakalama_fps:.1f}  |  İşlem FPS: {fps:.1f}  |  Kayıt: {toplam_kayit}"
            if yakalama_fps is not None
            else f"FPS: {fps:.1f}  |  Toplam Kayıt: {toplam_kayit}"
        )
        cv2.putText(kare, metin, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
