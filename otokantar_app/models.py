import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class PlakaKayit:
    plaka: str
    giris_tarih: str
    giris_saat: str
    giris_agirlik: float
    guven: float = 0.0
    cikis_tarih: Optional[str] = None
    cikis_saat: Optional[str] = None
    cikis_agirlik: Optional[float] = None
    net_agirlik: Optional[float] = None
    durum: str = "ICERIDE"
    operator: str = "AUTO"


@dataclass
class TespitSonucu:
    bbox: tuple
    ham_metin: str
    plaka: Optional[str]
    guven: float
    gecerli: bool = False


@dataclass
class DogrulamaDurumu:
    oylar: dict = field(default_factory=dict)
    hane: dict = field(default_factory=dict)
    okuma_sayisi: int = 0
    son_gorulme: float = field(default_factory=time.time)
    son_kayit: float = 0.0


@dataclass
class PlakaBuffer:
    plaka: str
    guven: float
    yolo_conf: float
    zaman: float = field(default_factory=time.time)

    def suresi_doldu_mu(self, ttl: float) -> bool:
        return (time.time() - self.zaman) > ttl

    def guncelle_eger_daha_iyi(self, plaka: str, guven: float, yolo_conf: float) -> bool:
        yeni_skor = 0.6 * guven + 0.4 * yolo_conf
        eski_skor = 0.6 * self.guven + 0.4 * self.yolo_conf
        if yeni_skor > eski_skor:
            self.plaka = plaka
            self.guven = guven
            self.yolo_conf = yolo_conf
            self.zaman = time.time()
            return True
        return False

    def guncule_eger_daha_iyi(self, plaka: str, guven: float, yolo_conf: float) -> bool:
        return self.guncelle_eger_daha_iyi(plaka, guven, yolo_conf)


@dataclass
class OcrGorevi:
    roi_bgr: np.ndarray
    arac_id: int
    yolo_conf: float
    bbox: tuple
