import abc
import re
import threading
import time
from typing import Optional

import serial

from otokantar_app.config import CONFIG
from otokantar_app.logger import log


class SatirCozumleme(abc.ABC):
    @abc.abstractmethod
    def cozumle(self, satir: str) -> Optional[float]:
        ...

    @property
    @abc.abstractmethod
    def ad(self) -> str:
        ...


class ErmetProtokol(SatirCozumleme):
    _RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:kg|k)", re.IGNORECASE)

    @property
    def ad(self) -> str:
        return "ermet"

    def cozumle(self, satir: str) -> Optional[float]:
        temiz = satir.strip()
        if temiz.upper().startswith("ERR"):
            return None
        m = self._RE.search(temiz)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
        return None


class TolpaProtokol(SatirCozumleme):
    _RE = re.compile(r"(?:W(?:EIGHT)?)[=:]\s*(\d+(?:\.\d+)?)", re.IGNORECASE)

    @property
    def ad(self) -> str:
        return "tolpa"

    def cozumle(self, satir: str) -> Optional[float]:
        m = self._RE.search(satir)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
        return None


_PROTOKOL_FABRIKASI: dict[str, type[SatirCozumleme]] = {
    "ermet": ErmetProtokol,
    "tolpa": TolpaProtokol,
}


def protokol_olustur(ad: str) -> SatirCozumleme:
    sinif = _PROTOKOL_FABRIKASI.get(ad.lower(), ErmetProtokol)
    nesne = sinif()
    log.info("Kantar protokolü seçildi: %s (%s)", nesne.ad, sinif.__name__)
    return nesne


class KantarOkuyucu(threading.Thread):
    YENIDEN_BAGLANTI_BEKLEME = 5.0
    DEBOUNCE_SURE = 3.0
    DEBOUNCE_TOLERANS = 20.0

    def __init__(self, protokol: Optional[SatirCozumleme] = None):
        super().__init__(name="KantarOkuyucu", daemon=True)
        self._protokol = protokol or protokol_olustur(CONFIG.get("KANTAR_PROTOKOL", "ermet"))
        self.guncel_agirlik: float = 0.0
        self._simule_agirlik: float = 0.0
        self.agirlik_sabit_mi: bool = False
        self._kilit = threading.Lock()
        self._dur = threading.Event()
        self._debounce_ref: float = 0.0
        self._debounce_bas: float = 0.0

    def _satiri_cozumle(self, satir: str) -> Optional[float]:
        return self._protokol.cozumle(satir)

    def _debounce_guncelle(self, yeni: float) -> None:
        simdi = time.time()
        if abs(yeni - self._debounce_ref) > self.DEBOUNCE_TOLERANS:
            self._debounce_ref = yeni
            self._debounce_bas = simdi
            sabit = False
        else:
            sabit = (simdi - self._debounce_bas) >= self.DEBOUNCE_SURE
        with self._kilit:
            self.guncel_agirlik = yeni
            self.agirlik_sabit_mi = sabit

    def simule_et(self, kg: float) -> None:
        with self._kilit:
            self._simule_agirlik = float(kg)

    def run(self):
        log.info(
            "KantarOkuyucu başlatıldı → port=%s baud=%s protokol=%s debounce=%.0fsn/±%.0fkg",
            CONFIG["KANTAR_PORT"], CONFIG["KANTAR_BAUD"], self._protokol.ad,
            self.DEBOUNCE_SURE, self.DEBOUNCE_TOLERANS,
        )
        if CONFIG.get("SIMULASYON_MODU"):
            log.info("KantarOkuyucu simülasyon modunda çalışıyor.")
            while not self._dur.is_set():
                self._debounce_guncelle(self._simule_agirlik)
                time.sleep(0.5)
            log.info("KantarOkuyucu durduruldu.")
            return
        while not self._dur.is_set():
            port = None
            try:
                port = serial.Serial(
                    port=CONFIG["KANTAR_PORT"],
                    baudrate=int(CONFIG["KANTAR_BAUD"]),
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=1.0,
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
        with self._kilit:
            return (self.guncel_agirlik, self.agirlik_sabit_mi)
