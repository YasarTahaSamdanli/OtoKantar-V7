import csv
import json
import re
import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from otokantar_app.config import CONFIG, _HARF_DUZELTME, _RAKAM_DUZELTME
from otokantar_app.db.mysql_manager import MySQLDBManager
from otokantar_app.logger import log
from otokantar_app.models import PlakaKayit


class KantarKaydedici:
    _PLAKA_REGEX = re.compile(r"^(0[1-9]|[1-7][0-9]|8[0-1])[A-Z]{1,3}\d{2,4}$")
    _ALNUM_DISI = re.compile(r"[^A-Z0-9]+")
    _TR_HARF_MAP = str.maketrans({
        "C": "C",
        "G": "G",
        "I": "I",
        "O": "O",
        "S": "S",
        "U": "U",
        "Ç": "C",
        "Ğ": "G",
        "İ": "I",
        "Ö": "O",
        "Ş": "S",
        "Ü": "U",
    })

    def __init__(self, csv_dosya: str, json_dosya: str, mysql_db=None):
        self.csv_dosya = csv_dosya
        self.json_dosya = json_dosya
        self.son_kayitlar: list = []
        self._kilit = threading.Lock()
        self._csv_aktif = True
        self._acik_seanslar: dict[str, dict] = {}
        self.mysql = mysql_db
        self._csv_baslik_yaz()
        self._csvden_durum_yukle()
        log.info("KantarKaydedici %s modu aktif.", "MySQL" if self.mysql is not None else "CSV")

    def _plaka_temizle(self, plaka: str) -> str:
        plaka = (plaka or "").upper().translate(self._TR_HARF_MAP)
        return self._ALNUM_DISI.sub("", plaka)

    def _plaka_harf_blok_duzelt(self, metin: str) -> str:
        return "".join(_HARF_DUZELTME.get(ch, ch) for ch in metin)

    def _plaka_rakam_blok_duzelt(self, metin: str) -> str:
        return "".join(_RAKAM_DUZELTME.get(ch, ch) for ch in metin)

    def _plaka_normalize(self, plaka: str) -> str:
        ham = self._plaka_temizle(plaka)
        if self._PLAKA_REGEX.match(ham):
            return ham

        adaylar = []
        for harf_uzunlugu in range(1, 4):
            rakam_uzunlugu = len(ham) - 2 - harf_uzunlugu
            if rakam_uzunlugu < 2 or rakam_uzunlugu > 4:
                continue

            il_kodu_ham = ham[:2]
            harf_ham = ham[2:2 + harf_uzunlugu]
            rakam_ham = ham[2 + harf_uzunlugu:]
            aday = (
                f"{self._plaka_rakam_blok_duzelt(il_kodu_ham)}"
                f"{self._plaka_harf_blok_duzelt(harf_ham)}"
                f"{self._plaka_rakam_blok_duzelt(rakam_ham)}"
            )
            if not self._PLAKA_REGEX.match(aday):
                continue
            degisim_sayisi = sum(1 for once, sonra in zip(ham, aday) if once != sonra)
            adaylar.append((degisim_sayisi, harf_uzunlugu, aday))

        if not adaylar:
            return ham

        adaylar.sort(key=lambda item: (item[0], item[1] != 2, item[1] != 3, item[1]))
        return adaylar[0][2]

    def plaka_kara_listede_mi(self, plaka: str) -> bool:
        if self.mysql is None:
            return self._plaka_normalize(plaka) in {
                self._plaka_normalize(p) for p in CONFIG.get("KARA_LISTE", [])
            }
        try:
            return self.mysql.kara_listede_mi(plaka)
        except Exception as e:
            log.warning("Kara liste sorgulanırken hata: %s", e)
            return False
    def acik_seans_getir(self, plaka: str) -> Optional[dict]:
        return self._acik_seanslar.get(self._plaka_normalize(plaka))

    def giris_kaydet(self, plaka: str, agirlik: float) -> PlakaKayit:
        plaka = self._plaka_normalize(plaka)
        simdi = datetime.now()

        kayit = PlakaKayit(
            plaka=plaka,
            giris_tarih=simdi.strftime("%Y-%m-%d"),
            giris_saat=simdi.strftime("%H:%M:%S"),
            giris_agirlik=float(agirlik),
            durum="ICERIDE",
        )
        self.gecis_kaydet(kayit)
        self._acik_seanslar[plaka] = {
            "plaka": plaka,
            "giris_tarih": kayit.giris_tarih,
            "giris_saat": kayit.giris_saat,
            "giris_agirlik": kayit.giris_agirlik,
            "guven": kayit.guven,
        }
        return kayit

    def cikis_kaydet(self, plaka: str, agirlik: float) -> Optional[PlakaKayit]:
        plaka = self._plaka_normalize(plaka)
        simdi = datetime.now()
        acik = self.acik_seans_getir(plaka)
        if acik is None:
            log.warning("Çıkış kaydı atlandı: açık seans yok (%s)", plaka)
            return None
        giris_agirlik = float(acik.get("giris_agirlik") or 0.0)
        cikis_agirlik = float(agirlik)
        kayit = PlakaKayit(
            plaka=plaka,
            giris_tarih=str(acik.get("giris_tarih") or simdi.strftime("%Y-%m-%d")),
            giris_saat=str(acik.get("giris_saat") or simdi.strftime("%H:%M:%S")),
            giris_agirlik=giris_agirlik,
            guven=float(acik.get("guven") or 0.0),
            cikis_tarih=simdi.strftime("%Y-%m-%d"),
            cikis_saat=simdi.strftime("%H:%M:%S"),
            cikis_agirlik=cikis_agirlik,
            net_agirlik=abs(giris_agirlik - cikis_agirlik),
            durum="TAMAMLANDI",
            firma_adi=acik.get("firma_adi"),
            sofor_adi=acik.get("sofor_adi"),
            sofor_tel=acik.get("sofor_tel"),
            malzeme_cinsi=acik.get("malzeme_cinsi"),
            irsaliye_no=acik.get("irsaliye_no"),
        )
        self.gecis_kaydet(kayit)
        self._acik_seanslar.pop(plaka, None)
        return kayit

    def kapat(self) -> None:
        log.info("KantarKaydedici kapatıldı.")

    # ------------------------------------------------------------------
    # CSV / JSON
    # ------------------------------------------------------------------
    def _csv_baslik_yaz(self):
        dosya_var = Path(self.csv_dosya).exists()
        try:
            with open(self.csv_dosya, mode="a", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f, delimiter=";")
                if not dosya_var:
                    w.writerow([
                        "Plaka", "Durum", "GirisTarih", "GirisSaat", "GirisAgirlik(kg)",
                        "CikisTarih", "CikisSaat", "CikisAgirlik(kg)", "NetAgirlik(kg)",
                        "Guven", "Operator", "FirmaAdi", "SoforAdi", "SoforTel",
                        "MalzemeCinsi", "IrsaliyeNo",
                    ])
        except PermissionError as e:
            self._csv_aktif = False
            log.warning("CSV erişilemedi, devre dışı: %s", e)

    def _csvden_durum_yukle(self) -> None:
        path = Path(self.csv_dosya)
        if not path.exists():
            return

        acik_seanslar: dict[str, dict] = {}
        son_kayitlar: list[PlakaKayit] = []
        try:
            with open(path, mode="r", newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f, delimiter=";")
                for row in reader:
                    kayit = self._csv_satirindan_kayit(row)
                    if kayit is None:
                        continue

                    son_kayitlar.append(kayit)
                    plaka = self._plaka_normalize(kayit.plaka)
                    if kayit.durum == "ICERIDE":
                        acik_seanslar[plaka] = {
                            "plaka": plaka,
                            "giris_tarih": kayit.giris_tarih,
                            "giris_saat": kayit.giris_saat,
                            "giris_agirlik": kayit.giris_agirlik,
                            "guven": kayit.guven,
                            "firma_adi": kayit.firma_adi,
                            "sofor_adi": kayit.sofor_adi,
                            "sofor_tel": kayit.sofor_tel,
                            "malzeme_cinsi": kayit.malzeme_cinsi,
                            "irsaliye_no": kayit.irsaliye_no,
                        }
                    elif kayit.durum == "TAMAMLANDI":
                        acik_seanslar.pop(plaka, None)
        except Exception as e:
            log.warning("CSV durum restore edilemedi: %s", e)
            return

        self._acik_seanslar = acik_seanslar
        self.son_kayitlar = son_kayitlar[-50:]
        if self._acik_seanslar:
            log.warning(
                "Acik kantar seansi CSV'den restore edildi: %s",
                ", ".join(sorted(self._acik_seanslar)),
            )
        if self.son_kayitlar:
            log.info("Son %d kantar kaydi CSV'den yuklendi.", len(self.son_kayitlar))

    def _csv_satirindan_kayit(self, row: dict) -> Optional[PlakaKayit]:
        plaka = self._plaka_normalize(str(row.get("Plaka") or ""))
        if not plaka:
            return None

        durum = str(row.get("Durum") or "ICERIDE").strip().upper()
        giris_agirlik = self._float_oku(row.get("GirisAgirlik(kg)"), 0.0)
        cikis_agirlik = self._float_oku(row.get("CikisAgirlik(kg)"))
        net_agirlik = self._float_oku(row.get("NetAgirlik(kg)"))

        return PlakaKayit(
            plaka=plaka,
            giris_tarih=str(row.get("GirisTarih") or ""),
            giris_saat=str(row.get("GirisSaat") or ""),
            giris_agirlik=giris_agirlik if giris_agirlik is not None else 0.0,
            guven=self._float_oku(row.get("Guven"), 0.0) or 0.0,
            cikis_tarih=self._bos_ise_none(row.get("CikisTarih")),
            cikis_saat=self._bos_ise_none(row.get("CikisSaat")),
            cikis_agirlik=cikis_agirlik,
            net_agirlik=net_agirlik,
            durum=durum,
            operator=str(row.get("Operator") or "AUTO"),
            firma_adi=self._bos_ise_none(row.get("FirmaAdi")),
            sofor_adi=self._bos_ise_none(row.get("SoforAdi")),
            sofor_tel=self._bos_ise_none(row.get("SoforTel")),
            malzeme_cinsi=self._bos_ise_none(row.get("MalzemeCinsi")),
            irsaliye_no=self._bos_ise_none(row.get("IrsaliyeNo")),
        )

    @staticmethod
    def _bos_ise_none(value) -> Optional[str]:
        text = "" if value is None else str(value).strip()
        return text or None

    @staticmethod
    def _float_oku(value, default: Optional[float] = None) -> Optional[float]:
        text = "" if value is None else str(value).strip().replace(",", ".")
        if not text:
            return default
        try:
            return float(text)
        except ValueError:
            return default

    def gecis_kaydet(self, kayit: PlakaKayit):
        with self._kilit:
            if self._csv_aktif:
                try:
                    with open(self.csv_dosya, mode="a", newline="", encoding="utf-8-sig") as f:
                        w = csv.writer(f, delimiter=";")
                        w.writerow([
                            kayit.plaka, kayit.durum, kayit.giris_tarih, kayit.giris_saat,
                            f"{kayit.giris_agirlik:.1f}", kayit.cikis_tarih or "", kayit.cikis_saat or "",
                            "" if kayit.cikis_agirlik is None else f"{kayit.cikis_agirlik:.1f}",
                            "" if kayit.net_agirlik is None else f"{kayit.net_agirlik:.1f}",
                            f"{kayit.guven:.2f}", kayit.operator,
                            kayit.firma_adi or "", kayit.sofor_adi or "", kayit.sofor_tel or "",
                            kayit.malzeme_cinsi or "", kayit.irsaliye_no or "",
                        ])
                except PermissionError as e:
                    self._csv_aktif = False
                    log.warning("CSV yazılamadı: %s", e)
            self.son_kayitlar.append(kayit)
            if len(self.son_kayitlar) > 50:
                self.son_kayitlar = self.son_kayitlar[-50:]
            self._json_guncelle(kayit)
            log.info(
                "KAYIT: %s | %s | giris=%.1fkg | cikis=%s | net=%s | firma=%s",
                kayit.plaka, kayit.durum, kayit.giris_agirlik,
                "-" if kayit.cikis_agirlik is None else f"{kayit.cikis_agirlik:.1f}kg",
                "-" if kayit.net_agirlik is None else f"{kayit.net_agirlik:.1f}kg",
                kayit.firma_adi or "-",
            )

    def kaydet(self, kayit: PlakaKayit):
        self.gecis_kaydet(kayit)

    def _json_guncelle(self, son_kayit: PlakaKayit):
        try:
            with open(self.json_dosya, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "son_guncelleme": datetime.now().isoformat(),
                        "son_kayit": asdict(son_kayit),
                        "son_10": [asdict(k) for k in self.son_kayitlar[-10:]],
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception as e:
            log.warning("JSON güncellenemedi: %s", e)
