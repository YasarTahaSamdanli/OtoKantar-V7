import re
import time

from otokantar_app.config import _HARF_DUZELTME, _RAKAM_DUZELTME
from otokantar_app.models import DogrulamaDurumu


class DogrulamaMotoru:
    TR_PLAKA_REGEX = re.compile(r"^(0[1-9]|[1-7][0-9]|8[0-1])[A-Z]{1,3}\d{2,4}$")
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

    def __init__(self, esik: int, min_toplam_guven: float, kayit_sonrasi_bekleme: float):
        self.esik = int(esik)
        self.min_toplam_guven = float(min_toplam_guven)
        self.bekleme = float(kayit_sonrasi_bekleme)
        self._durum: dict = {}
        self.bilinen_plakalar = set()

    def _temizle(self, plaka: str) -> str:
        plaka = (plaka or "").upper().translate(self._TR_HARF_MAP)
        return self._ALNUM_DISI.sub("", plaka)

    def _harf_blok_duzelt(self, metin: str) -> str:
        return "".join(_HARF_DUZELTME.get(ch, ch) for ch in metin)

    def _rakam_blok_duzelt(self, metin: str) -> str:
        return "".join(_RAKAM_DUZELTME.get(ch, ch) for ch in metin)

    def _normalize(self, plaka: str) -> str:
        ham = self._temizle(plaka)
        if self._tr_plaka_gecerli_mi(ham):
            return ham

        if len(ham) < 5:
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
                f"{self._rakam_blok_duzelt(il_kodu_ham)}"
                f"{self._harf_blok_duzelt(harf_ham)}"
                f"{self._rakam_blok_duzelt(rakam_ham)}"
            )

            if not self._tr_plaka_gecerli_mi(aday):
                continue

            degisim_sayisi = sum(1 for once, sonra in zip(ham, aday) if once != sonra)
            adaylar.append((degisim_sayisi, harf_uzunlugu, aday))

        if not adaylar:
            return ham

        adaylar.sort(key=lambda item: (item[0], item[1] != 2, item[1] != 3, item[1]))
        return adaylar[0][2]

    def hazirla_bilinen_plakalar(self, plakalar) -> set[str]:
        temiz_plakalar = set()
        for plaka in plakalar or []:
            aday = self._normalize(plaka)
            if self._tr_plaka_gecerli_mi(aday):
                temiz_plakalar.add(aday)
        return temiz_plakalar

    def _tr_plaka_gecerli_mi(self, plaka: str) -> bool:
        return bool(self.TR_PLAKA_REGEX.match(plaka))

    def _mesafe_hesapla(self, s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return self._mesafe_hesapla(s2, s1)
        if len(s2) == 0:
            return len(s1)
        onceki_satir = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            guncel_satir = [i + 1]
            for j, c2 in enumerate(s2):
                ekleme = onceki_satir[j + 1] + 1
                silme = guncel_satir[j] + 1
                degistirme = onceki_satir[j] + (c1 != c2)
                guncel_satir.append(min(ekleme, silme, degistirme))
            onceki_satir = guncel_satir
        return onceki_satir[-1]

    def _benzer_mi(self, p1: str, p2: str) -> bool:
        if not (self._tr_plaka_gecerli_mi(p1) and self._tr_plaka_gecerli_mi(p2)):
            return False

        if abs(len(p1) - len(p2)) > 1:
            return False

        return self._mesafe_hesapla(p1, p2) <= 2

    def _cluster(self, oylar: dict, hane: dict):
        kumeler = []

        for plaka in oylar:
            bulundu = False
            for kume in kumeler:
                if any(self._benzer_mi(plaka, uye) for uye in kume["uyeler"]):
                    kume["uyeler"].append(plaka)
                    bulundu = True
                    break

            if not bulundu:
                kumeler.append({
                    "merkez": plaka,
                    "uyeler": [plaka],
                })

        en_iyi = None
        en_skor = -1

        for kume in kumeler:
            toplam_guven = sum(oylar[p] for p in kume["uyeler"])
            toplam_hane = sum(hane[p] for p in kume["uyeler"])

            lider = max(
                kume["uyeler"],
                key=lambda p: (
                    p in self.bilinen_plakalar,
                    oylar[p],
                    hane[p],
                ),
            )

            skor = toplam_guven + (toplam_hane * 0.2)
            if lider in self.bilinen_plakalar:
                skor += 2.0

            if skor > en_skor:
                en_skor = skor
                en_iyi = (lider, toplam_guven, toplam_hane)

        return en_iyi

    def _oto_duzelt(self, plaka: str) -> str:
        if not self.bilinen_plakalar or plaka in self.bilinen_plakalar:
            return plaka

        best_aday = plaka
        best_score = 999

        for kayitli in self.bilinen_plakalar:
            dist = self._mesafe_hesapla(plaka, kayitli)
            if dist < best_score:
                best_score = dist
                best_aday = kayitli

        if best_score <= 1:
            return best_aday

        return plaka

    def isle(self, arac_id: int, plaka: str, guven: float) -> tuple:
        su_an = time.time()

        d = self._durum.get(arac_id)
        if d is None:
            d = DogrulamaDurumu()
            self._durum[arac_id] = d

        if su_an - d.son_kayit < self.bekleme:
            return (False, None, 0.0, 0)

        if su_an - d.son_gorulme > 5.0:
            d.oylar.clear()
            d.hane.clear()
            d.okuma_sayisi = 0

        d.son_gorulme = su_an
        d.okuma_sayisi += 1

        if guven < 0.45:
            return (False, None, 0.0, 0)

        plaka = self._normalize(plaka)
        if not self._tr_plaka_gecerli_mi(plaka):
            return (False, None, 0.0, 0)

        plaka = self._oto_duzelt(plaka)

        g = min(max(float(guven), 0.3), 0.95)

        d.oylar[plaka] = d.oylar.get(plaka, 0.0) + g
        d.hane[plaka] = d.hane.get(plaka, 0) + 1

        lider, toplam_guven, toplam_hane = self._cluster(d.oylar, d.hane)
        tamam = (
            toplam_hane >= self.esik or
            toplam_guven >= self.min_toplam_guven
        )

        if not tamam:
            return (False, None, 0.0, 0)

        d.son_kayit = su_an
        d.oylar.clear()
        d.hane.clear()
        d.okuma_sayisi = 0

        return (True, lider, toplam_guven, toplam_hane)

    def durum_ozeti(self, arac_id: int) -> tuple:
        d = self._durum.get(arac_id)

        if d is None or not d.oylar:
            return ("", 0, 0.0, self.esik, self.min_toplam_guven)

        lider, toplam_guven, _ = self._cluster(d.oylar, d.hane)
        return (lider, d.okuma_sayisi, toplam_guven, self.esik, self.min_toplam_guven)

    def temizle_eski(self, yasam_suresi: float = 30.0):
        su_an = time.time()
        silinecek = [
            aid for aid, d in self._durum.items()
            if su_an - d.son_gorulme > yasam_suresi
        ]
        for aid in silinecek:
            del self._durum[aid]

    def sifirla(self):
        self._durum.clear()
