import time
import re
from collections import defaultdict

from otokantar_app.models import DogrulamaDurumu


class DogrulamaMotoru:
    def __init__(self, esik: int, min_toplam_guven: float, kayit_sonrasi_bekleme: float):
        self.esik = int(esik)
        self.min_toplam_guven = float(min_toplam_guven)
        self.bekleme = float(kayit_sonrasi_bekleme)
        self._durum: dict = {}

        # 🚀 Veritabanındaki bilinen araçların tutulacağı küme (main.py'den beslenecek)
        self.bilinen_plakalar = set()

        # TR plaka regex
        self.TR_PLAKA_REGEX = re.compile(r"^(0[1-9]|[1-7][0-9]|8[0-1])([A-Z]{1,3})(\d{2,4})$")

        # OCR normalize map
        self.normalize_map = str.maketrans({
            "O": "0", "I": "1", "İ": "1", "B": "8", "S": "5",
            "G": "6", "Z": "2"
        })

    # 🔧 Normalize
    def _normalize(self, plaka: str) -> str:
        plaka = plaka.upper().replace(" ", "")
        return plaka.translate(self.normalize_map)

    # 📏 Levenshtein Mesafe Algoritması (Karakter farkını milimetrik hesaplar)
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
        # En fazla 2 karakter farkına kadar "Benzer" kabul et
        return self._mesafe_hesapla(p1, p2) <= 2

    # 🧠 Akıllı Kümeleme (Otorite Puanlı)
    def _cluster(self, oylar: dict, hane: dict):
        kumeler = []

        for plaka in oylar:
            bulundu = False
            for kume in kumeler:
                if self._benzer_mi(plaka, kume["merkez"]):
                    kume["uyeler"].append(plaka)
                    bulundu = True
                    break

            if not bulundu:
                kumeler.append({
                    "merkez": plaka,
                    "uyeler": [plaka]
                })

        en_iyi = None
        en_skor = -1

        for kume in kumeler:
            toplam_guven = sum(oylar[p] for p in kume["uyeler"])
            toplam_hane = sum(hane[p] for p in kume["uyeler"])

            # Eğer kümedeki plakalardan biri veritabanında kayıtlıysa onu lider yap!
            lider = max(
                kume["uyeler"],
                key=lambda p: (
                    p in self.bilinen_plakalar,  # 🔥 DB Önceliği
                    self.TR_PLAKA_REGEX.match(p) is not None,
                    oylar[p],
                    hane[p]
                )
            )

            skor = toplam_guven + (toplam_hane * 0.2)
            if lider in self.bilinen_plakalar:
                skor += 2.0  # 🔥 Veritabanında varsa puanını yapay olarak artır (Otorite)

            if skor > en_skor:
                en_skor = skor
                en_iyi = (lider, toplam_guven, toplam_hane)

        return en_iyi

    # 🤖 AI Hata Düzeltici
    def _oto_duzelt(self, plaka: str) -> str:
        """AI'ın okuduğu plakayı veritabanındaki kayıtlı araçlara benzetir."""
        if not self.bilinen_plakalar or plaka in self.bilinen_plakalar:
            return plaka

        en_iyi_aday = plaka
        en_kucuk_mesafe = 3  # Maksimum 2 karakter farka tolerans

        for kayitli in self.bilinen_plakalar:
            mesafe = self._mesafe_hesapla(plaka, kayitli)
            if mesafe < en_kucuk_mesafe:
                en_kucuk_mesafe = mesafe
                en_iyi_aday = kayitli

        return en_iyi_aday

    def isle(self, arac_id: int, plaka: str, guven: float) -> tuple:
        su_an = time.time()

        d = self._durum.get(arac_id)
        if d is None:
            d = DogrulamaDurumu()
            self._durum[arac_id] = d

        # Cooldown
        if su_an - d.son_kayit < self.bekleme:
            return (False, None, 0.0, 0)

        # Timeout reset
        if su_an - d.son_gorulme > 5.0:
            d.oylar.clear()
            d.hane.clear()
            d.okuma_sayisi = 0

        d.son_gorulme = su_an
        d.okuma_sayisi += 1

        # Normalize 
        plaka = self._normalize(plaka)
        
        # 🔥 OTO-DÜZELTME DEVREDE! (1-2 harf hatası varsa düzeltilir)
        plaka = self._oto_duzelt(plaka)

        g = min(max(float(guven), 0.3), 0.95)

        d.oylar[plaka] = d.oylar.get(plaka, 0.0) + g
        d.hane[plaka] = d.hane.get(plaka, 0) + 1

        # Hybrid karar
        lider, toplam_guven, toplam_hane = self._cluster(d.oylar, d.hane)

        tamam = (
            toplam_hane >= self.esik or
            toplam_guven >= self.min_toplam_guven
        )

        if not tamam:
            return (False, None, 0.0, 0)

        # Reset
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