import time
import re

from otokantar_app.models import DogrulamaDurumu


class DogrulamaMotoru:
    def __init__(self, esik: int, min_toplam_guven: float, kayit_sonrasi_bekleme: float):
        self.esik = int(esik)
        self.min_toplam_guven = float(min_toplam_guven)
        self.bekleme = float(kayit_sonrasi_bekleme)
        self._durum: dict = {}

        # 🚀 Veritabanındaki bilinen araçların tutulacağı küme
        self.bilinen_plakalar = set()

        # 🔥 FIX 2: İl kodunu da (01-81) native olarak kontrol eden saf Validation Regex'i
        self.TR_PLAKA_REGEX = re.compile(r"^(0[1-9]|[1-7][0-9]|8[0-1])[A-Z]{1,3}\d{2,4}$")

        # OCR normalize map
        self.normalize_map = str.maketrans({
            "O": "0", "I": "1", "İ": "1", "B": "8", "S": "5",
            "G": "6", "Z": "2"
        })

    # 🔧 Normalize
    def _normalize(self, plaka: str) -> str:
        plaka = plaka.upper().replace(" ", "")
        return plaka.translate(self.normalize_map)

    # 🔥 FIX 2: Sadeleştirilmiş Katı Format Denetleyicisi
    def _tr_plaka_gecerli_mi(self, plaka: str) -> bool:
        return bool(self.TR_PLAKA_REGEX.match(plaka))

    # 📏 Levenshtein Mesafe Algoritması
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

    # 🔥 FIX 4 Uyumlu Benzerlik Kontrolü
    def _benzer_mi(self, p1: str, p2: str) -> bool:
        if not (self._tr_plaka_gecerli_mi(p1) and self._tr_plaka_gecerli_mi(p2)):
            return False

        if abs(len(p1) - len(p2)) > 1:
            return False

        return self._mesafe_hesapla(p1, p2) <= 2

    # 🧠 Akıllı Kümeleme (Merkez Bağımlılığı Kaldırıldı)
    def _cluster(self, oylar: dict, hane: dict):
        kumeler = []

        for plaka in oylar:
            bulundu = False
            for kume in kumeler:
                # 🔥 FIX 4: Sabit merkeze değil, kümedeki HERHANGİ bir üyeye benzemesi yeterli
                if any(self._benzer_mi(plaka, u) for u in kume["uyeler"]):
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

            lider = max(
                kume["uyeler"],
                key=lambda p: (
                    p in self.bilinen_plakalar,
                    oylar[p],
                    hane[p]
                )
            )

            skor = toplam_guven + (toplam_hane * 0.2)
            if lider in self.bilinen_plakalar:
                skor += 2.0  # Veritabanı otorite puanı

            if skor > en_skor:
                en_skor = skor
                en_iyi = (lider, toplam_guven, toplam_hane)

        return en_iyi

    # 🔥 FIX 1: Overwrite Engelli Akıllı Oto-Düzeltme
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

        # Yalnızca 1 karakterlik minimal hataları düzeltir.
        # Mesafe 2 ise "False Plate Injection" riski taşır, orijinal plakayı bırakır.
        if best_score <= 1:
            return best_aday
        
        return plaka

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

        # 🔥 FIX 3: Orta kalite okumalar için Confidence Gate 0.45'e indirildi
        if guven < 0.45:
            return (False, None, 0.0, 0)

        # Doğru Akış: Normalize -> Validation -> Düzeltme
        plaka = self._normalize(plaka)

        if not self._tr_plaka_gecerli_mi(plaka):
            return (False, None, 0.0, 0)

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