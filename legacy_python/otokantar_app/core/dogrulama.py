import re
import time

from otokantar_app.config import CONFIG, _HARF_DUZELTME, _RAKAM_DUZELTME
from otokantar_app.logger import log
from otokantar_app.models import DogrulamaDurumu


class DogrulamaMotoru:
    TR_PLAKA_REGEX = re.compile(r"^(0[1-9]|[1-7][0-9]|8[0-1])[A-Z]{1,3}\d{2,4}$")
    _ALNUM_DISI = re.compile(r"[^A-Z0-9]+")
    _GUVENLI_KARISIMLAR = {
        frozenset(("0", "O")),
        frozenset(("0", "D")),
        frozenset(("1", "I")),
        frozenset(("2", "Z")),
        frozenset(("6", "G")),
        frozenset(("8", "B")),
    }
    _TR_HARF_MAP = str.maketrans({
        "Ç": "C", "Ğ": "G", "İ": "I", "Ö": "O", "Ş": "S", "Ü": "U",
    })

    def __init__(self, esik: int, min_toplam_guven: float, kayit_sonrasi_bekleme: float):
        self.esik = int(esik)
        self.min_toplam_guven = float(min_toplam_guven)
        self.bekleme = float(kayit_sonrasi_bekleme)
        self._durum: dict = {}
        self.bilinen_plakalar = set()

        # ── Hareket eden kamera için gevşetilmiş eşikler ──────────────────────
        # Sabit kamerada bir plaka onlarca kare görünür; hareketlide 3-8 kare
        # görünür. Bu yüzden:
        #   • okuma_penceresi   : oy toplama süresi uzatıldı (3s → 6s)
        #   • min_gecerli_guven : OCR güven alt sınırı düşürüldü (0.45 → 0.30)
        #   • erken_cikis_guven : tek güçlü okumada anında kabul eşiği
        self.okuma_penceresi   = 6.0   # saniye — eski 5.0
        self.min_gecerli_guven = 0.30  # eski 0.45
        self.min_duzeltme_guven = 0.55
        self.min_lider_hane = 2
        self.min_lider_guven = 1.0
        self.bilinen_plaka_otoduzelt = bool(CONFIG.get("BILINEN_PLAKA_OTODUZELT", False))
        self.bilinen_plaka_kume_bonusu = bool(CONFIG.get("BILINEN_PLAKA_KUME_BONUSU", False))
        self.manuel_duzeltmeler = self._manuel_duzeltmeler_yukle(
            CONFIG.get("PLAKA_MANUEL_DUZELTMELER", {})
        )
        self.erken_cikis_guven = 2.5   # toplam güven bu değere ulaşırsa
                                       # esik kare beklenmeden kabul edilir

    # ─────────────────────────────────────────────────────────────────────────
    # Metin temizleme / normalize
    # ─────────────────────────────────────────────────────────────────────────

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
            harf_ham    = ham[2:2 + harf_uzunlugu]
            rakam_ham   = ham[2 + harf_uzunlugu:]
            aday = (
                f"{self._rakam_blok_duzelt(il_kodu_ham)}"
                f"{self._harf_blok_duzelt(harf_ham)}"
                f"{self._rakam_blok_duzelt(rakam_ham)}"
            )
            if not self._tr_plaka_gecerli_mi(aday):
                continue
            degisim_sayisi = sum(1 for o, s in zip(ham, aday) if o != s)
            adaylar.append((degisim_sayisi, harf_uzunlugu, aday))

        if not adaylar:
            return ham
        adaylar.sort(key=lambda item: (item[0], item[1] != 2, item[1] != 3, item[1]))
        return adaylar[0][2]

    def hazirla_bilinen_plakalar(self, plakalar) -> set:
        temiz = set()
        for p in (plakalar or []):
            a = self._normalize(p)
            if self._tr_plaka_gecerli_mi(a):
                temiz.add(a)
        return temiz

    def _manuel_duzeltmeler_yukle(self, duzeltmeler) -> dict:
        if not isinstance(duzeltmeler, dict):
            return {}
        sonuc = {}
        for kaynak, hedef in duzeltmeler.items():
            kaynak_norm = self._normalize(str(kaynak))
            hedef_norm = self._normalize(str(hedef))
            if self._tr_plaka_gecerli_mi(kaynak_norm) and self._tr_plaka_gecerli_mi(hedef_norm):
                sonuc[kaynak_norm] = hedef_norm
        return sonuc

    def _tr_plaka_gecerli_mi(self, plaka: str) -> bool:
        return bool(self.TR_PLAKA_REGEX.match(plaka))

    # ─────────────────────────────────────────────────────────────────────────
    # Levenshtein mesafesi
    # ─────────────────────────────────────────────────────────────────────────

    def _mesafe_hesapla(self, s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return self._mesafe_hesapla(s2, s1)
        if len(s2) == 0:
            return len(s1)
        onceki_satir = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            guncel_satir = [i + 1]
            for j, c2 in enumerate(s2):
                guncel_satir.append(min(
                    onceki_satir[j + 1] + 1,
                    guncel_satir[j] + 1,
                    onceki_satir[j] + (c1 != c2),
                ))
            onceki_satir = guncel_satir
        return onceki_satir[-1]

    def _benzer_mi(self, p1: str, p2: str) -> bool:
        if not (self._tr_plaka_gecerli_mi(p1) and self._tr_plaka_gecerli_mi(p2)):
            return False
        if abs(len(p1) - len(p2)) > 1:
            return False
        return self._mesafe_hesapla(p1, p2) <= 2

    # ─────────────────────────────────────────────────────────────────────────
    # Cluster (değişmedi — sadece erken çıkış eklendi)
    # ─────────────────────────────────────────────────────────────────────────

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
                kumeler.append({"merkez": plaka, "uyeler": [plaka]})

        en_iyi = None
        en_skor = -1
        for kume in kumeler:
            toplam_guven = sum(oylar[p] for p in kume["uyeler"])
            toplam_hane  = sum(hane[p]  for p in kume["uyeler"])
            lider = max(
                kume["uyeler"],
                key=lambda p: (hane[p], oylar[p], p),
            )
            skor = toplam_guven + (toplam_hane * 0.2)
            if self.bilinen_plaka_kume_bonusu and lider in self.bilinen_plakalar:
                skor += 2.0
            if skor > en_skor:
                en_skor = skor
                en_iyi = (lider, toplam_guven, toplam_hane)

        return en_iyi

    # ─────────────────────────────────────────────────────────────────────────
    # Otomatik düzeltme (bilinen plakalara yaklaştırma)
    # ─────────────────────────────────────────────────────────────────────────

    def _bilinen_duzeltme_guvenli_mi(self, plaka: str, kayitli: str) -> bool:
        if len(plaka) != len(kayitli) or plaka[:2] != kayitli[:2]:
            return False
        farklar = [
            (src, dst)
            for src, dst in zip(plaka, kayitli)
            if src != dst
        ]
        if len(farklar) != 1:
            return False
        return frozenset(farklar[0]) in self._GUVENLI_KARISIMLAR

    def _oto_duzelt(self, plaka: str, guven: float) -> str:
        if (
            not self.bilinen_plaka_otoduzelt
            or not self.bilinen_plakalar
            or plaka in self.bilinen_plakalar
        ):
            return plaka
        if guven < self.min_duzeltme_guven:
            return plaka
        best_aday = plaka
        best_score = 999
        for kayitli in self.bilinen_plakalar:
            if not self._bilinen_duzeltme_guvenli_mi(plaka, kayitli):
                continue
            dist = self._mesafe_hesapla(plaka, kayitli)
            if dist < best_score:
                best_score = dist
                best_aday = kayitli
        return best_aday if best_score <= 1 else plaka

    # ─────────────────────────────────────────────────────────────────────────
    # Ana işleme — hareket eden kamera için düzeltmeler burada
    # ─────────────────────────────────────────────────────────────────────────

    def isle(self, arac_id: int, plaka: str, guven: float) -> tuple:
        su_an = time.time()
        gelen_plaka = plaka
        gelen_guven = float(guven or 0.0)

        d = self._durum.get(arac_id)
        if d is None:
            d = DogrulamaDurumu()
            self._durum[arac_id] = d
            log.debug("OCR_DOGRULAMA yeni_arac arac_id=%s", arac_id)

        # Son kayıttan bu yana bekleme süresi dolmadıysa işlem yapma
        if su_an - d.son_kayit < self.bekleme:
            log.debug(
                "OCR_DOGRULAMA red=bektirme arac_id=%s plaka=%s kalan=%.1f",
                arac_id, gelen_plaka, self.bekleme - (su_an - d.son_kayit),
            )
            return (False, None, 0.0, 0)

        # ── DÜZELTME 1: Pencere süresi uzatıldı ──────────────────────────────
        # Eski kod 5.0s'de sıfırlıyordu. Hareketli kamerada araç 3-8 kare
        # görünür, 5s'de yeterliydi ama kamera hızlı geçerken oylar birikmeden
        # pencere kapanıyordu. 6s'ye çıkardık.
        if su_an - d.son_gorulme > self.okuma_penceresi:
            log.debug(
                "OCR_DOGRULAMA pencere_sifirlandi arac_id=%s onceki_oylar=%s onceki_hane=%s",
                arac_id, dict(d.oylar), dict(d.hane),
            )
            d.oylar.clear()
            d.hane.clear()
            d.okuma_sayisi = 0

        d.son_gorulme  = su_an
        d.okuma_sayisi += 1

        # ── DÜZELTME 2: Güven alt sınırı düşürüldü ───────────────────────────
        # Hareketli kamerada motion blur nedeniyle OCR güveni 0.45'in altına
        # düşüyor fakat metin çoğunlukla doğru. 0.30'a düşürdük.
        if guven < self.min_gecerli_guven:
            log.debug(
                "OCR_DOGRULAMA red=dusuk_guven arac_id=%s plaka=%s guven=%.3f min=%.3f",
                arac_id, gelen_plaka, gelen_guven, self.min_gecerli_guven,
            )
            return (False, None, 0.0, 0)

        plaka = self._normalize(plaka)
        if not self._tr_plaka_gecerli_mi(plaka):
            log.debug(
                "OCR_DOGRULAMA red=normalize_gecersiz arac_id=%s ham=%s normalize=%s guven=%.3f",
                arac_id, gelen_plaka, plaka, gelen_guven,
            )
            return (False, None, 0.0, 0)

        normalize_plaka = plaka
        manuel_plaka = self.manuel_duzeltmeler.get(plaka)
        if manuel_plaka:
            log.debug(
                "OCR_DOGRULAMA manuel_duzelt arac_id=%s normalize=%s duzeltilen=%s guven=%.3f",
                arac_id, plaka, manuel_plaka, gelen_guven,
            )
            plaka = manuel_plaka

        oto_giris_plaka = plaka
        plaka = self._oto_duzelt(plaka, gelen_guven)
        if plaka != oto_giris_plaka:
            log.debug(
                "OCR_DOGRULAMA oto_duzelt arac_id=%s normalize=%s duzeltilen=%s guven=%.3f",
                arac_id, oto_giris_plaka, plaka, gelen_guven,
            )

        # ── DÜZELTME 3: Güven klamp aralığı genişletildi ─────────────────────
        # Eski: (0.3, 0.95). 0.30'un altı zaten yukarıda elendi.
        # min'i 0.25'e indirdik; böylece düşük güvenli ama tekrarlayan
        # okumaların birikimi engellenmiyor.
        g = min(max(float(guven), 0.25), 0.95)

        d.oylar[plaka] = d.oylar.get(plaka, 0.0) + g
        d.hane[plaka]  = d.hane.get(plaka, 0) + 1

        lider, toplam_guven, toplam_hane = self._cluster(d.oylar, d.hane)
        log.debug(
            "OCR_DOGRULAMA oy arac_id=%s gelen=%s normalize=%s oy_plaka=%s guven=%.3f klamp=%.3f "
            "lider=%s toplam_guven=%.3f toplam_hane=%d okuma_sayisi=%d oylar=%s hane=%s",
            arac_id, gelen_plaka, normalize_plaka, plaka, gelen_guven, g,
            lider, float(toplam_guven), int(toplam_hane), int(d.okuma_sayisi),
            dict(d.oylar), dict(d.hane),
        )

        # ── DÜZELTME 4: Erken çıkış — tek yüksek güvenli okuma yeterli ───────
        # Hareket eden kamerada bazen plaka sadece 1-2 kare net görünür.
        # OCR o karede güçlüyse (toplam güven erken_cikis_guven'i geçiyorsa)
        # esik kare beklenmeden kaydedilir.
        lider_oy = float(d.oylar.get(lider, 0.0))
        lider_hane = int(d.hane.get(lider, 0))

        kume_hazir = (
            toplam_hane  >= self.esik              # eski davranış: yeterli kare
            or toplam_guven >= self.min_toplam_guven   # eski davranış: güven toplamı
            or (toplam_hane >= 2 and toplam_guven >= self.erken_cikis_guven)
        )
        lider_guvenilir = (
            lider_hane >= self.min_lider_hane
            and lider_oy >= self.min_lider_guven
        )
        tamam = kume_hazir and lider_guvenilir

        if not tamam:
            log.debug(
                "OCR_DOGRULAMA bekle arac_id=%s lider=%s toplam_guven=%.3f toplam_hane=%d "
                "lider_oy=%.3f lider_hane=%d esik=%d min_toplam=%.3f erken=%.3f",
                arac_id, lider, float(toplam_guven), int(toplam_hane),
                lider_oy, lider_hane, self.esik, self.min_toplam_guven, self.erken_cikis_guven,
            )
            return (False, None, 0.0, 0)

        log.info(
            "OCR_DOGRULAMA kabul arac_id=%s final=%s toplam_guven=%.3f toplam_hane=%d "
            "lider_oy=%.3f lider_hane=%d oylar=%s",
            arac_id, lider, float(toplam_guven), int(toplam_hane),
            lider_oy, lider_hane, dict(d.oylar),
        )
        d.son_kayit    = su_an
        d.oylar.clear()
        d.hane.clear()
        d.okuma_sayisi = 0

        return (True, lider, toplam_guven, toplam_hane)

    # ─────────────────────────────────────────────────────────────────────────
    # Durum özeti (debug / arayüz için)
    # ─────────────────────────────────────────────────────────────────────────

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

    def sil(self, arac_id: int):
        self._durum.pop(arac_id, None)

    def sifirla(self):
        self._durum.clear()
