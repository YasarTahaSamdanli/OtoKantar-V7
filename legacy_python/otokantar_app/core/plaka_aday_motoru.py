"""
plaka_aday_motoru.py - OCR karakter karismasi icin sinirli duzeltme.

Bu modul plaka tahmini yapmaz. OCR metninde gorulen karakterleri ayni
pozisyonda, ayni uzunlukta ve yalnizca izinli karisma ciftleriyle duzeltir.
"""

from __future__ import annotations

import re
from typing import Optional

from otokantar_app.config import PLAKA_REGEX, PLAKA_REGEX_TAM

_ALNUM_DISI = re.compile(r"[^A-Z0-9]")

# TR_PLAKA_TAM artık config.py'den PLAKA_REGEX_TAM olarak geliyor.
# Geriye dönük uyumluluk için alias:
TR_PLAKA_TAM = PLAKA_REGEX_TAM

_HARF_ALTERNATIFLERI = {
    "0": ("O", "D"),
    "6": ("G",),
    "8": ("B",),
    "2": ("Z",),
    "1": ("I",),
}

_RAKAM_ALTERNATIFLERI = {
    "O": ("0",),
    "D": ("0",),
    "G": ("6",),
    "B": ("8",),
    "Z": ("2",),
    "I": ("1",),
}

_IZINLI_KARISMALAR = {
    frozenset(("0", "O")),
    frozenset(("0", "D")),
    frozenset(("6", "G")),
    frozenset(("8", "B")),
    frozenset(("2", "Z")),
    frozenset(("1", "I")),
}


def _temizle(metin: str) -> str:
    return _ALNUM_DISI.sub("", (metin or "").upper())


def _gecerli_uzunluk(plaka: str) -> bool:
    return 7 <= len(plaka) <= 9


def _tr_plaka_gecerli(plaka: str) -> bool:
    return bool(PLAKA_REGEX_TAM.match(plaka or ""))


def _karisma_maliyeti(kaynak_ch: str, hedef_ch: str) -> float:
    if kaynak_ch == hedef_ch:
        return 0.0
    if frozenset((kaynak_ch, hedef_ch)) in _IZINLI_KARISMALAR:
        return 0.35
    return 1.0


def _pozisyon_alternatifleri(ch: str, rol: str) -> tuple[str, ...]:
    if rol == "harf":
        return (ch, *_HARF_ALTERNATIFLERI.get(ch, ()))
    if rol == "rakam":
        return (ch, *_RAKAM_ALTERNATIFLERI.get(ch, ()))
    return (ch,)


def _duzeltme_adaylari(il: str, harf: str, rakam: str) -> set[str]:
    parcalar = (
        [(ch, "rakam") for ch in il]
        + [(ch, "harf") for ch in harf]
        + [(ch, "rakam") for ch in rakam]
    )
    adaylar = [""]
    for ch, rol in parcalar:
        siradaki = []
        for kok in adaylar:
            for alt in _pozisyon_alternatifleri(ch, rol):
                siradaki.append(kok + alt)
        adaylar = siradaki

    kaynak = il + harf + rakam
    return {
        aday
        for aday in adaylar
        if len(aday) == len(kaynak) and _gecerli_uzunluk(aday) and _tr_plaka_gecerli(aday)
    }


def _agirlikli_mesafe(kaynak: str, hedef: str) -> float:
    if len(kaynak) != len(hedef):
        return 999.0
    return sum(_karisma_maliyeti(src, dst) for src, dst in zip(kaynak, hedef))


def _uret_adaylar(ham: str) -> set[str]:
    ham = _temizle(ham)
    adaylar: set[str] = set()

    # PLAKA_REGEX_TAM (ankorsuz değil, tam eşleşme) kullanıyoruz.
    # Eski PLAKA_REGEX.fullmatch() çoğu durumda doğru çalışır çünkü
    # fullmatch zaten tüm metni kapsar; ancak PLAKA_REGEX_TAM açıklık sağlar
    # ve yanlışlıkla search() ile kullanılmasını önler.
    eslesen = PLAKA_REGEX_TAM.match(ham)
    if not eslesen:
        return adaylar

    il, harf, rakam = eslesen.group(1), eslesen.group(2), eslesen.group(3)
    kaynak = il + harf + rakam
    for aday in _duzeltme_adaylari(il, harf, rakam):
        if len(aday) == len(kaynak):
            adaylar.add(aday)

    return adaylar


def aday_skorla(
    aday: str,
    ham: str,
    ocr_guven: float,
    bilinen_plakalar: Optional[set[str]] = None,
    baseline: Optional[str] = None,
    aday_havuzu: Optional[set[str]] = None,
) -> tuple[float, float]:
    del bilinen_plakalar, baseline, aday_havuzu

    ham = _temizle(ham)
    maliyet = _agirlikli_mesafe(ham, aday) if PLAKA_REGEX_TAM.match(ham) else 999.0
    skor = float(ocr_guven) - maliyet
    return skor, maliyet


def _kirpma_denetimi(ham: str, aday: str) -> dict:
    """
    Ham OCR metni ile seçilen aday arasındaki karakter kırpmayı ölçer.

    Türk plakası karakterleri 1-1 dönüştürülebilir (O↔0, B↔8 vb.);
    dolayısıyla aday uzunluğu ham uzunluğuna eşit olmalıdır.
    Uzunluk farkı varsa ham metinden substring seçilmiş demektir — bu hata.

    Rapor kuralı:
      toplam == 0            → kabul
      toplam == 1            → belirsiz (skor cezalı, tek başına kayıt üretmez)
      toplam >= 2            → reddet
      oran > %10             → reddet
      her iki yönde kırpma   → reddet (örn. 060L61261 → 60L6126)

    Dönüş: {'sol': int, 'sag': int, 'toplam': int, 'oran': float, 'red': bool}
    """
    ham_len = len(ham)
    aday_len = len(aday)
    toplam = ham_len - aday_len

    if toplam < 0:
        # aday ham'dan uzun olamaz — tanımsız durum, güvenli taraf reddet
        return {"sol": 0, "sag": 0, "toplam": 0, "oran": 0.0, "red": True}

    if toplam == 0:
        return {"sol": 0, "sag": 0, "toplam": 0, "oran": 0.0, "red": False}

    # Sol kırpmayı bul: ham'ın başından aday[0] ile eşleşene kadar say
    sol = 0
    for i in range(min(toplam + 1, ham_len)):
        if i < ham_len and ham[i] == aday[0]:
            break
        sol += 1
    sol = min(sol, toplam)
    sag = toplam - sol

    oran = toplam / ham_len if ham_len > 0 else 0.0

    red = (
        toplam >= 2           # 2+ karakter kayıp → reddet
        or oran > 0.10        # ham'ın %10'undan fazlası kayıp → reddet
        or (sol > 0 and sag > 0)  # her iki yönden kırpma → reddet
    )

    return {
        "sol": sol, "sag": sag,
        "toplam": toplam, "oran": oran,
        "red": red,
    }


def en_iyi_aday_sec(
    ham_metin: str,
    ocr_guven: float,
    bilinen_plakalar: Optional[set[str]] = None,
) -> tuple[Optional[str], float, float, list[tuple[str, float, float]]]:
    del bilinen_plakalar

    ham = _temizle(ham_metin)
    if not ham:
        return None, 0.0, 0.0, []

    adaylar = _uret_adaylar(ham)
    if not adaylar:
        return None, 0.0, 0.0, []

    skorlu = []
    for aday in adaylar:
        kirpma = _kirpma_denetimi(ham, aday)

        if kirpma["red"]:
            # Rapor kuralı: toplam>=2, oran>%10 veya çift yön kırpma → reddet.
            # Yalnızca loglama; aday havuzuna alınmıyor.
            # (log modülü bu katmanda mevcut değil; caller log'lamalı)
            continue

        skor, maliyet = aday_skorla(aday, ham, ocr_guven)

        # Belirsiz (1 karakter kırpma) → skor cezası; tek başına kayıt tetiklemez,
        # ama oylamada birden fazla kare gelirse hâlâ kazanabilir.
        if kirpma["toplam"] == 1:
            skor -= 0.30

        skorlu.append((aday, skor, maliyet))

    if not skorlu:
        return None, 0.0, 0.0, []

    skorlu.sort(key=lambda item: (-item[1], item[2], item[0]))
    en_iyi, skor, maliyet = skorlu[0]
    return en_iyi, skor, maliyet, skorlu