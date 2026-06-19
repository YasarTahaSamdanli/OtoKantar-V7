"""
plaka_aday_motoru.py - OCR karakter karismasi icin sinirli duzeltme.

Bu modul plaka tahmini yapmaz. OCR metninde gorulen karakterleri ayni
pozisyonda, ayni uzunlukta ve yalnizca izinli karisma ciftleriyle duzeltir.
"""

from __future__ import annotations

import re
from typing import Optional

from otokantar_app.config import PLAKA_REGEX

TR_PLAKA_TAM = re.compile(
    r"^(0[1-9]|[1-7][0-9]|8[0-1])([A-Z]{1,3})(\d{2,4})$"
)

_ALNUM_DISI = re.compile(r"[^A-Z0-9]")

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
    return bool(TR_PLAKA_TAM.match(plaka or ""))


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

    eslesen = PLAKA_REGEX.fullmatch(ham)
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
    maliyet = _agirlikli_mesafe(ham, aday) if PLAKA_REGEX.fullmatch(ham) else 999.0
    skor = float(ocr_guven) - maliyet
    return skor, maliyet


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

    skorlu = [(aday, *aday_skorla(aday, ham, ocr_guven)) for aday in adaylar]
    skorlu.sort(key=lambda item: (-item[1], item[2], item[0]))
    en_iyi, skor, maliyet = skorlu[0]
    return en_iyi, skor, maliyet, skorlu
