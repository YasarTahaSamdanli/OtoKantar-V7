"""
tests/test_dogrulama.py — DogrulamaMotoru birim testleri
"""
import time

import pytest

from otokantar_app.core.dogrulama import DogrulamaMotoru

# Plaka seçimi: OCR normalize haritasında (O→0, I→1, B→8, S→5, G→6, Z→2)
# dönüşüm sonrasında bozulmayacak karakterler kullanılmalı.
_GECERLI_PLAKA = "34XY567"   # X, Y, 5, 6, 7 → değişmez; 34 geçerli il kodu
_GECERLI_PLAKA2 = "35DE890"  # D, E → değişmez
_GECERLI_PLAKA3 = "41KLM99"  # K, L, M → değişmez


@pytest.fixture
def motor():
    return DogrulamaMotoru(esik=3, min_toplam_guven=2.0, kayit_sonrasi_bekleme=5.0)


def test_gecerli_plaka_kabul_edilir(motor):
    """Eşiği geçen yeterli okuma sonucunda plaka kabul edilmeli."""
    kabul = False
    for _ in range(10):
        kabul, plaka, _, _ = motor.isle(arac_id=1, plaka=_GECERLI_PLAKA, ocr_guven=0.9)
        if kabul:
            break
    assert kabul is True
    assert plaka == _GECERLI_PLAKA


def test_gecersiz_plaka_reddedilir(motor):
    """Türk plaka formatına uymayan metin reddedilmeli."""
    for _ in range(10):
        kabul, _, _, _ = motor.isle(arac_id=2, plaka="XY9999ZZ", ocr_guven=0.95)
        assert kabul is False


def test_dusuk_guven_reddedilir(motor):
    """Kombine güven 0.40'ın altındaysa oy kaydedilmemeli."""
    for _ in range(10):
        kabul, plaka, _, _ = motor.isle(arac_id=3, plaka=_GECERLI_PLAKA, ocr_guven=0.10, yolo_guven=0.10)
        assert kabul is False


def test_cooldown_uygulanir(motor):
    """Plaka kabul edildikten sonra bekleme süresi içinde tekrar kabul edilmemeli."""
    for _ in range(10):
        kabul, _, _, _ = motor.isle(arac_id=4, plaka=_GECERLI_PLAKA, ocr_guven=0.9)
        if kabul:
            break

    # Hemen ardından tekrar gönder — cooldown aktif
    kabul2, _, _, _ = motor.isle(arac_id=4, plaka=_GECERLI_PLAKA, ocr_guven=0.9)
    assert kabul2 is False


def test_farkli_araclar_birbirini_etkilemiyor(motor):
    """İki farklı araç ID'si bağımsız çalışmalı."""
    for _ in range(10):
        kabul1, _, _, _ = motor.isle(arac_id=10, plaka="01AA987", ocr_guven=0.9)
        if kabul1:
            break

    # İkinci araç henüz birikmiş oy yok — ilk denemede kabul edilmemeli
    kabul2, _, _, _ = motor.isle(arac_id=11, plaka=_GECERLI_PLAKA2, ocr_guven=0.9)
    assert kabul2 is False


def test_normalize_ocr_karisikligi(motor):
    """'O' → '0' normalizasyonu plaka ID'sinde çalışmalı."""
    # "O1AE456" → normalize → "01AE456" (geçerli)
    kabul = False
    for _ in range(10):
        kabul, plaka, _, _ = motor.isle(arac_id=20, plaka="O1AE456", ocr_guven=0.9)
        if kabul:
            break
    assert kabul is True
    assert plaka == "01AE456"


def test_oto_duzelt_bilinen_plakaya_esler(motor):
    """Tek karakter farkıyla bilinen bir plakaya otomatik düzeltme yapılmalı."""
    motor.bilinen_plakalari_guncelle({"34XY568"})
    # "34XY567" → dist("34XY567","34XY568")=1 → "34XY568" olarak düzeltilmeli
    kabul = False
    for _ in range(10):
        kabul, plaka, _, _ = motor.isle(arac_id=30, plaka="34XY567", ocr_guven=0.9)
        if kabul:
            break
    assert kabul is True
    assert plaka == "34XY568"


def test_aktif_arac_sayisi(motor):
    motor.isle(arac_id=100, plaka="06RT28", ocr_guven=0.9)
    assert motor.aktif_arac_sayisi() >= 1

