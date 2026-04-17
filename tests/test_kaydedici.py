"""
tests/test_kaydedici.py — KantarKaydedici birim testleri (geçici DB kullanır)
"""
import tempfile
import os
import time

import pytest

from otokantar_app.db.kaydedici import KantarKaydedici


@pytest.fixture
def kaydedici(tmp_path):
    csv_path = str(tmp_path / "test.csv")
    json_path = str(tmp_path / "test.json")
    db_path = str(tmp_path / "test.db")
    kd = KantarKaydedici(csv_dosya=csv_path, json_dosya=json_path, db_dosya=db_path)
    yield kd
    kd.kapat()


def test_giris_kaydi(kaydedici):
    kayit = kaydedici.giris_kaydet("06ABC123", 12000.0)
    assert kayit.plaka == "06ABC123"
    assert kayit.giris_agirlik == 12000.0
    assert kayit.durum == "ICERIDE"


def test_acik_seans_bulunur(kaydedici):
    kaydedici.giris_kaydet("34XY567", 8000.0)
    time.sleep(0.2)  # DbWriter thread'e biraz zaman tanı
    seans = kaydedici.acik_seans_getir("34XY567")
    assert seans is not None
    assert seans["plaka"] == "34XY567"


def test_cikis_kaydi(kaydedici):
    kaydedici.giris_kaydet("35DE890", 10000.0)
    time.sleep(0.3)
    kayit = kaydedici.cikis_kaydet("35DE890", 40000.0)
    assert kayit is not None
    assert kayit.durum == "TAMAMLANDI"
    assert kayit.net_agirlik == 30000.0


def test_net_agirlik_dogru(kaydedici):
    kaydedici.giris_kaydet("41KLM99", 15000.0)
    time.sleep(0.3)
    kayit = kaydedici.cikis_kaydet("41KLM99", 50000.0)
    assert kayit.net_agirlik == 35000.0


def test_acik_seans_olmadan_cikis(kaydedici):
    """Açık seans yoksa çıkış kaydı None döndürmeli."""
    kayit = kaydedici.cikis_kaydet("BILINMEYEN123", 5000.0)
    assert kayit is None


def test_kara_liste_ekle_sorgula(kaydedici):
    import sqlite3
    con = sqlite3.connect(kaydedici.db_dosya)
    con.execute("INSERT INTO kara_liste (plaka) VALUES (?)", ("06YAS01",))
    con.commit()
    con.close()
    assert kaydedici.plaka_kara_listede_mi("06YAS01") is True
    assert kaydedici.plaka_kara_listede_mi("06TEMIZ01") is False


def test_plaka_buyuk_harf_normalize(kaydedici):
    """Küçük harfle gelen plaka büyük harfe normalize edilmeli."""
    kayit = kaydedici.giris_kaydet("06abc999", 5000.0)
    assert kayit.plaka == "06ABC999"


def test_son_kayitlar_hafiza(kaydedici):
    kaydedici.giris_kaydet("07ZZZ111", 3000.0)
    assert any(k.plaka == "07ZZZ111" for k in kaydedici.son_kayitlar)
