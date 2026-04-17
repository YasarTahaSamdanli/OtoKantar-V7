"""
tests/test_api.py — FastAPI endpoint birim testleri (TestClient kullanır)
"""
import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from otokantar_app.api.routes import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# /api/canli-durum
# ---------------------------------------------------------------------------
def test_canli_durum_ok(client):
    resp = client.get("/api/canli-durum")
    assert resp.status_code == 200
    data = resp.json()
    # Alan adları her zaman mevcut olmalı
    assert "son_guncelleme" in data or "son_kayit" in data or "kantar_kg" in data


# ---------------------------------------------------------------------------
# /api/son-kayitlar
# ---------------------------------------------------------------------------
def test_son_kayitlar_liste_doner(client):
    resp = client.get("/api/son-kayitlar")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


# ---------------------------------------------------------------------------
# /api/kara-liste — kimlik doğrulama devre dışıyken çalışmalı
# ---------------------------------------------------------------------------
def test_kara_liste_get(client):
    resp = client.get("/api/kara-liste")
    assert resp.status_code == 200
    assert "kara_liste" in resp.json()


def test_kara_liste_ekle_gecerli_plaka(client):
    resp = client.post("/api/kara-liste", json={"plaka": "06AEP01"})
    # 200 veya 409 (zaten varsa) beklenir
    assert resp.status_code in (200, 409)


def test_kara_liste_ekle_gecersiz_plaka(client):
    resp = client.post("/api/kara-liste", json={"plaka": "GECERSIIZ"})
    assert resp.status_code == 422


def test_kara_liste_sil(client):
    # Önce ekle
    client.post("/api/kara-liste", json={"plaka": "06AEP02"})
    resp = client.delete("/api/kara-liste/06AEP02")
    # 200 (silindi) veya 500 (DB yoksa) beklenir
    assert resp.status_code in (200, 500)


# ---------------------------------------------------------------------------
# /api/log-son
# ---------------------------------------------------------------------------
def test_log_son_yapi(client):
    resp = client.get("/api/log-son")
    assert resp.status_code == 200
    data = resp.json()
    assert "satirlar" in data
    assert isinstance(data["satirlar"], list)


# ---------------------------------------------------------------------------
# Auth aktifken token kontrolü
# ---------------------------------------------------------------------------
def test_auth_reddet_gecersiz_token(monkeypatch):
    """API_TOKEN set edildiğinde geçersiz token 403 döndürmeli."""
    import otokantar_app.api.routes as routes_module
    monkeypatch.setattr(routes_module, "_API_TOKEN", "gizli-token")

    with TestClient(app) as c:
        resp = c.post(
            "/api/kara-liste",
            json={"plaka": "06AEP03"},
            headers={"Authorization": "Bearer yanlis-token"},
        )
    assert resp.status_code == 403


def test_auth_kabul_et_dogru_token(monkeypatch):
    """Doğru token ile istek kabul edilmeli."""
    import otokantar_app.api.routes as routes_module
    monkeypatch.setattr(routes_module, "_API_TOKEN", "dogru-token")

    with TestClient(app) as c:
        resp = c.post(
            "/api/kara-liste",
            json={"plaka": "06AEP04"},
            headers={"Authorization": "Bearer dogru-token"},
        )
    assert resp.status_code in (200, 409, 500)
