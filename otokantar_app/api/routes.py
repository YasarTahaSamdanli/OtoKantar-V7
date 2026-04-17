import json
import secrets
import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from otokantar_app.config import CONFIG, PLAKA_REGEX
from otokantar_app.logger import log

app = FastAPI(title="OtoKantar V7 API")

# ---------------------------------------------------------------------------
# CORS — restrict to configured origins (default: same-origin only)
# ---------------------------------------------------------------------------
_cors_origins: list = CONFIG.get("CORS_ORIGINS", ["http://localhost:8000", "http://127.0.0.1:8000"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

# ---------------------------------------------------------------------------
# Optional Bearer-token auth for write endpoints
# ---------------------------------------------------------------------------
_API_TOKEN: str = str(CONFIG.get("API_TOKEN", "")).strip()
_bearer = HTTPBearer(auto_error=False)


def _yetki_kontrol(credentials: HTTPAuthorizationCredentials = Security(_bearer)) -> None:
    """
    If API_TOKEN is configured, every write request must carry a matching
    ``Authorization: Bearer <token>`` header.  When API_TOKEN is empty,
    the check is skipped (suitable for local / dev environments).
    """
    if not _API_TOKEN:
        return  # auth disabled
    if credentials is None or not secrets.compare_digest(credentials.credentials, _API_TOKEN):
        raise HTTPException(status_code=403, detail="Geçersiz veya eksik API token.")


# Multiprocessing nedeniyle bu referans alt süreçte (FastAPI) None kalabilir.
# Bu yüzden veriye erişimde her zaman Dosya/Veritabanı önceliklidir.
sistem_referansi = None

# Proje kök dizini (main.py'nin olduğu yer)
_PROJE_KOKU = Path(__file__).resolve().parents[2]

def sistem_referansi_ata(sistem) -> None:
    global sistem_referansi
    sistem_referansi = sistem

def _db_baglantisi_kur():
    """Veritabanına WAL modunda güvenli bağlantı açar."""
    db_yol = _PROJE_KOKU / CONFIG.get("DB_DOSYA", "otokantar.db")
    con = sqlite3.connect(db_yol, timeout=10)
    con.row_factory = sqlite3.Row
    # WAL modu dashboard okumalarını hızlandırır
    con.execute("PRAGMA journal_mode=WAL;")
    return con

def _canli_json_dosyadan_oku() -> dict:
    yol = _PROJE_KOKU / CONFIG.get("JSON_CANLI", "canli_durum.json")
    if not yol.is_file():
        return {"son_guncelleme": None, "son_kayit": None, "son_10": []}
    try:
        with open(yol, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.error(f"Canlı JSON okuma hatası: {e}")
        return {"son_guncelleme": None, "son_kayit": None, "son_10": []}

@app.get("/")
def index():
    return FileResponse(_PROJE_KOKU / "dashboard.html")

@app.get("/api/canli-durum")
def api_canli_durum():
    """
    Kantarın anlık durumunu döner. 
    Multiprocessing nedeniyle RAM yerine JSON dosyasına güvenir.
    """
    durum = _canli_json_dosyadan_oku()
    
    # Eğer sistem_referansi varsa (Linux/Unix fork veya tek process), ekstra detay ekle
    if sistem_referansi:
        ag, st = getattr(sistem_referansi, "kantar_okuyucu", type("_", (), {"veri": (None, None)})()).veri
        durum.update({
            "fps": round(getattr(sistem_referansi, "_fps", 0.0), 1),
            "kilitli": getattr(sistem_referansi, "_kantar_seans_kilitli", False),
            "plaka_buffer": sistem_referansi._plaka_buffer.plaka if sistem_referansi._plaka_buffer else None,
            "kantar_kg": round(ag, 1) if ag is not None else None,
            "kantar_sabit": st,
        })
    else:
        durum.setdefault("kantar_kg", None)
        durum.setdefault("kantar_sabit", None)
        durum.setdefault("kilitli", None)
        durum.setdefault("plaka_buffer", None)
    return durum

@app.get("/api/son-kayitlar")
def api_son_kayitlar():
    """Son 50 geçiş kaydını doğrudan veritabanından çeker."""
    try:
        with _db_baglantisi_kur() as con:
            rows = con.execute(
                "SELECT * FROM gecis_raporlari ORDER BY id DESC LIMIT 50"
            ).fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        log.error(f"Son kayıtlar DB hatası: {e}")
        # DB hatası durumunda bellekteki son kayıtları dene (varsa)
        if sistem_referansi and hasattr(sistem_referansi.kaydedici, "son_kayitlar"):
            return [asdict(k) for k in sistem_referansi.kaydedici.son_kayitlar]
        return []

class KaraListeEkleIstek(BaseModel):
    plaka: str

@app.get("/api/kara-liste")
def api_kara_liste_listele():
    """Veritabanındaki güncel kara listeyi döner."""
    try:
        with _db_baglantisi_kur() as con:
            rows = con.execute("SELECT plaka FROM kara_liste ORDER BY id DESC").fetchall()
            return {"kara_liste": [r["plaka"] for r in rows]}
    except Exception as e:
        log.error(f"Kara liste listeleme hatası: {e}")
        return {"kara_liste": []}

@app.post("/api/kara-liste")
def api_kara_liste_ekle(istek: KaraListeEkleIstek, _: None = Depends(_yetki_kontrol)):
    """Veritabanına yeni yasaklı plaka ekler."""
    plaka = istek.plaka.strip().upper()
    if not PLAKA_REGEX.fullmatch(plaka):
        raise HTTPException(422, f"Geçersiz Türk plaka formatı: '{plaka}'")
    
    try:
        with _db_baglantisi_kur() as con:
            con.execute("INSERT INTO kara_liste (plaka) VALUES (?)", (plaka,))
            con.commit()
        log.info("Kara listeye yeni plaka eklendi (API): %s", plaka)
        return {"mesaj": "Başarılı", "plaka": plaka}
    except sqlite3.IntegrityError:
        raise HTTPException(409, f"'{plaka}' zaten kara listede mevcut.")
    except Exception as e:
        log.error(f"Kara liste ekleme hatası: {e}")
        raise HTTPException(500, "Veritabanına yazılırken bir hata oluştu.")

@app.delete("/api/kara-liste/{plaka}")
def api_kara_liste_sil(plaka: str, _: None = Depends(_yetki_kontrol)):
    """Veritabanından plaka yasaklamasını kaldırır."""
    plaka = plaka.strip().upper()
    try:
        with _db_baglantisi_kur() as con:
            con.execute("DELETE FROM kara_liste WHERE plaka = ?", (plaka,))
            con.commit()
        return {"mesaj": "Silindi", "plaka": plaka}
    except Exception as e:
        log.error(f"Kara liste silme hatası: {e}")
        raise HTTPException(500, "Veritabanı hatası.")


@app.get("/api/log-son")
def api_log_son():
    """Son log satırlarını döner (dashboard için)."""
    yol = _PROJE_KOKU / CONFIG.get("LOG_DOSYA", "otokantar.log")
    if not yol.is_file():
        return {"satirlar": []}
    try:
        text = yol.read_text(encoding="utf-8", errors="ignore")
        tum = [s for s in text.splitlines() if s.strip()]
        return {"satirlar": tum[-8:]}
    except Exception:
        return {"satirlar": []}


@app.get("/canli_kare.jpg")
def api_canli_kare_dosyasi():
    """Anlık kamera karesini döner."""
    yol = _PROJE_KOKU / CONFIG.get("CANLI_KARE_DOSYA", "canli_kare.jpg")
    if not yol.is_file():
        raise HTTPException(404, "Canlı kare henüz yok.")
    return FileResponse(
        yol,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, max-age=0"},
    )
