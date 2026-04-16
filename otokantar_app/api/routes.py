import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from otokantar_app.config import CONFIG, PLAKA_REGEX
from otokantar_app.logger import log


app = FastAPI(title="OtoKantar V11 API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
sistem_referansi = None
# repo kökü (…/otokantar_app/api/routes.py -> …/OtoKantar_V7)
_PROJE_KOKU = Path(__file__).resolve().parents[2]


def sistem_referansi_ata(sistem) -> None:
    global sistem_referansi
    sistem_referansi = sistem


def _canli_json_dosyadan_oku() -> dict:
    yol = _PROJE_KOKU / CONFIG["JSON_CANLI"]
    if not yol.is_file():
        return {"son_guncelleme": None, "son_kayit": None, "son_10": []}
    try:
        with open(yol, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"son_guncelleme": None, "son_kayit": None, "son_10": []}


def _tail_son_satirlar(dosya: Path, n: int = 8, max_bytes: int = 256_000) -> list[str]:
    """
    Log dosyasının tamamını okumadan son N satırı döndür.
    Büyük loglarda /api/log-son çağrıları UI'ı kastırmasın diye dosya sonundan okunur.
    """
    n = max(1, int(n))
    if not dosya.is_file():
        return []
    try:
        with open(dosya, "rb") as f:
            f.seek(0, 2)
            end = f.tell()
            if end <= 0:
                return []
            chunk = 8192
            pos = end
            buf = b""
            while pos > 0 and len(buf) < max_bytes:
                read_size = chunk if pos >= chunk else pos
                pos -= read_size
                f.seek(pos)
                buf = f.read(read_size) + buf
                if buf.count(b"\n") >= (n + 1):
                    break
        text = buf.decode("utf-8", errors="ignore")
        lines = [s for s in text.splitlines() if s.strip()]
        return lines[-n:]
    except Exception:
        return []


@app.get("/")
def dashboard_sayfa():
    html_yol = _PROJE_KOKU / "dashboard.html"
    if not html_yol.is_file():
        raise HTTPException(404, "dashboard.html bulunamadı.")
    return FileResponse(html_yol, media_type="text/html; charset=utf-8")


@app.get("/dashboard.html")
def dashboard_sayfa_alias():
    return dashboard_sayfa()


@app.get("/api/canli-durum")
def api_canli_durum():
    veri = _canli_json_dosyadan_oku()
    if sistem_referansi is not None:
        ag, st = sistem_referansi.kantar_okuyucu.veri
        veri["kantar_kg"] = round(ag, 1)
        veri["kantar_sabit"] = st
        veri["plaka_buffer"] = sistem_referansi._plaka_buffer.plaka if sistem_referansi._plaka_buffer else None
        veri["seans_kilitli"] = sistem_referansi.seans_kilitli_mi
        kk = sistem_referansi.kaydedici.son_kayitlar
        if kk:
            veri["son_kayit"] = asdict(kk[-1])
            veri["son_10"] = [asdict(k) for k in kk[-10:]]
        veri["son_guncelleme"] = datetime.now().isoformat()
    else:
        veri.setdefault("kantar_kg", None)
        veri.setdefault("kantar_sabit", None)
        veri.setdefault("plaka_buffer", None)
        veri.setdefault("seans_kilitli", None)
    return veri


@app.get("/canli_kare.jpg")
def api_canli_kare_dosyasi():
    yol = _PROJE_KOKU / CONFIG["CANLI_KARE_DOSYA"]
    if not yol.is_file():
        raise HTTPException(404, "Canlı kare henüz yok.")
    return FileResponse(yol, media_type="image/jpeg", headers={"Cache-Control": "no-store, max-age=0"})


@app.get("/api/log-son")
def api_log_son():
    yol = _PROJE_KOKU / CONFIG["LOG_DOSYA"]
    if not yol.is_file():
        return {"satirlar": []}
    return {"satirlar": _tail_son_satirlar(yol, n=8)}


@app.get("/api/durum")
def api_durum():
    if sistem_referansi is None:
        return {"hata": "Sistem henüz başlatılmadı"}
    return {
        "kantar_kg": sistem_referansi.kantar_okuyucu.agirlik,
        "sabit": sistem_referansi.kantar_okuyucu.sabit,
        "seans_kilitli": sistem_referansi.seans_kilitli_mi,
        "plaka_buffer": sistem_referansi._plaka_buffer.plaka if sistem_referansi._plaka_buffer else None,
    }


@app.get("/api/son-kayitlar")
def api_son_kayitlar():
    if sistem_referansi is None:
        return []
    return [asdict(k) for k in sistem_referansi.kaydedici.son_kayitlar]


class KaraListeEkleIstek(BaseModel):
    plaka: str


@app.get("/api/kara-liste")
def api_kara_liste_listele():
    return {"kara_liste": list(CONFIG.get("KARA_LISTE", []))}


@app.post("/api/kara-liste")
def api_kara_liste_ekle(istek: KaraListeEkleIstek):
    plaka = istek.plaka.strip().upper()
    if not PLAKA_REGEX.fullmatch(plaka):
        raise HTTPException(422, f"Geçersiz Türk plaka formatı: '{plaka}'")
    kara_liste: list = CONFIG.setdefault("KARA_LISTE", [])
    if plaka in kara_liste:
        raise HTTPException(409, f"'{plaka}' zaten kara listede.")
    kara_liste.append(plaka)
    log.info("Kara listeye eklendi (API): %s", plaka)
    return {"eklendi": plaka, "kara_liste": list(kara_liste)}


@app.delete("/api/kara-liste/{plaka}")
def api_kara_liste_sil(plaka: str):
    plaka = plaka.strip().upper()
    kara_liste: list = CONFIG.get("KARA_LISTE", [])
    if plaka not in kara_liste:
        raise HTTPException(404, f"'{plaka}' kara listede bulunamadı.")
    kara_liste.remove(plaka)
    log.info("Kara listeden silindi (API): %s", plaka)
    return {"silindi": plaka, "kara_liste": list(kara_liste)}
