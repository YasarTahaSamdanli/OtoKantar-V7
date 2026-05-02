import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from otokantar_app.config import CONFIG, PLAKA_REGEX
from otokantar_app.db.mysql_manager import MySQLDBManager
from otokantar_app.logger import log

app = FastAPI(title="OtoKantar V11 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Multiprocessing nedeniyle bu referans alt süreçte (FastAPI) None kalabilir.
# Bu yüzden veriye erişimde her zaman Dosya/Veritabanı önceliklidir.
sistem_referansi = None

# Proje kök dizini (main.py'nin olduğu yer)
_PROJE_KOKU = Path(__file__).resolve().parents[2]


def sistem_referansi_ata(sistem) -> None:
    global sistem_referansi
    sistem_referansi = sistem


def _db_baglantisi_kur():
    """MySQL yöneticisi döndürür."""
    return MySQLDBManager.from_config(CONFIG)


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


# ──────────────────────────────────────────────────────────────────────
# MEVCUT ENDPOINTLER
# ──────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(_PROJE_KOKU / "dashboard.html")


@app.get("/api/canli-durum")
def api_canli_durum():
    """Kantarın anlık durumunu döner."""
    durum = _canli_json_dosyadan_oku()
    if sistem_referansi:
        durum.update({
            "fps": round(getattr(sistem_referansi, "_fps", 0.0), 1),
            "kilitli": getattr(sistem_referansi, "_kantar_seans_kilitli", False),
            "plaka_buffer": sistem_referansi._plaka_buffer.plaka if sistem_referansi._plaka_buffer else None,
        })
    return durum


@app.get("/api/son-kayitlar")
def api_son_kayitlar():
    """Son 50 geçiş kaydını MySQL'den çeker."""
    try:
        db = _db_baglantisi_kur()
        return db.son_gecisler(50)
    except Exception as e:
        log.error(f"Son kayıtlar DB hatası: {e}")
        return []


class KaraListeEkleIstek(BaseModel):
    plaka: str


@app.get("/api/kara-liste")
def api_kara_liste_listele():
    """MySQL'deki güncel kara listeyi döner."""
    try:
        db = _db_baglantisi_kur()
        return {"kara_liste": db.kara_liste_listele()}
    except Exception as e:
        log.error(f"Kara liste listeleme hatası: {e}")
        return {"kara_liste": []}


@app.post("/api/kara-liste")
def api_kara_liste_ekle(istek: KaraListeEkleIstek):
    """MySQL veritabanına yeni yasaklı plaka ekler."""
    plaka = istek.plaka.strip().upper()
    if not PLAKA_REGEX.fullmatch(plaka):
        raise HTTPException(422, f"Geçersiz Türk plaka formatı: '{plaka}'")
    try:
        db = _db_baglantisi_kur()
        db.kara_liste_guncelle(plaka, True)
        log.info("Kara listeye yeni plaka eklendi (API): %s", plaka)
        return {"mesaj": "Başarılı", "plaka": plaka}
    except Exception as e:
        log.error(f"Kara liste ekleme hatası: {e}")
        raise HTTPException(500, "Veritabanına yazılırken bir hata oluştu.")


@app.delete("/api/kara-liste/{plaka}")
def api_kara_liste_sil(plaka: str):
    """MySQL veritabanından plaka yasaklamasını kaldırır."""
    plaka = plaka.strip().upper()
    try:
        db = _db_baglantisi_kur()
        db.kara_liste_guncelle(plaka, False)
        return {"mesaj": "Silindi", "plaka": plaka}
    except Exception as e:
        log.error(f"Kara liste silme hatası: {e}")
        raise HTTPException(500, "Veritabanı hatası.")


# ──────────────────────────────────────────────────────────────────────
# YENİ ENDPOINTLER — ARAÇ SİCİL & LOJİSTİK VERİ
# ──────────────────────────────────────────────────────────────────────

@app.get("/api/arac/{plaka}")
def api_arac_bilgi_getir(plaka: str):
    """Verilen plakanın MySQL araç kaydını döner."""
    plaka = plaka.strip().upper()
    try:
        db = _db_baglantisi_kur()
        arac_id, kara_liste = db.upsert_arac(plaka)
        return {
            "plaka": plaka,
            "kayitli": True,
            "id": arac_id,
            "kara_liste": kara_liste,
        }
    except Exception as e:
        log.error("Araç bilgisi getirme hatası (%s): %s", plaka, e)
        raise HTTPException(500, "Veritabanı hatası.")


class AracGuncelleIstek(BaseModel):
    plaka: str
    firma_adi: Optional[str] = None
    sofor_adi: Optional[str] = None
    sofor_tel: Optional[str] = None


@app.post("/api/arac/guncelle")
def api_arac_guncelle(istek: AracGuncelleIstek):
    """
    Araç sicilini (firma / şoför / telefon) günceller veya ilk kez oluşturur.
    En az bir alan gönderilmelidir.
    """
    plaka = istek.plaka.strip().upper()
    if not PLAKA_REGEX.fullmatch(plaka):
        raise HTTPException(422, f"Geçersiz Türk plaka formatı: '{plaka}'")
    if istek.firma_adi is None and istek.sofor_adi is None and istek.sofor_tel is None:
        raise HTTPException(422, "MySQL şemasında sadece plaka/kara_liste mevcut.")
    try:
        db = _db_baglantisi_kur()
        arac_id, _ = db.upsert_arac(plaka)
        log.info("Araç sicil API ile güncellendi (yalnız plaka): %s", plaka)
        return {"mesaj": "Araç kaydı güncellendi.", "plaka": plaka, "id": arac_id}
    except Exception as e:
        log.error("Araç güncelleme hatası (%s): %s", plaka, e)
        raise HTTPException(500, "Veritabanı hatası.")


class EkVeriIstek(BaseModel):
    plaka: str
    malzeme_cinsi: Optional[str] = None
    irsaliye_no: Optional[str] = None


@app.post("/api/gecis/ek-veri-gir")
def api_gecis_ek_veri_gir(istek: EkVeriIstek):
    """
    'ICERIDE' durumundaki aktif kantara malzeme cinsi ve irsaliye no yazar.
    Plaka gönderilmezse veritabanındaki son açık seansı otomatik bulur.
    """
    plaka = istek.plaka.strip().upper() if istek.plaka else None
    if not plaka:
        raise HTTPException(422, "Plaka zorunludur.")
    if istek.malzeme_cinsi is None and istek.irsaliye_no is None:
        raise HTTPException(422, "En az bir alan gönderilmelidir.")
    try:
        log.info(
            "Ek veri endpoint çağrıldı fakat MySQL şeması bu alanları tutmuyor: %s",
            plaka,
        )
        return {"mesaj": "MySQL şemasında ek veri alanı yok.", "plaka": plaka}
    except HTTPException:
        raise
    except Exception as e:
        log.error("Ek veri kaydetme hatası (%s): %s", plaka, e)
        raise HTTPException(500, "Veritabanı hatası.")