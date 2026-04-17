import json
import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

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
    """Son 50 geçiş kaydını doğrudan veritabanından çeker."""
    try:
        with _db_baglantisi_kur() as con:
            rows = con.execute(
                "SELECT * FROM gecis_raporlari ORDER BY id DESC LIMIT 50"
            ).fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        log.error(f"Son kayıtlar DB hatası: {e}")
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
def api_kara_liste_ekle(istek: KaraListeEkleIstek):
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
def api_kara_liste_sil(plaka: str):
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


# ──────────────────────────────────────────────────────────────────────
# YENİ ENDPOINTLER — ARAÇ SİCİL & LOJİSTİK VERİ
# ──────────────────────────────────────────────────────────────────────

@app.get("/api/arac/{plaka}")
def api_arac_bilgi_getir(plaka: str):
    """
    Verilen plakaya ait kayıtlı şoför / firma bilgisini döner.
    Araç daha önce hiç kaydedilmemişse 404 yerine boş alan döner,
    böylece frontend kolayca ayırt edebilir.
    """
    plaka = plaka.strip().upper()
    try:
        with _db_baglantisi_kur() as con:
            row = con.execute(
                "SELECT firma_adi, sofor_adi, sofor_tel FROM kayitli_araclar WHERE plaka = ? LIMIT 1",
                (plaka,),
            ).fetchone()
        if row is None:
            return {"plaka": plaka, "kayitli": False, "firma_adi": None, "sofor_adi": None, "sofor_tel": None}
        return {
            "plaka": plaka,
            "kayitli": True,
            "firma_adi": row["firma_adi"],
            "sofor_adi": row["sofor_adi"],
            "sofor_tel": row["sofor_tel"],
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
        raise HTTPException(422, "En az bir alan (firma_adi, sofor_adi, sofor_tel) gönderilmelidir.")
    try:
        with _db_baglantisi_kur() as con:
            con.execute(
                "INSERT OR IGNORE INTO kayitli_araclar (plaka, ilk_kayit_tarihi) VALUES (?, ?)",
                (plaka, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            )
            if istek.firma_adi is not None:
                con.execute("UPDATE kayitli_araclar SET firma_adi=? WHERE plaka=?", (istek.firma_adi, plaka))
            if istek.sofor_adi is not None:
                con.execute("UPDATE kayitli_araclar SET sofor_adi=? WHERE plaka=?", (istek.sofor_adi, plaka))
            if istek.sofor_tel is not None:
                con.execute("UPDATE kayitli_araclar SET sofor_tel=? WHERE plaka=?", (istek.sofor_tel, plaka))
            con.commit()
        log.info("Araç sicil API ile güncellendi: %s | firma=%s | şoför=%s", plaka, istek.firma_adi, istek.sofor_adi)
        return {"mesaj": "Araç sicili güncellendi.", "plaka": plaka}
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
        raise HTTPException(422, "En az bir alan (malzeme_cinsi, irsaliye_no) gönderilmelidir.")
    try:
        with _db_baglantisi_kur() as con:
            # Aktif seans var mı kontrol et
            row = con.execute(
                "SELECT id FROM gecis_raporlari WHERE plaka=? AND durum='ICERIDE' ORDER BY id DESC LIMIT 1",
                (plaka,),
            ).fetchone()
            if row is None:
                raise HTTPException(404, f"'{plaka}' için aktif (ICERIDE) seans bulunamadı.")

            updates: list[str] = []
            params: list = []
            if istek.malzeme_cinsi is not None:
                updates.append("malzeme_cinsi=?")
                params.append(istek.malzeme_cinsi)
            if istek.irsaliye_no is not None:
                updates.append("irsaliye_no=?")
                params.append(istek.irsaliye_no)
            params.append(row["id"])

            con.execute(
                f"UPDATE gecis_raporlari SET {', '.join(updates)} WHERE id=?",
                params,
            )
            con.commit()
        log.info("Aktif seans ek veri (API): %s | malzeme=%s | irsaliye=%s",
                 plaka, istek.malzeme_cinsi, istek.irsaliye_no)
        return {"mesaj": "Ek veri kaydedildi.", "plaka": plaka}
    except HTTPException:
        raise
    except Exception as e:
        log.error("Ek veri kaydetme hatası (%s): %s", plaka, e)
        raise HTTPException(500, "Veritabanı hatası.")