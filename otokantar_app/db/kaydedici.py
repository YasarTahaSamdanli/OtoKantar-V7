import csv
import json
import queue
import sqlite3
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from otokantar_app.logger import log
from otokantar_app.models import PlakaKayit


class KantarKaydedici:
    _RETRY_GECIKME = [0.5, 1.0, 2.0]

    def __init__(self, csv_dosya: str, json_dosya: str, db_dosya: str):
        self.csv_dosya = csv_dosya
        self.json_dosya = json_dosya
        self.db_dosya = db_dosya
        self.son_kayitlar: list = []
        self._kilit = threading.Lock()
        self._csv_aktif = True
        self._db_kuyrugu: queue.Queue = queue.Queue()
        self._db_dur = threading.Event()
        self._db_thread = threading.Thread(target=self._db_writer_loop, name="DbWriter", daemon=True)
        self._db_kur_sema()
        self._csv_baslik_yaz()
        self._db_thread.start()
        log.info("DbWriter thread başlatıldı (WAL + retry aktif).")

    # ------------------------------------------------------------------
    # ŞEMA KURULUM & MİGRASYON
    # ------------------------------------------------------------------
    def _db_kur_sema(self):
        con = sqlite3.connect(self.db_dosya, timeout=10)
        try:
            con.execute("PRAGMA journal_mode=WAL;")
            con.execute("PRAGMA synchronous=NORMAL;")
            con.execute("PRAGMA wal_autocheckpoint=100;")

            # ── kayitli_araclar ────────────────────────────────────────
            con.execute("""
                CREATE TABLE IF NOT EXISTS kayitli_araclar (
                    plaka            TEXT PRIMARY KEY,
                    ilk_kayit_tarihi TEXT,
                    firma_adi        TEXT,
                    sofor_adi        TEXT,
                    sofor_tel        TEXT
                )
            """)

            # Eski tabloya eksik kolon migrasyonu
            mevcut_kolon = {
                str(r[1]).lower()
                for r in con.execute("PRAGMA table_info(kayitli_araclar)").fetchall()
            }
            for kolon, tip in [("firma_adi", "TEXT"), ("sofor_adi", "TEXT"), ("sofor_tel", "TEXT")]:
                if kolon not in mevcut_kolon:
                    con.execute(f"ALTER TABLE kayitli_araclar ADD COLUMN {kolon} {tip}")
                    log.info("Migrasyon: kayitli_araclar.%s eklendi.", kolon)

            # ── gecis_raporlari ────────────────────────────────────────
            con.execute("""
                CREATE TABLE IF NOT EXISTS gecis_raporlari (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    plaka         TEXT NOT NULL,
                    giris_tarih   TEXT NOT NULL,
                    giris_saat    TEXT NOT NULL,
                    giris_agirlik REAL NOT NULL DEFAULT 0.0,
                    cikis_tarih   TEXT,
                    cikis_saat    TEXT,
                    cikis_agirlik REAL,
                    net_agirlik   REAL,
                    durum         TEXT NOT NULL DEFAULT 'ICERIDE',
                    guven         REAL DEFAULT 0.0,
                    malzeme_cinsi TEXT,
                    irsaliye_no   TEXT
                )
            """)

            # Eski gecis_raporlari tablosuna eksik kolon migrasyonu
            gecis_kolonlar = {
                str(r[1]).lower()
                for r in con.execute("PRAGMA table_info(gecis_raporlari)").fetchall()
            }

            if "giris_tarih" not in gecis_kolonlar:
                # Çok eski şema — tam tablo yeniden oluştur
                log.warning("Eski gecis_raporlari şeması tespit edildi, migrasyon başlatılıyor...")
                con.execute("DROP TABLE IF EXISTS gecis_raporlari_yeni")
                con.execute("""
                    CREATE TABLE gecis_raporlari_yeni (
                        id            INTEGER PRIMARY KEY AUTOINCREMENT,
                        plaka         TEXT NOT NULL,
                        giris_tarih   TEXT NOT NULL,
                        giris_saat    TEXT NOT NULL,
                        giris_agirlik REAL NOT NULL DEFAULT 0.0,
                        cikis_tarih   TEXT,
                        cikis_saat    TEXT,
                        cikis_agirlik REAL,
                        net_agirlik   REAL,
                        durum         TEXT NOT NULL DEFAULT 'ICERIDE',
                        guven         REAL DEFAULT 0.0,
                        malzeme_cinsi TEXT,
                        irsaliye_no   TEXT
                    )
                """)
                tarih_src = "tarih" if "tarih" in gecis_kolonlar else "DATE('now')"
                saat_src = "saat" if "saat" in gecis_kolonlar else "TIME('now')"
                agirlik_src = "agirlik" if "agirlik" in gecis_kolonlar else "0.0"
                guven_src = "guven" if "guven" in gecis_kolonlar else "0.0"
                con.execute(f"""
                    INSERT INTO gecis_raporlari_yeni
                    (id, plaka, giris_tarih, giris_saat, giris_agirlik, durum, guven)
                    SELECT
                        id,
                        COALESCE(plaka, ''),
                        COALESCE({tarih_src}, DATE('now')),
                        COALESCE({saat_src}, TIME('now')),
                        COALESCE({agirlik_src}, 0.0),
                        'TAMAMLANDI',
                        COALESCE({guven_src}, 0.0)
                    FROM gecis_raporlari
                """)
                con.execute("DROP TABLE gecis_raporlari")
                con.execute("ALTER TABLE gecis_raporlari_yeni RENAME TO gecis_raporlari")
                log.info("gecis_raporlari tam migrasyon tamamlandı.")
            else:
                # Yeni yeterince taze tablo — sadece eksik kolonları ekle
                for kolon, tip in [("malzeme_cinsi", "TEXT"), ("irsaliye_no", "TEXT")]:
                    if kolon not in gecis_kolonlar:
                        con.execute(f"ALTER TABLE gecis_raporlari ADD COLUMN {kolon} {tip}")
                        log.info("Migrasyon: gecis_raporlari.%s eklendi.", kolon)

            # ── kara_liste ─────────────────────────────────────────────
            con.execute("""
                CREATE TABLE IF NOT EXISTS kara_liste (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    plaka          TEXT UNIQUE NOT NULL,
                    eklenme_tarihi DATETIME NOT NULL DEFAULT (datetime('now', 'localtime'))
                )
            """)

            # İndexler
            con.execute("CREATE INDEX IF NOT EXISTS idx_plaka ON kayitli_araclar(plaka)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_gecis_plaka_giris_tarih ON gecis_raporlari(plaka, giris_tarih)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_gecis_plaka_durum ON gecis_raporlari(plaka, durum)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_kara_liste_plaka ON kara_liste(plaka)")
            con.commit()
            log.debug("SQLite WAL şema hazır: %s", self.db_dosya)
        finally:
            con.close()

    # ------------------------------------------------------------------
    # KARA LİSTE
    # ------------------------------------------------------------------
    def plaka_kara_listede_mi(self, plaka: str) -> bool:
        con = sqlite3.connect(self.db_dosya, timeout=10)
        try:
            row = con.execute("SELECT 1 FROM kara_liste WHERE plaka = ? LIMIT 1", (plaka,)).fetchone()
            return row is not None
        except Exception as e:
            log.warning("Kara liste sorgulanırken hata: %s", e)
            return False
        finally:
            con.close()

    # ------------------------------------------------------------------
    # ARAÇ SİCİL (kayitli_araclar) OKUMA / YAZMA
    # ------------------------------------------------------------------
    def arac_bilgi_getir(self, plaka: str) -> Optional[dict]:
        """
        Plakaya ait şoför / firma bilgisini döner.
        Bulunamazsa None döner.
        """
        con = sqlite3.connect(self.db_dosya, timeout=10)
        con.row_factory = sqlite3.Row
        try:
            row = con.execute(
                "SELECT firma_adi, sofor_adi, sofor_tel FROM kayitli_araclar WHERE plaka = ? LIMIT 1",
                (plaka,),
            ).fetchone()
            return dict(row) if row else None
        except Exception as e:
            log.warning("Araç bilgisi getirme hatası (%s): %s", plaka, e)
            return None
        finally:
            con.close()

    def arac_bilgi_guncelle(self, plaka: str, firma_adi: Optional[str] = None,
                             sofor_adi: Optional[str] = None, sofor_tel: Optional[str] = None) -> None:
        """
        kayitli_araclar tablosuna UPSERT yapar.
        Sadece None olmayan alanlar güncellenir.
        """
        con = sqlite3.connect(self.db_dosya, timeout=10)
        try:
            con.execute("PRAGMA journal_mode=WAL;")
            # Kayıt yoksa oluştur
            con.execute(
                "INSERT OR IGNORE INTO kayitli_araclar (plaka, ilk_kayit_tarihi) VALUES (?, ?)",
                (plaka, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            )
            # Alanları güncelle
            if firma_adi is not None:
                con.execute("UPDATE kayitli_araclar SET firma_adi = ? WHERE plaka = ?", (firma_adi, plaka))
            if sofor_adi is not None:
                con.execute("UPDATE kayitli_araclar SET sofor_adi = ? WHERE plaka = ?", (sofor_adi, plaka))
            if sofor_tel is not None:
                con.execute("UPDATE kayitli_araclar SET sofor_tel = ? WHERE plaka = ?", (sofor_tel, plaka))
            con.commit()
            log.info("Araç sicil güncellendi: %s | firma=%s | şoför=%s", plaka, firma_adi, sofor_adi)
        except Exception as e:
            log.error("Araç bilgisi güncellenirken hata (%s): %s", plaka, e)
        finally:
            con.close()

    def aktif_seans_ek_veri_gir(self, plaka: str, malzeme_cinsi: Optional[str] = None,
                                  irsaliye_no: Optional[str] = None) -> bool:
        """
        'ICERIDE' durumundaki son geçiş kaydına malzeme ve irsaliye bilgisi yazar.
        Başarılıysa True, seans bulunamazsa False döner.
        """
        con = sqlite3.connect(self.db_dosya, timeout=10)
        try:
            con.execute("PRAGMA journal_mode=WAL;")
            updates = []
            params = []
            if malzeme_cinsi is not None:
                updates.append("malzeme_cinsi = ?")
                params.append(malzeme_cinsi)
            if irsaliye_no is not None:
                updates.append("irsaliye_no = ?")
                params.append(irsaliye_no)
            if not updates:
                return False
            params.append(plaka)
            sql = (
                f"UPDATE gecis_raporlari SET {', '.join(updates)} "
                "WHERE id = (SELECT id FROM gecis_raporlari WHERE plaka = ? AND durum = 'ICERIDE' ORDER BY id DESC LIMIT 1)"
            )
            cur = con.execute(sql, params)
            con.commit()
            if cur.rowcount > 0:
                log.info("Aktif seans ek veri eklendi: %s | malzeme=%s | irsaliye=%s", plaka, malzeme_cinsi, irsaliye_no)
                return True
            log.warning("Aktif seans bulunamadı (ek veri): %s", plaka)
            return False
        except Exception as e:
            log.error("Aktif seans ek veri hatası (%s): %s", plaka, e)
            return False
        finally:
            con.close()

    # ------------------------------------------------------------------
    # GEÇİŞ KAYITLARI
    # ------------------------------------------------------------------
    def acik_seans_getir(self, plaka: str) -> Optional[dict]:
        con = sqlite3.connect(self.db_dosya, timeout=10)
        con.row_factory = sqlite3.Row
        try:
            row = con.execute(
                "SELECT * FROM gecis_raporlari WHERE plaka = ? AND durum = 'ICERIDE' ORDER BY id DESC LIMIT 1",
                (plaka,),
            ).fetchone()
            return dict(row) if row is not None else None
        finally:
            con.close()

    def giris_kaydet(self, plaka: str, agirlik: float) -> PlakaKayit:
        plaka = plaka.strip().upper()
        simdi = datetime.now()

        # Araç daha önce kaydedilmişse sicil bilgilerini otomatik çek
        sicil = self.arac_bilgi_getir(plaka)

        kayit = PlakaKayit(
            plaka=plaka,
            giris_tarih=simdi.strftime("%Y-%m-%d"),
            giris_saat=simdi.strftime("%H:%M:%S"),
            giris_agirlik=float(agirlik),
            durum="ICERIDE",
            firma_adi=sicil.get("firma_adi") if sicil else None,
            sofor_adi=sicil.get("sofor_adi") if sicil else None,
            sofor_tel=sicil.get("sofor_tel") if sicil else None,
        )
        self.gecis_kaydet(kayit)
        return kayit

    def cikis_kaydet(self, plaka: str, agirlik: float) -> Optional[PlakaKayit]:
        plaka = plaka.strip().upper()
        simdi = datetime.now()
        acik = self.acik_seans_getir(plaka)
        if acik is None:
            log.warning("Çıkış kaydı atlandı: açık seans yok (%s)", plaka)
            return None
        giris_agirlik = float(acik.get("giris_agirlik") or 0.0)
        cikis_agirlik = float(agirlik)
        kayit = PlakaKayit(
            plaka=plaka,
            giris_tarih=str(acik.get("giris_tarih") or simdi.strftime("%Y-%m-%d")),
            giris_saat=str(acik.get("giris_saat") or simdi.strftime("%H:%M:%S")),
            giris_agirlik=giris_agirlik,
            guven=float(acik.get("guven") or 0.0),
            cikis_tarih=simdi.strftime("%Y-%m-%d"),
            cikis_saat=simdi.strftime("%H:%M:%S"),
            cikis_agirlik=cikis_agirlik,
            net_agirlik=abs(giris_agirlik - cikis_agirlik),
            durum="TAMAMLANDI",
            firma_adi=acik.get("firma_adi"),
            sofor_adi=acik.get("sofor_adi"),
            sofor_tel=acik.get("sofor_tel"),
            malzeme_cinsi=acik.get("malzeme_cinsi"),
            irsaliye_no=acik.get("irsaliye_no"),
        )
        self.gecis_kaydet(kayit)
        return kayit

    # ------------------------------------------------------------------
    # ASYNC DB YAZICI
    # ------------------------------------------------------------------
    def _db_writer_loop(self) -> None:
        def _yaz(con: sqlite3.Connection, kayit: PlakaKayit) -> None:
            cur = con.cursor()
            cur.execute("SELECT 1 FROM kayitli_araclar WHERE plaka = ? LIMIT 1", (kayit.plaka,))
            if cur.fetchone() is None:
                cur.execute(
                    """INSERT INTO kayitli_araclar
                       (plaka, ilk_kayit_tarihi, firma_adi, sofor_adi, sofor_tel)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        kayit.plaka,
                        f"{kayit.giris_tarih} {kayit.giris_saat}",
                        kayit.firma_adi,
                        kayit.sofor_adi,
                        kayit.sofor_tel,
                    ),
                )
            else:
                # Sicil bilgisi geldiyse güncelle (None değilse)
                if kayit.firma_adi is not None:
                    cur.execute("UPDATE kayitli_araclar SET firma_adi=? WHERE plaka=?", (kayit.firma_adi, kayit.plaka))
                if kayit.sofor_adi is not None:
                    cur.execute("UPDATE kayitli_araclar SET sofor_adi=? WHERE plaka=?", (kayit.sofor_adi, kayit.plaka))
                if kayit.sofor_tel is not None:
                    cur.execute("UPDATE kayitli_araclar SET sofor_tel=? WHERE plaka=?", (kayit.sofor_tel, kayit.plaka))

            if kayit.durum == "ICERIDE":
                cur.execute(
                    """INSERT INTO gecis_raporlari
                       (plaka, giris_tarih, giris_saat, giris_agirlik, durum, guven,
                        malzeme_cinsi, irsaliye_no)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        kayit.plaka, kayit.giris_tarih, kayit.giris_saat,
                        float(kayit.giris_agirlik), "ICERIDE", float(kayit.guven),
                        kayit.malzeme_cinsi, kayit.irsaliye_no,
                    ),
                )
            elif kayit.durum == "TAMAMLANDI":
                cur.execute(
                    """UPDATE gecis_raporlari
                       SET cikis_tarih=?, cikis_saat=?, cikis_agirlik=?, net_agirlik=?,
                           durum='TAMAMLANDI', malzeme_cinsi=?, irsaliye_no=?
                       WHERE id = (
                           SELECT id FROM gecis_raporlari
                           WHERE plaka=? AND durum='ICERIDE'
                           ORDER BY id DESC LIMIT 1
                       )""",
                    (
                        kayit.cikis_tarih, kayit.cikis_saat,
                        float(kayit.cikis_agirlik or 0.0),
                        float(kayit.net_agirlik or 0.0),
                        kayit.malzeme_cinsi, kayit.irsaliye_no,
                        kayit.plaka,
                    ),
                )
                if cur.rowcount == 0:
                    cur.execute(
                        """INSERT INTO gecis_raporlari
                           (plaka, giris_tarih, giris_saat, giris_agirlik,
                            cikis_tarih, cikis_saat, cikis_agirlik, net_agirlik,
                            durum, guven, malzeme_cinsi, irsaliye_no)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            kayit.plaka, kayit.giris_tarih, kayit.giris_saat,
                            float(kayit.giris_agirlik),
                            kayit.cikis_tarih, kayit.cikis_saat,
                            float(kayit.cikis_agirlik or 0.0),
                            float(kayit.net_agirlik or 0.0),
                            "TAMAMLANDI", float(kayit.guven),
                            kayit.malzeme_cinsi, kayit.irsaliye_no,
                        ),
                    )
            con.commit()

        con = sqlite3.connect(self.db_dosya, timeout=30)
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        con.execute("PRAGMA wal_autocheckpoint=100;")
        log.debug("DbWriter bağlantısı açıldı.")
        try:
            while not self._db_dur.is_set():
                try:
                    kayit = self._db_kuyrugu.get(timeout=1.0)
                except queue.Empty:
                    continue
                if kayit is None:
                    break
                basarili = False
                for deneme, bekleme in enumerate(self._RETRY_GECIKME, start=1):
                    try:
                        _yaz(con, kayit)
                        log.debug("DbWriter: kayıt yazıldı → %s (deneme %d)", kayit.plaka, deneme)
                        basarili = True
                        break
                    except sqlite3.OperationalError as e:
                        log.warning("DbWriter geçici SQLite hatası (deneme %d/%d): %s — %.1fsn bekle.", deneme, len(self._RETRY_GECIKME), e, bekleme)
                        try:
                            con.rollback()
                        except Exception:
                            pass
                        time.sleep(bekleme)
                    except sqlite3.IntegrityError as e:
                        log.warning("DbWriter IntegrityError (atlandı): %s → %s", e, kayit.plaka)
                        basarili = True
                        break
                    except Exception as e:
                        log.error("DbWriter beklenmedik hata (%s): %s", type(e).__name__, e)
                        break
                if not basarili:
                    log.error("DbWriter: %d deneme sonrası BAŞARISIZ → %s %.1fkg", len(self._RETRY_GECIKME), kayit.plaka, kayit.giris_agirlik)
        finally:
            con.close()
            log.debug("DbWriter bağlantısı kapatıldı.")

    def kapat(self) -> None:
        log.info("KantarKaydedici: kuyruk boşaltılıyor...")
        bitis = time.time() + 10.0
        while not self._db_kuyrugu.empty() and time.time() < bitis:
            time.sleep(0.1)
        self._db_dur.set()
        self._db_kuyrugu.put(None)
        self._db_thread.join(timeout=8.0)
        if self._db_thread.is_alive():
            log.warning("DbWriter thread 8sn içinde sonlanmadı — zorla bırakılıyor.")
        else:
            log.info("DbWriter thread temiz kapandı.")

    # ------------------------------------------------------------------
    # CSV / JSON
    # ------------------------------------------------------------------
    def _csv_baslik_yaz(self):
        dosya_var = Path(self.csv_dosya).exists()
        try:
            with open(self.csv_dosya, mode="a", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f, delimiter=";")
                if not dosya_var:
                    w.writerow([
                        "Plaka", "Durum", "GirisTarih", "GirisSaat", "GirisAgirlik(kg)",
                        "CikisTarih", "CikisSaat", "CikisAgirlik(kg)", "NetAgirlik(kg)",
                        "Guven", "Operator", "FirmaAdi", "SoforAdi", "SoforTel",
                        "MalzemeCinsi", "IrsaliyeNo",
                    ])
        except PermissionError as e:
            self._csv_aktif = False
            log.warning("CSV erişilemedi, devre dışı: %s", e)

    def gecis_kaydet(self, kayit: PlakaKayit):
        with self._kilit:
            try:
                self._db_kuyrugu.put_nowait(kayit)
            except queue.Full:
                log.error("DbWriter kuyruğu dolu — kayıt %s geçici atlandı!", kayit.plaka)
            if self._csv_aktif:
                try:
                    with open(self.csv_dosya, mode="a", newline="", encoding="utf-8-sig") as f:
                        w = csv.writer(f, delimiter=";")
                        w.writerow([
                            kayit.plaka, kayit.durum, kayit.giris_tarih, kayit.giris_saat,
                            f"{kayit.giris_agirlik:.1f}", kayit.cikis_tarih or "", kayit.cikis_saat or "",
                            "" if kayit.cikis_agirlik is None else f"{kayit.cikis_agirlik:.1f}",
                            "" if kayit.net_agirlik is None else f"{kayit.net_agirlik:.1f}",
                            f"{kayit.guven:.2f}", kayit.operator,
                            kayit.firma_adi or "", kayit.sofor_adi or "", kayit.sofor_tel or "",
                            kayit.malzeme_cinsi or "", kayit.irsaliye_no or "",
                        ])
                except PermissionError as e:
                    self._csv_aktif = False
                    log.warning("CSV yazılamadı: %s", e)
            self.son_kayitlar.append(kayit)
            if len(self.son_kayitlar) > 50:
                self.son_kayitlar = self.son_kayitlar[-50:]
            self._json_guncelle(kayit)
            log.info(
                "KAYIT: %s | %s | giris=%.1fkg | cikis=%s | net=%s | firma=%s",
                kayit.plaka, kayit.durum, kayit.giris_agirlik,
                "-" if kayit.cikis_agirlik is None else f"{kayit.cikis_agirlik:.1f}kg",
                "-" if kayit.net_agirlik is None else f"{kayit.net_agirlik:.1f}kg",
                kayit.firma_adi or "-",
            )

    def kaydet(self, kayit: PlakaKayit):
        self.gecis_kaydet(kayit)

    def _json_guncelle(self, son_kayit: PlakaKayit):
        try:
            with open(self.json_dosya, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "son_guncelleme": datetime.now().isoformat(),
                        "son_kayit": asdict(son_kayit),
                        "son_10": [asdict(k) for k in self.son_kayitlar[-10:]],
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception as e:
            log.warning("JSON güncellenemedi: %s", e)