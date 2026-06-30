import platform
import json
import re
import shutil
import sqlite3
import threading
import zipfile
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from otokantar_app.config import CONFIG
from otokantar_app.device_status import DeviceStatusStore
from otokantar_app.logger import log
from otokantar_app.models import PlakaKayit


if platform.system() == "Windows":
    try:
        import win32print
        _WIN32PRINT_OK = True
    except ImportError:
        _WIN32PRINT_OK = False
else:
    _WIN32PRINT_OK = False

try:
    import escpos.printer as escpos_printer
    _ESCPOS_OK = True
except ImportError:
    _ESCPOS_OK = False


class FisYazdirici:
    FIS_DOSYA = "kantar_fisi.txt"
    FIS_ARSIV_DIR = "fisler"
    FIS_ZIP_DIR = "fisler_zip"
    PRINT_LOG_DB = "receipt_log.sqlite"
    PRINTER_STATUS_DOSYA = "printer_status.json"
    GENISLIK = 42
    _ESC_INIT = b"\x1b\x40"
    _ESC_BOLD_ON = b"\x1b\x45\x01"
    _ESC_BOLD_OFF = b"\x1b\x45\x00"
    _ESC_CENTER = b"\x1b\x61\x01"
    _ESC_LEFT = b"\x1b\x61\x00"
    _ESC_FEED = b"\x1b\x64\x04"
    _ESC_CUT = b"\x1d\x56\x41\x00"

    def yazdir(self, kayit: PlakaKayit) -> None:
        backend = str(CONFIG.get("YAZICI_BACKEND", "file")).lower()
        if backend == "win32" and not _WIN32PRINT_OK:
            log.warning("win32print yok — 'file' moduna düşürüldü.")
            backend = "file"
        if backend == "escpos" and not _ESCPOS_OK:
            log.warning("python-escpos yok — 'file' moduna düşürüldü.")
            backend = "file"
        try:
            icerik = self._fis_metin_olustur(kayit)
            receipt_id = self._receipt_id(icerik)
            self._dosyaya_yaz(icerik)
            if self._receipt_success_var_mi(receipt_id):
                self._print_log_yaz(
                    receipt_id=receipt_id,
                    kayit=kayit,
                    backend=backend,
                    printer=str(CONFIG.get("YAZICI_ADI", "")).strip() or "(varsayılan)",
                    status="SKIPPED_DUPLICATE",
                    reason="Fiş daha önce başarıyla yazdırıldı.",
                )
                log.warning("Fiş tekrar yazdırılmadı; daha önce basılmış: %s", receipt_id)
                return
            if backend == "win32":
                self._win32_gonder_async(kayit, receipt_id)
            elif backend == "escpos":
                self._escpos_gonder_async(kayit, receipt_id)
            else:
                self._print_log_yaz(
                    receipt_id=receipt_id,
                    kayit=kayit,
                    backend=backend,
                    printer="file",
                    status="SAVED",
                )
                log.info("Yazıcı backend=file; fiş '%s' dosyasına kaydedildi.", self.FIS_DOSYA)
        except PermissionError as e:
            log.error("Fiş dosyası yazılamadı — izin hatası: %s", e)
        except OSError as e:
            log.error("Fiş dosyası yazılamadı — I/O hatası (errno %s): %s", e.errno, e.strerror)
        except Exception as e:
            log.error("FisYazdirici beklenmedik hata (%s): %s — devam ediyor.", type(e).__name__, e)

    def tekrar_yazdir(self, receipt_id: str) -> bool:
        receipt_id = self._receipt_id_temizle(receipt_id)
        if not receipt_id:
            log.warning("Tekrar yazdırma atlandı: geçersiz receipt_id.")
            return False

        found = self._arsivden_fis_oku(receipt_id)
        if found is None:
            log.warning("Tekrar yazdırma atlandı: fiş arşivde bulunamadı (%s).", receipt_id)
            return False

        icerik, kaynak = found
        kayit = self._kayit_fisten_olustur(icerik)
        ham_veri = self._fis_icerik_ham_olustur(icerik)
        backend = str(CONFIG.get("YAZICI_BACKEND", "file")).lower()

        if backend == "win32" and not _WIN32PRINT_OK:
            reason = "win32print yok"
            self._print_log_yaz(receipt_id, kayit, backend, "(varsayılan)", "MANUAL_REPRINT_FAILED", reason)
            log.warning("Tekrar yazdırma başarısız: %s", reason)
            return False
        if backend == "escpos" and not _ESCPOS_OK:
            reason = "python-escpos yok"
            self._print_log_yaz(receipt_id, kayit, backend, "escpos", "MANUAL_REPRINT_FAILED", reason)
            log.warning("Tekrar yazdırma başarısız: %s", reason)
            return False
        if backend not in {"win32", "escpos"}:
            reason = "Yazıcı backend=file; fiziksel baskı yapılmadı"
            self._print_log_yaz(receipt_id, kayit, backend, "file", "MANUAL_REPRINT_SAVED", reason)
            log.info("Tekrar yazdırma dosya modunda kaldı: %s", receipt_id)
            return False

        try:
            printer = self._win32_ham_gonder(ham_veri) if backend == "win32" else self._escpos_ham_gonder(ham_veri)
            self._print_log_yaz(
                receipt_id,
                kayit,
                backend,
                printer,
                "MANUAL_REPRINT_SUCCESS",
                f"Arşivden tekrar yazdırıldı: {kaynak}",
            )
            log.info("Fiş manuel tekrar yazdırıldı: %s", receipt_id)
            return True
        except Exception as e:
            printer = str(CONFIG.get("YAZICI_ADI", "")).strip() or backend
            reason = f"{type(e).__name__}: {e}"
            self._print_log_yaz(receipt_id, kayit, backend, printer, "MANUAL_REPRINT_FAILED", reason)
            log.error("Tekrar yazdırma başarısız (%s): %s", receipt_id, reason)
            return False

    def _fis_metin_olustur(self, kayit: PlakaKayit) -> str:
        sep = "=" * self.GENISLIK
        dash = "-" * self.GENISLIK
        cikis_agirlik = kayit.cikis_agirlik if kayit.cikis_agirlik is not None else 0.0
        net_agirlik = kayit.net_agirlik if kayit.net_agirlik is not None else 0.0
        return "\n".join([
            sep, "         BRİKET FABRİKASI KANTAR FİŞİ", sep,
            f"  Giriş T. : {kayit.giris_tarih}", f"  Giriş S. : {kayit.giris_saat}", dash,
            f"  Plaka    : {kayit.plaka}", f"  Giriş Kg : {kayit.giris_agirlik:.1f} kg",
            f"  Çıkış Kg : {cikis_agirlik:.1f} kg", f"  Net Kg   : {net_agirlik:.1f} kg",
            f"  Durum    : {kayit.durum}", f"  Operatör : {kayit.operator}", dash,
            f"  Güven    : %{kayit.guven * 100:.1f}", sep, "        Teşekkür Ederiz — İyi Yolculuklar", sep, "",
        ])

    def _dosyaya_yaz(self, icerik: str) -> None:
        with open(self.FIS_DOSYA, "w", encoding="utf-8") as f:
            f.write(icerik)
        self._arsive_yaz(icerik)
        log.debug("Fiş dosyaya yazıldı: %s", self.FIS_DOSYA)

    def _arsive_yaz(self, icerik: str) -> Path:
        receipt_date = self._receipt_date(icerik)
        arsiv_dir = Path(self.FIS_ARSIV_DIR) / f"{receipt_date.year:04d}" / f"{receipt_date.month:02d}"
        arsiv_dir.mkdir(parents=True, exist_ok=True)
        hedef = arsiv_dir / f"{self._receipt_id(icerik)}.txt"
        hedef.write_text(icerik, encoding="utf-8")
        log.debug("Fiş arşive yazıldı: %s", hedef)
        self._eski_aylari_ziple()
        return hedef

    def _receipt_id(self, icerik: str) -> str:
        plaka = self._fis_alani(icerik, "Plaka") or "PLAKASIZ"
        tarih = self._fis_alani(icerik, "Giriş T.") or "TARIHSIZ"
        saat = self._fis_alani(icerik, "Giriş S.") or "SAATSIZ"
        durum = self._fis_alani(icerik, "Durum") or "DURUMSUZ"
        raw = f"{tarih}_{saat}_{plaka}_{durum}"
        return re.sub(r"[^A-Za-z0-9_-]+", "-", raw).strip("-")[:120]

    def _receipt_id_temizle(self, receipt_id: str) -> str:
        rid = str(receipt_id or "").strip()
        if rid.lower().endswith(".txt"):
            rid = rid[:-4]
        if re.fullmatch(r"[A-Za-z0-9_-]{1,120}", rid):
            return rid
        return ""

    def _arsivden_fis_oku(self, receipt_id: str) -> Optional[tuple[str, str]]:
        receipt_id = self._receipt_id_temizle(receipt_id)
        if not receipt_id:
            return None

        file_name = f"{receipt_id}.txt"
        root = Path(self.FIS_ARSIV_DIR)
        if root.is_dir():
            for file in sorted(root.rglob(file_name)):
                if file.is_file():
                    return file.read_text(encoding="utf-8"), str(file)

        zip_root = Path(self.FIS_ZIP_DIR)
        if zip_root.is_dir():
            for zip_path in sorted(zip_root.glob("*.zip")):
                try:
                    with zipfile.ZipFile(zip_path) as zf:
                        for member in zf.namelist():
                            if Path(member).name == file_name:
                                return zf.read(member).decode("utf-8"), f"{zip_path}!{member}"
                except Exception as e:
                    log.warning("Fiş ZIP arşivi okunamadı (%s): %s", zip_path, e)
        return None

    def _kayit_fisten_olustur(self, icerik: str) -> PlakaKayit:
        def kg(alan: str) -> Optional[float]:
            raw = self._fis_alani(icerik, alan).replace("kg", "").strip()
            if not raw:
                return None
            try:
                return float(raw.replace(",", "."))
            except ValueError:
                return None

        guven_raw = self._fis_alani(icerik, "Güven").replace("%", "").strip()
        try:
            guven = float(guven_raw.replace(",", ".")) / 100 if guven_raw else 0.0
        except ValueError:
            guven = 0.0

        return PlakaKayit(
            plaka=self._fis_alani(icerik, "Plaka") or "PLAKASIZ",
            giris_tarih=self._fis_alani(icerik, "Giriş T.") or date.today().isoformat(),
            giris_saat=self._fis_alani(icerik, "Giriş S.") or "00:00:00",
            giris_agirlik=kg("Giriş Kg") or 0.0,
            guven=guven,
            cikis_agirlik=kg("Çıkış Kg"),
            net_agirlik=kg("Net Kg"),
            durum=self._fis_alani(icerik, "Durum") or "TAMAMLANDI",
            operator=self._fis_alani(icerik, "Operatör") or "AUTO",
        )

    @staticmethod
    def _fis_alani(icerik: str, alan: str) -> str:
        pattern = rf"^\s*{re.escape(alan)}\s*:\s*(.+?)\s*$"
        match = re.search(pattern, icerik, flags=re.MULTILINE)
        return match.group(1).strip() if match else ""

    def _connect_print_log(self) -> sqlite3.Connection:
        path = Path(self.PRINT_LOG_DB)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _print_log_connection(self):
        conn = self._connect_print_log()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS receipt_print_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    receipt_id TEXT NOT NULL,
                    plate TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    printer TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reason TEXT,
                    printed_at TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_receipt_print_log_receipt
                ON receipt_print_log (receipt_id, created_at)
                """
            )
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _print_log_yaz(
        self,
        receipt_id: str,
        kayit: PlakaKayit,
        backend: str,
        printer: str,
        status: str,
        reason: Optional[str] = None,
    ) -> None:
        self._printer_status_yaz(
            receipt_id=receipt_id,
            kayit=kayit,
            backend=backend,
            printer=printer,
            status=status,
            reason=reason,
        )
        try:
            now = datetime.now().isoformat(timespec="seconds")
            with self._print_log_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO receipt_print_log (
                        receipt_id, plate, backend, printer, status,
                        reason, printed_at, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        receipt_id,
                        kayit.plaka,
                        backend,
                        printer,
                        status,
                        reason,
                        now if status in {"SUCCESS", "MANUAL_REPRINT_SUCCESS"} else None,
                        now,
                    ),
                )
            log.info(
                "Fiş yazdırma logu: Printer=%s Status=%s Vehicle=%s Reason=%s",
                printer,
                status,
                kayit.plaka,
                reason or "-",
            )
        except Exception as e:
            log.warning("Fiş yazdırma logu yazılamadı: %s", e)

    def _receipt_success_var_mi(self, receipt_id: str) -> bool:
        try:
            with self._print_log_connection() as conn:
                row = conn.execute(
                    """
                    SELECT 1
                    FROM receipt_print_log
                    WHERE receipt_id = ? AND status = 'SUCCESS'
                    LIMIT 1
                    """,
                    (receipt_id,),
                ).fetchone()
            return row is not None
        except Exception as e:
            log.warning("Fiş duplicate kontrolu yapılamadı: %s", e)
            return False

    def _printer_status_yaz(
        self,
        receipt_id: str,
        kayit: PlakaKayit,
        backend: str,
        printer: str,
        status: str,
        reason: Optional[str] = None,
    ) -> None:
        try:
            if status == "FAILED":
                level = "error"
                message = "Yazıcı çevrimdışı veya hata verdi."
            elif status == "MANUAL_REPRINT_FAILED":
                level = "error"
                message = "Fiş manuel tekrar yazdırılamadı."
            elif status == "SUCCESS":
                level = "ok"
                message = "Fiş yazıcıya gönderildi."
            elif status == "MANUAL_REPRINT_SUCCESS":
                level = "ok"
                message = "Fiş manuel tekrar yazdırıldı."
            elif status == "SKIPPED_DUPLICATE":
                level = "warning"
                message = "Fiş daha önce yazdırıldığı için tekrar basılmadı."
            elif status == "MANUAL_REPRINT_SAVED":
                level = "info"
                message = "Fiş manuel tekrar yazdırma isteği dosya modunda kaldı."
            else:
                level = "info"
                message = "Fiş dosyaya kaydedildi."
            payload = {
                "receipt_id": receipt_id,
                "plate": kayit.plaka,
                "backend": backend,
                "printer": printer,
                "status": status,
                "level": level,
                "message": message,
                "reason": reason,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            DeviceStatusStore.update(
                device_key="printer",
                status="FAILED" if status in {"FAILED", "MANUAL_REPRINT_FAILED"} else "OK",
                level=level,
                message=message,
                payload=payload,
                last_error=reason,
            )
            hedef = Path(self.PRINTER_STATUS_DOSYA)
            tmp = hedef.with_name(hedef.name + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            tmp.replace(hedef)
        except Exception as e:
            log.warning("Yazıcı durum dosyası yazılamadı: %s", e)

    @classmethod
    def yazici_durum_oku(cls) -> dict:
        device_status = DeviceStatusStore.read("printer")
        if device_status:
            payload = dict(device_status.get("payload") or {})
            fallback_status = "FAILED" if device_status["status"] == "FAILED" else "SUCCESS"
            payload.setdefault("status", fallback_status)
            payload.setdefault("level", device_status["level"])
            payload.setdefault("message", device_status["message"])
            payload.setdefault("reason", device_status.get("last_error"))
            payload.setdefault("updated_at", device_status["updated_at"])
            return payload

        path = Path(cls.PRINTER_STATUS_DOSYA)
        if not path.is_file():
            return {
                "status": "UNKNOWN",
                "level": "info",
                "message": "Yazıcı durumu henüz yok.",
            }
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {
                "status": "UNKNOWN",
                "level": "warning",
                "message": "Yazıcı durum dosyası okunamadı.",
            }
        except Exception as e:
            return {
                "status": "UNKNOWN",
                "level": "warning",
                "message": f"Yazıcı durum dosyası okunamadı: {e}",
            }

    def _receipt_date(self, icerik: str) -> date:
        tarih = self._fis_alani(icerik, "Giriş T.")
        try:
            return datetime.strptime(tarih, "%Y-%m-%d").date()
        except ValueError:
            return date.today()

    def _eski_aylari_ziple(
        self,
        today: date | None = None,
        older_than_days: int | None = None,
    ) -> None:
        today = today or date.today()
        older_than_days = int(
            older_than_days
            if older_than_days is not None
            else CONFIG.get("FIS_ARSIV_ZIP_OLDER_THAN_DAYS", 45)
        )
        cutoff = today - timedelta(days=max(0, older_than_days))
        root = Path(self.FIS_ARSIV_DIR)
        zip_root = Path(self.FIS_ZIP_DIR)
        if not root.is_dir():
            return

        for year_dir in sorted(root.iterdir()):
            if not year_dir.is_dir() or not year_dir.name.isdigit():
                continue
            for month_dir in sorted(year_dir.iterdir()):
                if not month_dir.is_dir() or not month_dir.name.isdigit():
                    continue
                try:
                    year = int(year_dir.name)
                    month = int(month_dir.name)
                    next_month = date(year + int(month == 12), 1 if month == 12 else month + 1, 1)
                    month_end = next_month - timedelta(days=1)
                except ValueError:
                    continue

                if month_end >= cutoff:
                    continue

                files = [p for p in sorted(month_dir.rglob("*")) if p.is_file()]
                if not files:
                    continue

                zip_root.mkdir(parents=True, exist_ok=True)
                zip_path = zip_root / f"{year:04d}-{month:02d}.zip"
                try:
                    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                        for file in files:
                            zf.write(file, arcname=str(file.relative_to(month_dir)))
                    shutil.rmtree(month_dir)
                    log.info("Eski fiş ayı ZIP arşive alındı: %s", zip_path)
                except Exception as e:
                    log.warning("Fiş ayı ZIP arşive alınamadı (%s): %s", month_dir, e)

    def _escpos_ham_olustur(self, kayit: PlakaKayit) -> bytes:
        enc = "cp857"
        sep = ("=" * self.GENISLIK + "\n").encode(enc, errors="replace")
        dash = ("-" * self.GENISLIK + "\n").encode(enc, errors="replace")
        cikis_agirlik = kayit.cikis_agirlik if kayit.cikis_agirlik is not None else 0.0
        net_agirlik = kayit.net_agirlik if kayit.net_agirlik is not None else 0.0

        def satir(m: str) -> bytes:
            return (m + "\n").encode(enc, errors="replace")

        return (
            self._ESC_INIT + self._ESC_CENTER + self._ESC_BOLD_ON + satir("BRİKET FABRİKASI KANTAR FİŞİ")
            + self._ESC_BOLD_OFF + self._ESC_LEFT + sep
            + satir(f"  Giris T. : {kayit.giris_tarih}") + satir(f"  Giris S. : {kayit.giris_saat}") + dash
            + self._ESC_BOLD_ON + satir(f"  Plaka    : {kayit.plaka}") + satir(f"  Giris Kg : {kayit.giris_agirlik:.1f} kg")
            + satir(f"  Cikis Kg : {cikis_agirlik:.1f} kg") + satir(f"  Net Kg   : {net_agirlik:.1f} kg")
            + self._ESC_BOLD_OFF + satir(f"  Durum    : {kayit.durum}") + satir(f"  Operatör : {kayit.operator}")
            + dash + satir(f"  Güven    : %{kayit.guven * 100:.1f}") + sep + self._ESC_CENTER
            + satir("Teşekkür Ederiz — İyi Yolculuklar") + self._ESC_FEED + self._ESC_CUT
        )

    def _win32_gonder_async(self, kayit: PlakaKayit, receipt_id: str) -> None:
        def _gonder():
            printer = str(CONFIG.get("YAZICI_ADI", "")).strip() or "(varsayılan)"
            try:
                printer = self._win32_gonder(kayit)
                self._print_log_yaz(receipt_id, kayit, "win32", printer, "SUCCESS")
            except FileNotFoundError:
                yazici_adi = str(CONFIG.get("YAZICI_ADI", "")).strip()
                reason = "Yazıcı bulunamadı"
                self._print_log_yaz(receipt_id, kayit, "win32", yazici_adi or printer, "FAILED", reason)
                log.error("win32print: yazıcı bulunamadı → '%s'", yazici_adi or "(varsayılan)")
            except OSError as e:
                reason = e.strerror or str(e)
                self._print_log_yaz(receipt_id, kayit, "win32", printer, "FAILED", reason)
                log.error("win32print OSError (errno %s): %s", e.errno, reason)
            except Exception as e:
                reason = f"{type(e).__name__}: {e}"
                self._print_log_yaz(receipt_id, kayit, "win32", printer, "FAILED", reason)
                log.error("win32print beklenmedik hata (%s): %s", type(e).__name__, e)

        try:
            t = threading.Thread(target=_gonder, name="FisGonderici-win32", daemon=True)
            t.start()

            def _izle():
                t.join(30)
                if t.is_alive():
                    log.warning("FisGonderici-win32: 30sn timeout aşıldı.")
            threading.Thread(target=_izle, daemon=True).start()
        except Exception as e:
            log.error("win32 thread başlatılamadı (%s): %s", type(e).__name__, e)

    def _win32_gonder(self, kayit: PlakaKayit) -> str:
        ham_veri = self._escpos_ham_olustur(kayit)
        yazici_adi = str(CONFIG.get("YAZICI_ADI", "")).strip()
        hedef = yazici_adi if yazici_adi else win32print.GetDefaultPrinter()
        handle = win32print.OpenPrinter(hedef)
        try:
            win32print.StartDocPrinter(handle, 1, ("KantarFisi", None, "RAW"))
            try:
                win32print.StartPagePrinter(handle)
                win32print.WritePrinter(handle, ham_veri)
                win32print.EndPagePrinter(handle)
            finally:
                win32print.EndDocPrinter(handle)
        finally:
            win32print.ClosePrinter(handle)
        log.info("Fiş win32print ile gönderildi → '%s'", hedef)
        return hedef

    def _escpos_gonder_async(self, kayit: PlakaKayit, receipt_id: str) -> None:
        ham_veri = self._escpos_ham_olustur(kayit)
        vendor = int(CONFIG.get("ESCPOS_USB_VENDOR", 0x04B8))
        product = int(CONFIG.get("ESCPOS_USB_PRODUCT", 0x0202))
        printer = f"USB {vendor:04X}:{product:04X}"

        def _gonder():
            try:
                p = escpos_printer.Usb(vendor, product)
                p._raw(ham_veri)
                self._print_log_yaz(receipt_id, kayit, "escpos", printer, "SUCCESS")
                log.info("Fiş escpos ile gönderildi → %s", printer)
            except Exception as e:
                self._print_log_yaz(
                    receipt_id,
                    kayit,
                    "escpos",
                    printer,
                    "FAILED",
                    f"{type(e).__name__}: {e}",
                )
                log.error("escpos hata (%s): %s", type(e).__name__, e)

        try:
            threading.Thread(target=_gonder, name="FisGonderici-escpos", daemon=True).start()
        except Exception as e:
            log.error("escpos thread başlatılamadı (%s): %s", type(e).__name__, e)
