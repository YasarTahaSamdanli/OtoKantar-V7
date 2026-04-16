import platform
import threading

from otokantar_app.config import CONFIG
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
            self._dosyaya_yaz(icerik)
            if backend == "win32":
                self._win32_gonder_async(kayit)
            elif backend == "escpos":
                self._escpos_gonder_async(kayit)
            else:
                log.info("Yazıcı backend=file; fiş '%s' dosyasına kaydedildi.", self.FIS_DOSYA)
        except PermissionError as e:
            log.error("Fiş dosyası yazılamadı — izin hatası: %s", e)
        except OSError as e:
            log.error("Fiş dosyası yazılamadı — I/O hatası (errno %s): %s", e.errno, e.strerror)
        except Exception as e:
            log.error("FisYazdirici beklenmedik hata (%s): %s — devam ediyor.", type(e).__name__, e)

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
        log.debug("Fiş dosyaya yazıldı: %s", self.FIS_DOSYA)

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

    def _win32_gonder_async(self, kayit: PlakaKayit) -> None:
        ham_veri = self._escpos_ham_olustur(kayit)
        yazici_adi = str(CONFIG.get("YAZICI_ADI", "")).strip()

        def _gonder():
            try:
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
            except FileNotFoundError:
                log.error("win32print: yazıcı bulunamadı → '%s'", yazici_adi or "(varsayılan)")
            except OSError as e:
                log.error("win32print OSError (errno %s): %s", e.errno, e.strerror)
            except Exception as e:
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

    def _escpos_gonder_async(self, kayit: PlakaKayit) -> None:
        ham_veri = self._escpos_ham_olustur(kayit)
        vendor = int(CONFIG.get("ESCPOS_USB_VENDOR", 0x04B8))
        product = int(CONFIG.get("ESCPOS_USB_PRODUCT", 0x0202))

        def _gonder():
            try:
                p = escpos_printer.Usb(vendor, product)
                p._raw(ham_veri)
                log.info("Fiş escpos ile gönderildi → USB %04X:%04X", vendor, product)
            except Exception as e:
                log.error("escpos hata (%s): %s", type(e).__name__, e)

        try:
            threading.Thread(target=_gonder, name="FisGonderici-escpos", daemon=True).start()
        except Exception as e:
            log.error("escpos thread başlatılamadı (%s): %s", type(e).__name__, e)
