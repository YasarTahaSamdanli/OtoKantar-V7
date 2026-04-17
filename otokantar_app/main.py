"""
OtoKantar V12 — Production-Grade Refactor
==========================================
Kritik düzeltmeler:
  1. Thread-safe durum yönetimi (threading.Lock)
  2. Atomik kuyruk operasyonu (race condition giderildi)
  3. Tek yetkili cikis_istendi Event'i (dur_event kaldırıldı)
  4. FastAPI sürecine graceful SIGTERM, timeout sonrası SIGKILL
  5. FPS ölçümü rolling-average ile stabilize edildi
  6. _son_plaka_listesi Lock ile korundu
  7. plaka_hafizasi ve _plaka_buffer tüm erişimlerde Lock altında
  8. Snapshot IO ayrı thread'e taşındı (ana döngüyü bloklamıyor)
  9. canli_kare_yaz atomik tmp→rename ile güvenli yapıldı
 10. CONFIG değerleri __init__'te bir kez okunup cache'lendi
 11. Logging: her çağrıda float() dönüşüm hatası önlendi
 12. _kapat: _cikis_istendi ve dur_event birleştirildi
"""

import multiprocessing
import os
import platform
import queue
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import uvicorn

from otokantar_app.api.routes import app, sistem_referansi_ata
from otokantar_app.config import CONFIG
from otokantar_app.core.ai_motoru import OcrWorker, PlakaCozucu, PlakaTespitci
from otokantar_app.core.dogrulama import DogrulamaMotoru
from otokantar_app.core.tracker import CentroidTracker
from otokantar_app.db.kaydedici import KantarKaydedici
from otokantar_app.donanim.kantar import KantarOkuyucu
from otokantar_app.donanim.yazici import FisYazdirici
from otokantar_app.logger import periyodik_temizlik_baslat, eski_snapshot_temizle, log
from otokantar_app.models import OcrGorevi, PlakaBuffer, PlakaKayit
from otokantar_app.utils.cizici import EkranCizici

if platform.system() == "Windows":
    import winsound
    _WINSOUND_OK = True
else:
    _WINSOUND_OK = False

# ---------------------------------------------------------------------------
# FastAPI alt-süreci
# ---------------------------------------------------------------------------

def _fastapi_sureci_hedef(host: str, port: int) -> None:
    """
    DÜZELTME: SIGINT görmezden gel, SIGTERM ile temiz çıkış yap.
    Orijinalde aynıydı — burada değişiklik yok, yalnızca belgelendi.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    uconf = uvicorn.Config(app=app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(uconf)

    def _sigterm_handler(signum, frame):
        server.should_exit = True

    signal.signal(signal.SIGTERM, _sigterm_handler)
    server.run()


# ---------------------------------------------------------------------------
# Ana sınıf
# ---------------------------------------------------------------------------

class OtoKantar:
    def __init__(self):
        log.info("=" * 60)
        log.info("OtoKantar V12 (Prodüksiyon) başlatılıyor...")

        # ------------------------------------------------------------------
        # FIX #1: Tek yetkili çıkış sinyali.
        # Orijinal kodda hem self._cikis_istendi hem de _kapat'a geçilen
        # ayrı bir `dur_event` vardı. İkisi arasındaki senkronizasyon
        # eksikliği, üretici thread'in durmamasına yol açabiliyordu.
        # ------------------------------------------------------------------
        self._cikis_istendi = threading.Event()

        eski_snapshot_temizle()
        self.temizlik_thread = periyodik_temizlik_baslat(
            self._cikis_istendi, aralik_saat=6.0
        )

        # ------------------------------------------------------------------
        # FIX #2: CONFIG değerlerini bir kez oku, float/int dönüşümlerini
        # burada yap. Döngü içinde tekrar tekrar float(CONFIG["..."]) çağrısı
        # hem yavaşlatıyordu hem de KeyError'ı geç fark ettiriyordu.
        # ------------------------------------------------------------------
        self._cfg_min_kilit_agirlik   = float(CONFIG["MIN_KILIT_AGIRLIK"])
        self._cfg_seans_sifir_bekleme = float(CONFIG["SEANS_SIFIR_BEKLEME"])
        self._cfg_plaka_buffer_ttl    = float(CONFIG["PLAKA_BUFFER_TTL"])
        self._cfg_ocr_kare_atlama     = int(CONFIG["OCR_KARE_ATLAMA"])
        self._cfg_canli_kare_aralik   = max(1, int(CONFIG["CANLI_KARE_ARALIK"]))
        self._cfg_canli_kare_dosya    = str(CONFIG["CANLI_KARE_DOSYA"])

        # Modeller
        self.tespitci    = PlakaTespitci(
            CONFIG["PLATE_WEIGHTS_URL"], CONFIG["MODELS_DIR"],
            CONFIG["YOLO_CONF"], CONFIG["OCR_GPU"],
        )
        self.cozucu      = PlakaCozucu(
            CONFIG["OCR_DILLER"], CONFIG["OCR_GPU"], CONFIG["MIN_OCR_CONF"]
        )
        _ocr_kuyruk = int(CONFIG.get("OCR_WORKER_KUYRUK", 4))
        self._ocr_cikis_kuyrugu = queue.Queue(maxsize=max(2, _ocr_kuyruk * 2))
        self.ocr_worker = OcrWorker(
            self.cozucu,
            self._ocr_cikis_kuyrugu,
            kuyruk_boyutu=_ocr_kuyruk,
        )
        self.dogrulama   = DogrulamaMotoru(
            CONFIG["ESIK_DEGERI"],
            CONFIG["OYLAMA_MIN_TOPLAM_GUVEN"],
            CONFIG["BEKLEME_SURESI_SONRA"],
        )
        self.kantar_okuyucu = KantarOkuyucu()
        self.fis_yazdirici  = FisYazdirici()
        self.kaydedici      = KantarKaydedici(
            CONFIG["CSV_DOSYA"], CONFIG["JSON_CANLI"], CONFIG["DB_DOSYA"]
        )
        self.cizici  = EkranCizici()
        self.tracker = CentroidTracker()
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=False
        )

        # ------------------------------------------------------------------
        # FIX #3: Paylaşılan durum değişkenleri için tek bir kilit.
        # Orijinal kodda plaka_hafizasi, _plaka_buffer, _kantar_seans_kilitli
        # ve _seans_sifir_baslangic herhangi bir senkronizasyon olmadan
        # ana thread (consumer) ve FastAPI thread'inden (sistem_referansi
        # üzerinden) okunup yazılıyordu → veri bozulması / race condition.
        # ------------------------------------------------------------------
        self._durum_lock = threading.Lock()

        self.plaka_hafizasi: dict          = {}
        self._plaka_buffer: Optional[PlakaBuffer] = None
        self._kantar_seans_kilitli: bool   = False
        self._seans_sifir_baslangic: Optional[float] = None
        self._kantar_roi_piksel: Optional[tuple]      = None

        # Saf sayaçlar (yalnızca ana thread'de değişir, lock gerekmez)
        self._kare_sayaci      = 0
        self._canli_kare_sayac = 0

        # ------------------------------------------------------------------
        # FIX #4: FPS'i tek-kare farkı yerine rolling-average ile ölç.
        # Tek-kare fark, bir kare geciktiğinde 0 fps veya 999 fps gösterir.
        # ------------------------------------------------------------------
        self._fps_pencere_boyutu = 30
        self._fps_zamanlar: list = []
        self._fps               = 0.0
        self._yakalama_fps_zamanlar: list = []
        self._yakalama_fps      = 0.0

        self._aktif_bbox        = None
        self._son_plaka_listesi: list = []
        # _son_plaka_listesi'ne iki thread'den erişilmiyor; ancak
        # gelecekte tracker thread'e taşınırsa bu lock gerekecek.
        self._plaka_listesi_lock = threading.Lock()

        self._fastapi_proc: Optional[multiprocessing.Process] = None

        # ------------------------------------------------------------------
        # FIX #5: Ağır IO (snapshot kaydetme) ana döngüyü bloklamasın.
        # ------------------------------------------------------------------
        self._io_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="SnapshotIO")
        # Veritabanındaki bilinen araçları doğrulamaya besle (Fuzzy Match İçin)
        import sqlite3
        try:
            db_baglanti = sqlite3.connect(CONFIG.get("DB_DOSYA", "otokantar.db"))
            satirlar = db_baglanti.execute("SELECT plaka FROM kayitli_araclar").fetchall()
            self.dogrulama.bilinen_plakalar = set(r[0] for r in satirlar)
            db_baglanti.close()
            log.info(f"Oto-Düzeltme Aktif: Veritabanından {len(self.dogrulama.bilinen_plakalar)} araç hafızaya alındı.")
        except Exception as e:
            log.warning("Oto-düzeltme için plakalar okunamadı: %s", e)

    # -----------------------------------------------------------------------
    # Property: FastAPI thread'inden de güvenle okunabilir
    # -----------------------------------------------------------------------
    @property
    def seans_kilitli_mi(self) -> bool:
        with self._durum_lock:
            return self._kantar_seans_kilitli

    # -----------------------------------------------------------------------
    # Kayıt
    # -----------------------------------------------------------------------
    def _kayit_yap(
        self,
        kare: np.ndarray,
        bbox: tuple,
        plaka: str,
        agirlik: float,
        final_conf: float,
        kaynak: str = "DOĞRUDAN",
    ) -> Optional[PlakaKayit]:
        kara_listede = self.kaydedici.plaka_kara_listede_mi(plaka)
        acik_seans   = self.kaydedici.acik_seans_getir(plaka)

        if acik_seans is None:
            kayit = self.kaydedici.giris_kaydet(plaka, agirlik)
            log.info(
                "GİRİŞ TARTIMI  [%s]: %s - %.1f kg  güven=%.2f",
                kaynak, plaka, agirlik, final_conf,
            )
            self.cizici.giris_yapildi(kare, bbox, plaka)
        else:
            kayit = self.kaydedici.cikis_kaydet(plaka, agirlik)
            if kayit is None:
                return None
            net_kg = float(kayit.net_agirlik or 0.0)
            log.info(
                "ÇIKIŞ TARTIMI  [%s]: %s - Net: %.1f kg  güven=%.2f",
                kaynak, plaka, net_kg, final_conf,
            )
            self.cizici.cikis_yapildi(kare, bbox, plaka, net_kg)
            try:
                self.fis_yazdirici.yazdir(kayit)
            except Exception as e:
                log.error("Fiş yazdırma hatası (%s): %s", type(e).__name__, e)

        # ------------------------------------------------------------------
        # FIX #5 uygulama: snapshot IO'yu thread-pool'a gönder.
        # Orijinalde cv2.imwrite() doğrudan ana döngü içinde çağrılıyordu;
        # disk yazma latency'si (5-50 ms) frame drop'a neden oluyordu.
        # ------------------------------------------------------------------
        kare_kopya = kare.copy()   # IO thread'ine güvenli kopya
        etiket     = "CIKIS" if acik_seans else "GIRIS"
        self._io_executor.submit(
            self._snapshot_kaydet, kare_kopya, plaka, etiket
        )

        if _WINSOUND_OK:
            try:
                winsound.Beep(500 if kara_listede else 1000, 1000 if kara_listede else 300)
            except Exception:
                pass

        return kayit

    @staticmethod
    def _snapshot_kaydet(kare: np.ndarray, plaka: str, etiket: str) -> None:
        """Arka planda (IO thread) snapshot yazar."""
        try:
            simdi_dt = datetime.now()
            Path("captures").mkdir(exist_ok=True)
            snap = (
                f"captures/{plaka}_{simdi_dt.strftime('%Y-%m-%d_%H-%M-%S')}_{etiket}.jpg"
            )
            cv2.imwrite(snap, kare)
            log.info("Snapshot: %s", snap)
        except Exception as e:
            log.warning("Snapshot yazılamadı: %s", e)

    # -----------------------------------------------------------------------
    # Sinyal işleyici
    # -----------------------------------------------------------------------
    def _signal_handler(self, signum, frame) -> None:
        log.info("Signal %s alındı — graceful shutdown başlıyor...", signum)
        self._cikis_istendi.set()

    # -----------------------------------------------------------------------
    # FastAPI süreci
    # -----------------------------------------------------------------------
    def _fastapi_baslat(self) -> multiprocessing.Process:
        host = str(CONFIG.get("FASTAPI_HOST", "0.0.0.0"))
        port = int(CONFIG.get("FASTAPI_PORT", 8000))
        proc = multiprocessing.Process(
            target=_fastapi_sureci_hedef,
            args=(host, port),
            name="FastAPIProcess",
            daemon=True,
        )
        proc.start()
        log.info(
            "FastAPI süreci başlatıldı (PID=%d) → http://%s:%s",
            proc.pid, host, port,
        )
        return proc

    # -----------------------------------------------------------------------
    # Graceful shutdown
    # FIX #6: Tek Event ile hem üreticiyi hem tüm sistemi kapat.
    # -----------------------------------------------------------------------
    def _kapat(self, uretici_thread: threading.Thread) -> None:
        log.info("Kapatma süreci başladı...")

        # _cikis_istendi hem üreticiyi hem de döngüyü durdurur.
        self._cikis_istendi.set()

        if hasattr(self, "temizlik_thread") and self.temizlik_thread.is_alive():
            self.temizlik_thread.join(timeout=2.0)

        uretici_thread.join(timeout=5.0)
        if uretici_thread.is_alive():
            log.warning("Kamera üreticisi 5 sn içinde sonlanmadı.")

        self.ocr_worker.durdur()
        self.ocr_worker.join(timeout=5.0)
        if self.ocr_worker.is_alive():
            log.warning("OcrWorker 5 sn içinde sonlanmadı.")

        self.kantar_okuyucu.durdur()
        self.kantar_okuyucu.join(timeout=3.0)

        # IO thread-pool'u kapat (devam eden snapshotları tamamla)
        self._io_executor.shutdown(wait=True, cancel_futures=False)

        self.kaydedici.kapat()

        if self._fastapi_proc is not None and self._fastapi_proc.is_alive():
            log.info(
                "FastAPI süreci (PID=%d) kapatılıyor...", self._fastapi_proc.pid
            )
            self._fastapi_proc.terminate()          # SIGTERM → graceful
            self._fastapi_proc.join(timeout=5.0)
            if self._fastapi_proc.is_alive():
                log.warning("FastAPI SIGTERM'e yanıt vermedi → SIGKILL.")
                self._fastapi_proc.kill()
                self._fastapi_proc.join(timeout=2.0)
            else:
                log.info("FastAPI süreci temiz kapandı.")

        cv2.destroyAllWindows()
        log.info("OtoKantar V12 temiz kapandı.")

    # -----------------------------------------------------------------------
    # ROI
    # -----------------------------------------------------------------------
    def _roi_piksel_guncelle(self, kare_h: int, kare_w: int) -> tuple:
        # Lock gerekmiyor: yalnızca ana thread'de çağrılır.
        if self._kantar_roi_piksel is not None:
            return self._kantar_roi_piksel
        l, t, r, b = CONFIG["KANTAR_ROI_NORM"]
        roi = (
            int(l * kare_w), int(t * kare_h),
            int(r * kare_w), int(b * kare_h),
        )
        self._kantar_roi_piksel = roi
        log.info(
            "Dinamik ROI hesaplandı: norm=%s → piksel=%s (%dx%d)",
            CONFIG["KANTAR_ROI_NORM"], roi, kare_w, kare_h,
        )
        return roi

    # -----------------------------------------------------------------------
    # Seans sıfırlama — Lock altında
    # -----------------------------------------------------------------------
    def _seans_temizle(self) -> None:
        with self._durum_lock:
            self._kantar_seans_kilitli   = False
            self._seans_sifir_baslangic  = None
            self._plaka_buffer           = None
            self._kantar_roi_piksel      = None
            self.plaka_hafizasi.clear()
        self.dogrulama.sifirla()
        self.tracker.sifirla()
        log.info("Seans sıfırlandı — guard tamamlandı, tüm hafızalar temizlendi.")

    # -----------------------------------------------------------------------
    # Kamera açma
    # -----------------------------------------------------------------------
    def _kamera_ac(self) -> cv2.VideoCapture:
        kamera_index  = int(CONFIG["KAMERA_INDEX"])
        warmup_frames = int(CONFIG.get("KAMERA_WARMUP_FRAMES", 8))
        siyah_esik    = float(CONFIG.get("KAMERA_SIYAH_ESIK", 2.0))
        warmup_sleep  = float(CONFIG.get("KAMERA_WARMUP_SLEEP", 0.02))

        if platform.system() == "Windows":
            backendler = [
                ("DSHOW", cv2.CAP_DSHOW),
                ("MSMF", cv2.CAP_MSMF),
                ("AUTO", None),
            ]
        else:
            backendler = [("AUTO", None)]

        def _siyah_mi(kare: np.ndarray) -> bool:
            try:
                if kare is None or kare.size == 0:
                    return True
                gri = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)
                return float(gri.mean()) < siyah_esik
            except Exception:
                return True

        for deneme in range(CONFIG["KAMERA_YENIDEN_BAGLANTI_DENEMESI"]):
            for backend_ad, backend in backendler:
                kamera = (
                    cv2.VideoCapture(kamera_index, backend)
                    if backend is not None
                    else cv2.VideoCapture(kamera_index)
                )
                if not kamera.isOpened():
                    kamera.release()
                    continue
                son_kare = None
                for _ in range(max(1, warmup_frames)):
                    ret, kare = kamera.read()
                    if not ret:
                        son_kare = None
                        break
                    son_kare = kare
                    if warmup_sleep > 0:
                        time.sleep(warmup_sleep)

                if son_kare is not None and not _siyah_mi(son_kare):
                    log.info("Kamera bağlantısı kuruldu (%s).", backend_ad)
                    return kamera

                kamera.release()

            log.warning(
                "Kamera açılamadı/okunamadı. Deneme %d/%d",
                deneme + 1,
                CONFIG["KAMERA_YENIDEN_BAGLANTI_DENEMESI"],
            )
            time.sleep(CONFIG["KAMERA_BEKLEME_SURESI"])

        raise RuntimeError("Kamera açılamadı. Cihazı kontrol edin.")

    # -----------------------------------------------------------------------
    # FPS — Rolling average
    # FIX #4 uygulama: 1/Δt yerine son N karenin ortalaması
    # -----------------------------------------------------------------------
    def _fps_guncelle(self) -> None:
        su_an = time.monotonic()
        self._fps_zamanlar.append(su_an)
        if len(self._fps_zamanlar) > self._fps_pencere_boyutu:
            self._fps_zamanlar.pop(0)
        if len(self._fps_zamanlar) >= 2:
            toplam_sure = self._fps_zamanlar[-1] - self._fps_zamanlar[0]
            if toplam_sure > 0:
                self._fps = (len(self._fps_zamanlar) - 1) / toplam_sure

    def _yakalama_fps_guncelle(self) -> None:
        su_an = time.monotonic()
        self._yakalama_fps_zamanlar.append(su_an)
        if len(self._yakalama_fps_zamanlar) > self._fps_pencere_boyutu:
            self._yakalama_fps_zamanlar.pop(0)
        if len(self._yakalama_fps_zamanlar) >= 2:
            toplam_sure = (
                self._yakalama_fps_zamanlar[-1] - self._yakalama_fps_zamanlar[0]
            )
            if toplam_sure > 0:
                self._yakalama_fps = (
                    len(self._yakalama_fps_zamanlar) - 1
                ) / toplam_sure

    # -----------------------------------------------------------------------
    # Kuyruğa kare ekle — Atomik
    # FIX #7: get_nowait → put_nowait arasında başka thread kare alırsa
    # ikinci put_nowait da Full verebilir; orijinal kod bunu yutuyordu.
    # Çözüm: tek mutex ile get+put'u atomik yap.
    # -----------------------------------------------------------------------
    def __init_kuyruk_lock(self):
        """Queue işlemleri için lock (calistir'da queue yaratıldıktan sonra atanır)."""
        self._kuyruk_lock = threading.Lock()

    def _kuyruga_kare_koy(self, q: queue.Queue, kare: np.ndarray) -> None:
        """
        Gerçek zamanlı sistemlerde eski kareyi at, yeni kareyi koy.
        Lock ile get→put arası atomik yapılır.
        """
        yuk = kare.copy()
        with self._kuyruk_lock:
            if q.full():
                try:
                    q.get_nowait()          # eski kareyi at
                except queue.Empty:
                    pass
            try:
                q.put_nowait(yuk)
            except queue.Full:
                # Teorik olarak buraya gelmemeli; paranoya guard.
                log.debug("Kuyruk beklenmedik şekilde dolu, kare atlandı.")

    # -----------------------------------------------------------------------
    # Kamera üretici thread
    # -----------------------------------------------------------------------
    def _kamera_uretici(self, kare_kuyrugu: queue.Queue) -> None:
        """
        FIX #6: dur_event parametresi kaldırıldı.
        Yalnızca self._cikis_istendi kullanılır.
        """
        kamera = self._kamera_ac()
        try:
            while not self._cikis_istendi.is_set():
                ret, kare = kamera.read()
                if not ret:
                    log.warning("Kare okunamadı — yeniden bağlanılıyor...")
                    kamera.release()
                    time.sleep(0.35)
                    try:
                        kamera = self._kamera_ac()
                    except RuntimeError as e:
                        log.error(str(e))
                        break
                    continue
                self._yakalama_fps_guncelle()
                self._kuyruga_kare_koy(kare_kuyrugu, kare)
        finally:
            kamera.release()
            log.debug("Kamera (üretici) serbest bırakıldı.")

    # -----------------------------------------------------------------------
    # Canlı kare yazma — Atomik tmp → rename
    # FIX #8: Doğrudan imwrite() kısmi-yazma riski yaratır.
    # tmp dosyaya yaz, ardından atomik rename et.
    # -----------------------------------------------------------------------
    def _canli_kare_yaz(self, kare: np.ndarray) -> None:
        self._canli_kare_sayac += 1
        if self._canli_kare_sayac % self._cfg_canli_kare_aralik != 0:
            return
        hedef = Path(self._cfg_canli_kare_dosya)
        tmp   = hedef.with_suffix(".tmp.jpg")
        try:
            cv2.imwrite(str(tmp), kare)
            tmp.replace(hedef)          # POSIX'te atomik; Windows'ta da çalışır
        except Exception as e:
            log.warning("Canlı kare yazılamadı: %s", e)
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    # -----------------------------------------------------------------------
    # OCR sonuçlarını işle — Durum Lock altında güncelle
    # -----------------------------------------------------------------------
    def _ocr_sonuclarini_isle(
        self,
        kare: np.ndarray,
        guncel_kg: float,
        agirlik_sabit: bool,
        kare_w: int,
        kare_h: int,
    ) -> None:
        sonuclar = self.ocr_worker.sonuclari_topla()
        for (arac_id, sonuc, yolo_conf, bbox) in sonuclar:
            if not sonuc.gecerli or sonuc.plaka is None:
                continue
            plaka = sonuc.plaka
            kara_listede_okunan = self.kaydedici.plaka_kara_listede_mi(plaka)
            if kara_listede_okunan:
                log.warning(
                    "ALARM: KARA LİSTEDEKİ ARAÇ TESPİT EDİLDİ! (%s)", plaka
                )

            kaydet, final_plaka, kazan_toplam, kazan_n = self.dogrulama.isle(
                arac_id, plaka, float(sonuc.guven or 0.0)
            )
            if not kaydet:
                lider, oku_n, lider_oy, esik_c, min_g = self.dogrulama.durum_ozeti(
                    arac_id
                )
                self.cizici.plaka_kutusu(
                    kare, bbox, plaka, oku_n, esik_c,
                    arac_id=arac_id, kara_liste=kara_listede_okunan,
                    oy_lider=lider, oy_lider_toplam=lider_oy, oy_min_guven=min_g,
                )
                continue

            if final_plaka is None:
                continue

            ort_ocr    = float(kazan_toplam) / max(1, int(kazan_n))
            final_conf = 0.6 * ort_ocr + 0.4 * float(yolo_conf)
            kantar_dolu = guncel_kg >= self._cfg_min_kilit_agirlik

            # -- Durum güncellemesi Lock altında --
            with self._durum_lock:
                if kantar_dolu:
                    if self._plaka_buffer is None:
                        self._plaka_buffer = PlakaBuffer(
                            plaka=final_plaka,
                            guven=ort_ocr,
                            yolo_conf=float(yolo_conf),
                        )
                    else:
                        self._plaka_buffer.guncule_eger_daha_iyi(
                            final_plaka, ort_ocr, float(yolo_conf)
                        )

                seans_kilitli = self._kantar_seans_kilitli

            if agirlik_sabit and not seans_kilitli:
                kayit = self._kayit_yap(
                    kare=kare, bbox=bbox, plaka=final_plaka,
                    agirlik=guncel_kg, final_conf=final_conf,
                    kaynak="DOĞRUDAN",
                )
                if kayit is not None:
                    with self._durum_lock:
                        self.plaka_hafizasi[arac_id] = final_plaka
                        self._kantar_seans_kilitli   = True
                        self._plaka_buffer           = None

    # -----------------------------------------------------------------------
    # Ana kare işleme döngüsü
    # -----------------------------------------------------------------------
    def _kare_isle(self, kare: np.ndarray) -> None:
        self._kare_sayaci += 1
        self._fps_guncelle()
        kare_h, kare_w = kare.shape[:2]
        simdi = time.monotonic()

        roi_x1, roi_y1, roi_x2, roi_y2 = self._roi_piksel_guncelle(kare_h, kare_w)
        guncel_kg, agirlik_sabit = self.kantar_okuyucu.veri
        kantar_dolu = guncel_kg >= self._cfg_min_kilit_agirlik

        _olcek_ref = kare_h / 720.0
        tx         = int(kare_w * 0.013)
        y1         = int(kare_h * 0.090)
        y2         = int(kare_h * 0.135)
        y3         = int(kare_h * 0.180)
        fs_normal  = round(0.65 * _olcek_ref, 2)
        fs_kucuk   = round(0.55 * _olcek_ref, 2)
        pad_y_bbox = max(int(kare_h * 0.055), 14)

        # Okuma ile birlikte Lock al, kopyala, hemen bırak
        with self._durum_lock:
            seans_kilitli         = self._kantar_seans_kilitli
            seans_sifir_baslangic = self._seans_sifir_baslangic
            plaka_buffer_var      = self._plaka_buffer is not None
            plaka_buffer_ttl_ok   = (
                self._plaka_buffer is not None
                and not self._plaka_buffer.suresi_doldu_mu(self._cfg_plaka_buffer_ttl)
            )

        # Seans sıfır guard
        if not kantar_dolu and seans_kilitli:
            if seans_sifir_baslangic is None:
                with self._durum_lock:
                    self._seans_sifir_baslangic = simdi
            else:
                gecen = simdi - seans_sifir_baslangic
                if gecen >= self._cfg_seans_sifir_bekleme:
                    self._seans_temizle()
                else:
                    kalan = self._cfg_seans_sifir_bekleme - gecen
                    cv2.putText(
                        kare,
                        f"ARAÇ İNİYOR — Guard: {kalan:.1f}sn kaldı",
                        (tx, y3),
                        cv2.FONT_HERSHEY_SIMPLEX, fs_kucuk, (0, 200, 255), 1,
                    )
        elif kantar_dolu and seans_sifir_baslangic is not None:
            with self._durum_lock:
                self._seans_sifir_baslangic = None

        # Buffer TTL
        if not kantar_dolu and plaka_buffer_var and not plaka_buffer_ttl_ok:
            with self._durum_lock:
                self._plaka_buffer = None

        if not kantar_dolu and self.tracker.is_empty():
            self._aktif_bbox = None
            cv2.putText(
                kare,
                f"DURUM: KANTAR BOŞ ({guncel_kg:.0f} kg)",
                (tx, y1),
                cv2.FONT_HERSHEY_SIMPLEX, fs_normal, (0, 255, 0), 2,
            )
            self.ocr_worker.sonuclari_topla()
            return

        if kantar_dolu:
            if agirlik_sabit:
                cv2.putText(
                    kare,
                    f"KANTARDA ARAÇ VAR: {guncel_kg:.0f} kg  [SABİT]",
                    (tx, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, fs_normal, (0, 165, 255), 2,
                )
            else:
                cv2.putText(
                    kare,
                    f"AĞIRLIK BEKLENİYOR... {guncel_kg:.0f} kg",
                    (tx, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, fs_normal, (0, 200, 255), 2,
                )

            with self._durum_lock:
                buf            = self._plaka_buffer
                seans_kilitli2 = self._kantar_seans_kilitli

            if (
                agirlik_sabit
                and buf is not None
                and not seans_kilitli2
                and not buf.suresi_doldu_mu(self._cfg_plaka_buffer_ttl)
            ):
                final_conf = 0.6 * buf.guven + 0.4 * buf.yolo_conf
                buf_bbox   = self._aktif_bbox or (
                    tx, int(kare_h * 0.125),
                    int(kare_w * 0.556), int(kare_h * 0.181),
                )
                kayit = self._kayit_yap(
                    kare=kare, bbox=buf_bbox, plaka=buf.plaka,
                    agirlik=guncel_kg, final_conf=final_conf, kaynak="BUFFER",
                )
                if kayit is not None:
                    with self._durum_lock:
                        self._kantar_seans_kilitli = True
                        self._plaka_buffer         = None

        with self._durum_lock:
            seans_kilitli_son = self._kantar_seans_kilitli

        if seans_kilitli_son:
            cv2.putText(
                kare,
                "SEANS KİLİTLİ — ARAÇ ÇIKIŞI BEKLENİYOR",
                (tx, y2),
                cv2.FONT_HERSHEY_SIMPLEX, fs_normal, (0, 80, 255), 2,
            )
            self.ocr_worker.sonuclari_topla()
            return

        self._ocr_sonuclarini_isle(kare, guncel_kg, agirlik_sabit, kare_w, kare_h)

        # Plaka tespiti (her 2 karede bir)
        if self._kare_sayaci % 2 == 0:
            plaka_listesi = self.tespitci.plakalari_bul(kare)
            with self._plaka_listesi_lock:
                self._son_plaka_listesi = plaka_listesi
        else:
            with self._plaka_listesi_lock:
                plaka_listesi = list(self._son_plaka_listesi)

        if not plaka_listesi:
            self._aktif_bbox = None
            return

        for silinen_id in self.tracker.purge_expired():
            with self._durum_lock:
                self.plaka_hafizasi.pop(silinen_id, None)

        self._aktif_bbox = None
        ocr_calis = self._kare_sayaci % self._cfg_ocr_kare_atlama == 0

        for x1, y1_det, x2, y2_det, yolo_conf in plaka_listesi:
            ix1 = max(0, int(x1));   iy1 = max(0, int(y1_det))
            ix2 = min(kare_w, int(x2)); iy2 = min(kare_h, int(y2_det))
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            cx, cy = (ix1 + ix2) / 2.0, (iy1 + iy2) / 2.0
            if not (roi_x1 <= cx <= roi_x2 and roi_y1 <= cy <= roi_y2):
                continue
            bbox    = (ix1, iy1, ix2, iy2)
            arac_id = self.tracker.assign_id(bbox)
            if self._aktif_bbox is None:
                self._aktif_bbox = bbox
            cv2.rectangle(kare, (ix1, iy1), (ix2, iy2), (128, 0, 128), 1)

            with self._durum_lock:
                plaka_cached = self.plaka_hafizasi.get(arac_id)

            if plaka_cached is not None:
                kara_listede = self.kaydedici.plaka_kara_listede_mi(plaka_cached)
                if kara_listede:
                    log.warning("ALARM: KARA LİSTEDEKİ ARAÇ TESPİT EDİLDİ!")
                self.cizici.basarili_kayit(kare, bbox, plaka_cached)
                cv2.putText(
                    kare,
                    f"ID:{arac_id} - {plaka_cached}",
                    (ix1, max(pad_y_bbox, iy1 - pad_y_bbox)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    round(0.80 * _olcek_ref, 2),
                    (0, 0, 255) if kara_listede else (0, 255, 0),
                    2,
                )
                continue

            if not ocr_calis:
                continue

            w_roi = ix2 - ix1
            pad   = int(w_roi * 0.10)
            roi   = kare[iy1:iy2, max(0, ix1 - pad):min(kare_w, ix2 + pad)]
            if roi.size == 0:
                continue

            gonderildi = self.ocr_worker.gorevi_gonder(
                OcrGorevi(
                    roi_bgr=roi.copy(),
                    arac_id=arac_id,
                    yolo_conf=float(yolo_conf),
                    bbox=bbox,
                )
            )
            if not gonderildi:
                log.debug(
                    "OCR kuyruğu dolu — kare %d için araç %d atlandı.",
                    self._kare_sayaci, arac_id,
                )

        if self._kare_sayaci % 300 == 0:
            self.dogrulama.temizle_eski()

    # -----------------------------------------------------------------------
    # Ana döngü
    # -----------------------------------------------------------------------
    def calistir(self) -> None:
        sistem_referansi_ata(self)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self._fastapi_proc = self._fastapi_baslat()
        self.kantar_okuyucu.start()
        log.info("KantarOkuyucu başlatıldı.")
        self.ocr_worker.start()
        log.info("OcrWorker başlatıldı.")

        kare_kuyrugu: queue.Queue = queue.Queue(
            maxsize=max(1, int(CONFIG.get("KARE_KUYRUK_BOYUTU", 2)))
        )
        # FIX #7: kuyruk lock'unu başlat
        self.__init_kuyruk_lock()

        # FIX #6: Tek Event — dur parametresi yok
        uretici = threading.Thread(
            target=self._kamera_uretici,
            args=(kare_kuyrugu,),
            name="KameraUretici",
            daemon=True,
        )
        uretici.start()
        log.info(
            "Producer–Consumer aktif (kuyruk boyutu=%d). Çıkmak için 'q' veya Ctrl+C.",
            kare_kuyrugu.maxsize,
        )

        try:
            while not self._cikis_istendi.is_set():
                try:
                    kare = kare_kuyrugu.get(timeout=0.25)
                except queue.Empty:
                    tus = cv2.waitKey(1) & 0xFF
                    if tus == ord("0"):
                        self.kantar_okuyucu.simule_et(0.0)
                    elif tus == ord("1"):
                        self.kantar_okuyucu.simule_et(12000.0)
                    elif tus == ord("2"):
                        self.kantar_okuyucu.simule_et(42000.0)
                    elif tus == ord("q"):
                        log.info("Kullanıcı 'q' ile çıkış yaptı.")
                        break
                    continue

                self._kare_isle(kare)
                self.cizici.fps_ve_bilgi(
                    kare, self._fps,
                    len(self.kaydedici.son_kayitlar),
                    yakalama_fps=self._yakalama_fps,
                )
                self._canli_kare_yaz(kare)
                cv2.imshow("OtoKantar V12", kare)
                tus = cv2.waitKey(1) & 0xFF
                if tus == ord("0"):
                    self.kantar_okuyucu.simule_et(0.0)
                elif tus == ord("1"):
                    self.kantar_okuyucu.simule_et(12000.0)
                elif tus == ord("2"):
                    self.kantar_okuyucu.simule_et(42000.0)
                elif tus == ord("q"):
                    log.info("Kullanıcı 'q' ile çıkış yaptı.")
                    break

        except KeyboardInterrupt:
            log.info("Ctrl+C (KeyboardInterrupt) — graceful shutdown tetiklendi.")
        finally:
            # FIX #6: dur_event parametresi yok
            self._kapat(uretici)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    multiprocessing.freeze_support()
    sistem = OtoKantar()
    sistem.calistir()