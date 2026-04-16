import multiprocessing
import os
import platform
import queue
import signal
import threading
import time
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

def _fastapi_sureci_hedef(host: str, port: int) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    uconf = uvicorn.Config(app=app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(uconf)

    def _sigterm_handler(signum, frame):
        server.should_exit = True

    signal.signal(signal.SIGTERM, _sigterm_handler)
    server.run()

class OtoKantar:
    def __init__(self):
        log.info("=" * 60)
        log.info("OtoKantar V11 (Prodüksiyon) başlatılıyor...")
        
        self._cikis_istendi = threading.Event()
        eski_snapshot_temizle()
        self.temizlik_thread = periyodik_temizlik_baslat(self._cikis_istendi, aralik_saat=6.0)
        
        self.tespitci = PlakaTespitci(CONFIG["PLATE_WEIGHTS_URL"], CONFIG["MODELS_DIR"], CONFIG["YOLO_CONF"], CONFIG["OCR_GPU"])
        self.cozucu = PlakaCozucu(CONFIG["OCR_DILLER"], CONFIG["OCR_GPU"], CONFIG["MIN_OCR_CONF"])
        self.ocr_worker = OcrWorker(self.cozucu, int(CONFIG.get("OCR_WORKER_KUYRUK", 4)))
        self.dogrulama = DogrulamaMotoru(CONFIG["ESIK_DEGERI"], CONFIG["OYLAMA_MIN_TOPLAM_GUVEN"], CONFIG["BEKLEME_SURESI_SONRA"])
        self.kantar_okuyucu = KantarOkuyucu()
        self.fis_yazdirici = FisYazdirici()
        self.kaydedici = KantarKaydedici(CONFIG["CSV_DOSYA"], CONFIG["JSON_CANLI"], CONFIG["DB_DOSYA"])
        self.cizici = EkranCizici()
        self.tracker = CentroidTracker()
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
        self.plaka_hafizasi: dict = {}
        self._plaka_buffer: Optional[PlakaBuffer] = None
        self._kantar_seans_kilitli: bool = False
        self._seans_sifir_baslangic: Optional[float] = None
        self._kantar_roi_piksel: Optional[tuple] = None
        self._kare_sayaci = 0
        self._fps_onceki = time.time()
        self._fps = 0.0
        self._yakalama_fps_onceki = time.time()
        self._yakalama_fps = 0.0
        self._aktif_bbox = None
        self._canli_kare_sayac = 0
        self._son_plaka_listesi: list = []
        self._fastapi_proc: Optional[multiprocessing.Process] = None

    @property
    def seans_kilitli_mi(self) -> bool:
        return self._kantar_seans_kilitli

    def _kayit_yap(self, kare: np.ndarray, bbox: tuple, plaka: str, agirlik: float, final_conf: float, kaynak: str = "DOĞRUDAN") -> Optional[PlakaKayit]:
        kara_listede = self.kaydedici.plaka_kara_listede_mi(plaka)
        acik_seans = self.kaydedici.acik_seans_getir(plaka)
        if acik_seans is None:
            kayit = self.kaydedici.giris_kaydet(plaka, agirlik)
            log.info("GİRİŞ TARTIMI  [%s]: %s - %.1f kg  güven=%.2f", kaynak, plaka, agirlik, final_conf)
            self.cizici.giris_yapildi(kare, bbox, plaka)
        else:
            kayit = self.kaydedici.cikis_kaydet(plaka, agirlik)
            if kayit is None:
                return None
            net_kg = float(kayit.net_agirlik or 0.0)
            log.info("ÇIKIŞ TARTIMI  [%s]: %s - Net: %.1f kg  güven=%.2f", kaynak, plaka, net_kg, final_conf)
            self.cizici.cikis_yapildi(kare, bbox, plaka, net_kg)
            try:
                self.fis_yazdirici.yazdir(kayit)
            except Exception as e:
                log.error("Fiş yazdırma hatası (%s): %s", type(e).__name__, e)
        try:
            simdi_dt = datetime.now()
            Path("captures").mkdir(exist_ok=True)
            snap = f"captures/{plaka}_{simdi_dt.strftime('%Y-%m-%d_%H-%M-%S')}_{'CIKIS' if acik_seans else 'GIRIS'}.jpg"
            cv2.imwrite(snap, kare)
            log.info("Snapshot: %s", snap)
        except Exception as e:
            log.warning("Snapshot yazılamadı: %s", e)
        if _WINSOUND_OK:
            try:
                winsound.Beep(500 if kara_listede else 1000, 1000 if kara_listede else 300)
            except Exception:
                pass
        return kayit

    def _signal_handler(self, signum, frame) -> None:
        log.info("Signal %s alındı — graceful shutdown başlıyor...", signum)
        self._cikis_istendi.set()

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
        log.info("FastAPI süreci başlatıldı (PID=%d) → http://%s:%s", proc.pid, host, port)
        return proc

    def _kapat(self, dur_event: threading.Event, uretici_thread: threading.Thread) -> None:
        log.info("Kapatma süreci başladı...")
        dur_event.set()
        
        if hasattr(self, 'temizlik_thread') and self.temizlik_thread.is_alive():
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
        self.kaydedici.kapat()
        
        if self._fastapi_proc is not None and self._fastapi_proc.is_alive():
            log.info("FastAPI süreci (PID=%d) kapatılıyor...", self._fastapi_proc.pid)
            self._fastapi_proc.terminate()
            self._fastapi_proc.join(timeout=5.0)
            if self._fastapi_proc.is_alive():
                self._fastapi_proc.kill()
                self._fastapi_proc.join(timeout=2.0)
            else:
                log.info("FastAPI süreci temiz kapandı.")

        cv2.destroyAllWindows()
        log.info("OtoKantar V11 temiz kapandı.")

    def _roi_piksel_guncelle(self, kare_h: int, kare_w: int) -> tuple:
        if self._kantar_roi_piksel is not None:
            return self._kantar_roi_piksel
        l, t, r, b = CONFIG["KANTAR_ROI_NORM"]
        roi = (int(l * kare_w), int(t * kare_h), int(r * kare_w), int(b * kare_h))
        self._kantar_roi_piksel = roi
        log.info("Dinamik ROI hesaplandı: norm=%s → piksel=%s (%dx%d)", CONFIG["KANTAR_ROI_NORM"], roi, kare_w, kare_h)
        return roi

    def _seans_temizle(self) -> None:
        self._kantar_seans_kilitli = False
        self._seans_sifir_baslangic = None
        self._plaka_buffer = None
        self._kantar_roi_piksel = None
        self.plaka_hafizasi.clear()
        self.dogrulama.sifirla()
        self.tracker.sifirla()
        log.info("Seans sıfırlandı — guard tamamlandı, tüm hafızalar temizlendi.")

    def _kamera_ac(self) -> cv2.VideoCapture:
        kamera_index = int(CONFIG["KAMERA_INDEX"])
        if platform.system() == "Windows":
            backendler = [
                ("DSHOW", cv2.CAP_DSHOW),
                ("MSMF", cv2.CAP_MSMF),
                ("AUTO", None),
            ]
        else:
            backendler = [("AUTO", None)]
        warmup_frames = int(CONFIG.get("KAMERA_WARMUP_FRAMES", 8))
        siyah_esik = float(CONFIG.get("KAMERA_SIYAH_ESIK", 2.0))
        warmup_sleep = float(CONFIG.get("KAMERA_WARMUP_SLEEP", 0.02))

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
                ok = False
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
                    ok = True

                if ok:
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

    def _fps_guncelle(self):
        su_an = time.time()
        gecen = su_an - self._fps_onceki
        if gecen > 0:
            self._fps = 1.0 / gecen
        self._fps_onceki = su_an

    def _yakalama_fps_guncelle(self):
        su_an = time.time()
        gecen = su_an - self._yakalama_fps_onceki
        if gecen > 0:
            self._yakalama_fps = 1.0 / gecen
        self._yakalama_fps_onceki = su_an

    @staticmethod
    def _kuyruga_kare_koy(q: queue.Queue, kare: np.ndarray) -> None:
        yuk = kare.copy()
        try:
            q.put_nowait(yuk)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(yuk)
            except queue.Full:
                pass

    def _kamera_uretici(self, kare_kuyrugu: queue.Queue, dur: threading.Event) -> None:
        kamera = self._kamera_ac()
        try:
            while not dur.is_set():
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

    def _canli_kare_yaz(self, kare: np.ndarray):
        self._canli_kare_sayac += 1
        if self._canli_kare_sayac % max(1, int(CONFIG["CANLI_KARE_ARALIK"])) != 0:
            return
        try:
            cv2.imwrite(CONFIG["CANLI_KARE_DOSYA"], kare)
        except Exception as e:
            log.warning("Canlı kare yazılamadı: %s", e)

    def _ocr_sonuclarini_isle(self, kare: np.ndarray, guncel_kg: float, agirlik_sabit: bool, kare_w: int, kare_h: int) -> None:
        sonuclar = self.ocr_worker.sonuclari_topla()
        for (arac_id, sonuc, yolo_conf, bbox) in sonuclar:
            if not sonuc.gecerli or sonuc.plaka is None:
                continue
            plaka = sonuc.plaka
            kara_listede_okunan = self.kaydedici.plaka_kara_listede_mi(plaka)
            if kara_listede_okunan:
                log.warning("ALARM: KARA LİSTEDEKİ ARAÇ TESPİT EDİLDİ! (%s)", plaka)
            kaydet, final_plaka, kazan_toplam, kazan_n = self.dogrulama.isle(arac_id, plaka, float(sonuc.guven or 0.0))
            if not kaydet:
                lider, oku_n, lider_oy, esik_c, min_g = self.dogrulama.durum_ozeti(arac_id)
                self.cizici.plaka_kutusu(kare, bbox, plaka, oku_n, esik_c, arac_id=arac_id, kara_liste=kara_listede_okunan, oy_lider=lider, oy_lider_toplam=lider_oy, oy_min_guven=min_g)
                continue
            if final_plaka is None:
                continue
            ort_ocr = float(kazan_toplam) / max(1, int(kazan_n))
            final_conf = 0.6 * ort_ocr + 0.4 * float(yolo_conf)
            kantar_dolu = guncel_kg >= float(CONFIG["MIN_KILIT_AGIRLIK"])
            if kantar_dolu:
                if self._plaka_buffer is None:
                    self._plaka_buffer = PlakaBuffer(plaka=final_plaka, guven=ort_ocr, yolo_conf=float(yolo_conf))
                else:
                    self._plaka_buffer.guncule_eger_daha_iyi(final_plaka, ort_ocr, float(yolo_conf))
            if agirlik_sabit and not self._kantar_seans_kilitli:
                kayit = self._kayit_yap(kare=kare, bbox=bbox, plaka=final_plaka, agirlik=guncel_kg, final_conf=final_conf, kaynak="DOĞRUDAN")
                if kayit is not None:
                    self.plaka_hafizasi[arac_id] = final_plaka
                    self._kantar_seans_kilitli = True
                    self._plaka_buffer = None
            elif not agirlik_sabit and kantar_dolu:
                ix1, iy1 = bbox[0], bbox[1]
                _pad_y  = max(int(kare_h * 0.055), 14)
                _fs     = round(0.72 * (kare_h / 720.0), 2)
                cv2.putText(kare, f"PLAKA HAZIR ({final_plaka}) — AĞIRLIK SABİTLENİYOR...", (ix1, max(_pad_y, iy1 - _pad_y)), cv2.FONT_HERSHEY_SIMPLEX, _fs, (0, 200, 255), 2)

    def _kare_isle(self, kare: np.ndarray):
        self._kare_sayaci += 1
        self._fps_guncelle()
        kare_h, kare_w = kare.shape[:2]
        simdi = time.time()
        roi_x1, roi_y1, roi_x2, roi_y2 = self._roi_piksel_guncelle(kare_h, kare_w)
        guncel_kg, agirlik_sabit = self.kantar_okuyucu.veri
        esik_kg = float(CONFIG["MIN_KILIT_AGIRLIK"])
        kantar_dolu = guncel_kg >= esik_kg

        _olcek_ref  = kare_h / 720.0
        tx          = int(kare_w * 0.013)
        y1          = int(kare_h * 0.090)
        y2          = int(kare_h * 0.135)
        y3          = int(kare_h * 0.180)
        fs_normal   = round(0.65 * _olcek_ref, 2)
        fs_kucuk    = round(0.55 * _olcek_ref, 2)
        pad_y_bbox  = max(int(kare_h * 0.055), 14)

        if not kantar_dolu and self._kantar_seans_kilitli:
            if self._seans_sifir_baslangic is None:
                self._seans_sifir_baslangic = simdi
            else:
                gecen = simdi - self._seans_sifir_baslangic
                if gecen >= float(CONFIG["SEANS_SIFIR_BEKLEME"]):
                    self._seans_temizle()
                else:
                    kalan = float(CONFIG["SEANS_SIFIR_BEKLEME"]) - gecen
                    cv2.putText(kare, f"ARAÇ İNİYOR — Guard: {kalan:.1f}sn kaldı", (tx, y3), cv2.FONT_HERSHEY_SIMPLEX, fs_kucuk, (0, 200, 255), 1)
        elif kantar_dolu and self._seans_sifir_baslangic is not None:
            self._seans_sifir_baslangic = None
        
        if not kantar_dolu and self._plaka_buffer is not None:
            if self._plaka_buffer.suresi_doldu_mu(float(CONFIG["PLAKA_BUFFER_TTL"])):
                self._plaka_buffer = None
        
        if not kantar_dolu and self.tracker.is_empty():
            self._aktif_bbox = None
            cv2.putText(kare, f"DURUM: KANTAR BOŞ ({guncel_kg:.0f} kg)", (tx, y1), cv2.FONT_HERSHEY_SIMPLEX, fs_normal, (0, 255, 0), 2)
            self.ocr_worker.sonuclari_topla()
            return
        
        if kantar_dolu:
            if agirlik_sabit:
                cv2.putText(kare, f"KANTARDA ARAÇ VAR: {guncel_kg:.0f} kg  [SABİT]", (tx, y1), cv2.FONT_HERSHEY_SIMPLEX, fs_normal, (0, 165, 255), 2)
            else:
                cv2.putText(kare, f"AĞIRLIK BEKLENİYOR... {guncel_kg:.0f} kg", (tx, y1), cv2.FONT_HERSHEY_SIMPLEX, fs_normal, (0, 200, 255), 2)
            if agirlik_sabit and self._plaka_buffer is not None and not self._kantar_seans_kilitli and not self._plaka_buffer.suresi_doldu_mu(float(CONFIG["PLAKA_BUFFER_TTL"])):
                buf = self._plaka_buffer
                final_conf = 0.6 * buf.guven + 0.4 * buf.yolo_conf
                buf_bbox = self._aktif_bbox or (tx, int(kare_h * 0.125), int(kare_w * 0.556), int(kare_h * 0.181))
                kayit = self._kayit_yap(kare=kare, bbox=buf_bbox, plaka=buf.plaka, agirlik=guncel_kg, final_conf=final_conf, kaynak="BUFFER")
                if kayit is not None:
                    self._kantar_seans_kilitli = True
                    self._plaka_buffer = None
        
        if self._kantar_seans_kilitli:
            cv2.putText(kare, "SEANS KİLİTLİ — ARAÇ ÇIKIŞI BEKLENİYOR", (tx, y2), cv2.FONT_HERSHEY_SIMPLEX, fs_normal, (0, 80, 255), 2)
            self.ocr_worker.sonuclari_topla()
            return
        
        self._ocr_sonuclarini_isle(kare, guncel_kg, agirlik_sabit, kare_w, kare_h)
        
        if self._kare_sayaci % 2 == 0:
            plaka_listesi = self.tespitci.plakalari_bul(kare)
            self._son_plaka_listesi = plaka_listesi
        else:
            plaka_listesi = self._son_plaka_listesi
        if not plaka_listesi:
            self._aktif_bbox = None
            return
        
        for silinen_id in self.tracker.purge_expired():
            self.plaka_hafizasi.pop(silinen_id, None)
        self._aktif_bbox = None
        ocr_calis = self._kare_sayaci % CONFIG["OCR_KARE_ATLAMA"] == 0
        
        for x1, y1, x2, y2, yolo_conf in plaka_listesi:
            ix1, iy1 = max(0, int(x1)), max(0, int(y1))
            ix2, iy2 = min(kare_w, int(x2)), min(kare_h, int(y2))
            if ix2 <= ix1 or iy2 <= iy1: continue
            cx, cy = (ix1 + ix2) / 2.0, (iy1 + iy2) / 2.0
            if not (roi_x1 <= cx <= roi_x2 and roi_y1 <= cy <= roi_y2): continue
            bbox = (ix1, iy1, ix2, iy2)
            arac_id = self.tracker.assign_id(bbox)
            if self._aktif_bbox is None: self._aktif_bbox = bbox
            cv2.rectangle(kare, (ix1, iy1), (ix2, iy2), (128, 0, 128), 1)
            
            if arac_id in self.plaka_hafizasi:
                plaka_cached = self.plaka_hafizasi[arac_id]
                kara_listede = self.kaydedici.plaka_kara_listede_mi(plaka_cached)
                if kara_listede:
                    log.warning("ALARM: KARA LİSTEDEKİ ARAÇ TESPİT EDİLDİ!")
                self.cizici.basarili_kayit(kare, bbox, plaka_cached)
                cv2.putText(kare, f"ID:{arac_id} - {plaka_cached}", (ix1, max(pad_y_bbox, iy1 - pad_y_bbox)), cv2.FONT_HERSHEY_SIMPLEX, round(0.80 * _olcek_ref, 2), (0, 0, 255) if kara_listede else (0, 255, 0), 2)
                continue
            
            if not ocr_calis: continue
            w_roi = ix2 - ix1
            pad = int(w_roi * 0.10)
            roi = kare[iy1:iy2, max(0, ix1 - pad):min(kare_w, ix2 + pad)]
            if roi.size == 0: continue
            gonderildi = self.ocr_worker.gorevi_gonder(OcrGorevi(roi_bgr=roi.copy(), arac_id=arac_id, yolo_conf=float(yolo_conf), bbox=bbox))
            if not gonderildi:
                log.debug("OCR kuyruğu dolu — kare %d için araç %d atlandı.", self._kare_sayaci, arac_id)
        
        if self._kare_sayaci % 300 == 0:
            self.dogrulama.temizle_eski()

    def calistir(self):
        sistem_referansi_ata(self)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self._fastapi_proc = self._fastapi_baslat()
        
        self.kantar_okuyucu.start()
        log.info("KantarOkuyucu başlatıldı.")
        self.ocr_worker.start()
        log.info("OcrWorker başlatıldı.")
        kare_kuyrugu: queue.Queue = queue.Queue(maxsize=max(1, int(CONFIG.get("KARE_KUYRUK_BOYUTU", 2))))
        dur = threading.Event()
        uretici = threading.Thread(target=self._kamera_uretici, args=(kare_kuyrugu, dur), name="KameraUretici", daemon=True)
        uretici.start()
        log.info("Producer–Consumer aktif (kuyruk boyutu=%d). Çıkmak için 'q' veya Ctrl+C.", kare_kuyrugu.maxsize)
        try:
            while not self._cikis_istendi.is_set():
                try:
                    kare = kare_kuyrugu.get(timeout=0.25)
                except queue.Empty:
                    tus = cv2.waitKey(1) & 0xFF
                    if tus == ord("0"): self.kantar_okuyucu.simule_et(0.0)
                    elif tus == ord("1"): self.kantar_okuyucu.simule_et(12000.0)
                    elif tus == ord("2"): self.kantar_okuyucu.simule_et(42000.0)
                    elif tus == ord("q"):
                        log.info("Kullanıcı 'q' ile çıkış yaptı.")
                        break
                    continue
                self._kare_isle(kare)
                self.cizici.fps_ve_bilgi(kare, self._fps, len(self.kaydedici.son_kayitlar), yakalama_fps=self._yakalama_fps)
                self._canli_kare_yaz(kare)
                cv2.imshow("OtoKantar V11", kare)
                tus = cv2.waitKey(1) & 0xFF
                if tus == ord("0"): self.kantar_okuyucu.simule_et(0.0)
                elif tus == ord("1"): self.kantar_okuyucu.simule_et(12000.0)
                elif tus == ord("2"): self.kantar_okuyucu.simule_et(42000.0)
                elif tus == ord("q"):
                    log.info("Kullanıcı 'q' ile çıkış yaptı.")
                    break
        except KeyboardInterrupt:
            log.info("Ctrl+C (KeyboardInterrupt) — graceful shutdown tetiklendi.")
        finally:
            self._kapat(dur, uretici)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sistem = OtoKantar()
    sistem.calistir()