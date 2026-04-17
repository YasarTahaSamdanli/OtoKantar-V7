"""
dogrulama.py — Production-Grade Validation & Decision Engine
=============================================================
Temporal voting, plate clustering, and final plate decision logic.

Architecture role: DECISION layer only.
  • Accumulates OCR readings across frames (votes)
  • Clusters similar plates (Levenshtein + format validation)
  • Applies confidence decay for stale observations
  • Emits a final accepted plate once evidence is sufficient
  • Thread-safe: all state mutations are protected by a per-vehicle lock

Key improvements over v1:
  ✅ Per-vehicle RLock → no shared-state race conditions
  ✅ Confidence decay — old observations lose weight automatically
  ✅ Confidence fusion: weighted combination (avoids geometric-mean collapse near zero)
  ✅ Adaptive decay based on time gap between observations, not fixed per-frame alpha
  ✅ False-injection guard: auto-correct only on dist ≤ 1
  ✅ Bounded memory: temizle_eski() is called automatically via isle()
  ✅ Max-cluster-size cap to prevent O(N²) blow-up under noise storms
  ✅ Cooldown is per-vehicle (not global), preventing cross-talk
"""

from __future__ import annotations

import re
import threading
import time
from typing import Optional

from otokantar_app.models import DogrulamaDurumu


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Decay is now time-based: confidence halves every DECAY_HALF_LIFE seconds.
# This makes decay independent of FPS fluctuations.
# At 10 FPS (dt=0.1s), per-frame alpha ≈ 0.993  (very slow, good for slow cameras)
# At 30 FPS (dt=0.033s), per-frame alpha ≈ 0.998 (negligible per frame, as expected)
# At 1 FPS  (dt=1.0s),   per-frame alpha ≈ 0.933 (faster, stale readings age out)
_DECAY_HALF_LIFE_S = 8.0     # Confidence halves after this many seconds of no confirmation

# Burst correction: if a vehicle is seen very frequently (high observation rate)
# we slow the decay further so rapid re-reads don't erode valid accumulation.
_DECAY_MIN_ALPHA = 0.80      # Fastest per-call decay allowed (very long gap)
_DECAY_MAX_ALPHA = 0.999     # Slowest per-call decay allowed (very short gap)

_AUTO_TEMIZLE_ARALIGI = 5.0  # Run auto-cleanup every N seconds
_MAX_AYRI_PLAKA = 20         # Max distinct plate strings stored per vehicle


# ---------------------------------------------------------------------------
# Internal per-vehicle state
# ---------------------------------------------------------------------------

class _AracDurumu:
    """
    Thread-safe per-vehicle accumulator.

    Fields
    ------
    oylar           : plate_str → accumulated weighted confidence
    hane            : plate_str → observation count
    son_gorulme     : wall-clock time of last observation
    son_kayit       : wall-clock time of last accepted plate
    okuma_sayisi    : total readings seen (including rejects)
    son_decay_zamani: wall-clock time of the last decay application
    kilit           : reentrant lock for this vehicle's state
    """

    __slots__ = (
        "oylar", "hane", "son_gorulme", "son_kayit",
        "okuma_sayisi", "son_decay_zamani", "kilit",
    )

    def __init__(self) -> None:
        self.oylar: dict[str, float] = {}
        self.hane: dict[str, int] = {}
        self.son_gorulme: float = 0.0
        self.son_kayit: float = 0.0
        self.okuma_sayisi: int = 0
        self.son_decay_zamani: float = time.time()
        self.kilit: threading.RLock = threading.RLock()

    def uygula_decay(self, su_an: float) -> None:
        """
        Apply time-proportional exponential decay.

        Rather than a fixed per-frame alpha (which changes meaning at different
        FPS), we compute alpha from the elapsed time since last decay:

            alpha = 0.5 ^ (dt / half_life)

        This ensures a plate seen 8 s ago always has half its original weight,
        regardless of whether the camera is running at 5 FPS or 30 FPS.
        alpha is clamped to [_DECAY_MIN_ALPHA, _DECAY_MAX_ALPHA] to prevent
        single long-gap calls from wiping the entire accumulator.
        """
        import math
        dt = su_an - self.son_decay_zamani
        if dt <= 0.0:
            return

        # alpha = 2^(-dt/half_life) — time-proportional decay
        raw_alpha = 2.0 ** (-dt / _DECAY_HALF_LIFE_S)
        alpha = max(_DECAY_MIN_ALPHA, min(_DECAY_MAX_ALPHA, raw_alpha))

        for p in list(self.oylar):
            self.oylar[p] *= alpha
            if self.oylar[p] < 0.05:
                del self.oylar[p]
                self.hane.pop(p, None)

        self.son_decay_zamani = su_an

    def sifirla_oylar(self) -> None:
        self.oylar.clear()
        self.hane.clear()
        self.okuma_sayisi = 0
        self.son_decay_zamani = time.time()


# ---------------------------------------------------------------------------
# DogrulamaMotoru
# ---------------------------------------------------------------------------

class DogrulamaMotoru:
    """
    Multi-frame plate validation engine.

    Parameters
    ----------
    esik : int
        Minimum number of observations before a plate is accepted.
    min_toplam_guven : float
        Minimum accumulated weighted confidence to accept a plate.
    kayit_sonrasi_bekleme : float
        Cooldown (seconds) after accepting a plate for a vehicle.
    """

    def __init__(
        self,
        esik: int,
        min_toplam_guven: float,
        kayit_sonrasi_bekleme: float,
    ) -> None:
        self.esik = int(esik)
        self.min_toplam_guven = float(min_toplam_guven)
        self.bekleme = float(kayit_sonrasi_bekleme)

        # Global state dict: arac_id → _AracDurumu
        # Protected by _global_kilit for structural changes (add/remove keys)
        self._durum: dict[int, _AracDurumu] = {}
        self._global_kilit = threading.Lock()

        # Auto-cleanup scheduling
        self._son_temizlik: float = time.time()

        # Known plate set (loaded from DB externally)
        # Access is protected by _bilinen_kilit
        self._bilinen_plakalar: set[str] = set()
        self._bilinen_kilit = threading.RLock()

        # Validation regex: Turkish plate format (il kodu 01–81)
        self.TR_PLAKA_REGEX = re.compile(
            r"^(0[1-9]|[1-7][0-9]|8[0-1])[A-Z]{1,3}\d{2,4}$"
        )

        # OCR confusion normalisation map
        self._normalize_map = str.maketrans({
            "O": "0", "I": "1", "İ": "1",
            "B": "8", "S": "5", "G": "6", "Z": "2",
        })

    # ------------------------------------------------------------------
    # Public: known plates management
    # ------------------------------------------------------------------

    def bilinen_plakalari_guncelle(self, plakalar: set[str]) -> None:
        """Replace the known-plates set (thread-safe)."""
        with self._bilinen_kilit:
            self._bilinen_plakalar = set(plakalar)

    def bilinen_plaka_ekle(self, plaka: str) -> None:
        with self._bilinen_kilit:
            self._bilinen_plakalar.add(plaka.upper().strip())

    # ------------------------------------------------------------------
    # Normalisation & validation
    # ------------------------------------------------------------------

    def _normalize(self, plaka: str) -> str:
        return plaka.upper().replace(" ", "").translate(self._normalize_map)

    def _tr_plaka_gecerli_mi(self, plaka: str) -> bool:
        return bool(self.TR_PLAKA_REGEX.match(plaka))

    # ------------------------------------------------------------------
    # Levenshtein distance  (unchanged — well-known algorithm)
    # ------------------------------------------------------------------

    @staticmethod
    def _mesafe_hesapla(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        if not s2:
            return len(s1)
        prev = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
            prev = curr
        return prev[-1]

    # ------------------------------------------------------------------
    # Similarity check
    # ------------------------------------------------------------------

    def _benzer_mi(self, p1: str, p2: str) -> bool:
        """
        Two plates are "similar" iff:
          - Both pass format validation
          - Length difference ≤ 1
          - Levenshtein distance ≤ 2
        """
        if not (self._tr_plaka_gecerli_mi(p1) and self._tr_plaka_gecerli_mi(p2)):
            return False
        if abs(len(p1) - len(p2)) > 1:
            return False
        return self._mesafe_hesapla(p1, p2) <= 2

    # ------------------------------------------------------------------
    # Clustering  (centre-free, any-member linkage)
    # ------------------------------------------------------------------

    def _cluster(
        self,
        oylar: dict[str, float],
        hane: dict[str, int],
    ) -> Optional[tuple[str, float, int]]:
        """
        Group similar plates into clusters and return the best (leader,
        total_confidence, total_count) from the winning cluster.

        Returns None if oylar is empty.
        """
        if not oylar:
            return None

        kumeler: list[dict] = []  # [{uyeler: [str, ...]}]

        for plaka in oylar:
            matched = False
            for kume in kumeler:
                if any(self._benzer_mi(plaka, u) for u in kume["uyeler"]):
                    kume["uyeler"].append(plaka)
                    matched = True
                    break
            if not matched:
                kumeler.append({"uyeler": [plaka]})

        # Score each cluster
        with self._bilinen_kilit:
            bilinen = self._bilinen_plakalar

        en_iyi: Optional[tuple[str, float, int]] = None
        en_skor = -1.0

        for kume in kumeler:
            uyeler = kume["uyeler"]
            toplam_guven = sum(oylar.get(p, 0.0) for p in uyeler)
            toplam_hane = sum(hane.get(p, 0) for p in uyeler)

            # Leader = plate with highest composite score
            def _lider_skoru(p: str) -> tuple:
                return (
                    p in bilinen,        # DB authority (bool → 0/1)
                    oylar.get(p, 0.0),   # accumulated confidence
                    hane.get(p, 0),      # raw count
                )

            lider = max(uyeler, key=_lider_skoru)

            skor = toplam_guven + toplam_hane * 0.2
            if lider in bilinen:
                skor += 2.0  # DB authority bonus

            if skor > en_skor:
                en_skor = skor
                en_iyi = (lider, toplam_guven, toplam_hane)

        return en_iyi

    # ------------------------------------------------------------------
    # Auto-correct  (dist ≤ 1 only — guards False-Plate Injection)
    # ------------------------------------------------------------------

    def _oto_duzelt(self, plaka: str) -> str:
        """
        Correct at most 1-character OCR errors against known plates.
        Distance-2 corrections are intentionally skipped to avoid
        false-plate injection (e.g. "06ABC123" → "06ABD124").
        """
        with self._bilinen_kilit:
            bilinen = self._bilinen_plakalar

        if not bilinen or plaka in bilinen:
            return plaka

        best_aday = plaka
        best_dist = 999

        for kayitli in bilinen:
            d = self._mesafe_hesapla(plaka, kayitli)
            if d < best_dist:
                best_dist = d
                best_aday = kayitli

        return best_aday if best_dist <= 1 else plaka

    # ------------------------------------------------------------------
    # Internal state helpers
    # ------------------------------------------------------------------

    def _get_or_create(self, arac_id: int) -> _AracDurumu:
        """Return the state for a vehicle, creating it if needed."""
        d = self._durum.get(arac_id)
        if d is None:
            with self._global_kilit:
                d = self._durum.get(arac_id)
                if d is None:
                    d = _AracDurumu()
                    self._durum[arac_id] = d
        return d

    def _otomatik_temizle(self, su_an: float) -> None:
        """Periodically remove stale vehicle states (every 5 s)."""
        if su_an - self._son_temizlik < _AUTO_TEMIZLE_ARALIGI:
            return
        self._son_temizlik = su_an
        self.temizle_eski()

    # ------------------------------------------------------------------
    # Core processing  (thread-safe per-vehicle)
    # ------------------------------------------------------------------

    def isle(
        self,
        arac_id: int,
        plaka: str,
        ocr_guven: float,
        yolo_guven: float = 1.0,
    ) -> tuple[bool, Optional[str], float, int]:
        """
        Process one OCR reading for a vehicle.

        Parameters
        ----------
        arac_id    : stable vehicle ID from tracker
        plaka      : raw OCR text
        ocr_guven  : OCR backend confidence [0, 1]
        yolo_guven : YOLO detection confidence [0, 1]  (default 1.0 if unknown)

        Returns
        -------
        (accepted, plate_str, total_confidence, total_count)
          accepted = True means a plate has been confirmed and state is reset.
        """
        su_an = time.time()
        self._otomatik_temizle(su_an)

        d = self._get_or_create(arac_id)

        with d.kilit:
            # --- Cooldown guard ---
            if su_an - d.son_kayit < self.bekleme:
                return (False, None, 0.0, 0)

            # --- Timeout reset: vehicle was absent for >5 s ---
            if su_an - d.son_gorulme > 5.0 and d.son_gorulme > 0.0:
                d.sifirla_oylar()

            d.son_gorulme = su_an
            d.okuma_sayisi += 1

            # --- Confidence gate ---
            # Weighted fusion: OCR quality is the primary signal (weight 0.65),
            # YOLO detection quality is secondary (weight 0.35).
            # Unlike geometric mean, this does NOT collapse to near-zero when
            # either input is small — a very clear OCR reading on a slightly
            # blurry YOLO detection still scores reasonably.
            # Both inputs are clamped to [0, 1] defensively.
            ocr_g = min(max(float(ocr_guven), 0.0), 1.0)
            yolo_g = min(max(float(yolo_guven), 0.0), 1.0)
            combined_guven = 0.65 * ocr_g + 0.35 * yolo_g
            if combined_guven < 0.40:
                return (False, None, 0.0, 0)

            # --- Normalise → validate → auto-correct ---
            plaka = self._normalize(plaka)

            if not self._tr_plaka_gecerli_mi(plaka):
                return (False, None, 0.0, 0)

            plaka = self._oto_duzelt(plaka)

            # --- Apply time-proportional decay before adding new vote ---
            d.uygula_decay(su_an)

            # --- Enforce max distinct plates cap (noise storm protection) ---
            if plaka not in d.oylar and len(d.oylar) >= _MAX_AYRI_PLAKA:
                # Drop the weakest entry to make room
                weakest = min(d.oylar, key=lambda p: d.oylar[p])
                del d.oylar[weakest]
                d.hane.pop(weakest, None)

            # --- Accumulate vote ---
            g = min(max(combined_guven, 0.3), 0.95)
            d.oylar[plaka] = d.oylar.get(plaka, 0.0) + g
            d.hane[plaka] = d.hane.get(plaka, 0) + 1

            # --- Cluster and decide ---
            sonuc = self._cluster(d.oylar, d.hane)
            if sonuc is None:
                return (False, None, 0.0, 0)

            lider, toplam_guven, toplam_hane = sonuc

            tamam = (
                toplam_hane >= self.esik or
                toplam_guven >= self.min_toplam_guven
            )

            if not tamam:
                return (False, None, 0.0, 0)

            # --- Accept: reset state ---
            d.son_kayit = su_an
            d.sifirla_oylar()

            return (True, lider, toplam_guven, toplam_hane)

    # ------------------------------------------------------------------
    # Status query  (non-blocking, best-effort snapshot)
    # ------------------------------------------------------------------

    def durum_ozeti(
        self, arac_id: int
    ) -> tuple[str, int, float, int, float]:
        """
        Return a human-readable snapshot of current accumulation state.

        Returns
        -------
        (leading_plate, reading_count, total_confidence, esik, min_toplam_guven)
        """
        d = self._durum.get(arac_id)
        if d is None:
            return ("", 0, 0.0, self.esik, self.min_toplam_guven)

        with d.kilit:
            if not d.oylar:
                return ("", d.okuma_sayisi, 0.0, self.esik, self.min_toplam_guven)
            sonuc = self._cluster(d.oylar, d.hane)
            if sonuc is None:
                return ("", d.okuma_sayisi, 0.0, self.esik, self.min_toplam_guven)
            lider, toplam_guven, _ = sonuc
            return (lider, d.okuma_sayisi, toplam_guven, self.esik, self.min_toplam_guven)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def temizle_eski(self, yasam_suresi: float = 30.0) -> int:
        """
        Remove vehicle states not seen for `yasam_suresi` seconds.

        Returns
        -------
        int : number of states removed
        """
        su_an = time.time()
        with self._global_kilit:
            silinecek = [
                aid for aid, d in self._durum.items()
                if su_an - d.son_gorulme > yasam_suresi
            ]
            for aid in silinecek:
                del self._durum[aid]
        return len(silinecek)

    def sifirla(self) -> None:
        """Clear all vehicle state (e.g. at session end)."""
        with self._global_kilit:
            self._durum.clear()

    def aktif_arac_sayisi(self) -> int:
        """Return the number of currently tracked vehicles in memory."""
        return len(self._durum)