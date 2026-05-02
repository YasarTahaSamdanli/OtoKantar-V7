import time


class CentroidTracker:
    """
    Araç takip sınıfı — Hareket eden kamera için düzenlendi.

    Sabit kamerada araç görüntü içinde hareket eder; merkez kayması beklenir.
    Hareketli kamerada ise hem araç hem kamera hareket ettiğinden:
      • centroid iki kare arasında çok daha fazla kayabilir  → max_distance artırıldı
      • araç görüntüde çok kısa kalır                        → gorunmezlik_max düşürüldü
      • IoU örtüşmesi düşer (bbox ölçeği değişir)            → iou_esik düşürüldü
    """

    def __init__(
        self,
        max_distance: float = 120.0,   # eski: 60  — hareketli kamerada bbox sıçraması büyük
        max_age_s: float = 10.0,
        iou_esik: float = 0.10,        # eski: 0.20 — hareket blur'unda IoU düşer
        gorunmezlik_max: float = 1.5,  # eski: 2.0  — kısa süre görünen araçlar hızlı temizlenir
    ):
        self.max_distance   = float(max_distance)
        self.max_age_s      = float(max_age_s)
        self.iou_esik       = float(iou_esik)
        self._purge_after_s = min(float(max_age_s), float(gorunmezlik_max))
        self._next_id       = 1
        self._tracks: dict  = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Statik yardımcılar
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _centroid(bbox: tuple) -> tuple:
        x1, y1, x2, y2 = bbox
        return ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)

    @staticmethod
    def _dist(a: tuple, b: tuple) -> float:
        dx, dy = a[0] - b[0], a[1] - b[1]
        return (dx * dx + dy * dy) ** 0.5

    @staticmethod
    def _iou(a: tuple, b: tuple) -> float:
        ax1, ay1, ax2, ay2 = (float(v) for v in a)
        bx1, by1, bx2, by2 = (float(v) for v in b)
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        aa    = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        ab    = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = aa + ab - inter
        return inter / union if union > 0.0 else 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Eski track'leri temizle
    # ─────────────────────────────────────────────────────────────────────────

    def _purge(self, now: float) -> list:
        stale = [
            tid for tid, tr in self._tracks.items()
            if now - float(tr["last_matched"]) > self._purge_after_s
        ]
        for tid in stale:
            del self._tracks[tid]
        return stale

    def purge_expired(self) -> list:
        return self._purge(time.time())

    def is_empty(self) -> bool:
        return len(self._tracks) == 0

    def sifirla(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    # ─────────────────────────────────────────────────────────────────────────
    # ID atama — hareketli kamera için geliştirilmiş eşleştirme
    # ─────────────────────────────────────────────────────────────────────────

    def assign_id(self, bbox: tuple) -> int:
        now    = time.time()
        self._purge(now)
        bbox_f = tuple(float(v) for v in bbox)
        c_new  = self._centroid(bbox_f)

        # ── Adım 1: IoU eşleştirme ────────────────────────────────────────────
        # Hareketli kamerada bbox ölçeği değişse de, aynı karedeki en iyi
        # örtüşmeye bak. iou_esik 0.10'a düşürüldü.
        best_iou_id, best_iou = None, self.iou_esik
        for tid, tr in self._tracks.items():
            iou = self._iou(bbox_f, tr["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_iou_id = tid

        if best_iou_id is not None:
            chosen = best_iou_id

        else:
            # ── Adım 2: Centroid mesafe eşleştirme ───────────────────────────
            # max_distance 120px'e çıkarıldı. Kamera hareket edince centroid
            # birkaç kare içinde 60-100px kayabilir.
            best_c_id, best_d = None, None
            for tid, tr in self._tracks.items():
                d = self._dist(c_new, self._centroid(tr["bbox"]))
                if d <= self.max_distance and (best_d is None or d < best_d):
                    best_d, best_c_id = d, tid

            # ── Adım 3: Ölçek benzerliği kontrolü (YENİ) ─────────────────────
            # Centroid yakın ama bbox boyutu çok farklıysa (kamera zoom/scale
            # değişimi) yanlış eşleştirme olabilir. Boyut oranı 2x'ten fazla
            # farklıysa yeni bir ID aç.
            if best_c_id is not None:
                mevcut_bbox = self._tracks[best_c_id]["bbox"]
                w_yeni    = max(1.0, bbox_f[2]  - bbox_f[0])
                h_yeni    = max(1.0, bbox_f[3]  - bbox_f[1])
                w_eski    = max(1.0, mevcut_bbox[2] - mevcut_bbox[0])
                h_eski    = max(1.0, mevcut_bbox[3] - mevcut_bbox[1])
                en_oran   = max(w_yeni / w_eski, w_eski / w_yeni)
                boy_oran  = max(h_yeni / h_eski, h_eski / h_yeni)
                if en_oran > 2.0 or boy_oran > 2.0:
                    best_c_id = None   # boyut çok farklı → yeni ID

            if best_c_id is not None:
                chosen = best_c_id
            else:
                chosen = self._next_id
                self._next_id += 1

        self._tracks[chosen] = {"bbox": bbox_f, "last_matched": now}
        return chosen