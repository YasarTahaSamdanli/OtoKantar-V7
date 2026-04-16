import time


class CentroidTracker:
    def __init__(
        self,
        max_distance: float = 60.0,
        max_age_s: float = 10.0,
        iou_esik: float = 0.2,
        gorunmezlik_max: float = 2.0,
    ):
        self.max_distance = float(max_distance)
        self.max_age_s = float(max_age_s)
        self.iou_esik = float(iou_esik)
        self._purge_after_s = min(float(max_age_s), float(gorunmezlik_max))
        self._next_id = 1
        self._tracks: dict = {}

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
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        ab = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = aa + ab - inter
        return inter / union if union > 0.0 else 0.0

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

    def assign_id(self, bbox: tuple) -> int:
        now = time.time()
        self._purge(now)
        bbox_f = tuple(float(v) for v in bbox)
        c_new = self._centroid(bbox_f)

        best_iou_id, best_iou = None, self.iou_esik
        for tid, tr in self._tracks.items():
            iou = self._iou(bbox_f, tr["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_iou_id = tid

        if best_iou_id is not None:
            chosen = best_iou_id
        else:
            best_c_id, best_d = None, None
            for tid, tr in self._tracks.items():
                d = self._dist(c_new, self._centroid(tr["bbox"]))
                if d <= self.max_distance and (best_d is None or d < best_d):
                    best_d, best_c_id = d, tid
            chosen = best_c_id if best_c_id is not None else self._next_id
            if best_c_id is None:
                self._next_id += 1

        self._tracks[chosen] = {"bbox": bbox_f, "last_matched": now}
        return chosen
