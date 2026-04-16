import logging
import logging.handlers
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from otokantar_app.config import CONFIG


def _logger_kur(log_dosya: str) -> logging.Logger:
    logger = logging.getLogger("OtoKantar")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.handlers.RotatingFileHandler(
        filename=log_dosya,
        maxBytes=int(CONFIG.get("LOG_MAX_BYTES", 5 * 1024 * 1024)),
        backupCount=int(CONFIG.get("LOG_BACKUP_COUNT", 7)),
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


log = _logger_kur(CONFIG["LOG_DOSYA"])


def eski_snapshot_temizle(klasor: str = "captures", retention_days: Optional[int] = None) -> int:
    if retention_days is None:
        retention_days = int(CONFIG.get("CAPTURES_RETENTION_DAYS", 30))
    klasor_yol = Path(klasor)
    if not klasor_yol.is_dir():
        return 0
    sinir = datetime.now() - timedelta(days=retention_days)
    silinen = 0
    for dosya in klasor_yol.glob("*"):
        if dosya.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        try:
            mtime = datetime.fromtimestamp(dosya.stat().st_mtime)
            if mtime < sinir:
                dosya.unlink()
                silinen += 1
        except Exception as e:
            log.warning("Snapshot silinemedi (%s): %s", dosya.name, e)
    if silinen:
        log.info("[Cleanup] %d eski snapshot silindi (>%d gün).", silinen, retention_days)
    return silinen


def _periyodik_temizlik_baslat(aralik_saat: float = 6.0) -> threading.Thread:
    def _dongu():
        while True:
            time.sleep(aralik_saat * 3600)
            eski_snapshot_temizle()

    t = threading.Thread(target=_dongu, name="SnapshotCleaner", daemon=True)
    t.start()
    return t
