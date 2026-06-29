from __future__ import annotations

import atexit
import json
import logging
import logging.handlers
import os
import statistics
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from otokantar_app.config import CONFIG

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None


STRESS_LOG_DIR = Path(__file__).resolve().parents[2] / "stress_test_logs"


def stress_logging_enabled() -> bool:
    raw = os.getenv("STRESS_TEST_LOGGING", str(CONFIG.get("STRESS_TEST_LOGGING", False)))
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "debug"}


class _MessageFilter(logging.Filter):
    def __init__(self, *needles: str) -> None:
        super().__init__()
        self.needles = tuple(n.lower() for n in needles)

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage().lower()
        return any(needle in message for needle in self.needles)


class _ExceptionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage().lower()
        return record.exc_info is not None or record.levelno >= logging.ERROR or "exception" in message or "hata" in message


def setup_stress_logging(logger: logging.Logger) -> None:
    if not stress_logging_enabled():
        return

    STRESS_LOG_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s [%(threadName)s] [%(module)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    existing = {
        str(getattr(handler, "baseFilename", ""))
        for handler in logger.handlers
        if getattr(handler, "baseFilename", None)
    }

    def add_file(name: str, level: int = logging.DEBUG, flt: Optional[logging.Filter] = None) -> None:
        path = STRESS_LOG_DIR / name
        if str(path) in existing:
            return
        handler = logging.handlers.RotatingFileHandler(
            filename=path,
            maxBytes=int(CONFIG.get("STRESS_LOG_MAX_BYTES", 20 * 1024 * 1024)),
            backupCount=int(CONFIG.get("STRESS_LOG_BACKUP_COUNT", 10)),
            encoding="utf-8",
        )
        handler.setLevel(level)
        handler.setFormatter(formatter)
        if flt is not None:
            handler.addFilter(flt)
        logger.addHandler(handler)

    add_file("python.log")
    add_file("ocr.log", flt=_MessageFilter("ocr", "paddle", "easyocr"))
    add_file("yolo.log", flt=_MessageFilter("yolo", "bbox", "tespit"))
    add_file("api.log", flt=_MessageFilter("remote sync", "api", "http", "request", "post"))
    add_file("live_ingest.log", flt=_MessageFilter("canli", "live", "remote sync", "ingest"))
    add_file("exceptions.log", level=logging.WARNING, flt=_ExceptionFilter())
    add_file("threads.log", flt=_MessageFilter("thread", "worker", "producer", "consumer", "kuyruk", "queue"))
    add_file("queue.log", flt=_MessageFilter("kuyruk", "queue", "overflow", "back-pressure", "pending"))


class StressMetrics:
    def __init__(self) -> None:
        self.enabled = stress_logging_enabled()
        self.log_dir = STRESS_LOG_DIR
        self._lock = threading.Lock()
        self._start_wall = datetime.now()
        self._start = time.monotonic()
        self._fps_values: list[float] = []
        self._max_cpu_percent = 0.0
        self._max_ram_mb = 0.0
        self._ocr_success = 0
        self._ocr_error = 0
        self._api_error = 0
        self._exception_count = 0
        self._yolo_count = 0
        self._ocr_durations_ms: list[float] = []
        self._yolo_durations_ms: list[float] = []
        self._summary_written = False
        self._stop_event = threading.Event()
        self._perf_thread: Optional[threading.Thread] = None
        self._process = psutil.Process(os.getpid()) if self.enabled and psutil else None

        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.event("python", "stress_metrics_started", log_dir=str(self.log_dir))
            atexit.register(self.write_summary)

    def start_performance_sampler(self, interval: float = 5.0) -> None:
        if not self.enabled or self._perf_thread is not None:
            return
        self._perf_thread = threading.Thread(
            target=self._performance_loop,
            args=(max(1.0, float(interval)),),
            name="StressPerfSampler",
            daemon=True,
        )
        self._perf_thread.start()

    def stop_performance_sampler(self) -> None:
        self._stop_event.set()
        if self._perf_thread is not None:
            self._perf_thread.join(timeout=2.0)

    def event(self, category: str, event: str, **data: Any) -> None:
        if not self.enabled:
            return
        payload = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "uptime_sec": round(time.monotonic() - self._start, 3),
            "thread": threading.current_thread().name,
            "event": event,
            **data,
        }
        path = self.log_dir / f"{category}.log"
        try:
            with self._lock:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass

    def record_fps(self, fps: float, capture_fps: float = 0.0, queue_size: Optional[int] = None) -> None:
        if not self.enabled or fps <= 0:
            return
        with self._lock:
            self._fps_values.append(float(fps))
        self.event("fps", "fps_sample", fps=round(float(fps), 3), capture_fps=round(float(capture_fps), 3), queue_size=queue_size)

    def record_ocr(self, duration_ms: float, success: bool, **data: Any) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._ocr_durations_ms.append(float(duration_ms))
            if success:
                self._ocr_success += 1
            else:
                self._ocr_error += 1
        self.event("ocr", "ocr_duration", duration_ms=round(float(duration_ms), 3), success=bool(success), **data)

    def record_yolo(self, duration_ms: float, detections: int, **data: Any) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._yolo_count += 1
            self._yolo_durations_ms.append(float(duration_ms))
        self.event("yolo", "yolo_duration", duration_ms=round(float(duration_ms), 3), detections=int(detections), **data)

    def record_api_error(self, **data: Any) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._api_error += 1
        self.event("api", "api_error", **data)

    def record_exception(self, **data: Any) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._exception_count += 1
        self.event("exceptions", "exception", **data)

    def _performance_loop(self, interval: float) -> None:
        if self._process is not None:
            self._process.cpu_percent(interval=None)
        while not self._stop_event.wait(interval):
            self.sample_performance()

    def sample_performance(self) -> None:
        if not self.enabled:
            return
        cpu_percent = 0.0
        ram_mb = 0.0
        system_ram_percent = None
        if self._process is not None:
            try:
                cpu_percent = float(self._process.cpu_percent(interval=None))
                ram_mb = float(self._process.memory_info().rss) / (1024 * 1024)
                system_ram_percent = float(psutil.virtual_memory().percent) if psutil else None
            except Exception as exc:
                self.event("performance", "performance_sample_error", error=str(exc))

        with self._lock:
            self._max_cpu_percent = max(self._max_cpu_percent, cpu_percent)
            self._max_ram_mb = max(self._max_ram_mb, ram_mb)

        self.event(
            "performance",
            "performance_sample",
            process_cpu_percent=round(cpu_percent, 2),
            process_ram_mb=round(ram_mb, 2),
            system_ram_percent=round(system_ram_percent, 2) if system_ram_percent is not None else None,
            active_threads=threading.active_count(),
        )

    def summary(self, total_vehicles: int = 0) -> dict[str, Any]:
        with self._lock:
            fps_values = list(self._fps_values)
            ocr_durations = list(self._ocr_durations_ms)
            yolo_durations = list(self._yolo_durations_ms)
            return {
                "started_at": self._start_wall.isoformat(timespec="seconds"),
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "runtime_sec": round(time.monotonic() - self._start, 3),
                "total_vehicles": int(total_vehicles),
                "average_fps": round(statistics.fmean(fps_values), 3) if fps_values else 0.0,
                "min_fps": round(min(fps_values), 3) if fps_values else 0.0,
                "max_fps": round(max(fps_values), 3) if fps_values else 0.0,
                "max_cpu_percent": round(self._max_cpu_percent, 2),
                "max_ram_mb": round(self._max_ram_mb, 2),
                "ocr_success_count": self._ocr_success,
                "ocr_error_count": self._ocr_error,
                "api_error_count": self._api_error,
                "exception_count": self._exception_count,
                "yolo_inference_count": self._yolo_count,
                "average_ocr_ms": round(statistics.fmean(ocr_durations), 3) if ocr_durations else 0.0,
                "average_yolo_ms": round(statistics.fmean(yolo_durations), 3) if yolo_durations else 0.0,
                "log_dir": str(self.log_dir),
            }

    def write_summary(self, total_vehicles: int = 0) -> dict[str, Any]:
        if not self.enabled:
            return {}
        if self._summary_written:
            return self.summary(total_vehicles)
        self.sample_performance()
        summary = self.summary(total_vehicles)
        try:
            with self._lock:
                (self.log_dir / "summary.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                self._summary_written = True
        except Exception:
            pass
        return summary


stress_metrics = StressMetrics()
