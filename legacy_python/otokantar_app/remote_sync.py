from __future__ import annotations

import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

from otokantar_app.logger import log

try:
    import requests
except Exception as e:  # pragma: no cover
    requests = None
    requests_import_error = e

_RETRY_DELAYS = (2.0, 4.0, 8.0)
_CIRCUIT_FAILURE_THRESHOLD = 3
_CIRCUIT_COOLDOWN_SEC = 15.0
_QUEUE_PATH = Path(__file__).resolve().parent.parent / "sync_queue.jsonl"


class RemoteCanliSync:
    def __init__(
        self,
        enabled: bool,
        url: str,
        token: str,
        timeout: float = 4.0,
        min_interval: float = 0.5,
    ) -> None:
        self.enabled = bool(enabled and url and token)
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = float(timeout)
        self.min_interval = max(0.0, float(min_interval))
        self._last_submit = 0.0
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="RemoteCanliSync")
        self._inflight = None
        self._pending: Optional[tuple[dict, Optional[Path], Optional[Callable[[], None]]]] = None
        self._circuit_lock = threading.Lock()
        self._circuit_failure_streak = 0
        self._circuit_open_until = 0.0
        self._queue_lock = threading.Lock()

        if enabled and requests is None:
            self.enabled = False
            log.warning("Remote sync devre disi: requests kurulu degil (%s)", requests_import_error)
        elif self.enabled:
            log.info("Remote sync aktif: %s", self.url)
        else:
            log.info("Remote sync kapali.")

    def gonder(
        self,
        payload: Optional[dict] = None,
        image_path: Optional[str | Path] = None,
        zorla: bool = False,
        on_success: Optional[Callable[[], None]] = None,
    ) -> None:
        if not self.enabled:
            return

        payload_copy = dict(payload or {})
        image = Path(image_path) if image_path else None

        if self._circuit_is_open():
            self._enqueue(payload_copy, image, quiet=True)
            return

        job = (payload_copy, image, on_success)

        simdi = time.monotonic()
        if self._inflight is not None and not self._inflight.done():
            self._pending_job_kuyruga_al()
            self._pending = job
            return
        if not zorla and (simdi - self._last_submit) < self.min_interval:
            self._pending_job_kuyruga_al()
            self._pending = job
            return

        self._last_submit = simdi
        self._submit(job)

    def kapat(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _submit(
        self,
        job: tuple[dict, Optional[Path], Optional[Callable[[], None]]],
    ) -> None:
        payload, image_path, on_success = job
        self._inflight = self._executor.submit(
            self._gonder_sync,
            payload,
            image_path,
            None,
            on_success,
        )

    def _pending_job_kuyruga_al(self) -> None:
        if self._pending is None:
            return
        payload, image_path, _on_success = self._pending
        self._enqueue(payload, image_path, quiet=True)

    def _flush_pending(self) -> None:
        if self._pending is None:
            return
        job = self._pending
        self._pending = None
        payload, image_path, on_success = job
        if self._circuit_is_open():
            self._enqueue(payload, image_path, quiet=True)
            return
        self._last_submit = time.monotonic()
        self._submit(job)

    def _circuit_is_open(self) -> bool:
        with self._circuit_lock:
            return time.monotonic() < self._circuit_open_until

    def _circuit_record_success(self) -> None:
        with self._circuit_lock:
            self._circuit_failure_streak = 0

    def _circuit_record_failure(self) -> bool:
        with self._circuit_lock:
            self._circuit_failure_streak += 1
            if self._circuit_failure_streak < _CIRCUIT_FAILURE_THRESHOLD:
                return False
            self._circuit_failure_streak = 0
            self._circuit_open_until = time.monotonic() + _CIRCUIT_COOLDOWN_SEC
            return True

    def _gonder_sync(
        self,
        payload: dict,
        image_path: Optional[Path],
        queue_id: Optional[str] = None,
        on_success: Optional[Callable[[], None]] = None,
    ) -> None:
        try:
            if self._circuit_is_open():
                if queue_id is None:
                    self._enqueue(payload, image_path, quiet=True)
                return
            if queue_id is None:
                queue_id = self._find_queue_id(payload, image_path)
            self._drain_queue(exclude_id=queue_id)
            self._send_with_retry(
                payload,
                image_path,
                queue_id=queue_id,
                on_success=on_success,
            )
        finally:
            self._flush_pending()

    def _drain_queue(self, exclude_id: Optional[str] = None) -> None:
        if self._circuit_is_open():
            return
        for entry in self._read_queue():
            entry_id = entry.get("id")
            if not entry_id or entry_id == exclude_id:
                continue
            payload = dict(entry.get("payload") or {})
            image_raw = entry.get("image_path")
            image = Path(image_raw) if image_raw else None
            self._send_with_retry(payload, image, queue_id=entry_id)

    def _send_with_retry(
        self,
        payload: dict,
        image_path: Optional[Path],
        queue_id: Optional[str] = None,
        on_success: Optional[Callable[[], None]] = None,
    ) -> bool:
        if self._circuit_is_open():
            if queue_id is None:
                self._enqueue(payload, image_path, quiet=True)
            return False

        total_attempts = 1 + len(_RETRY_DELAYS)
        last_status: Optional[int] = None
        last_error: Optional[str] = None

        for attempt in range(total_attempts):
            if attempt > 0:
                time.sleep(_RETRY_DELAYS[attempt - 1])

            success, status_code, error = self._attempt_post(payload, image_path)
            if success:
                self._circuit_record_success()
                self._remove_from_queue(queue_id, payload, image_path)
                if on_success is not None:
                    try:
                        on_success()
                    except Exception as e:
                        log.warning("Remote sync on_success hatasi: %s", e)
                return True

            last_status = status_code
            last_error = error

        if last_status is not None:
            log.warning(
                "Remote sync %s deneme sonrasi basarisiz: HTTP %s",
                total_attempts,
                last_status,
            )
        else:
            log.warning(
                "Remote sync %s deneme sonrasi basarisiz: %s",
                total_attempts,
                last_error or "ag hatasi",
            )

        circuit_opened = self._circuit_record_failure()
        if circuit_opened:
            log.warning(
                "Remote sync circuit breaker acildi: %.0fs cooldown",
                _CIRCUIT_COOLDOWN_SEC,
            )

        if queue_id is None:
            self._enqueue(payload, image_path)
        else:
            log.debug("Remote sync kuyrukta bekliyor: %s", queue_id)

        return False

    def _attempt_post(
        self,
        payload: dict,
        image_path: Optional[Path],
    ) -> tuple[bool, Optional[int], Optional[str]]:
        assert requests is not None

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
        }
        data = {}
        files = {}
        image_handle = None

        try:
            if payload:
                data["json"] = json.dumps(payload, ensure_ascii=False)

            if image_path is not None and image_path.is_file():
                image_handle = open(image_path, "rb")
                files["image"] = ("canli_kare.jpg", image_handle, "image/jpeg")

            response = requests.post(
                self.url,
                headers=headers,
                data=data,
                files=files,
                timeout=self.timeout,
            )
            if response.status_code >= 400:
                return False, response.status_code, response.text[:200]
            return True, response.status_code, None
        except Exception as e:
            return False, None, str(e)
        finally:
            if image_handle is not None:
                image_handle.close()

    @staticmethod
    def _atomik_replace(tmp: Path, hedef: Path, deneme: int = 5) -> None:
        son_hata = None
        for _ in range(max(1, deneme)):
            try:
                tmp.replace(hedef)
                return
            except OSError as e:
                if getattr(e, "winerror", None) not in {5, 32}:
                    raise
                son_hata = e
                time.sleep(0.05)
        if son_hata is not None:
            raise son_hata

    def _read_queue_unlocked(self) -> list[dict]:
        if not _QUEUE_PATH.is_file():
            return []

        entries: list[dict] = []
        try:
            with open(_QUEUE_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        log.warning("Remote sync kuyruk satiri okunamadi, atlandi.")
        except Exception as e:
            log.warning("Remote sync kuyruk okunamadi: %s", e)
        return entries

    def _read_queue(self) -> list[dict]:
        with self._queue_lock:
            return self._read_queue_unlocked()

    def _write_queue_unlocked(self, entries: list[dict]) -> None:
        _QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _QUEUE_PATH.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._atomik_replace(tmp, _QUEUE_PATH)

    def _write_queue(self, entries: list[dict]) -> None:
        try:
            with self._queue_lock:
                self._write_queue_unlocked(entries)
        except Exception as e:
            log.warning("Remote sync kuyruk yazilamadi: %s", e)

    def _enqueue(
        self,
        payload: dict,
        image_path: Optional[Path],
        quiet: bool = False,
    ) -> Optional[str]:
        entry_key = self._entry_key(payload, image_path)
        entry = {
            "id": str(uuid.uuid4()),
            "dedup_key": entry_key,
            "payload": payload,
            "image_path": str(image_path) if image_path is not None else None,
            "queued_at": time.time(),
        }
        try:
            with self._queue_lock:
                entries = self._read_queue_unlocked()
                for existing in entries:
                    if self._entry_key_for_entry(existing) == entry_key:
                        existing_id = existing.get("id")
                        log.debug(
                            "Remote sync kuyruk duplicate atlandi: %s",
                            existing_id,
                        )
                        return existing_id
                entries.append(entry)
                self._write_queue_unlocked(entries)
            if quiet:
                log.debug("Remote sync cooldown, kuyruga eklendi: %s", entry["id"])
            else:
                log.warning("Remote sync kuyruga eklendi: %s", entry["id"])
            return entry["id"]
        except Exception as e:
            log.warning("Remote sync kuyruga eklenemedi: %s", e)
            return None

    def _find_queue_id(self, payload: dict, image_path: Optional[Path]) -> Optional[str]:
        entry_key = self._entry_key(payload, image_path)
        with self._queue_lock:
            for existing in self._read_queue_unlocked():
                if self._entry_key_for_entry(existing) == entry_key:
                    return existing.get("id")
        return None

    def _entry_key_for_entry(self, entry: dict) -> str:
        if entry.get("dedup_key"):
            return str(entry["dedup_key"])
        return self._entry_key(
            dict(entry.get("payload") or {}),
            Path(entry["image_path"]) if entry.get("image_path") else None,
        )

    def _entry_key(self, payload: dict, image_path: Optional[Path]) -> str:
        return json.dumps(
            {
                "payload": payload,
                "image_path": str(image_path) if image_path is not None else None,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def _remove_from_queue(
        self,
        queue_id: Optional[str],
        payload: dict,
        image_path: Optional[Path],
    ) -> None:
        with self._queue_lock:
            entries = self._read_queue_unlocked()
            if not entries:
                return

            if queue_id is not None:
                remaining = [e for e in entries if e.get("id") != queue_id]
            else:
                target_key = self._entry_key(payload, image_path)
                remaining = [
                    e for e in entries if self._entry_key_for_entry(e) != target_key
                ]

            if len(remaining) == len(entries):
                return

            try:
                self._write_queue_unlocked(remaining)
            except Exception as e:
                log.warning("Remote sync kuyruk guncellenemedi: %s", e)
