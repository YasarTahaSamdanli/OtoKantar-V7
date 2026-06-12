from __future__ import annotations

import json
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
        job = (payload_copy, image, on_success)

        simdi = time.monotonic()
        if self._inflight is not None and not self._inflight.done():
            self._pending = job
            return
        if not zorla and (simdi - self._last_submit) < self.min_interval:
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

    def _flush_pending(self) -> None:
        if self._pending is None:
            return
        job = self._pending
        self._pending = None
        self._last_submit = time.monotonic()
        self._submit(job)

    def _gonder_sync(
        self,
        payload: dict,
        image_path: Optional[Path],
        queue_id: Optional[str] = None,
        on_success: Optional[Callable[[], None]] = None,
    ) -> None:
        try:
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
        total_attempts = 1 + len(_RETRY_DELAYS)

        for attempt in range(total_attempts):
            if attempt > 0:
                time.sleep(_RETRY_DELAYS[attempt - 1])

            success, status_code = self._attempt_post(payload, image_path)
            if success:
                self._remove_from_queue(queue_id, payload, image_path)
                if on_success is not None:
                    try:
                        on_success()
                    except Exception as e:
                        log.warning("Remote sync on_success hatasi: %s", e)
                return True

            if status_code is not None:
                log.warning(
                    "Remote sync deneme %s/%s basarisiz: HTTP %s",
                    attempt + 1,
                    total_attempts,
                    status_code,
                )
            else:
                log.warning(
                    "Remote sync deneme %s/%s basarisiz: ag hatasi",
                    attempt + 1,
                    total_attempts,
                )

        if queue_id is None:
            self._enqueue(payload, image_path)
        else:
            log.warning("Remote sync kuyrukta bekliyor: %s", queue_id)

        return False

    def _attempt_post(self, payload: dict, image_path: Optional[Path]) -> tuple[bool, Optional[int]]:
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
                log.warning("Remote sync hata: HTTP %s - %s", response.status_code, response.text[:200])
                return False, response.status_code
            return True, response.status_code
        except Exception as e:
            log.warning("Remote sync gonderilemedi: %s", e)
            return False, None
        finally:
            if image_handle is not None:
                image_handle.close()

    def _read_queue(self) -> list[dict]:
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

    def _write_queue(self, entries: list[dict]) -> None:
        try:
            _QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = _QUEUE_PATH.with_suffix(".jsonl.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            tmp.replace(_QUEUE_PATH)
        except Exception as e:
            log.warning("Remote sync kuyruk yazilamadi: %s", e)

    def _enqueue(self, payload: dict, image_path: Optional[Path]) -> None:
        entry = {
            "id": str(uuid.uuid4()),
            "payload": payload,
            "image_path": str(image_path) if image_path is not None else None,
            "queued_at": time.time(),
        }
        try:
            _QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_QUEUE_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            log.warning("Remote sync kuyruga eklendi: %s", entry["id"])
        except Exception as e:
            log.warning("Remote sync kuyruga eklenemedi: %s", e)

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
        entries = self._read_queue()
        if not entries:
            return

        if queue_id is not None:
            remaining = [e for e in entries if e.get("id") != queue_id]
        else:
            target_key = self._entry_key(payload, image_path)
            remaining = [
                e
                for e in entries
                if self._entry_key(
                    dict(e.get("payload") or {}),
                    Path(e["image_path"]) if e.get("image_path") else None,
                )
                != target_key
            ]

        if len(remaining) == len(entries):
            return

        self._write_queue(remaining)
