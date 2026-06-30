from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Callable, Optional

from otokantar_app.logger import log
from otokantar_app.stress_logger import stress_metrics

try:
    import requests
except Exception as e:  # pragma: no cover
    requests = None
    requests_import_error = e

_RETRY_DELAYS = (2.0, 4.0, 8.0)
_CIRCUIT_FAILURE_THRESHOLD = 3
_CIRCUIT_COOLDOWN_SEC = 15.0
_QUEUE_PATH = Path(__file__).resolve().parent.parent / "sync_queue.jsonl"
_OUTBOX_PATH = Path(__file__).resolve().parent.parent / "offline_outbox.sqlite"


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
        self._queue_lock = threading.RLock()
        self._outbox_ready = False

        if enabled and requests is None:
            self.enabled = False
            log.warning("Remote sync devre disi: requests kurulu degil (%s)", requests_import_error)
        elif self.enabled:
            self._ensure_outbox()
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
        if self._invalid_image_only_job(payload_copy, image):
            log.debug("Remote sync payloadsiz JPG atlandi: %s", image)
            return

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
        deadline = time.monotonic() + max(2.0, self.timeout + 2.0)

        while time.monotonic() < deadline:
            if self._pending is not None and (
                self._inflight is None or self._inflight.done()
            ):
                try:
                    self._flush_pending()
                except RuntimeError as e:
                    log.warning("Remote sync bekleyen is gonderilemedi: %s", e)
                    break

            future = self._inflight
            if future is None:
                if self._pending is None:
                    break
                continue

            if future.done():
                try:
                    future.result()
                except Exception as e:
                    log.debug("Remote sync kapanis islemi hata ile bitti: %s", e)
                if self._pending is None:
                    break
                continue

            remaining = max(0.1, deadline - time.monotonic())
            try:
                future.result(timeout=min(1.0, remaining))
            except FuturesTimeoutError:
                continue
            except Exception as e:
                log.debug("Remote sync kapanis islemi hata ile bitti: %s", e)

        if self._pending is not None:
            payload, image_path, _on_success = self._pending
            if not self._status_only_job(payload, image_path):
                self._enqueue(payload, image_path, quiet=True)
            self._pending = None

        self._executor.shutdown(wait=True, cancel_futures=False)

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
        if self._status_only_job(payload, image_path):
            return
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
        for entry in self._sorted_queue_entries():
            entry_id = entry.get("id")
            if not entry_id or entry_id == exclude_id:
                continue
            payload = dict(entry.get("payload") or {})
            image_raw = entry.get("image_path")
            image = Path(image_raw) if image_raw else None
            if self._invalid_image_only_job(payload, image):
                log.warning("Remote sync gecersiz payloadsiz JPG kuyruktan silindi: %s", entry_id)
                self._remove_from_queue(entry_id, payload, image)
                continue
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
            if queue_id is not None:
                self._record_queue_failure(queue_id, status_code, error)
            stress_metrics.record_api_error(
                url=self.url,
                status_code=status_code,
                error=error,
                attempt=attempt + 1,
                queue_id=queue_id,
            )
            if status_code is not None and 400 <= status_code < 500 and status_code != 429:
                log.warning(
                    "Remote sync kalici hata nedeniyle kuyruktan silindi: HTTP %s",
                    status_code,
                )
                self._remove_from_queue(queue_id, payload, image_path)
                return False

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

    def _connect_outbox(self) -> sqlite3.Connection:
        _OUTBOX_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(_OUTBOX_PATH, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    @contextmanager
    def _outbox_connection(self):
        conn = self._connect_outbox()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _ensure_outbox(self) -> None:
        if self._outbox_ready:
            return

        with self._queue_lock:
            if self._outbox_ready:
                return
            with self._outbox_connection() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS outbox_events (
                        id TEXT PRIMARY KEY,
                        dedup_key TEXT NOT NULL UNIQUE,
                        event_type TEXT NOT NULL DEFAULT 'remote_sync',
                        payload_json TEXT NOT NULL,
                        image_path TEXT,
                        status TEXT NOT NULL DEFAULT 'pending',
                        attempts INTEGER NOT NULL DEFAULT 0,
                        last_error TEXT,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        sent_at REAL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_outbox_status_created
                    ON outbox_events (status, created_at)
                    """
                )
                self._migrate_jsonl_queue_unlocked(conn)
            self._outbox_ready = True

    def _migrate_jsonl_queue_unlocked(self, conn: sqlite3.Connection) -> None:
        if not _QUEUE_PATH.is_file():
            return

        migrated = 0
        try:
            with open(_QUEUE_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        log.warning("Remote sync kuyruk satiri okunamadi, atlandi.")
                        continue

                    payload = dict(entry.get("payload") or {})
                    image_raw = entry.get("image_path")
                    image = Path(image_raw) if image_raw else None
                    dedup_key = str(entry.get("dedup_key") or self._entry_key(payload, image))
                    queued_at = float(entry.get("queued_at") or time.time())
                    cursor = conn.execute(
                        """
                        INSERT OR IGNORE INTO outbox_events (
                            id, dedup_key, event_type, payload_json, image_path,
                            status, attempts, last_error, created_at, updated_at, sent_at
                        ) VALUES (?, ?, ?, ?, ?, 'pending', 0, NULL, ?, ?, NULL)
                        """,
                        (
                            str(entry.get("id") or uuid.uuid4()),
                            dedup_key,
                            "remote_sync",
                            json.dumps(payload, ensure_ascii=False),
                            str(image) if image is not None else None,
                            queued_at,
                            queued_at,
                        ),
                    )
                    migrated += int(cursor.rowcount > 0)
            if migrated:
                log.info("Remote sync JSONL kuyrugu SQLite outbox'a tasindi: %s kayit", migrated)
        except Exception as e:
            log.warning("Remote sync JSONL kuyruk migrasyonu atlandi: %s", e)

    def _sorted_queue_entries(self) -> list[dict]:
        entries = self._read_queue()

        def priority(entry: dict) -> tuple[int, float]:
            payload = dict(entry.get("payload") or {})
            image_raw = entry.get("image_path")
            image = Path(image_raw) if image_raw else None
            if self._invalid_image_only_job(payload, image):
                return (0, float(entry.get("queued_at") or 0))
            if image is not None and self._is_transition_payload(payload):
                return (1, float(entry.get("queued_at") or 0))
            if self._is_transition_payload(payload):
                return (2, float(entry.get("queued_at") or 0))
            return (3, float(entry.get("queued_at") or 0))

        return sorted(entries, key=priority)

    @staticmethod
    def _is_transition_payload(payload: dict) -> bool:
        event_type = str(
            payload.get("event_type")
            or payload.get("olay_tipi")
            or payload.get("_event_type")
            or ""
        ).strip().upper()
        return event_type in {"GIRIS", "CIKIS"}

    def _invalid_image_only_job(self, payload: dict, image_path: Optional[Path]) -> bool:
        return image_path is not None and not self._is_transition_payload(payload)

    def _status_only_job(self, payload: dict, image_path: Optional[Path]) -> bool:
        return image_path is None and not self._is_transition_payload(payload)

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
        try:
            self._ensure_outbox()
            with self._outbox_connection() as conn:
                rows = conn.execute(
                    """
                    SELECT id, dedup_key, payload_json, image_path, created_at
                    FROM outbox_events
                    WHERE status = 'pending'
                    ORDER BY created_at ASC
                    """
                ).fetchall()
            entries = []
            for row in rows:
                try:
                    payload = json.loads(row["payload_json"] or "{}")
                except json.JSONDecodeError:
                    payload = {}
                entries.append(
                    {
                        "id": row["id"],
                        "dedup_key": row["dedup_key"],
                        "payload": payload,
                        "image_path": row["image_path"],
                        "queued_at": float(row["created_at"] or 0),
                    }
                )
            return entries
        except Exception as e:
            log.warning("Remote sync kuyruk okunamadi: %s", e)
            return []

    def _read_queue(self) -> list[dict]:
        with self._queue_lock:
            return self._read_queue_unlocked()

    def _write_queue_unlocked(self, entries: list[dict]) -> None:
        self._ensure_outbox()
        now = time.time()
        with self._outbox_connection() as conn:
            conn.execute("DELETE FROM outbox_events WHERE status = 'pending'")
            for entry in entries:
                payload = dict(entry.get("payload") or {})
                image_raw = entry.get("image_path")
                image = Path(image_raw) if image_raw else None
                queued_at = float(entry.get("queued_at") or now)
                conn.execute(
                    """
                    INSERT OR IGNORE INTO outbox_events (
                        id, dedup_key, event_type, payload_json, image_path,
                        status, attempts, last_error, created_at, updated_at, sent_at
                    ) VALUES (?, ?, ?, ?, ?, 'pending', 0, NULL, ?, ?, NULL)
                    """,
                    (
                        str(entry.get("id") or uuid.uuid4()),
                        str(entry.get("dedup_key") or self._entry_key(payload, image)),
                        "remote_sync",
                        json.dumps(payload, ensure_ascii=False),
                        str(image) if image is not None else None,
                        queued_at,
                        now,
                    ),
                )

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
                self._ensure_outbox()
                with self._outbox_connection() as conn:
                    existing = conn.execute(
                        """
                        SELECT id FROM outbox_events
                        WHERE dedup_key = ?
                        LIMIT 1
                        """,
                        (entry_key,),
                    ).fetchone()
                    if existing is not None:
                        existing_id = existing["id"]
                        log.debug(
                            "Remote sync kuyruk duplicate atlandi: %s",
                            existing_id,
                        )
                        return existing_id
                    conn.execute(
                        """
                        INSERT INTO outbox_events (
                            id, dedup_key, event_type, payload_json, image_path,
                            status, attempts, last_error, created_at, updated_at, sent_at
                        ) VALUES (?, ?, ?, ?, ?, 'pending', 0, NULL, ?, ?, NULL)
                        """,
                        (
                            entry["id"],
                            entry["dedup_key"],
                            "remote_sync",
                            json.dumps(payload, ensure_ascii=False),
                            entry["image_path"],
                            entry["queued_at"],
                            entry["queued_at"],
                        ),
                    )
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
            self._ensure_outbox()
            try:
                with self._outbox_connection() as conn:
                    row = conn.execute(
                        """
                        SELECT id FROM outbox_events
                        WHERE dedup_key = ? AND status = 'pending'
                        LIMIT 1
                        """,
                        (entry_key,),
                    ).fetchone()
                    if row is not None:
                        return row["id"]
            except Exception as e:
                log.warning("Remote sync kuyruk id okunamadi: %s", e)
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
            try:
                self._ensure_outbox()
                with self._outbox_connection() as conn:
                    now = time.time()
                    if queue_id is not None:
                        conn.execute(
                            """
                            UPDATE outbox_events
                            SET status = 'sent', sent_at = ?, updated_at = ?
                            WHERE id = ? AND status = 'pending'
                            """,
                            (now, now, queue_id),
                        )
                    else:
                        conn.execute(
                            """
                            UPDATE outbox_events
                            SET status = 'sent', sent_at = ?, updated_at = ?
                            WHERE dedup_key = ? AND status = 'pending'
                            """,
                            (now, now, self._entry_key(payload, image_path)),
                        )
            except Exception as e:
                log.warning("Remote sync kuyruk guncellenemedi: %s", e)

    def _record_queue_failure(
        self,
        queue_id: str,
        status_code: Optional[int],
        error: Optional[str],
    ) -> None:
        try:
            with self._queue_lock:
                self._ensure_outbox()
                message = f"HTTP {status_code}" if status_code is not None else (error or "ag hatasi")
                with self._outbox_connection() as conn:
                    conn.execute(
                        """
                        UPDATE outbox_events
                        SET attempts = attempts + 1, last_error = ?, updated_at = ?
                        WHERE id = ? AND status = 'pending'
                        """,
                        (message[:500], time.time(), queue_id),
                    )
        except Exception as e:
            log.debug("Remote sync kuyruk hata bilgisi yazilamadi: %s", e)
