from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from otokantar_app.logger import log


class DeviceStatusStore:
    DB_PATH = "device_status.sqlite"

    @classmethod
    def _connect(cls) -> sqlite3.Connection:
        path = Path(cls.DB_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    @classmethod
    @contextmanager
    def _connection(cls):
        conn = cls._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS device_status (
                    device_key TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    last_error TEXT,
                    last_seen TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_device_status_level
                ON device_status (level, updated_at)
                """
            )
            yield conn
            conn.commit()
        finally:
            conn.close()

    @classmethod
    def update(
        cls,
        device_key: str,
        status: str,
        level: str,
        message: str,
        payload: Optional[dict] = None,
        last_error: Optional[str] = None,
    ) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        payload_json = json.dumps(payload or {}, ensure_ascii=False, sort_keys=True)
        try:
            with cls._connection() as conn:
                conn.execute(
                    """
                    INSERT INTO device_status (
                        device_key, status, level, message, last_error,
                        last_seen, updated_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(device_key) DO UPDATE SET
                        status = excluded.status,
                        level = excluded.level,
                        message = excluded.message,
                        last_error = excluded.last_error,
                        last_seen = excluded.last_seen,
                        updated_at = excluded.updated_at,
                        payload_json = excluded.payload_json
                    """,
                    (
                        device_key,
                        status,
                        level,
                        message,
                        last_error,
                        now,
                        now,
                        payload_json,
                    ),
                )
        except Exception as e:
            log.warning("Cihaz durum veritabanı yazılamadı (%s): %s", device_key, e)

    @classmethod
    def read(cls, device_key: str) -> dict:
        path = Path(cls.DB_PATH)
        if not path.is_file():
            return {}

        try:
            with cls._connection() as conn:
                row = conn.execute(
                    """
                    SELECT device_key, status, level, message, last_error,
                           last_seen, updated_at, payload_json
                    FROM device_status
                    WHERE device_key = ?
                    """,
                    (device_key,),
                ).fetchone()
            if row is None:
                return {}
            try:
                payload = json.loads(row["payload_json"] or "{}")
            except json.JSONDecodeError:
                payload = {}
            return {
                "device_key": row["device_key"],
                "status": row["status"],
                "level": row["level"],
                "message": row["message"],
                "last_error": row["last_error"],
                "last_seen": row["last_seen"],
                "updated_at": row["updated_at"],
                "payload": payload,
            }
        except Exception as e:
            log.warning("Cihaz durum veritabanı okunamadı (%s): %s", device_key, e)
            return {}
