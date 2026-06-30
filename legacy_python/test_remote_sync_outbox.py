from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from otokantar_app import remote_sync
from otokantar_app.remote_sync import RemoteCanliSync


class RemoteSyncOutboxTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.outbox_path = self.root / "offline_outbox.sqlite"
        self.queue_path = self.root / "sync_queue.jsonl"
        self.patches = [
            patch.object(remote_sync, "_OUTBOX_PATH", self.outbox_path),
            patch.object(remote_sync, "_QUEUE_PATH", self.queue_path),
            patch.object(remote_sync, "_RETRY_DELAYS", ()),
        ]
        for patcher in self.patches:
            patcher.start()

    def tearDown(self) -> None:
        for patcher in reversed(self.patches):
            patcher.stop()
        self.tmp.cleanup()

    def make_sync(self) -> RemoteCanliSync:
        sync = RemoteCanliSync(
            enabled=True,
            url="https://example.test/live-ingest",
            token="secret",
            timeout=0.1,
            min_interval=0.0,
        )
        self.addCleanup(sync.kapat)
        return sync

    def rows(self) -> list[sqlite3.Row]:
        conn = sqlite3.connect(self.outbox_path)
        conn.row_factory = sqlite3.Row
        try:
            return conn.execute(
                """
                SELECT id, dedup_key, payload_json, image_path, status,
                       attempts, last_error, sent_at
                FROM outbox_events
                ORDER BY created_at ASC
                """
            ).fetchall()
        finally:
            conn.close()

    def test_enqueue_persists_once_and_preserves_queue_shape(self) -> None:
        sync = self.make_sync()
        payload = {"event_type": "GIRIS", "event_id": "evt-1", "plaka": "34ABC123"}

        first_id = sync._enqueue(payload, None)
        second_id = sync._enqueue(dict(payload), None)

        self.assertEqual(first_id, second_id)
        entries = sync._read_queue()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["id"], first_id)
        self.assertEqual(entries[0]["payload"], payload)
        self.assertEqual(entries[0]["image_path"], None)

        rows = self.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "pending")
        self.assertEqual(json.loads(rows[0]["payload_json"]), payload)

    def test_existing_jsonl_queue_is_migrated_to_sqlite_outbox(self) -> None:
        payload = {"event_type": "CIKIS", "event_id": "evt-old", "plaka": "06XYZ06"}
        self.queue_path.write_text(
            json.dumps(
                {
                    "id": "legacy-id",
                    "dedup_key": json.dumps(
                        {"payload": payload, "image_path": None},
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "payload": payload,
                    "image_path": None,
                    "queued_at": 123.0,
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        sync = self.make_sync()

        entries = sync._read_queue()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["id"], "legacy-id")
        self.assertEqual(entries[0]["payload"], payload)
        self.assertEqual(self.rows()[0]["status"], "pending")

    def test_retry_failure_updates_attempt_metadata_and_success_marks_sent(self) -> None:
        sync = self.make_sync()
        payload = {"event_type": "GIRIS", "event_id": "evt-retry", "plaka": "35TST35"}
        queue_id = sync._enqueue(payload, None)
        self.assertIsNotNone(queue_id)

        sync._attempt_post = lambda _payload, _image: (False, None, "internet yok")
        self.assertFalse(sync._send_with_retry(payload, None, queue_id=queue_id))

        rows = self.rows()
        self.assertEqual(rows[0]["status"], "pending")
        self.assertEqual(rows[0]["attempts"], 1)
        self.assertEqual(rows[0]["last_error"], "internet yok")
        self.assertEqual(len(sync._read_queue()), 1)

        sync._attempt_post = lambda _payload, _image: (True, 200, None)
        self.assertTrue(sync._send_with_retry(payload, None, queue_id=queue_id))

        rows = self.rows()
        self.assertEqual(rows[0]["status"], "sent")
        self.assertIsNotNone(rows[0]["sent_at"])
        self.assertEqual(sync._read_queue(), [])


if __name__ == "__main__":
    unittest.main()
