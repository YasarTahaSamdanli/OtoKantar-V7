from __future__ import annotations

import os
import json
import sqlite3
import sys
import tempfile
import unittest
import zipfile
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from otokantar_app.donanim import yazici
from otokantar_app.donanim.yazici import FisYazdirici
from otokantar_app.models import PlakaKayit


def sample_kayit() -> PlakaKayit:
    return PlakaKayit(
        plaka="34ABC123",
        giris_tarih="2026-06-30",
        giris_saat="15:30:00",
        giris_agirlik=12000.0,
        guven=0.91,
        cikis_tarih="2026-06-30",
        cikis_saat="16:00:00",
        cikis_agirlik=42000.0,
        net_agirlik=30000.0,
        durum="TAMAMLANDI",
        operator="AUTO",
    )


class FakeWin32Print:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.written = b""

    def GetDefaultPrinter(self) -> str:
        self.calls.append(("GetDefaultPrinter",))
        return "Default Receipt Printer"

    def OpenPrinter(self, name: str) -> str:
        self.calls.append(("OpenPrinter", name))
        return "printer-handle"

    def StartDocPrinter(self, handle: str, level: int, doc_info: tuple) -> None:
        self.calls.append(("StartDocPrinter", handle, level, doc_info))

    def StartPagePrinter(self, handle: str) -> None:
        self.calls.append(("StartPagePrinter", handle))

    def WritePrinter(self, handle: str, data: bytes) -> None:
        self.calls.append(("WritePrinter", handle, data))
        self.written += data

    def EndPagePrinter(self, handle: str) -> None:
        self.calls.append(("EndPagePrinter", handle))

    def EndDocPrinter(self, handle: str) -> None:
        self.calls.append(("EndDocPrinter", handle))

    def ClosePrinter(self, handle: str) -> None:
        self.calls.append(("ClosePrinter", handle))


class FisYazdiriciTest(unittest.TestCase):
    def print_log_rows(self, db_path: Path) -> list[sqlite3.Row]:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            return conn.execute(
                """
                SELECT receipt_id, plate, backend, printer, status,
                       reason, printed_at, created_at
                FROM receipt_print_log
                ORDER BY id ASC
                """
            ).fetchall()
        finally:
            conn.close()

    def device_status_row(self, db_path: Path, device_key: str = "printer") -> sqlite3.Row:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                """
                SELECT device_key, status, level, message, last_error, payload_json
                FROM device_status
                WHERE device_key = ?
                """,
                (device_key,),
            ).fetchone()
            self.assertIsNotNone(row)
            return row
        finally:
            conn.close()

    def test_yazdir_writes_receipt_file_and_triggers_win32_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                printer = FisYazdirici()
                with patch.dict(yazici.CONFIG, {"YAZICI_BACKEND": "win32", "YAZICI_ADI": ""}):
                    with patch.object(yazici, "_WIN32PRINT_OK", True):
                        with patch.object(printer, "_win32_gonder_async") as send:
                            printer.yazdir(sample_kayit())

                receipt = Path(tmp) / "kantar_fisi.txt"
                self.assertTrue(receipt.is_file())
                content = receipt.read_text(encoding="utf-8")
                self.assertIn("34ABC123", content)
                self.assertIn("Net Kg   : 30000.0 kg", content)
                archive_files = list((Path(tmp) / "fisler" / "2026" / "06").glob("*.txt"))
                self.assertEqual(len(archive_files), 1)
                archive_content = archive_files[0].read_text(encoding="utf-8")
                self.assertIn("34ABC123", archive_content)
                self.assertIn("Net Kg   : 30000.0 kg", archive_content)
                self.assertIn("2026-06-30_15-30-00_34ABC123_TAMAMLANDI", archive_files[0].stem)
                send.assert_called_once()
                self.assertEqual(send.call_args.args[1], archive_files[0].stem)
            finally:
                os.chdir(old_cwd)

    def test_file_backend_records_saved_print_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                with patch.dict(yazici.CONFIG, {"YAZICI_BACKEND": "file"}):
                    FisYazdirici().yazdir(sample_kayit())

                rows = self.print_log_rows(Path("receipt_log.sqlite"))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["plate"], "34ABC123")
                self.assertEqual(rows[0]["backend"], "file")
                self.assertEqual(rows[0]["printer"], "file")
                self.assertEqual(rows[0]["status"], "SAVED")
                self.assertIsNone(rows[0]["printed_at"])
                status = json.loads(Path("printer_status.json").read_text(encoding="utf-8"))
                self.assertEqual(status["status"], "SAVED")
                self.assertEqual(status["level"], "info")
                self.assertEqual(status["plate"], "34ABC123")
                device_status = self.device_status_row(Path("device_status.sqlite"))
                self.assertEqual(device_status["device_key"], "printer")
                self.assertEqual(device_status["status"], "OK")
                self.assertEqual(device_status["level"], "info")
                self.assertIsNone(device_status["last_error"])
                device_payload = json.loads(device_status["payload_json"])
                self.assertEqual(device_payload["status"], "SAVED")
                self.assertEqual(device_payload["plate"], "34ABC123")
            finally:
                os.chdir(old_cwd)

    def test_yazdir_skips_printer_when_receipt_was_already_successful(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                kayit = sample_kayit()
                printer = FisYazdirici()
                receipt_id = printer._receipt_id(printer._fis_metin_olustur(kayit))
                printer._print_log_yaz(receipt_id, kayit, "win32", "Epson TM-T20III", "SUCCESS")

                with patch.dict(yazici.CONFIG, {"YAZICI_BACKEND": "win32", "YAZICI_ADI": "Epson TM-T20III"}):
                    with patch.object(yazici, "_WIN32PRINT_OK", True):
                        with patch.object(printer, "_win32_gonder_async") as send:
                            printer.yazdir(kayit)

                send.assert_not_called()
                rows = self.print_log_rows(Path("receipt_log.sqlite"))
                self.assertEqual([row["status"] for row in rows], ["SUCCESS", "SKIPPED_DUPLICATE"])
                self.assertEqual(rows[1]["receipt_id"], receipt_id)
                self.assertEqual(rows[1]["reason"], "Fiş daha önce başarıyla yazdırıldı.")
                status = FisYazdirici.yazici_durum_oku()
                self.assertEqual(status["status"], "SKIPPED_DUPLICATE")
                self.assertEqual(status["level"], "warning")
            finally:
                os.chdir(old_cwd)

    def test_old_month_receipts_are_zipped_and_source_month_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                old_month = Path("fisler") / "2026" / "04"
                old_month.mkdir(parents=True)
                (old_month / "old-receipt.txt").write_text("old receipt", encoding="utf-8")

                FisYazdirici()._eski_aylari_ziple(
                    today=date(2026, 6, 30),
                    older_than_days=45,
                )

                zip_path = Path("fisler_zip") / "2026-04.zip"
                self.assertTrue(zip_path.is_file())
                self.assertFalse(old_month.exists())
                with zipfile.ZipFile(zip_path) as zf:
                    self.assertEqual(zf.read("old-receipt.txt").decode("utf-8"), "old receipt")
            finally:
                os.chdir(old_cwd)

    def test_win32_sender_uses_default_printer_when_name_is_empty(self) -> None:
        fake = FakeWin32Print()
        with patch.dict(yazici.CONFIG, {"YAZICI_ADI": ""}):
            with patch.object(yazici, "win32print", fake, create=True):
                printer = FisYazdirici()._win32_gonder(sample_kayit())

        self.assertEqual(printer, "Default Receipt Printer")
        self.assertIn(("GetDefaultPrinter",), fake.calls)
        self.assertIn(("OpenPrinter", "Default Receipt Printer"), fake.calls)
        self.assertTrue(fake.written.startswith(FisYazdirici._ESC_INIT))
        self.assertIn(b"34ABC123", fake.written)

    def test_win32_sender_uses_configured_printer_name(self) -> None:
        fake = FakeWin32Print()
        with patch.dict(yazici.CONFIG, {"YAZICI_ADI": "USB Receipt Cutter"}):
            with patch.object(yazici, "win32print", fake, create=True):
                printer = FisYazdirici()._win32_gonder(sample_kayit())

        self.assertEqual(printer, "USB Receipt Cutter")
        self.assertNotIn(("GetDefaultPrinter",), fake.calls)
        self.assertIn(("OpenPrinter", "USB Receipt Cutter"), fake.calls)

    def test_win32_async_records_failed_print_log_without_raising(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                printer = FisYazdirici()
                with patch.dict(yazici.CONFIG, {"YAZICI_ADI": "Offline Printer"}):
                    with patch.object(printer, "_win32_gonder", side_effect=OSError("Printer Offline")):
                        printer._win32_gonder_async(sample_kayit(), "receipt-1")
                        for thread in list(yazici.threading.enumerate()):
                            if thread.name == "FisGonderici-win32":
                                thread.join(2)

                rows = self.print_log_rows(Path("receipt_log.sqlite"))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["receipt_id"], "receipt-1")
                self.assertEqual(rows[0]["plate"], "34ABC123")
                self.assertEqual(rows[0]["backend"], "win32")
                self.assertEqual(rows[0]["printer"], "Offline Printer")
                self.assertEqual(rows[0]["status"], "FAILED")
                self.assertIn("Printer Offline", rows[0]["reason"])
                self.assertIsNone(rows[0]["printed_at"])
                status = json.loads(Path("printer_status.json").read_text(encoding="utf-8"))
                self.assertEqual(status["status"], "FAILED")
                self.assertEqual(status["level"], "error")
                self.assertEqual(status["printer"], "Offline Printer")
                self.assertIn("Printer Offline", status["reason"])
                device_status = self.device_status_row(Path("device_status.sqlite"))
                self.assertEqual(device_status["status"], "FAILED")
                self.assertEqual(device_status["level"], "error")
                self.assertIn("Printer Offline", device_status["last_error"])
            finally:
                os.chdir(old_cwd)

    def test_print_log_records_success_with_printed_at(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                kayit = sample_kayit()
                printer = FisYazdirici()
                printer._print_log_yaz("receipt-ok", kayit, "win32", "Epson TM-T20III", "SUCCESS")

                rows = self.print_log_rows(Path("receipt_log.sqlite"))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["status"], "SUCCESS")
                self.assertEqual(rows[0]["printer"], "Epson TM-T20III")
                self.assertEqual(rows[0]["plate"], "34ABC123")
                self.assertIsNotNone(rows[0]["printed_at"])
                status = FisYazdirici.yazici_durum_oku()
                self.assertEqual(status["status"], "SUCCESS")
                self.assertEqual(status["level"], "ok")
                self.assertEqual(status["printer"], "Epson TM-T20III")
                device_status = self.device_status_row(Path("device_status.sqlite"))
                self.assertEqual(device_status["status"], "OK")
                self.assertEqual(device_status["level"], "ok")
                self.assertEqual(json.loads(device_status["payload_json"])["printer"], "Epson TM-T20III")
            finally:
                os.chdir(old_cwd)

    def test_printer_status_reader_returns_unknown_when_file_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                status = FisYazdirici.yazici_durum_oku()
                self.assertEqual(status["status"], "UNKNOWN")
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
