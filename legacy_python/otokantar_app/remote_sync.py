from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from otokantar_app.logger import log

try:
    import requests
except Exception as e:  # pragma: no cover
    requests = None
    requests_import_error = e


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

        if enabled and requests is None:
            self.enabled = False
            log.warning("Remote sync devre disi: requests kurulu degil (%s)", requests_import_error)
        elif self.enabled:
            log.info("Remote sync aktif: %s", self.url)
        else:
            log.info("Remote sync kapali.")

    def gonder(self, payload: Optional[dict] = None, image_path: Optional[str | Path] = None, zorla: bool = False) -> None:
        if not self.enabled:
            return

        simdi = time.monotonic()
        if not zorla and (simdi - self._last_submit) < self.min_interval:
            return
        if self._inflight is not None and not self._inflight.done():
            return
        self._last_submit = simdi

        payload_copy = dict(payload or {})
        image = Path(image_path) if image_path else None
        self._inflight = self._executor.submit(self._gonder_sync, payload_copy, image)

    def kapat(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _gonder_sync(self, payload: dict, image_path: Optional[Path]) -> None:
        assert requests is not None

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
        }
        data = {}
        files = {}

        if payload:
            data["json"] = json.dumps(payload, ensure_ascii=False)

        try:
            image_handle = None
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
        except Exception as e:
            log.warning("Remote sync gonderilemedi: %s", e)
        finally:
            if image_handle is not None:
                image_handle.close()
