import atexit
import os
import subprocess
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = 8080
OTOKANTAR_SCRIPT = ROOT / "Otokantar.py"

_proc_lock = threading.Lock()
_proc: subprocess.Popen | None = None


def _is_running() -> bool:
    return _proc is not None and _proc.poll() is None


def ensure_otokantar_started() -> bool:
    global _proc
    with _proc_lock:
        if _is_running():
            return True
        if not OTOKANTAR_SCRIPT.exists():
            return False
        _proc = subprocess.Popen(
            ["python", str(OTOKANTAR_SCRIPT)],
            cwd=str(ROOT),
            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0,
        )
        return True


def stop_otokantar():
    global _proc
    with _proc_lock:
        if _is_running():
            _proc.terminate()
        _proc = None


atexit.register(stop_otokantar)


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def _json_response(self, status_code: int, payload: str):
        body = payload.encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/ensure-otokantar":
            ok = ensure_otokantar_started()
            running = _is_running()
            self._json_response(
                200 if ok else 500,
                '{"ok": %s, "running": %s}' % ("true" if ok else "false", "true" if running else "false"),
            )
            return
        if path == "/api/otokantar-status":
            running = _is_running()
            self._json_response(200, '{"running": %s}' % ("true" if running else "false"))
            return
        super().do_GET()


def main():
    httpd = ThreadingHTTPServer((HOST, PORT), DashboardHandler)
    print(f"Dashboard server: http://{HOST}:{PORT}/dashboard.html")
    print("Istek geldiginde Otokantar.py otomatik baslatilacak.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
