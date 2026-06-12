"""
OtoKantar giris noktasi.

Bu dosya yalnizca Python arka plan surecini baslatir.
Web sunucusu acmaz; canli durum verisini ortak klasordeki
JSON / CSV / JPG dosyalari uzerinden uretir.
"""

import os
from pathlib import Path

# Paddle (Windows CPU) oneDNN/PIR — import öncesi ayarlanmalı
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_onednn", "0")

os.chdir(Path(__file__).resolve().parent)

from otokantar_app.main import OtoKantar


def main() -> None:
    sistem = OtoKantar()
    sistem.calistir()


if __name__ == "__main__":
    main()
