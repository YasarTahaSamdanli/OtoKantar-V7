"""
OtoKantar giris noktasi.

Bu dosya yalnizca Python arka plan surecini baslatir.
Web sunucusu acmaz; canli durum verisini ortak klasordeki
JSON / CSV / JPG dosyalari uzerinden uretir.
"""

import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)

from otokantar_app.main import OtoKantar


def main() -> None:
    sistem = OtoKantar()
    sistem.calistir()


if __name__ == "__main__":
    main()
