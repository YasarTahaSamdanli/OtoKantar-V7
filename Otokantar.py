"""
OtoKantar V7 — Uyumluluk giriş noktası
=======================================
Bu dosya yalnızca geriye dönük uyumluluk için mevcuttur.
Asıl uygulama ``otokantar_app/main.py`` içindedir.

Doğrudan başlatmak için:
    python -m otokantar_app.main
veya:
    python Otokantar.py
"""
import multiprocessing

from otokantar_app.main import OtoKantar

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sistem = OtoKantar()
    sistem.calistir()
