from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

from otokantar_app.logger import log

try:
    import mysql.connector  # type: ignore
    from mysql.connector import errorcode  # type: ignore
except Exception as e:  # pragma: no cover
    mysql = None  # type: ignore
    mysql_connector_import_error = e


@dataclass(frozen=True)
class MySQLConfig:
    host: str = "127.0.0.1"
    port: int = 3306
    user: str = "root"
    password: str = ""
    database: str = "otokantar"
    connect_timeout: int = 5


class MySQLDBManager:
    """
    MySQL schema + plate logging manager.

    Tables:
      - araclar(id, plaka, kara_liste, ilk_kayit)
      - gecisler(id, yon, gecis_zamani, guven)  # id -> araclar.id
    """

    _RETRY_DELAYS = (0.2, 0.5, 1.0, 2.0)

    def __init__(self, cfg: MySQLConfig) -> None:
        if mysql is None:  # pragma: no cover
            raise RuntimeError(
                "mysql-connector-python kurulu değil. "
                "Kurulum: pip install mysql-connector-python. "
                f"Import hatası: {mysql_connector_import_error}"
            )

        self.cfg = cfg
        self._cnx = None
        self._ensure_database_and_schema()

    @classmethod
    def from_config(cls, config_dict: dict) -> "MySQLDBManager":
        cfg = MySQLConfig(
            host=str(config_dict.get("MYSQL_HOST", "127.0.0.1")),
            port=int(config_dict.get("MYSQL_PORT", 3306)),
            user=str(config_dict.get("MYSQL_USER", "root")),
            password=str(config_dict.get("MYSQL_PASS", "")),
            database=str(config_dict.get("MYSQL_DB", "otokantar")),
            connect_timeout=int(config_dict.get("MYSQL_CONNECT_TIMEOUT", 5)),
        )
        return cls(cfg)

    # ---------------------------------------------------------------------
    # Connection management
    # ---------------------------------------------------------------------
    def _connect(self, with_db: bool) -> None:
        kwargs = dict(
            host=self.cfg.host,
            port=self.cfg.port,
            user=self.cfg.user,
            password=self.cfg.password,
            connection_timeout=self.cfg.connect_timeout,
            autocommit=False,
        )
        if with_db:
            kwargs["database"] = self.cfg.database
        self._cnx = mysql.connector.connect(**kwargs)

    def _close(self) -> None:
        try:
            if self._cnx is not None:
                self._cnx.close()
        except Exception:
            pass
        self._cnx = None

    def _ensure_connection(self) -> None:
        """
        Ensure a healthy connection (reconnect on lost/timeout).
        """
        if self._cnx is None:
            self._connect(with_db=True)
            return
        try:
            self._cnx.ping(reconnect=True, attempts=1, delay=0)
        except Exception:
            self._close()
            self._connect(with_db=True)

    def _run_with_retry(self, fn, *args, **kwargs):
        last_err = None
        for i, delay in enumerate((0.0,) + self._RETRY_DELAYS):
            if delay:
                time.sleep(delay)
            try:
                self._ensure_connection()
                return fn(*args, **kwargs)
            except Exception as e:
                last_err = e
                self._close()
                if i >= len(self._RETRY_DELAYS):
                    break
        raise last_err  # type: ignore[misc]

    # ---------------------------------------------------------------------
    # Schema
    # ---------------------------------------------------------------------
    def _ensure_database_and_schema(self) -> None:
        # 1) Create database if missing (connect without DB)
        for delay in (0.0,) + self._RETRY_DELAYS:
            if delay:
                time.sleep(delay)
            try:
                self._close()
                self._connect(with_db=False)
                cur = self._cnx.cursor()
                try:
                    cur.execute(
                        f"CREATE DATABASE IF NOT EXISTS `{self.cfg.database}` "
                        "CHARACTER SET utf8mb4 COLLATE utf8mb4_turkish_ci"
                    )
                    self._cnx.commit()
                finally:
                    cur.close()
                break
            except Exception as e:
                last = e
                continue
        else:  # pragma: no cover
            raise last  # type: ignore[misc]

        # 2) Connect with DB and create tables
        self._close()
        self._connect(with_db=True)
        self.init_schema()

    def init_schema(self) -> None:
        """
        Create required tables (InnoDB) if absent.
        """
        self._ensure_connection()
        cur = self._cnx.cursor()
        try:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS araclar (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    plaka VARCHAR(20) NOT NULL UNIQUE,
                    kara_liste BOOLEAN NOT NULL DEFAULT FALSE,
                    ilk_kayit TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS gecisler (
                    id INT NOT NULL,
                    yon VARCHAR(10) NOT NULL,
                    gecis_zamani TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    guven FLOAT,
                    CONSTRAINT fk_gecis_arac
                      FOREIGN KEY (id) REFERENCES araclar(id)
                      ON DELETE RESTRICT ON UPDATE CASCADE,
                    PRIMARY KEY (id, gecis_zamani, yon),
                    INDEX idx_gecis_arac_zaman (id, gecis_zamani),
                    INDEX idx_gecis_yon (yon)
                ) ENGINE=InnoDB
                """
            )
            self._migrate_gecisler_if_has_legacy_id(cur)
            self._migrate_gecisler_arac_id_to_id(cur)
            self._cnx.commit()
        finally:
            cur.close()

    def _migrate_gecisler_if_has_legacy_id(self, cur) -> None:
        """
        Legacy migration:
        If `gecisler` has AUTO_INCREMENT `id`, recreate table to shared-id model.
        """
        cur.execute(
            """
            SELECT EXTRA
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = %s
              AND TABLE_NAME = 'gecisler'
              AND COLUMN_NAME = 'id'
            LIMIT 1
            """,
            (self.cfg.database,),
        )
        row = cur.fetchone()
        if not row:
            return
        extra = str(row[0] or "").lower()
        if "auto_increment" not in extra:
            return

        log.warning("MySQL migrasyon: legacy gecisler.id (auto_increment) ortak id modeline dönüştürülüyor.")
        cur.execute("DROP TABLE IF EXISTS gecisler_yeni")
        cur.execute(
            """
            CREATE TABLE gecisler_yeni (
                id INT NOT NULL,
                yon VARCHAR(10) NOT NULL,
                gecis_zamani TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                guven FLOAT,
                CONSTRAINT fk_gecis_arac_yeni
                  FOREIGN KEY (id) REFERENCES araclar(id)
                  ON DELETE RESTRICT ON UPDATE CASCADE,
                PRIMARY KEY (id, gecis_zamani, yon),
                INDEX idx_gecis_arac_zaman (id, gecis_zamani),
                INDEX idx_gecis_yon (yon)
            ) ENGINE=InnoDB
            """
        )
        cur.execute(
            """
            INSERT INTO gecisler_yeni (id, yon, gecis_zamani, guven)
            SELECT arac_id, yon, gecis_zamani, MAX(guven) AS guven
            FROM gecisler
            GROUP BY arac_id, yon, gecis_zamani
            ORDER BY arac_id, gecis_zamani, yon
            """
        )
        cur.execute("RENAME TABLE gecisler TO gecisler_eski, gecisler_yeni TO gecisler")
        cur.execute("DROP TABLE gecisler_eski")

    def _migrate_gecisler_arac_id_to_id(self, cur) -> None:
        """
        Rename legacy shared FK column from `arac_id` to `id`.
        """
        cur.execute(
            """
            SELECT COUNT(*)
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = %s
              AND TABLE_NAME = 'gecisler'
              AND COLUMN_NAME = 'arac_id'
            """,
            (self.cfg.database,),
        )
        has_arac_id = int(cur.fetchone()[0]) > 0
        if not has_arac_id:
            return

        log.warning("MySQL migrasyon: gecisler.arac_id -> gecisler.id (ortak ID).")
        cur.execute("DROP TABLE IF EXISTS gecisler_yeni")
        cur.execute(
            """
            CREATE TABLE gecisler_yeni (
                id INT NOT NULL,
                yon VARCHAR(10) NOT NULL,
                gecis_zamani TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                guven FLOAT,
                CONSTRAINT fk_gecis_arac_yeni2
                  FOREIGN KEY (id) REFERENCES araclar(id)
                  ON DELETE RESTRICT ON UPDATE CASCADE,
                PRIMARY KEY (id, gecis_zamani, yon),
                INDEX idx_gecis_arac_zaman (id, gecis_zamani),
                INDEX idx_gecis_yon (yon)
            ) ENGINE=InnoDB
            """
        )
        cur.execute(
            """
            INSERT INTO gecisler_yeni (id, yon, gecis_zamani, guven)
            SELECT arac_id, yon, gecis_zamani, MAX(guven) AS guven
            FROM gecisler
            GROUP BY arac_id, yon, gecis_zamani
            ORDER BY arac_id, gecis_zamani, yon
            """
        )
        cur.execute("RENAME TABLE gecisler TO gecisler_eski2, gecisler_yeni TO gecisler")
        cur.execute("DROP TABLE gecisler_eski2")

    # ---------------------------------------------------------------------
    # Business logic (requested flow)
    # ---------------------------------------------------------------------
    def upsert_arac(self, plaka: str) -> Tuple[int, bool]:
        """
        Insert plate into `araclar` if missing, and return (id, kara_liste).
        Uses ON DUPLICATE KEY UPDATE to avoid new IDs.
        """

        def _op() -> Tuple[int, bool]:
            assert self._cnx is not None
            cur = self._cnx.cursor()
            try:
                # Upsert (no-op update ensures LAST_INSERT_ID trick works)
                cur.execute(
                    """
                    INSERT INTO araclar (plaka)
                    VALUES (%s)
                    ON DUPLICATE KEY UPDATE id = LAST_INSERT_ID(id)
                    """,
                    (plaka,),
                )
                arac_id = int(cur.lastrowid)
                cur.execute("SELECT kara_liste FROM araclar WHERE id=%s LIMIT 1", (arac_id,))
                row = cur.fetchone()
                kara_liste = bool(row[0]) if row else False
                self._cnx.commit()
                return arac_id, kara_liste
            finally:
                cur.close()

        return self._run_with_retry(_op)

    def gecis_ekle(self, arac_id: int, yon: str, guven: Optional[float]) -> None:
        def _op() -> None:
            assert self._cnx is not None
            cur = self._cnx.cursor()
            try:
                cur.execute(
                    """
                    INSERT INTO gecisler (id, yon, guven)
                    VALUES (%s, %s, %s)
                    """,
                    (int(arac_id), str(yon), None if guven is None else float(guven)),
                )
                self._cnx.commit()
                return None
            finally:
                cur.close()

        return self._run_with_retry(_op)

    def plaka_okundu(self, plaka: str, yon: str, guven: Optional[float] = None) -> dict:
        """
        Requested sequence:
          1) upsert araclar (get id + kara_liste)
          2) warn if kara_liste
          3) insert gecisler row
        Returns a dict for logging/diagnostics.
        """
        plaka = (plaka or "").strip().upper()
        yon = (yon or "").strip().upper()

        arac_id, kara_liste = self.upsert_arac(plaka)
        if kara_liste:
            log.warning("ALARM (MySQL): kara listede plaka okundu → %s", plaka)
        self.gecis_ekle(arac_id, yon, guven)
        return {
            "arac_id": arac_id,
            "kara_liste": kara_liste,
            "plaka": plaka,
            "yon": yon,
            "guven": None if guven is None else float(guven),
        }

    def kara_listede_mi(self, plaka: str) -> bool:
        def _op() -> bool:
            assert self._cnx is not None
            cur = self._cnx.cursor()
            try:
                cur.execute(
                    "SELECT kara_liste FROM araclar WHERE plaka=%s LIMIT 1",
                    ((plaka or "").strip().upper(),),
                )
                row = cur.fetchone()
                return bool(row[0]) if row else False
            finally:
                cur.close()

        return self._run_with_retry(_op)

    def kara_liste_guncelle(self, plaka: str, kara_liste: bool) -> dict:
        plaka = (plaka or "").strip().upper()
        arac_id, _ = self.upsert_arac(plaka)

        def _op() -> None:
            assert self._cnx is not None
            cur = self._cnx.cursor()
            try:
                cur.execute(
                    "UPDATE araclar SET kara_liste=%s WHERE id=%s",
                    (1 if kara_liste else 0, int(arac_id)),
                )
                self._cnx.commit()
            finally:
                cur.close()

        self._run_with_retry(_op)
        return {"arac_id": arac_id, "plaka": plaka, "kara_liste": bool(kara_liste)}

    def kara_liste_listele(self) -> list[str]:
        def _op() -> list[str]:
            assert self._cnx is not None
            cur = self._cnx.cursor()
            try:
                cur.execute("SELECT plaka FROM araclar WHERE kara_liste=1 ORDER BY id DESC")
                return [str(r[0]) for r in cur.fetchall()]
            finally:
                cur.close()

        return self._run_with_retry(_op)

    def tum_plakalar(self) -> list[str]:
        def _op() -> list[str]:
            assert self._cnx is not None
            cur = self._cnx.cursor()
            try:
                cur.execute("SELECT plaka FROM araclar")
                return [str(r[0]) for r in cur.fetchall()]
            finally:
                cur.close()

        return self._run_with_retry(_op)

    def son_gecisler(self, limit: int = 50) -> list[dict]:
        def _op() -> list[dict]:
            assert self._cnx is not None
            cur = self._cnx.cursor(dictionary=True)
            try:
                cur.execute(
                    """
                    SELECT g.id AS arac_id, a.plaka, a.kara_liste, g.yon, g.gecis_zamani, g.guven
                    FROM gecisler g
                    JOIN araclar a ON a.id = g.id
                    ORDER BY g.gecis_zamani DESC
                    LIMIT %s
                    """,
                    (int(limit),),
                )
                return list(cur.fetchall())
            finally:
                cur.close()

        return self._run_with_retry(_op)

