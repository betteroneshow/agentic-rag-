# -*- coding: utf-8 -*-
"""长期记忆关系数据访问层：SQLite/MySQL 可替换 Repository。"""

from __future__ import annotations

import datetime as dt
import os
import sqlite3
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Iterator


MEMORY_COLUMNS = (
    "id", "text", "type", "importance", "created_at", "last_accessed_at",
    "user_id", "status", "expires_at", "updated_at", "access_count",
    "sync_status", "source_type", "confidence",
)


class MemoryRepository(ABC):
    """长期记忆元数据仓储接口，业务层不感知具体数据库。"""

    @abstractmethod
    def initialize(self) -> None: ...

    @abstractmethod
    def list_not_deleted(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    def list_active_by_type(self, user_id: str, memory_type: str) -> list[dict[str, Any]]: ...

    @abstractmethod
    def insert(self, record: dict[str, Any], *, preserve_id: bool = False) -> int: ...

    @abstractmethod
    def update_duplicate(self, memory_id: int, importance: int, confidence: float, now: dt.datetime) -> None: ...

    @abstractmethod
    def mark_synced(self, memory_ids: list[int]) -> None: ...

    @abstractmethod
    def delete(self, memory_id: int, user_id: str | None = None) -> bool: ...

    @abstractmethod
    def active_count(self, user_id: str, now: dt.datetime) -> int: ...

    @abstractmethod
    def get_active_by_ids(self, ids: list[int], user_id: str, now: dt.datetime) -> dict[int, dict[str, Any]]: ...

    @abstractmethod
    def touch(self, memory_ids: list[int], now: dt.datetime) -> None: ...

    @abstractmethod
    def get(self, memory_id: int, user_id: str) -> dict[str, Any] | None: ...

    @abstractmethod
    def view(self, limit: int, user_id: str, include_inactive: bool, now: dt.datetime) -> list[dict[str, Any]]: ...

    @abstractmethod
    def find_expired_ids(self, now: dt.datetime) -> list[int]: ...

    @abstractmethod
    def mark_expired(self, ids: list[int], now: dt.datetime) -> None: ...

    @abstractmethod
    def upsert_many(self, records: list[dict[str, Any]]) -> int: ...


class SQLiteMemoryRepository(MemoryRepository):
    def __init__(self, path: str = "long_term_memory.sqlite"):
        self.path = path

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _add_column(self, conn: sqlite3.Connection, name: str, declaration: str) -> None:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(memories)")}
        if name not in columns:
            conn.execute(f"ALTER TABLE memories ADD COLUMN {name} {declaration}")

    def initialize(self) -> None:
        with self._connection() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL,
                type TEXT NOT NULL DEFAULT 'fact', importance INTEGER NOT NULL DEFAULT 5,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_accessed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
            additions = {
                "user_id": "TEXT NOT NULL DEFAULT 'default'", "status": "TEXT NOT NULL DEFAULT 'active'",
                "expires_at": "TIMESTAMP", "updated_at": "TIMESTAMP",
                "access_count": "INTEGER NOT NULL DEFAULT 0", "sync_status": "TEXT NOT NULL DEFAULT 'active'",
                "source_type": "TEXT NOT NULL DEFAULT 'inferred'", "confidence": "REAL NOT NULL DEFAULT 0.7",
            }
            for name, declaration in additions.items():
                self._add_column(conn, name, declaration)
            conn.execute("UPDATE memories SET updated_at=COALESCE(updated_at, created_at)")
            conn.execute("UPDATE memories SET user_id='default' WHERE user_id IS NULL OR user_id=''")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_user_status ON memories(user_id,status,sync_status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_expiry ON memories(expires_at)")

    @staticmethod
    def _dicts(rows) -> list[dict[str, Any]]:
        return [dict(row) for row in rows]

    def list_not_deleted(self):
        with self._connection() as conn:
            return self._dicts(conn.execute("SELECT * FROM memories WHERE status!='deleted'").fetchall())

    def list_active_by_type(self, user_id, memory_type):
        with self._connection() as conn:
            return self._dicts(conn.execute(
                "SELECT id,text FROM memories WHERE user_id=? AND type=? AND status='active' AND sync_status='active'",
                (user_id, memory_type)).fetchall())

    def insert(self, record, *, preserve_id=False):
        columns = [c for c in MEMORY_COLUMNS if c != "id" or preserve_id]
        with self._connection() as conn:
            cursor = conn.execute(
                f"INSERT INTO memories ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                tuple(record.get(c) for c in columns))
            return int(record["id"] if preserve_id else cursor.lastrowid)

    def update_duplicate(self, memory_id, importance, confidence, now):
        with self._connection() as conn:
            conn.execute("UPDATE memories SET importance=MAX(importance,?),confidence=MAX(confidence,?),updated_at=? WHERE id=?",
                         (importance, confidence, now, memory_id))

    def mark_synced(self, memory_ids):
        if not memory_ids: return
        with self._connection() as conn:
            conn.executemany("UPDATE memories SET status='active',sync_status='active' WHERE id=?", [(i,) for i in memory_ids])

    def delete(self, memory_id, user_id=None):
        with self._connection() as conn:
            if user_id is None:
                cursor = conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
            else:
                cursor = conn.execute("DELETE FROM memories WHERE id=? AND user_id=?", (memory_id, user_id))
            return cursor.rowcount > 0

    def active_count(self, user_id, now):
        with self._connection() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM memories WHERE user_id=? AND status='active' AND sync_status='active' AND (expires_at IS NULL OR expires_at>?)", (user_id, now)).fetchone()[0])

    def get_active_by_ids(self, ids, user_id, now):
        if not ids: return {}
        placeholders = ",".join("?" for _ in ids)
        with self._connection() as conn:
            rows = conn.execute(f"SELECT * FROM memories WHERE id IN ({placeholders}) AND user_id=? AND status='active' AND sync_status='active' AND (expires_at IS NULL OR expires_at>?)", (*ids, user_id, now)).fetchall()
            return {int(r["id"]): dict(r) for r in rows}

    def touch(self, memory_ids, now):
        with self._connection() as conn:
            conn.executemany("UPDATE memories SET last_accessed_at=?,access_count=access_count+1 WHERE id=?", [(now, i) for i in memory_ids])

    def get(self, memory_id, user_id):
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM memories WHERE id=? AND user_id=?", (memory_id, user_id)).fetchone()
            return dict(row) if row else None

    def view(self, limit, user_id, include_inactive, now):
        with self._connection() as conn:
            if include_inactive:
                rows = conn.execute("SELECT * FROM memories WHERE user_id=? ORDER BY updated_at DESC LIMIT ?", (user_id, limit)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM memories WHERE user_id=? AND status='active' AND sync_status='active' AND (expires_at IS NULL OR expires_at>?) ORDER BY updated_at DESC LIMIT ?", (user_id, now, limit)).fetchall()
            return self._dicts(rows)

    def find_expired_ids(self, now):
        with self._connection() as conn:
            return [int(r[0]) for r in conn.execute("SELECT id FROM memories WHERE status='active' AND expires_at IS NOT NULL AND expires_at<=?", (now,)).fetchall()]

    def mark_expired(self, ids, now):
        with self._connection() as conn:
            conn.executemany("UPDATE memories SET status='expired',sync_status='inactive',updated_at=? WHERE id=?", [(now, i) for i in ids])

    def upsert_many(self, records):
        count = 0
        with self._connection() as conn:
            for record in records:
                columns = list(MEMORY_COLUMNS)
                conn.execute(f"INSERT OR REPLACE INTO memories ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})", tuple(record.get(c) for c in columns))
                count += 1
        return count


class MySQLMemoryRepository(MemoryRepository):
    def __init__(self, **config):
        self.config = config or mysql_config_from_env()

    @contextmanager
    def _connection(self):
        try:
            import pymysql
        except ImportError as exc:
            raise RuntimeError("MySQL 后端需要安装 PyMySQL：pip install PyMySQL") from exc
        conn = pymysql.connect(**self.config, cursorclass=pymysql.cursors.DictCursor, autocommit=False)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self):
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute("""CREATE TABLE IF NOT EXISTS memories (
                id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
                text LONGTEXT NOT NULL, type VARCHAR(32) NOT NULL DEFAULT 'fact',
                importance TINYINT UNSIGNED NOT NULL DEFAULT 5,
                created_at DATETIME(6) NOT NULL, last_accessed_at DATETIME(6) NOT NULL,
                user_id VARCHAR(128) NOT NULL DEFAULT 'default', status VARCHAR(24) NOT NULL DEFAULT 'active',
                expires_at DATETIME(6) NULL, updated_at DATETIME(6) NOT NULL,
                access_count BIGINT UNSIGNED NOT NULL DEFAULT 0,
                sync_status VARCHAR(24) NOT NULL DEFAULT 'active', source_type VARCHAR(32) NOT NULL DEFAULT 'inferred',
                confidence DECIMAL(5,4) NOT NULL DEFAULT 0.7000,
                INDEX idx_memory_user_status (user_id,status,sync_status), INDEX idx_memory_expiry (expires_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci""")

    def _fetchall(self, sql, params=()):
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params); return list(cur.fetchall())

    def _execute(self, sql, params=()):
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(sql, params); return cur.rowcount, cur.lastrowid

    def list_not_deleted(self): return self._fetchall("SELECT * FROM memories WHERE status!='deleted'")
    def list_active_by_type(self, user_id, memory_type): return self._fetchall("SELECT id,text FROM memories WHERE user_id=%s AND type=%s AND status='active' AND sync_status='active'", (user_id,memory_type))
    def insert(self, record, *, preserve_id=False):
        columns = [c for c in MEMORY_COLUMNS if c != "id" or preserve_id]
        _, last_id = self._execute(f"INSERT INTO memories ({','.join(columns)}) VALUES ({','.join('%s' for _ in columns)})", tuple(record.get(c) for c in columns))
        return int(record["id"] if preserve_id else last_id)
    def update_duplicate(self, memory_id, importance, confidence, now): self._execute("UPDATE memories SET importance=GREATEST(importance,%s),confidence=GREATEST(confidence,%s),updated_at=%s WHERE id=%s", (importance,confidence,now,memory_id))
    def mark_synced(self, memory_ids):
        if memory_ids:
            with self._connection() as conn, conn.cursor() as cur: cur.executemany("UPDATE memories SET status='active',sync_status='active' WHERE id=%s", [(i,) for i in memory_ids])
    def delete(self, memory_id, user_id=None):
        sql, params = ("DELETE FROM memories WHERE id=%s", (memory_id,)) if user_id is None else ("DELETE FROM memories WHERE id=%s AND user_id=%s", (memory_id,user_id))
        return self._execute(sql, params)[0] > 0
    def active_count(self, user_id, now): return int(self._fetchall("SELECT COUNT(*) AS n FROM memories WHERE user_id=%s AND status='active' AND sync_status='active' AND (expires_at IS NULL OR expires_at>%s)", (user_id,now))[0]["n"])
    def get_active_by_ids(self, ids, user_id, now):
        if not ids: return {}
        rows = self._fetchall(f"SELECT * FROM memories WHERE id IN ({','.join('%s' for _ in ids)}) AND user_id=%s AND status='active' AND sync_status='active' AND (expires_at IS NULL OR expires_at>%s)", (*ids,user_id,now))
        return {int(r["id"]): r for r in rows}
    def touch(self, memory_ids, now):
        if memory_ids:
            with self._connection() as conn, conn.cursor() as cur: cur.executemany("UPDATE memories SET last_accessed_at=%s,access_count=access_count+1 WHERE id=%s", [(now,i) for i in memory_ids])
    def get(self, memory_id, user_id):
        rows = self._fetchall("SELECT * FROM memories WHERE id=%s AND user_id=%s", (memory_id,user_id)); return rows[0] if rows else None
    def view(self, limit, user_id, include_inactive, now):
        if include_inactive: return self._fetchall("SELECT * FROM memories WHERE user_id=%s ORDER BY updated_at DESC LIMIT %s", (user_id,limit))
        return self._fetchall("SELECT * FROM memories WHERE user_id=%s AND status='active' AND sync_status='active' AND (expires_at IS NULL OR expires_at>%s) ORDER BY updated_at DESC LIMIT %s", (user_id,now,limit))
    def find_expired_ids(self, now): return [int(r["id"]) for r in self._fetchall("SELECT id FROM memories WHERE status='active' AND expires_at IS NOT NULL AND expires_at<=%s", (now,))]
    def mark_expired(self, ids, now):
        if ids:
            with self._connection() as conn, conn.cursor() as cur: cur.executemany("UPDATE memories SET status='expired',sync_status='inactive',updated_at=%s WHERE id=%s", [(now,i) for i in ids])
    def upsert_many(self, records):
        if not records: return 0
        columns = list(MEMORY_COLUMNS)
        updates = ",".join(f"{c}=VALUES({c})" for c in columns if c != "id")
        sql = f"INSERT INTO memories ({','.join(columns)}) VALUES ({','.join('%s' for _ in columns)}) ON DUPLICATE KEY UPDATE {updates}"
        with self._connection() as conn, conn.cursor() as cur:
            cur.executemany(sql, [tuple(r.get(c) for c in columns) for r in records])
        return len(records)


def mysql_config_from_env() -> dict[str, Any]:
    return {
        "host": os.getenv("MEMORY_MYSQL_HOST", "127.0.0.1"),
        "port": int(os.getenv("MEMORY_MYSQL_PORT", "3306")),
        "user": os.getenv("MEMORY_MYSQL_USER", "root"),
        "password": os.getenv("MEMORY_MYSQL_PASSWORD", ""),
        "database": os.getenv("MEMORY_MYSQL_DATABASE", "agentic_rag"),
        "charset": "utf8mb4",
        "connect_timeout": int(os.getenv("MEMORY_MYSQL_CONNECT_TIMEOUT", "10")),
    }


def get_memory_repository(backend: str | None = None) -> MemoryRepository:
    backend = (backend or os.getenv("MEMORY_DB_BACKEND", "sqlite")).strip().lower()
    if backend == "sqlite":
        return SQLiteMemoryRepository(os.getenv("MEMORY_SQLITE_PATH", "long_term_memory.sqlite"))
    if backend == "mysql":
        return MySQLMemoryRepository()
    raise ValueError(f"不支持的 MEMORY_DB_BACKEND: {backend}")
