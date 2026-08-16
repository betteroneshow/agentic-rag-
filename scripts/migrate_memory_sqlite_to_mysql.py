# -*- coding: utf-8 -*-
"""将 SQLite 长期记忆元数据幂等迁移至 MySQL，保留原始 ID。"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from agentic_rag.memory_repository import MySQLMemoryRepository, SQLiteMemoryRepository


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description="SQLite 长期记忆迁移至 MySQL")
    parser.add_argument("--sqlite-path", default=os.getenv("MEMORY_SQLITE_PATH", "long_term_memory.sqlite"))
    parser.add_argument("--dry-run", action="store_true", help="只读取并统计，不写入 MySQL")
    args = parser.parse_args()

    source = SQLiteMemoryRepository(args.sqlite_path)
    source.initialize()
    records = source.list_not_deleted()
    print(f"SQLite: {Path(args.sqlite_path).resolve()}")
    print(f"待迁移记录: {len(records)}")
    if args.dry_run:
        print("dry-run 完成，未连接或写入 MySQL。")
        return

    target = MySQLMemoryRepository()
    target.initialize()
    migrated = target.upsert_many(records)
    target_count = len(target.list_not_deleted())
    print(f"本次迁移/更新: {migrated}")
    print(f"MySQL 当前记录: {target_count}")
    if target_count < len(records):
        raise RuntimeError("迁移校验失败：MySQL 记录数少于 SQLite 源记录数")
    print("迁移完成。验证后将 MEMORY_DB_BACKEND 改为 mysql。")


if __name__ == "__main__":
    main()
