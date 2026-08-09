"""SQLite persistence for analysis results and history viewing."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

DEFAULT_DB_PATH = Path("output") / "form_analyzer.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    upload_datetime TEXT NOT NULL,
    status TEXT NOT NULL,
    score REAL NOT NULL,
    ocr_text TEXT,
    missing_items TEXT,
    warnings TEXT,
    result_json TEXT NOT NULL,
    processing_duration_seconds REAL
);
"""


class FormDatabase:
    """Thin wrapper around a SQLite database storing analysis history."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(SCHEMA)

    def insert_result(
        self,
        filename: str,
        status: str,
        score: float,
        ocr_text: str,
        missing_items: List[str],
        warnings: List[str],
        result_dict: Dict[str, Any],
        processing_duration_seconds: float,
    ) -> int:
        """Insert one analysis result row. Returns the new row id."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO analysis_results
                    (filename, upload_datetime, status, score, ocr_text,
                     missing_items, warnings, result_json, processing_duration_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    filename,
                    datetime.now().isoformat(timespec="seconds"),
                    status,
                    score,
                    ocr_text,
                    json.dumps(missing_items),
                    json.dumps(warnings),
                    json.dumps(result_dict),
                    processing_duration_seconds,
                ),
            )
            return int(cursor.lastrowid)

    def fetch_history(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Return the most recent analysis results, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, filename, upload_datetime, status, score,
                       missing_items, warnings, processing_duration_seconds
                FROM analysis_results
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def fetch_result_detail(self, result_id: int) -> Optional[Dict[str, Any]]:
        """Return the full stored JSON result for one row, or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT result_json FROM analysis_results WHERE id = ?",
                (result_id,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["result_json"])

    def clear_history(self) -> int:
        """Delete all stored analysis history. Returns number of rows removed."""
        with self._connect() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM analysis_results")
            count = cursor.fetchone()[0]
            conn.execute("DELETE FROM analysis_results")
        return int(count)
