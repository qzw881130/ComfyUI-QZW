"""Persistent reference-upload URLs; no image bytes or API credentials on disk."""

import hashlib
import logging
import math
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


LOG = logging.getLogger("ComfyUI-QZW")
EXPIRY_MARGIN = 600
MAX_LIFETIME = 86400


def upload_expiry(url, expires_in, started_at):
    deadlines = []
    try:
        if expires_in is not None:
            deadlines.append(started_at + float(expires_in))
        query = parse_qs(urlsplit(url).query, keep_blank_values=True)
        # COS signatures and signing keys can expire before the API's stated TTL.
        for field in ("q-sign-time", "q-key-time"):
            for value in query.get(field, []):
                start, end = value.split(";")
                start, end = float(start), float(end)
                if not math.isfinite(start) or end <= start:
                    return None
                deadlines.append(end)
    except (TypeError, ValueError):
        return None
    if not deadlines or not all(math.isfinite(t) for t in deadlines):
        return None
    return min(*deadlines, started_at + MAX_LIFETIME)


class UploadCache:
    def __init__(self, path, base_url, api_key, *, file_names=False):
        self.path = Path(path)
        self.scope = hashlib.sha256((base_url + "\0" + api_key).encode("utf-8")).hexdigest()
        self.file_names = file_names
        self.table = "uploaded_files" if file_names else "uploads"
        self.column = "file_name" if file_names else "url"

    def get(self, digest):
        try:
            if not self.path.exists():
                return None
            with closing(sqlite3.connect(self.path, timeout=5)) as db:
                row = db.execute(f"SELECT {self.column}, expires_at FROM {self.table} WHERE scope=? AND md5=?",
                                 (self.scope, digest)).fetchone()
            if row and row[1] > time.time() + EXPIRY_MARGIN:
                if self.file_names:
                    return row[0] if isinstance(row[0], str) and row[0].strip() else None
                parsed = urlsplit(row[0])
                if parsed.scheme in ("http", "https") and parsed.hostname:
                    return row[0]
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            LOG.warning("[QZW] 上传缓存读取失败（%s），将正常上传。", type(exc).__name__)
        return None

    def put(self, digest, url, expires_in, started_at, *, file_name=None):
        deadline = upload_expiry(url, expires_in, started_at)
        now = time.time()
        cacheable = deadline is not None and deadline > now + EXPIRY_MARGIN
        value = file_name if self.file_names else url
        cacheable = cacheable and isinstance(value, str) and bool(value.strip())
        try:
            if not cacheable and not self.path.exists():
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(self.path, timeout=5)) as db, db:
                db.execute(f"CREATE TABLE IF NOT EXISTS {self.table} (scope TEXT NOT NULL, md5 TEXT NOT NULL, "
                           f"{self.column} TEXT NOT NULL, expires_at REAL NOT NULL, PRIMARY KEY (scope, md5))")
                db.execute(f"DELETE FROM {self.table} WHERE expires_at<=?", (now + EXPIRY_MARGIN,))
                db.execute(f"DELETE FROM {self.table} WHERE scope=? AND md5=?", (self.scope, digest))
                if cacheable:
                    db.execute(f"INSERT INTO {self.table} VALUES (?, ?, ?, ?)",
                               (self.scope, digest, value, deadline))
        except (OSError, sqlite3.Error) as exc:
            LOG.warning("[QZW] 上传缓存保存失败（%s），本次使用已上传的链接。", type(exc).__name__)
