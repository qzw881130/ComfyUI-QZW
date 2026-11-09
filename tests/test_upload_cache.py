import importlib.util
import socket
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("qzw_cache_tests", Path(__file__).resolve().parents[1] / "upload_cache.py")
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)


class UploadCacheTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "uploads.sqlite3"
        self.guard = patch.object(socket.socket, "connect", side_effect=AssertionError("No network"))
        self.guard.start()
        self.addCleanup(self.guard.stop)
        self.clock = patch.object(cache.time, "time", return_value=10000)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.store = cache.UploadCache(self.path, "https://api.aitudou.net", "fake-secret")

    def test_expiry_uses_earliest_ttl_or_cos_signature(self):
        # Signature is created after the upload started; its start time is not the TTL origin.
        url = "https://cdn.test/a.png?q-sign-time=10010%3B14000&q-key-time=10010%3B13000"
        self.assertEqual(cache.upload_expiry(url, 86400, 10000), 13000)
        self.assertEqual(cache.upload_expiry(url, 2000, 10000), 12000)
        self.assertEqual(cache.upload_expiry(url, None, 10000), 13000)
        self.assertEqual(cache.upload_expiry("https://cdn.test/a.png", "3600", 10000), 13600)
        self.assertEqual(cache.upload_expiry("https://cdn.test/a.png", 172800, 10000), 96400)
        self.assertEqual(cache.upload_expiry("https://cdn.test/a.png?q-sign-time=10000%3B182800", None, 10000), 96400)
        for url, ttl in (("https://cdn.test/a.png", None), (url, "invalid"), (url, float("inf")),
                         ("https://cdn.test/a.png?q-sign-time=invalid", 86400)):
            self.assertIsNone(cache.upload_expiry(url, ttl, 10000))

    def test_account_and_base_url_isolation(self):
        self.store.put("digest", "https://cdn.test/a.png", 3600, 10000)
        self.assertEqual(cache.UploadCache(self.path, "https://api.aitudou.net", "fake-secret").get("digest"),
                         "https://cdn.test/a.png")
        self.assertIsNone(cache.UploadCache(self.path, "https://api.aitudou.net", "different-secret").get("digest"))
        self.assertIsNone(cache.UploadCache(self.path, "https://another.test", "fake-secret").get("digest"))
        self.assertIsNone(self.store.get("different-image"))
        self.assertNotIn(b"fake-secret", self.path.read_bytes())

    def test_unknown_or_short_expiry_cannot_leave_old_cached_url(self):
        for ttl in (None, 0, -100, 600, "invalid"):
            self.store.put("digest", "https://cdn.test/old.png", 86400, 10000)
            self.store.put("digest", "https://cdn.test/new.png", ttl, 10000)
            self.assertIsNone(self.store.get("digest"))

    def test_expired_entries_are_pruned_on_write(self):
        self.store.put("old", "https://cdn.test/old.png", 3600, 10000)
        with patch.object(cache.time, "time", return_value=13000):
            self.assertIsNone(self.store.get("old"))
            self.store.put("new", "https://cdn.test/new.png", 3600, 13000)
        db = sqlite3.connect(self.path)
        try:
            self.assertEqual(db.execute("SELECT md5 FROM uploads").fetchall(), [("new",)])
        finally:
            db.close()

    def test_concurrent_writes_keep_all_entries(self):
        def write(index):
            store = cache.UploadCache(self.path, "https://api.aitudou.net", "fake-secret")
            store.put(str(index), f"https://cdn.test/{index}.png", 3600, 10000)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(write, range(12)))
        for index in range(12):
            self.assertEqual(self.store.get(str(index)), f"https://cdn.test/{index}.png")


if __name__ == "__main__":
    unittest.main()
