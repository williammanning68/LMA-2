import unittest
from pathlib import Path
from unittest import mock
import tempfile
import send_email


class MergeWindowsTest(unittest.TestCase):
    def test_merge_only_close_windows(self):
        wins = [
            [0, 1, {"a"}, {1}],
            [2, 3, {"b"}, {2}],
            [6, 7, {"c"}, {3}],
        ]
        merged = send_email._merge_windows_far_only(wins, gap_gt=2)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0][:2], [0, 3])
        self.assertEqual(merged[1][:2], [6, 7])


class TemplateDecodeTest(unittest.TestCase):
    def test_decoding_error_raises(self):
        class Bad:
            def decode(self, *args, **kwargs):
                raise UnicodeDecodeError("test", b"", 0, 1, "bad")

        class DummyPath:
            def read_bytes(self):
                return Bad()

        with self.assertRaises(RuntimeError):
            send_email._load_template_text(DummyPath())

    def test_utf8_fallback(self):
        class Fake:
            def decode(self, encoding, errors="strict"):
                if encoding == "windows-1252":
                    raise UnicodeDecodeError("test", b"", 0, 1, "bad")
                return "ok"

        class DummyPath:
            def read_bytes(self):
                return Fake()

        self.assertEqual(send_email._load_template_text(DummyPath()), "ok")


class EncodingIntegrationTest(unittest.TestCase):
    def test_dash_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            transcript = Path(tmp) / "House_of_Assembly_test.txt"
            transcript.write_bytes("Mr SPEAKER:\n dash – test dash.".encode("windows-1252"))
            html, _total, _counts = send_email.build_digest_html([str(transcript)], ["dash"])
            self.assertIn("charset=utf-8", html)
            self.assertTrue("–" in html, "dash missing")


class TranscriptDecodeTest(unittest.TestCase):
    def test_decoding_error_raises(self):
        class Bad:
            def decode(self, *args, **kwargs):
                raise UnicodeDecodeError("test", b"", 0, 1, "bad")

        class DummyPath:
            def read_bytes(self):
                return Bad()

        with self.assertRaises(RuntimeError):
            send_email._load_transcript_text(DummyPath())

    def test_utf8_fallback(self):
        class Fake:
            def decode(self, encoding, errors="strict"):
                if encoding == "windows-1252":
                    raise UnicodeDecodeError("test", b"", 0, 1, "bad")
                return "ok"

        class DummyPath:
            def read_bytes(self):
                return Fake()

        self.assertEqual(send_email._load_transcript_text(DummyPath()), "ok")


if __name__ == "__main__":
    unittest.main()
