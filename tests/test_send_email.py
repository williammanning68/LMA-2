import unittest
from pathlib import Path
from unittest import mock
import tempfile
import re
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


class TranscriptDecodeTest(unittest.TestCase):
    def test_invalid_bytes_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.txt"
            bad.write_bytes(b"\x81")
            with self.assertRaises(RuntimeError) as cm:
                send_email.build_digest_html([str(bad)], ["kw"])
            self.assertIn("Windows-1252 or UTF-8", str(cm.exception))


class EncodingIntegrationTest(unittest.TestCase):
    def test_dash_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            transcript = Path(tmp) / "House_of_Assembly_test.txt"
            transcript.write_text("Mr SPEAKER:\n dash – test dash.", encoding="utf-8")
            html, _total, _counts = send_email.build_digest_html([str(transcript)], ["dash"])
            self.assertIn("charset=utf-8", html)
            self.assertTrue("–" in html, "dash missing")


class BuildDigestHtmlTest(unittest.TestCase):
    def test_detection_table_contains_expected_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            transcript = Path(tmp) / "House_of_Assembly_simple.txt"
            transcript.write_text("Mr SPEAKER:\napple banana apple.", encoding="utf-8")
            html, _total, _counts = send_email.build_digest_html(
                [str(transcript)], ["apple", "banana", "cherry"]
            )
            expected_rows = "".join(
                [
                    send_email._build_detection_row("apple", 1, 0, 1),
                    send_email._build_detection_row("banana", 1, 0, 1),
                    send_email._build_detection_row("cherry", 0, 0, 0),
                ]
            )
            m = re.search(
                r"<!--\s*DETECTION_SUMMARY_TABLE_START\s*-->.*?Keyword.*?</tr>(.*?)</table>",
                html,
                re.S,
            )
            self.assertIsNotNone(m)
            self.assertEqual(m.group(1), expected_rows)

    def test_sample_section_replaced_with_file_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            transcript = Path(tmp) / "House_of_Assembly_sample.txt"
            transcript.write_text("Mr SPEAKER:\napple.", encoding="utf-8")
            html, _total, _counts = send_email.build_digest_html([str(transcript)], ["apple"])
            self.assertNotIn("SAMPLE_SECTION_START", html)
            self.assertNotIn("SAMPLE_SECTION_END", html)
            self.assertNotIn("Sample_file.txt", html)
            self.assertIn(transcript.name, html)
            self.assertGreater(
                html.index(transcript.name),
                html.index("<!-- DETECTION_SUMMARY_TABLE_END -->"),
            )


if __name__ == "__main__":
    unittest.main()
