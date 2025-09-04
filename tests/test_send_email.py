import os
import unittest
from pathlib import Path
from unittest import mock
from html.parser import HTMLParser
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
        class DummyPath:
            def read_bytes(self):
                return b"\xff"

        with self.assertRaises(RuntimeError):
            send_email._load_template_text(DummyPath())

    def test_decodes_with_chosen_encoding(self):
        class DummyPath:
            def read_bytes(self):
                return "ok".encode(send_email.ENCODING)

        self.assertEqual(send_email._load_template_text(DummyPath()), "ok")


class EncodingIntegrationTest(unittest.TestCase):
    def test_dash_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            transcript = Path(tmp) / "House_of_Assembly_test.txt"
            transcript.write_text("Mr SPEAKER:\n dash – test dash.", encoding=send_email.ENCODING)
            html, _total, _counts = send_email.build_digest_html([str(transcript)], ["dash"])
            self.assertIn(f"charset={send_email.ENCODING}", html)
            self.assertTrue("–" in html, "dash missing")


class SendEmailIntegrationTest(unittest.TestCase):
    def test_yagmail_receives_encoded_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                Path("transcripts").mkdir()
                transcript = Path("transcripts") / "House_of_Assembly_test.txt"
                transcript.write_text("Mr SPEAKER:\n dash – test dash.", encoding=send_email.ENCODING)

                env = {
                    "EMAIL_USER": "u",
                    "EMAIL_PASS": "p",
                    "EMAIL_TO": "t@example.com",
                    "KEYWORDS": "dash",
                }
                captured = {}

                class DummySMTP:
                    def __init__(self, *args, **kwargs):
                        pass

                    def send(self, **kwargs):
                        captured.update(kwargs)

                with mock.patch.dict(os.environ, env, clear=True), mock.patch(
                    "send_email.yagmail.SMTP", return_value=DummySMTP()
                ):
                    send_email.main()

                self.assertIn("contents", captured)
                content = captured["contents"][0]
                self.assertIsInstance(content, bytes)
                html = content.decode(send_email.ENCODING)

                class Collector(HTMLParser):
                    def __init__(self):
                        super().__init__()
                        self.text = []

                    def handle_data(self, data):
                        self.text.append(data)

                parser = Collector()
                parser.feed(html)
                text = "".join(parser.text)
                self.assertIn("–", text)
                self.assertIn(f"charset={send_email.ENCODING}", html)
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
