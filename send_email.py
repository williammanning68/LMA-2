
from __future__ import annotations
import os
import re
import sys
import glob
import html
import smtplib
import datetime as dt
from email.message import EmailMessage
from email.utils import make_msgid, formatdate
from pathlib import Path
from typing import List, Tuple, Dict

# ---------- Config ----------
VERSION_STR = "Hansard Monitor – BETA Version 18.3"
DATE_FMT = "%d %B %Y"  # 03 September 2025
ROOT = Path(os.getcwd())
TRANSCRIPTS_DIR = Path(os.getenv("TRANSCRIPTS_DIR", "transcripts"))
DEFAULT_TEMPLATE_CANDIDATES = [
    Path(os.getenv("TEMPLATE_PATH", "")).expanduser(),
    ROOT / "email_template.html",
    ROOT / "EmailTemplate.html",
]

# ---------- Utilities ----------

def _load_keywords() -> List[str]:
    env = os.getenv("KEYWORDS", "").strip()
    if env:
        kws = [k.strip() for k in env.split(',') if k.strip()]
        return sorted(set(kws), key=str.lower)
    for name in ("keywords.txt", "Keywords.txt"):
        p = ROOT / name
        if p.exists():
            text = p.read_text(encoding="utf-8", errors="ignore")
            kws = [k.strip() for k in text.splitlines() if k.strip() and not k.strip().startswith('#')]
            return sorted(set(kws), key=str.lower)
    return []


def _find_template() -> Path:
    for p in DEFAULT_TEMPLATE_CANDIDATES:
        if p and p.exists():
            return p
    # fallback: first .html in root
    htmls = sorted(ROOT.glob("*.html"))
    if htmls:
        return htmls[0]
    raise FileNotFoundError("email_template.html not found")


def _list_transcripts() -> List[Path]:
    if not TRANSCRIPTS_DIR.exists():
        return []
    return sorted(p for p in TRANSCRIPTS_DIR.glob("*.txt") if p.is_file())


# ---------- Parsing & Extraction (simple + robust) ----------
SENT_SPLIT = re.compile(r"(?s)(?<=[\.!?])\s+(?=[A-Z0-9\[\(])")


def _normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _find_hits(text: str, keywords: List[str]) -> Dict[str, List[Tuple[int, str]]]:
    """Return {filename_keyword: [(line_no, snippet), ...]}.
    Very conservative: sentence windows around keyword, HTML-safe with <b> highlights.
    """
    hits: Dict[str, List[Tuple[int, str]]] = {}
    # Precompute lowercase text for matching
    lower = text.lower()
    lines = text.splitlines()

    # Map line offsets for snippets
    offsets = []
    pos = 0
    for ln, line in enumerate(lines, 1):
        offsets.append((ln, pos))
        pos += len(line) + 1

    for kw in keywords:
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        for m in pattern.finditer(lower):
            # Locate the line number roughly by counting newlines up to m.start()
            line_no = lower[: m.start()].count("\n") + 1
            # Grab a small window around the match
            start = max(0, m.start() - 120)
            end = min(len(text), m.end() + 120)
            snippet_raw = text[start:end]
            # Expand to sentence boundaries if possible
            # (safe: don't rely on complex heuristics)
            left = snippet_raw.rfind(". ")
            right = snippet_raw.find(". ", m.end() - start)
            if left != -1:
                snippet_raw = snippet_raw[left + 2 :]
            if right != -1:
                snippet_raw = snippet_raw[: right + 1]
            # HTML-escape and highlight
            esc = html.escape(snippet_raw)
            esc = re.sub(re.escape(html.escape(kw)), r"<b>\g<0></b>", esc, flags=re.IGNORECASE)
            key = kw
            hits.setdefault(key, []).append((line_no, esc))
    return hits


def _chamber_from_filename(name: str) -> str:
    n = name.lower()
    if "house_of_assembly" in n or "hoa" in n:
        return "House of Assembly"
    if "legislative_council" in n or "lc" in n:
        return "Legislative Council"
    return "Unknown"


# ---------- HTML assembly via anchors ----------
ANCHOR_ROWS_BEGIN = "<!-- BEGIN: DETECTION_TABLE_ROWS -->"
ANCHOR_ROWS_END = "<!-- END: DETECTION_TABLE_ROWS -->"
ANCHOR_SECTIONS_BEGIN = "<!-- BEGIN: SECTIONS -->"
ANCHOR_SECTIONS_END = "<!-- END: SECTIONS -->"


def _replace_between(src: str, begin: str, end: str, payload: str) -> str:
    b = src.find(begin)
    e = src.find(end)
    if b == -1 or e == -1 or e < b:
        raise ValueError(f"Template anchors missing: {begin}..{end}")
    return src[: b + len(begin)] + "\n" + payload + "\n" + src[e:]


def _build_detection_rows(counts: Dict[str, Dict[str, int]]) -> str:
    """counts = {keyword: {"House of Assembly": n, "Legislative Council": m}}"""
    rows = []
    for kw in sorted(counts.keys(), key=str.lower):
        hoa = counts[kw].get("House of Assembly", 0)
        lc = counts[kw].get("Legislative Council", 0)
        total = hoa + lc
        rows.append(
            f"""
            <tr>
              <td style="padding:6pt 8pt; border:1pt solid #C9CED6; font-family:'Segoe UI', Arial, sans-serif; font-size:10pt;">{html.escape(kw)}</td>
              <td style="padding:6pt 8pt; border:1pt solid #C9CED6; font-family:'Segoe UI', Arial, sans-serif; font-size:10pt; text-align:center;">{hoa}</td>
              <td style="padding:6pt 8pt; border:1pt solid #C9CED6; font-family:'Segoe UI', Arial, sans-serif; font-size:10pt; text-align:center;">{lc}</td>
              <td style="padding:6pt 8pt; border:1pt solid #C9CED6; font-family:'Segoe UI', Arial, sans-serif; font-size:10pt; text-align:center; font-weight:600;">{total}</td>
            </tr>
            """
        )
    return "\n".join(rows) if rows else """<tr><td colspan=4 style="padding:8pt; font-family:'Segoe UI', Arial, sans-serif; font-size:10pt; color:#6B7280;">No detections.</td></tr>"""


def _build_sections(per_file_hits: Dict[Path, Dict[str, List[Tuple[int, str]]]]) -> str:
    blocks = []
    for file, by_kw in per_file_hits.items():
        chamber = _chamber_from_filename(file.name)
        # Header for file
        header = f"""
        <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="91%" align="center" style="margin:0 auto 6pt auto;">
          <tr>
            <td style="font-family:'Segoe UI', Arial, sans-serif; font-size:11pt; color:#111827; padding:4pt 0 2pt 0;">{html.escape(file.name)} — {html.escape(chamber)}</td>
          </tr>
        </table>
        """
        cards = []
        idx = 1
        for kw in sorted(by_kw.keys(), key=str.lower):
            for (line_no, snippet_html) in by_kw[kw]:
                card = f"""
                <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="91%" align="center" style="margin:0 auto 10pt auto; border:1pt solid #C9CED6; border-radius:8pt;">
                  <tr>
                    <td style="padding:8pt 10pt; font-family:'Segoe UI', Arial, sans-serif;">
                      <div style="font-size:9pt; color:#475560;">#{idx} • <span style=\"background-color:#D9D9D9; mso-highlight:lightgrey;\">{html.escape(kw)}</span> • line {line_no}</div>
                      <div style="font-size:11pt; color:#111827; padding-top:2pt;">{snippet_html}</div>
                    </td>
                  </tr>
                </table>
                """
                cards.append(card)
                idx += 1
        # Spacer after each file’s section
        spacer = """
        <table role=\"presentation\" cellpadding=\"0\" cellspacing=\"0\" border=\"0\" width=\"91%\" align=\"center\">
          <tr><td style=\"font-size:0; line-height:0; height:8pt\">&nbsp;</td></tr>
        </table>
        """
        blocks.append(header + "".join(cards) + spacer)
    return "".join(blocks)


# ---------- Email composition ----------

def _compose_html(date_str: str, rows_html: str, sections_html: str) -> str:
    template_path = _find_template()
    raw = template_path.read_text(encoding="utf-8")
    # Simple token replacement for the date label
    html_doc = raw.replace("%%PROGRAM_RUN_DATE%%", html.escape(date_str))
    # Inject rows and sections into anchored regions
    html_doc = _replace_between(html_doc, ANCHOR_ROWS_BEGIN, ANCHOR_ROWS_END, rows_html)
    html_doc = _replace_between(html_doc, ANCHOR_SECTIONS_BEGIN, ANCHOR_SECTIONS_END, sections_html)
    return html_doc


def _build_message(subject: str, html_body: str, attachments: List[Path]) -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.getenv("EMAIL_USER")
    msg["To"] = os.getenv("EMAIL_TO")
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid()

    # Provide a minimal plain text fallback
    plain = re.sub(r"<[^>]+>", "", html_body)
    msg.set_content(plain, charset="windows-1252")
    msg.add_alternative(html_body, subtype="html", charset="windows-1252")

    for p in attachments:
        try:
            data = p.read_bytes()
            msg.add_attachment(
                data,
                maintype="text",
                subtype="plain",
                filename=p.name,
            )
        except Exception as e:
            print(f"WARN: failed to attach {p}: {e}")

    return msg


def _send(msg: EmailMessage) -> None:
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    use_ssl = os.getenv("SMTP_SSL", "0") == "1"
    use_starttls = os.getenv("SMTP_STARTTLS", "1") == "1"
    user = os.getenv("EMAIL_USER")
    pw = os.getenv("EMAIL_PASS")

    if use_ssl:
        with smtplib.SMTP_SSL(host, port) as s:
            if user and pw:
                s.login(user, pw)
            s.send_message(msg)
            return

    with smtplib.SMTP(host, port) as s:
        s.ehlo()
        if use_starttls:
            s.starttls()
            s.ehlo()
        if user and pw:
            s.login(user, pw)
        s.send_message(msg)


# ---------- Main ----------

def main() -> int:
    keywords = _load_keywords()
    transcripts = _list_transcripts()

    # Aggregate detections per keyword and per file
    per_kw_counts: Dict[str, Dict[str, int]] = {}
    per_file_hits: Dict[Path, Dict[str, List[Tuple[int, str]]]] = {}

    for p in transcripts:
        text = p.read_text(encoding="utf-8", errors="ignore")
        by_kw = _find_hits(text, keywords) if keywords else {}
        if by_kw:
            per_file_hits[p] = by_kw
            # Update counts per chamber
            chamber = _chamber_from_filename(p.name)
            for kw, items in by_kw.items():
                per_kw_counts.setdefault(kw, {})
                per_kw_counts[kw][chamber] = per_kw_counts[kw].get(chamber, 0) + len(items)

    # Compose HTML using anchors
    today = dt.datetime.now().strftime(DATE_FMT)
    rows_html = _build_detection_rows(per_kw_counts)
    sections_html = _build_sections(per_file_hits)
    html_body = _compose_html(today, rows_html, sections_html)

    # Subject mirrors Version 1 style
    subject_prefix = os.getenv("SUBJECT_PREFIX", VERSION_STR)
    subject = f"{subject_prefix} — {today}"

    # Build and send
    msg = _build_message(subject, html_body, list(per_file_hits.keys()))
    _send(msg)

    print("Email sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    
