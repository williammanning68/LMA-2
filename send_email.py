#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
from datetime import datetime
from pathlib import Path
from bisect import bisect_right
from typing import List, Tuple, Dict, Set
from email.mime.text import MIMEText

import yagmail  # transport
from yagmail.raw import raw as yag_raw  # <- use this to pass raw MIME parts (no rewriting)

# Optionally silence cssutils (pulled by premailer via yagmail); with raw MIME we shouldn't hit it,
# but keeping this avoids noisy CI logs in any case.
try:
    import logging, cssutils  # type: ignore
    cssutils.log.setLevel(logging.FATAL)
except Exception:
    pass

# -----------------------------------------------------------------------------
# Constants / paths
# -----------------------------------------------------------------------------
TEMPLATE_HTML_PATH = Path("email_template.htm")
KEYWORDS_PATH      = Path("keywords.txt")
LOG_FILE           = Path("sent.log")
TRANSCRIPTS_DIR    = Path("transcripts")

DEFAULT_TITLE = "Hansard Monitor – BETA"
VERSION_STR   = os.environ.get("VERSION_STR", "18.3")

# -----------------------------------------------------------------------------
# Utility helpers (template-agnostic; no visual HTML here)
# -----------------------------------------------------------------------------

def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
         .replace("'", "&#39;")
    )

def _tighten_outlook_whitespace(html: str) -> str:
    """
    Gentle whitespace cleanup that won't strip spacer cells or intentional gaps.
    - Remove ghost empty Word paragraphs.
    - Collapse runs of <br>.
    - Remove whitespace between adjacent tables (safe).
    """
    EMPTY_MSOP_RE = re.compile(
        r"<p\b[^>]*>\s*(?:&nbsp;|\s|<span\b[^>]*>\s*</span>|<o:p>\s*</o:p>|<br[^>]*>)*\s*</p>",
        flags=re.I
    )
    html = EMPTY_MSOP_RE.sub("", html)
    html = re.sub(r"(?:\s*<br[^>]*>\s*){2,}", "<br>", html, flags=re.I)
    html = re.sub(r"(</table>)\s+(?=(?:<!--.*?-->\s*)*<table\b)", r"\1", html, flags=re.I | re.S)
    return html

def _minify_inter_tag_whitespace(html: str) -> str:
    return re.sub(r">\s+<", "><", html)

def _parse_chamber_from_filename(name: str) -> str:
    lower = name.lower()
    if "legislative" in lower:
        return "Legislative Council"
    if "assembly" in lower:
        return "House of Assembly"
    return "House of Assembly"

def load_keywords() -> List[str]:
    env_kw = os.environ.get("KEYWORDS", "").strip()
    parts = []
    if env_kw:
        parts.extend([p.strip() for p in env_kw.split(",") if p.strip()])
    if KEYWORDS_PATH.exists():
        for ln in KEYWORDS_PATH.read_text(encoding="utf-8", errors="ignore").splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts.append(ln)
    seen = set()
    uniq = []
    for p in parts:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq

# -----------------------------------------------------------------------------
# Matching (data only)
# -----------------------------------------------------------------------------

Match = Tuple[Set[str], str, str, List[int], int, int]
# (keywords_matched, excerpt_html, speaker, line_numbers, start_index, end_index)

def _keyword_regexes(keywords: List[str]) -> List[Tuple[str, re.Pattern]]:
    regs = []
    for kw in keywords:
        regs.append((kw, re.compile(re.escape(kw), flags=re.I)))
    return regs

def extract_matches(text: str, keywords: List[str]) -> List[Match]:
    """
    Minimal extractor: find lines containing any keyword; return a short
    excerpt for each hit. Keeps logic template-agnostic (no HTML here).
    """
    regs = _keyword_regexes(keywords)
    matches: List[Match] = []
    lines = text.splitlines()
    for i, line in enumerate(lines, start=1):
        found: Set[str] = set()
        for kw, rx in regs:
            if rx.search(line):
                found.add(kw)
        if not found:
            continue
        speaker = ""  # unknown; can be enhanced later
        snippet = _html_escape(line.strip())
        matches.append((found, snippet, speaker, [i], 0, 0))
    return matches

# -----------------------------------------------------------------------------
# Template-driven builder (discover & duplicate rows; no hard-coded layout)
# -----------------------------------------------------------------------------

def build_digest_html(files: List[str], keywords: List[str]) -> Tuple[str, int, Dict[str, Dict[str, int]]]:
    """
    Build final HTML strictly by filling placeholders in email_template.htm.
    """
    # Load template (Word/Outlook often uses Windows-1252)
    try:
        template_html = TEMPLATE_HTML_PATH.read_text(encoding="windows-1252", errors="ignore")
    except Exception:
        template_html = TEMPLATE_HTML_PATH.read_text(encoding="utf-8", errors="ignore")

    # Keep only the <body> inner HTML (prevents head/XML from showing as text)
    m_body = re.search(r'<body\b[^>]*>(?P<body>[\s\S]*?)</body>', template_html, flags=re.I)
    if m_body:
        template_html = m_body.group('body')

    # Insert the current date
    run_date = datetime.now().strftime("%d %B %Y")
    template_html = template_html.replace("[DATE]", run_date)

    # Scan files → counts
    counts: Dict[str, Dict[str, int]] = {kw: {"House of Assembly": 0, "Legislative Council": 0} for kw in keywords}
    file_matches: List[Tuple[str, List[Match]]] = []
    total_matches = 0

    for f in files:
        text = Path(f).read_text(encoding="utf-8", errors="ignore")
        matches = extract_matches(text, keywords)
        if not matches:
            continue
        total_matches += len(matches)
        chamber = _parse_chamber_from_filename(Path(f).name)
        for kw_set, _excerpt_html, _speaker, _lines, _s, _e in matches:
            for kw in kw_set:
                counts.setdefault(kw, {"House of Assembly": 0, "Legislative Council": 0})
                if chamber in counts[kw]:
                    counts[kw][chamber] += 1
        file_matches.append((Path(f).name, matches))

    # Detection table rows
    m_kw = re.search(r"\[\s*Keyword\s*\]", template_html, flags=re.I)
    if m_kw:
        idx_kw = m_kw.start()
        tr_start = template_html.rfind("<tr", 0, idx_kw)
        tr_end = template_html.find("</tr>", idx_kw) + len("</tr>")
        det_row_template = template_html[tr_start:tr_end]

        det_rows_html = []
        for kw in keywords:
            hoa = counts.get(kw, {}).get("House of Assembly", 0)
            lc  = counts.get(kw, {}).get("Legislative Council", 0)
            tot = hoa + lc
            row_html = det_row_template
            row_html = re.sub(r"\[\s*Keyword\s*\]", _html_escape(kw), row_html, flags=re.I)
            row_html = re.sub(r"\[\s*House\s+of\s+Assembly\s+count\s*\]", str(hoa), row_html, flags=re.I)
            row_html = re.sub(r"\[\s*Legislative\s+Council\s+count\s*\]", str(lc), row_html, flags=re.I)
            row_html = re.sub(r"\[\s*Total\s+count\s*\]", str(tot), row_html, flags=re.I)
            det_rows_html.append(row_html)

        template_html = template_html.replace(det_row_template, "".join(det_rows_html), 1)

    # Transcript cards
    trans_match = re.search(r"\[\s*Transcript\s+filename\s*\]", template_html, flags=re.I)
    if trans_match:
        idx_t = trans_match.start()
        table_start = template_html.rfind("<table", 0, idx_t)

        # find matching </table>
        open_tables = 0
        pos = table_start
        end_table = None
        while pos < len(template_html):
            if template_html.startswith("<table", pos):
                open_tables += 1; pos += 6; continue
            if template_html.startswith("</table>", pos):
                open_tables -= 1; pos += 8
                if open_tables == 0:
                    end_table = pos; break
                continue
            pos += 1

        if end_table is not None:
            card_template = template_html[table_start:end_table]
            m_match_no = re.search(r"\[\s*Match\s*#\s*\]", card_template, flags=re.I)
            snippet_idx = card_template.lower().find("/snippet")
            if m_match_no and snippet_idx != -1:
                idx_m = m_match_no.start()
                start_tr1 = card_template.rfind("<tr", 0, idx_m)
                end_tr1   = card_template.find("</tr>", idx_m) + len("</tr>")
                start_tr2 = card_template.rfind("<tr", 0, snippet_idx)
                end_tr2   = card_template.find("</tr>", snippet_idx) + len("</tr>")

                match_template = card_template[start_tr1:end_tr2]
                card_before = card_template[:start_tr1]
                card_after  = card_template[end_tr2:]

                cards_html = []
                for fname, matches in file_matches:
                    if not matches:
                        continue
                    rows = []
                    for i, (_kw_set, excerpt_html, speaker, line_list, _s, _e) in enumerate(matches, start=1):
                        line_txt = f"line {line_list[0]}" if len(line_list) == 1 else "lines " + ", ".join(str(n) for n in line_list)
                        row_html = match_template
                        row_html = re.sub(r"\[\s*Match\s*#\s*\]", str(i), row_html, flags=re.I)
                        row_html = re.sub(r"\[\s*SPEAKER\s+NAME\s*\]", _html_escape(speaker) if speaker else "UNKNOWN", row_html, flags=re.I)
                        row_html = re.sub(r"\[\s*Line\s+number\(s\)\s*\]", _html_escape(line_txt), row_html, flags=re.I)
                        row_html = re.sub(r"\[[^\]]*snippet[^\]]*\]", excerpt_html, row_html, flags=re.I)
                        rows.append(row_html)

                    card_html = card_before + "".join(rows) + card_after
                    card_html = re.sub(r"\[\s*Transcript\s+filename\s*\]", _html_escape(fname), card_html, flags=re.I)
                    card_html = re.sub(r"\[\s*Match\s+count\s*\]", str(len(matches)), card_html, flags=re.I)
                    cards_html.append(card_html)

                template_html = template_html.replace(card_template, "".join(cards_html), 1)

    # Final safe cleanups
    template_html = _tighten_outlook_whitespace(template_html)
    template_html = _minify_inter_tag_whitespace(template_html)

    return template_html, total_matches, counts

# -----------------------------------------------------------------------------
# Sent-log helpers
# -----------------------------------------------------------------------------

def load_sent_log():  # -> Set[str]
    if LOG_FILE.exists():
        with open(LOG_FILE, encoding="utf-8") as f:
            return {ln.strip() for ln in f if ln.strip()}
    return set()

def update_sent_log(files: List[str]) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        for fp in files:
            f.write(Path(fp).name + "\n")

# -----------------------------------------------------------------------------
# Main: input → sort → send (no visual HTML here; template controls visuals)
# -----------------------------------------------------------------------------

def main() -> None:
    EMAIL_USER = os.environ["EMAIL_USER"]
    EMAIL_PASS = os.environ["EMAIL_PASS"]
    EMAIL_TO   = os.environ["EMAIL_TO"]

    yag = yagmail.SMTP(
        user=EMAIL_USER,
        password=EMAIL_PASS,
        host=os.environ.get("SMTP_HOST", "smtp.gmail.com"),
        port=int(os.environ.get("SMTP_PORT", "587")),
        smtp_starttls=os.environ.get("SMTP_STARTTLS", "1").lower() in ("1", "true", "yes"),
        smtp_ssl=os.environ.get("SMTP_SSL", "0").lower() in ("1", "true", "yes"),
    )

    keywords = load_keywords()
    if not keywords:
        raise SystemExit("No keywords found (keywords.txt or KEYWORDS env var).")

    TRANSCRIPTS_DIR.mkdir(exist_ok=True)
    all_files = sorted(glob.glob(str(TRANSCRIPTS_DIR / "*.txt")))
    if not all_files:
        print("No transcripts found in transcripts/")
        return

    sent = load_sent_log()
    files = [f for f in all_files if Path(f).name not in sent]
    if not files:
        print("No new transcripts to email.")
        return

    # Build HTML (body fragment only)
    body_html, total_hits, _counts = build_digest_html(files, keywords)

    # Wrap as raw HTML MIME part and mark raw for yagmail to send without processing
    html_part = MIMEText(body_html, "html", "utf-8")
    subject = f"{DEFAULT_TITLE} – Version {VERSION_STR} — {datetime.now().strftime('%d %b %Y')}"
    to_list = [addr.strip() for addr in re.split(r"[,\s]+", EMAIL_TO) if addr.strip()]

    yag.send(
        to=to_list,
        subject=subject,
        contents=[yag_raw(html_part)],   # <- CRITICAL: pass-through raw MIME; no premailer/cssutils
        attachments=files,
    )

    update_sent_log(files)
    print(f"✅ Email sent to {EMAIL_TO} with {len(files)} file(s), {total_hits} match(es).")

if __name__ == "__main__":
    main()
