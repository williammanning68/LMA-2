#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hansard Monitor — send_email.py

Template-driven email renderer/sender.

Guarantees:
- No visual HTML constructed here; we only inject dynamic data into placeholders
  that already exist in the HTML template.
- Prefer a Gmail-round-tripped template ('gmail sent.htm' / 'gmail sent.html')
  if present (best cross-client rendering); otherwise fall back to 'email_template.htm'.
- Keep only the <body> markup, strip broken <style> blocks (e.g., Word CSS with <br>)
  to avoid any "code showing" and to silence cssutils/premailer errors.
- Send as raw HTML (no CSS inlining), preserving template visuals.

Placeholders expected in the template body (case-insensitive inside [ ... ]):
  [DATE]
  [Keyword]
  [House of Assembly count]
  [Legislative Council count]
  [Total count]

  [Transcript filename]
  [Match count]

  [Match #]
  [SPEAKER NAME]
  [Line number(s)]
  [...snippet...]   (any placeholder containing 'snippet' inside [ ])

Assumptions:
- Transcripts are *.txt under ./transcripts/
- keywords.txt exists (one keyword per line; # for comments)
- EMAIL_USER / EMAIL_PASS / EMAIL_TO provided as env vars
"""

from __future__ import annotations

import os
import re
import glob
import html
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Set, Optional

import yagmail  # requirements.txt includes yagmail

# -----------------------------------------------------------------------------
# Paths & constants
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
TRANSCRIPT_DIR = BASE_DIR / "transcripts"
KEYWORDS_PATH = BASE_DIR / "keywords.txt"
LOG_PATH = BASE_DIR / "sent.log"

GMAIL_TEMPLATE_CANDIDATES = [
    BASE_DIR / "gmail sent.htm",
    BASE_DIR / "gmail sent.html",
]
FALLBACK_TEMPLATE = BASE_DIR / "email_template.htm"

# Snippet/extraction
MAX_SNIPPET_CHARS = 800
CONTEXT_CHARS = 280  # chars before/after a match window

# Liberal speaker header (plain-text transcripts)
SPEAKER_HEADER_RE = re.compile(r"^(?P<speaker>[A-Z][A-Za-z .,'()\-]+):\s*$", re.M)


@dataclass
class Match:
    keywords: Set[str]
    excerpt_html: str
    speaker: str
    line_numbers: List[int]
    start_idx: int
    end_idx: int


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _resolve_template_path() -> Path:
    for p in GMAIL_TEMPLATE_CANDIDATES:
        if p.exists():
            return p
    return FALLBACK_TEMPLATE

def _html_escape(s: str) -> str:
    return html.escape(s, quote=True)

def _only_body(html_text: str) -> str:
    """Return only inner <body> contents."""
    m = re.search(r"<body\b[^>]*>(?P<body>[\s\S]*?)</body\s*>", html_text, flags=re.I)
    return m.group("body") if m else html_text

def _remove_broken_style_blocks(body_html: str) -> str:
    """
    Remove <style>...</style> blocks that contain '<br' or look invalid
    (Word/Outlook sometimes inject <br> into CSS which breaks parsers).
    """
    def strip_if_broken(m: re.Match) -> str:
        inner = m.group(1)
        if "<br" in inner.lower() or "<" in inner.replace("<style", "").lower():
            return ""
        return m.group(0)

    return re.sub(r"<style\b[^>]*>([\s\S]*?)</style\s*>", strip_if_broken, body_html, flags=re.I)

def _compute_line_number_indices(text: str) -> List[int]:
    starts = [0]
    for m in re.finditer(r"\n", text):
        starts.append(m.end())
    return starts

def _idx_to_line(starts: List[int], idx: int) -> int:
    lo, hi = 0, len(starts) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if starts[mid] <= idx:
            lo = mid + 1
        else:
            hi = mid - 1
    return hi + 1  # 1-based

def _highlight(text: str, keywords: Iterable[str]) -> str:
    """Highlight keywords safely in HTML, case-insensitive, whole-word-ish."""
    esc = _html_escape(text)
    kws = sorted({k for k in keywords if k}, key=len, reverse=True)
    if not kws:
        return esc

    pat = r"(?i)(?<![A-Za-z0-9])(" + "|".join(re.escape(k) for k in kws) + r")(?![A-Za-z0-9])"
    def repl(m: re.Match) -> str:
        return f'<span style="background:#fff0a6">{m.group(0)}</span>'
    return re.sub(pat, repl, esc)

def _parse_chamber_from_filename(name: str) -> str:
    """Heuristic chamber parsing from filename."""
    n = name.lower()
    if "legislative council" in n or "lc" in n or "council" in n:
        return "Legislative Council"
    return "House of Assembly"

# -----------------------------------------------------------------------------
# Core matching (no visuals)
# -----------------------------------------------------------------------------

def extract_matches(text: str, keywords: List[str]) -> List[Match]:
    """
    - Find occurrences of any keyword (case-insensitive, word-ish)
    - Attribute to nearest 'SPEAKER:' header above hit if present
    - Merge overlapping windows; build excerpt; cap length; highlight
    """
    matches: List[Match] = []
    if not text or not keywords:
        return matches

    kw_pat = re.compile(
        r"(?i)(?<![A-Za-z0-9])(" + "|".join(re.escape(k) for k in keywords if k) + r")(?![A-Za-z0-9])"
    )

    hits = [m for m in kw_pat.finditer(text)]
    if not hits:
        return matches

    # Speaker headers
    speaker_by_line: Dict[int, str] = {}
    for sm in SPEAKER_HEADER_RE.finditer(text):
        line_no = text.count("\n", 0, sm.start()) + 1
        speaker_by_line[line_no] = sm.group("speaker").strip()

    line_starts = _compute_line_number_indices(text)

    # Merge nearby hits into windows
    windows: List[Tuple[int, int, Set[str]]] = []
    for h in hits:
        start = max(0, h.start() - CONTEXT_CHARS)
        end = min(len(text), h.end() + CONTEXT_CHARS)
        kws = {h.group(0)}
        if windows and start <= windows[-1][1]:
            prev_s, prev_e, prev_k = windows[-1]
            windows[-1] = (prev_s, max(prev_e, end), prev_k | kws)
        else:
            windows.append((start, end, kws))

    for s, e, kws in windows:
        raw = text[s:e]
        # Find nearest speaker header above
        start_line = _idx_to_line(line_starts, s)
        speaker_lines = [ln for ln in speaker_by_line if ln <= start_line]
        speaker = speaker_by_line[max(speaker_lines)] if speaker_lines else ""

        if len(raw) > MAX_SNIPPET_CHARS:
            mid = (e + s) // 2
            half = MAX_SNIPPET_CHARS // 2
            raw = text[max(0, mid - half):min(len(text), mid + half)]

        snippet = _highlight(raw, kws)
        first_line = _idx_to_line(line_starts, s)
        last_line = _idx_to_line(line_starts, e - 1)
        line_numbers = [first_line] if first_line == last_line else list(range(first_line, min(first_line + 3, last_line + 1)))

        matches.append(Match(
            keywords={k.strip() for k in kws if k.strip()},
            excerpt_html=snippet,
            speaker=speaker,
            line_numbers=line_numbers,
            start_idx=s,
            end_idx=e,
        ))

    return matches

# -----------------------------------------------------------------------------
# Template-driven rendering (no visuals here; clone from template)
# -----------------------------------------------------------------------------

def _read_template_html(path: Path) -> str:
    try:
        return path.read_text(encoding="windows-1252", errors="ignore")
    except Exception:
        return path.read_text(encoding="utf-8", errors="ignore")

def _inject_date(body: str) -> str:
    run_date = datetime.now().strftime("%d %B %Y")
    return re.sub(r"\[\s*DATE\s*\]", _html_escape(run_date), body, flags=re.I)

def _clone_detection_rows(body: str, keywords: List[str], counts: Dict[str, Dict[str, int]]) -> str:
    m_kw = re.search(r"\[\s*Keyword\s*\]", body, flags=re.I)
    if not m_kw:
        return body

    tr_start = body.rfind("<tr", 0, m_kw.start())
    tr_end = body.find("</tr>", m_kw.end())
    if tr_start == -1 or tr_end == -1:
        return body
    tr_end += len("</tr>")

    row_tpl = body[tr_start:tr_end]
    rows = []
    for kw in keywords:
        hoa = counts.get(kw, {}).get("House of Assembly", 0)
        lc = counts.get(kw, {}).get("Legislative Council", 0)
        total = hoa + lc
        row = row_tpl
        row = re.sub(r"\[\s*Keyword\s*\]", _html_escape(kw), row, flags=re.I)
        row = re.sub(r"\[\s*House\s+of\s+Assembly\s+count\s*\]", str(hoa), row, flags=re.I)
        row = re.sub(r"\[\s*Legislative\s+Council\s+count\s*\]", str(lc), row, flags=re.I)
        row = re.sub(r"\[\s*Total\s+count\s*\]", str(total), row, flags=re.I)
        rows.append(row)

    return body.replace(row_tpl, "".join(rows), 1)

def _find_enclosing_table(html_text: str, idx: int) -> Optional[Tuple[int, int]]:
    """Find the outermost <table>...</table> that encloses idx. Returns (start, end)."""
    # Walk back to nearest <table
    tbl_start = html_text.rfind("<table", 0, idx)
    if tbl_start == -1:
        return None
    # Walk forward with a simple stack
    pos = tbl_start
    depth = 0
    end_tbl = None
    while pos < len(html_text):
        if html_text.startswith("<table", pos):
            depth += 1
            pos = html_text.find(">", pos) + 1
            continue
        if html_text.startswith("</table", pos):
            depth -= 1
            close_end = html_text.find(">", pos) + 1
            pos = close_end
            if depth == 0:
                end_tbl = close_end
                break
            continue
        pos += 1
    return (tbl_start, end_tbl) if end_tbl else None

def _clone_transcript_cards(body: str, file_matches: List[Tuple[str, List[Match]]]) -> str:
    m_tf = re.search(r"\[\s*Transcript\s+filename\s*\]", body, flags=re.I)
    if not m_tf:
        return body

    tbl_bounds = _find_enclosing_table(body, m_tf.start())
    if not tbl_bounds:
        return body

    tbl_start, tbl_end = tbl_bounds
    card_tpl = body[tbl_start:tbl_end]

    # Find match <tr> template inside this card (row containing [Match #] or 'snippet')
    m_row = re.search(r"\[\s*Match\s*#\s*\]|\[\s*.*snippet.*\s*\]", card_tpl, flags=re.I)
    match_row_tpl = None
    if m_row:
        row_start = card_tpl.rfind("<tr", 0, m_row.start())
        row_end = card_tpl.find("</tr>", m_row.end())
        if row_start != -1 and row_end != -1:
            row_end += len("</tr>")
            match_row_tpl = card_tpl[row_start:row_end]

    cards = []
    for fname, matches in file_matches:
        card_html = card_tpl
        card_html = re.sub(r"\[\s*Transcript\s+filename\s*\]", _html_escape(fname), card_html, flags=re.I)
        card_html = re.sub(r"\[\s*Match\s+count\s*\]", str(len(matches)), card_html, flags=re.I)

        if match_row_tpl:
            rows = []
            for i, m in enumerate(matches, start=1):
                r = match_row_tpl
                r = re.sub(r"\[\s*Match\s*#\s*\]", str(i), r, flags=re.I)
                r = re.sub(r"\[\s*SPEAKER\s+NAME\s*\]", _html_escape(m.speaker or ""), r, flags=re.I)
                r = re.sub(r"\[\s*Line\s+number\(s\)\s*\]", _html_escape(", ".join(str(x) for x in m.line_numbers)), r, flags=re.I)
                # Replace any [..snippet..]-like placeholder with HTML snippet
                r = re.sub(r"\[\s*.*snippet.*\s*\]", m.excerpt_html, r, flags=re.I)
                rows.append(r)
            # Replace the first instance of the row template with all rows
            card_html = card_html.replace(match_row_tpl, "".join(rows), 1)

        cards.append(card_html)

    # Replace the first instance of the card template with all cards
    return body.replace(card_tpl, "".join(cards), 1)

def build_digest_html(files: List[str], keywords: List[str]) -> Tuple[str, int, Dict[str, Dict[str, int]]]:
    """
    Render the final HTML by cloning structures from the template itself.
    We do NOT hard-code any visual HTML; we only replace placeholders.
    """
    template_path = _resolve_template_path()
    raw_html = _read_template_html(template_path)

    # Keep only <body> content and remove broken <style> blocks
    body = _only_body(raw_html)
    body = _remove_broken_style_blocks(body)

    # Insert date
    body = _inject_date(body)

    # Pass 1: extract matches & counts
    counts: Dict[str, Dict[str, int]] = {kw: {"House of Assembly": 0, "Legislative Council": 0} for kw in keywords}
    file_matches: List[Tuple[str, List[Match]]] = []
    total_matches = 0

    for f in files:
        text = Path(f).read_text(encoding="utf-8", errors="ignore")
        ms = extract_matches(text, keywords)
        if not ms:
            continue
        total_matches += len(ms)
        chamber = _parse_chamber_from_filename(Path(f).name)
        for m in ms:
            for kw in m.keywords:
                # map to canonical keyword for counting (case-insensitive)
                for target in keywords:
                    if kw.lower() == target.lower():
                        counts.setdefault(target, {"House of Assembly": 0, "Legislative Council": 0})
                        counts[target][chamber] = counts[target].get(chamber, 0) + 1
                        break
        file_matches.append((Path(f).name, ms))

    # Pass 2: detection table rows
    body = _clone_detection_rows(body, keywords, counts)

    # Pass 3: transcript cards & match rows
    body = _clone_transcript_cards(body, file_matches)

    # Optionally fill [Total count] outside the table if present
    grand_total = sum(sum(c.values()) for c in counts.values())
    body = re.sub(r"\[\s*Total\s+count\s*\]", str(grand_total), body, flags=re.I)

    return body, total_matches, counts

# -----------------------------------------------------------------------------
# IO: keywords / transcripts
# -----------------------------------------------------------------------------

def load_keywords() -> List[str]:
    kws: List[str] = []
    if KEYWORDS_PATH.exists():
        for line in KEYWORDS_PATH.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith('"') and s.endswith('"') and len(s) >= 2:
                s = s[1:-1]
            kws.append(s)
    # Keep input order
    return kws

def list_transcripts() -> List[str]:
    # keep deterministic order
    paths = sorted(glob.glob(str(TRANSCRIPT_DIR / "*.txt")))
    return paths

# -----------------------------------------------------------------------------
# Send
# -----------------------------------------------------------------------------

def send_email():
    user = os.getenv("EMAIL_USER")
    pwd = os.getenv("EMAIL_PASS")
    to_raw = os.getenv("EMAIL_TO", "")
    if not user or not pwd or not to_raw.strip():
        raise SystemExit("Missing EMAIL_USER / EMAIL_PASS / EMAIL_TO environment variables.")

    recipients = [a.strip() for a in re.split(r"[;,]", to_raw) if a.strip()]

    keywords = load_keywords()
    files = list_transcripts()

    html_body, total_matches, _counts = build_digest_html(files, keywords)

    # Subject: keep non-visuals here
    today = datetime.now().strftime("%d %b %Y")
    subject = f"Hansard Monitor — {today}"

    # Send as raw HTML to bypass premailer/cssutils entirely
    contents = [yagmail.raw(html_body)]

    with yagmail.SMTP(user=user, password=pwd) as yag:
        yag.send(to=recipients, subject=subject, contents=contents)

    print(f"✅ Email sent to {', '.join(recipients)} with {len(files)} file(s), {total_matches} match(es).")

    # Log
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        ts = datetime.now().isoformat(timespec="seconds")
        f.write(f"{ts}\tto={','.join(recipients)}\tfiles={len(files)}\tmatches={total_matches}\n")

if __name__ == "__main__":
    send_email()
