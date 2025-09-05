#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hansard Monitor — send_email.py
Template-driven email renderer/sender.

Key guarantees:
- No visual HTML constructed in Python; we only inject dynamic data into
  placeholders discovered from the HTML template itself.
- Prefer the Gmail-round-tripped template ('gmail sent.htm') if present, as
  it renders best in Outlook/Gmail; otherwise fall back to 'email_template.htm'.
- Strip <head> and any broken <style> blocks (e.g., Word CSS with <br>) to
  prevent "code showing in the email".
- Force UTF-8 when sending (do not pass a custom 'encoding' to yagmail).

Placeholders expected in the template body:
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
  [...snippet...]   (any placeholder containing the word 'snippet' inside [ ])

This script assumes your pipeline has already created .txt transcripts under
./transcripts/ and a keywords list in keywords.txt (one per line).
"""

from __future__ import annotations

import os
import re
import glob
import html
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Set

import yagmail  # requirements.txt includes yagmail

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
TRANSCRIPT_DIR = BASE_DIR / "transcripts"
KEYWORDS_PATH = BASE_DIR / "keywords.txt"
LOG_PATH = BASE_DIR / "sent.log"

GMAIL_TEMPLATE_CANDIDATES = [
    BASE_DIR / "gmail sent.htm",
    BASE_DIR / "gmail sent.html",
]
FALLBACK_TEMPLATE = BASE_DIR / "email_template.htm"

# Snippet/extraction tunables (kept close to your previous semantics)
MAX_SNIPPET_CHARS = 800
CONTEXT_CHARS = 280  # chars before/after a match if speaker/utterance not found

# Regex to find a plausible speaker header. We keep this liberal and ASCII-safe.
SPEAKER_HEADER_RE = re.compile(r"^(?P<speaker>[A-Z][A-Za-z .,'()\-]+):\s*$", re.M)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _resolve_template_path() -> Path:
    """Prefer the Gmail-round-tripped template, else fallback."""
    for p in GMAIL_TEMPLATE_CANDIDATES:
        if p.exists():
            return p
    return FALLBACK_TEMPLATE

def _html_escape(s: str) -> str:
    return html.escape(s, quote=True)

def _only_body(html_text: str) -> str:
    """
    Return only inner <body> HTML to avoid Word XML/CSS in <head> rendering as text.
    """
    m = re.search(r"<body\b[^>]*>(?P<body>[\s\S]*?)</body\s*>", html_text, flags=re.I)
    return m.group("body") if m else html_text

def _remove_broken_style_blocks(body_html: str) -> str:
    """
    Remove <style>...</style> blocks that contain <br> or look invalid.
    This is safe because we're relying on the Gmail/Word inline styles already.
    """
    out = []
    pos = 0
    while True:
        start = body_html.find("<style", pos)
        if start == -1:
            out.append(body_html[pos:])
            break
        out.append(body_html[pos:start])
        end = body_html.find("</style>", start)
        if end == -1:
            # No closing tag; drop the remainder (safer than showing CSS as text)
            break
        block = body_html[start:end + len("</style>")]
        if "<br" in block.lower() or "&lt;br" in block.lower():
            # Drop this broken style block
            pass
        else:
            # Keep valid style blocks
            out.append(block)
        pos = end + len("</style>")
    return "".join(out)

def _minify_inter_tag_whitespace(body_html: str) -> str:
    """
    Collapse pure inter-tag whitespace to keep Outlook from injecting odd gaps.
    """
    return re.sub(r">\s+<", "><", body_html)

def _tighten_outlook_whitespace(body_html: str) -> str:
    # Strip <o:p> wrappers commonly emitted by Word/Outlook
    body_html = re.sub(r"</?o:p>", "", body_html, flags=re.I)
    # Normalize multiple non-breaking spaces
    body_html = body_html.replace("&nbsp;&nbsp;", "&nbsp;")
    return body_html

def _parse_chamber_from_filename(fname: str) -> str:
    fn = fname.lower()
    if "legislative council" in fn or "lc_" in fn or "legislative_council" in fn:
        return "Legislative Council"
    return "House of Assembly"

def _load_keywords(path: Path) -> List[str]:
    kws: List[str] = []
    if not path.exists():
        return kws
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        kws.append(s)
    return kws

def _read_transcripts() -> List[Path]:
    files = sorted(Path(p) for p in glob.glob(str(TRANSCRIPT_DIR / "*.txt")))
    return [f for f in files if f.is_file()]

# ---------------------------------------------------------------------------
# Extraction (kept simple & robust; does not touch formatting/template)
# ---------------------------------------------------------------------------

@dataclass
class Match:
    keywords: Set[str]
    excerpt_html: str
    speaker: str
    line_numbers: List[int]
    start_idx: int
    end_idx: int

def _highlight(text: str, keywords: Iterable[str]) -> str:
    """
    Highlight keywords safely in HTML, case-insensitive, whole-word-ish.
    """
    esc = _html_escape(text)

    # Build a combined regex of all keywords, longest-first
    kws = sorted(set(k for k in keywords if k), key=len, reverse=True)
    if not kws:
        return esc

    def repl(m: re.Match) -> str:
        return f"<span style=\"background:#fff0a6\">{m.group(0)}</span>"

    # Use case-insensitive, word-boundary-ish (tolerate punctuation)
    pat = r"(?i)(?<![A-Za-z0-9])(" + "|".join(re.escape(k) for k in kws) + r")(?![A-Za-z0-9])"
    return re.sub(pat, repl, esc)

def _compute_line_number_indices(text: str) -> List[int]:
    """
    Return a list of indices where each line starts for fast line-number lookup.
    """
    starts = [0]
    for m in re.finditer(r"\n", text):
        starts.append(m.end())
    return starts

def _idx_to_line(starts: List[int], idx: int) -> int:
    # Binary search manually (avoid bisect import if not needed)
    lo, hi = 0, len(starts) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if starts[mid] <= idx:
            lo = mid + 1
        else:
            hi = mid - 1
    return hi + 1  # 1-based line number

def extract_matches(text: str, keywords: List[str]) -> List[Match]:
    """
    Minimal, robust matching:
    - Find occurrences of any keyword (case-insensitive, word-ish)
    - Try to attribute to a 'speaker:' line above the hit if present
    - Build an excerpt with ~±CONTEXT_CHARS around the hit (merged if overlapping)
    - Cap to MAX_SNIPPET_CHARS and highlight keywords
    """
    matches: List[Match] = []
    if not text or not keywords:
        return matches

    # Precompute regex for all keywords
    kw_pat = re.compile(
        r"(?i)(?<![A-Za-z0-9])(" + "|".join(re.escape(k) for k in keywords if k) + r")(?![A-Za-z0-9])"
    )

    # Find all hits
    hits = [m for m in kw_pat.finditer(text)]
    if not hits:
        return matches

    # Speaker headers, if present
    speaker_by_line: Dict[int, str] = {}
    for sm in SPEAKER_HEADER_RE.finditer(text):
        line_no = text.count("\n", 0, sm.start()) + 1
        speaker_by_line[line_no] = sm.group("speaker").strip()

    # Prepare line starts for mapping indices to line numbers
    line_starts = _compute_line_number_indices(text)

    # Merge nearby hits into windows and create excerpts
    windows: List[Tuple[int, int, Set[str]]] = []
    for h in hits:
        start = max(0, h.start() - CONTEXT_CHARS)
        end = min(len(text), h.end() + CONTEXT_CHARS)
        kws = {h.group(0)}
        # Merge with prior window if overlapping
        if windows and start <= windows[-1][1]:
            prev_s, prev_e, prev_k = windows[-1]
            windows[-1] = (prev_s, max(prev_e, end), prev_k.union(kws))
        else:
            windows.append((start, end, kws))

    for i, (s, e, kws) in enumerate(windows, start=1):
        raw = text[s:e]
        # Determine speaker (nearest header above start index)
        start_line = _idx_to_line(line_starts, s)
        speaker_lines = [ln for ln in speaker_by_line.keys() if ln <= start_line]
        speaker = speaker_by_line[max(speaker_lines)] if speaker_lines else ""

        # Bound size
        if len(raw) > MAX_SNIPPET_CHARS:
            mid = (e + s) // 2
            half = MAX_SNIPPET_CHARS // 2
            raw = text[max(0, mid - half):min(len(text), mid + half)]

        snippet = _highlight(raw, kws)
        # Line numbers covered by this window (simple: first line only, or a small range)
        first_line = _idx_to_line(line_starts, s)
        last_line = _idx_to_line(line_starts, e - 1)
        line_numbers = [first_line] if first_line == last_line else list(range(first_line, min(first_line + 3, last_line + 1)))

        matches.append(Match(keywords=set(k.strip() for k in kws if k.strip()),
                             excerpt_html=snippet,
                             speaker=speaker,
                             line_numbers=line_numbers,
                             start_idx=s,
                             end_idx=e))
    return matches

# ---------------------------------------------------------------------------
# Template-driven rendering
# ---------------------------------------------------------------------------

def build_digest_html(files: List[str], keywords: List[str]) -> Tuple[str, int, Dict[str, Dict[str, int]]]:
    """
    Render the final HTML by cloning structures from the template itself.
    We do NOT hard-code any visual HTML; we only replace placeholders.
    """
    template_path = _resolve_template_path()
    # Word/Outlook exports are often Windows-1252, but the Gmail copy may be UTF-8-ish.
    # Read permissively; we'll *send* as UTF-8 regardless.
    try:
        raw_html = template_path.read_text(encoding="windows-1252", errors="ignore")
    except Exception:
        raw_html = template_path.read_text(encoding="utf-8", errors="ignore")

    # Keep only <body> content
    body = _only_body(raw_html)

    # Remove style blocks that contain <br> (Word sometimes sprays these)
    body = _remove_broken_style_blocks(body)

    # Insert date
    run_date = datetime.now().strftime("%d %B %Y")
    body = re.sub(r"\[\s*DATE\s*\]", _html_escape(run_date), body, flags=re.I)

    # Pass 1: extract matches and keyword counts
    counts: Dict[str, Dict[str, int]] = {kw: {"House of Assembly": 0, "Legislative Council": 0} for kw in keywords}
    file_matches: List[Tuple[str, List[Match]]] = []
    total_matches = 0

    for f in files:
        text = Path(f).read_text(encoding="utf-8", errors="ignore")
        ms = extract_matches(text, keywords)
        if not ms:
            continue
        total_matches += len(ms)
        ch = _parse_chamber_from_filename(Path(f).name)
        for m in ms:
            for kw in m.keywords:
                # Map kw back to one of the original keywords for counting (case-insensitive)
                for target in keywords:
                    if kw.lower() == target.lower():
                        counts.setdefault(target, {"House of Assembly": 0, "Legislative Council": 0})
                        counts[target][ch] = counts[target].get(ch, 0) + 1
                        break
        file_matches.append((Path(f).name, ms))

    # Pass 2: detection table rows
    # Find a <tr> that contains [Keyword] — use that as our row template.
    m_kw = re.search(r"\[\s*Keyword\s*\]", body, flags=re.I)
    if m_kw:
        tr_start = body.rfind("<tr", 0, m_kw.start())
        tr_end = body.find("</tr>", m_kw.end())
        if tr_start != -1 and tr_end != -1:
            tr_end += len("</tr>")
            det_row_tpl = body[tr_start:tr_end]
            det_rows = []
            for kw in keywords:
                hoa = counts.get(kw, {}).get("House of Assembly", 0)
                lc = counts.get(kw, {}).get("Legislative Council", 0)
                total = hoa + lc
                row = det_row_tpl
                row = re.sub(r"\[\s*Keyword\s*\]", _html_escape(kw), row, flags=re.I)
                row = re.sub(r"\[\s*House\s+of\s+Assembly\s+count\s*\]", str(hoa), row, flags=re.I)
                row = re.sub(r"\[\s*Legislative\s+Council\s+count\s*\]", str(lc), row, flags=re.I)
                row = re.sub(r"\[\s*Total\s+count\s*\]", str(total), row, flags=re.I)
                det_rows.append(row)
            # Replace the first template row with all rows
            body = body.replace(det_row_tpl, "".join(det_rows), 1)

    # Pass 3: transcript "card" sections
    # Locate the outermost <table> that encloses [Transcript filename]
    m_tf = re.search(r"\[\s*Transcript\s+filename\s*\]", body, flags=re.I)
    if m_tf:
        # Find the start of enclosing <table>
        tbl_start = body.rfind("<table", 0, m_tf.start())
        if tbl_start != -1:
            # Find the matching closing </table> using a simple stack
            pos = tbl_start
            depth = 0
            end_tbl = None
            while pos < len(body):
                if body.startswith("<table", pos):
                    depth += 1
                    pos += 6
                    continue
                if body.startswith("</table>", pos):
                    depth -= 1
                    pos += 8
                    if depth == 0:
                        end_tbl = pos
                        break
                    continue
                pos += 1
            if end_tbl:
                card_template = body[tbl_start:end_tbl]

                # Inside the card, find the match sub-rows: the row that holds [Match #]
                # and the following row that holds the snippet placeholder (something with 'snippet' inside [ ])
                m_mn = re.search(r"\[\s*Match\s*#\s*\]", card_template, flags=re.I)
                # Find any [...] that contains 'snippet'
                m_sn = re.search(r"\[[^\]]*snippet[^\]]*\]", card_template, flags=re.I)
                if m_mn and m_sn:
                    # Extract the two rows that form one "match entry"
                    r1_start = card_template.rfind("<tr", 0, m_mn.start())
                    r1_end = card_template.find("</tr>", m_mn.end())
                    r2_start = card_template.rfind("<tr", 0, m_sn.start())
                    r2_end = card_template.find("</tr>", m_sn.end())
                    if -1 not in (r1_start, r1_end, r2_start, r2_end):
                        r1_end += len("</tr>")
                        r2_end += len("</tr>")
                        match_rows_tpl = card_template[r1_start:r2_end]
                        card_before = card_template[:r1_start]
                        card_after = card_template[r2_end:]

                        # Build one card per file, one duplicated set of rows per match
                        cards = []
                        for fname, ms in file_matches:
                            if not ms:
                                continue
                            rows = []
                            for idx, m in enumerate(ms, start=1):
                                row_html = match_rows_tpl
                                row_html = re.sub(r"\[\s*Match\s*#\s*\]", str(idx), row_html, flags=re.I)
                                row_html = re.sub(r"\[\s*SPEAKER\s+NAME\s*\]",
                                                  _html_escape(m.speaker) if m.speaker else "UNKNOWN",
                                                  row_html, flags=re.I)
                                # Allow [Line number(s)] with any whitespace
                                line_txt = "line " + ", ".join(str(n) for n in m.line_numbers) if len(m.line_numbers) == 1 \
                                           else "lines " + ", ".join(str(n) for n in m.line_numbers)
                                row_html = re.sub(r"\[\s*Line\s+number\(s\)\s*\]", _html_escape(line_txt),
                                                  row_html, flags=re.I)
                                # Replace the snippet placeholder (any [...] containing 'snippet')
                                row_html = re.sub(r"\[[^\]]*snippet[^\]]*\]", m.excerpt_html, row_html, flags=re.I)
                                rows.append(row_html)
                            card_html = card_before + "".join(rows) + card_after
                            card_html = re.sub(r"\[\s*Transcript\s+filename\s*\]", _html_escape(fname),
                                               card_html, flags=re.I)
                            card_html = re.sub(r"\[\s*Match\s+count\s*\]", str(len(ms)),
                                               card_html, flags=re.I)
                            cards.append(card_html)

                        # Replace the original template card with all cards
                        body = body.replace(card_template, "".join(cards), 1)

    # Final: small whitespace cleanup
    body = _tighten_outlook_whitespace(body)
    body = _minify_inter_tag_whitespace(body)

    # Wrap in minimal HTML shell (safe; doesn’t change visuals)
    final_html = f"<!DOCTYPE html><html><body>{body}</body></html>"
    return final_html, total_matches, counts

# ---------------------------------------------------------------------------
# Sender
# ---------------------------------------------------------------------------

def main() -> None:
    # Gather transcripts and keywords
    files = [str(p) for p in _read_transcripts()]
    keywords = _load_keywords(KEYWORDS_PATH)

    # Render HTML
    html_out, total_matches, counts = build_digest_html(files, keywords)

    # Prepare subject
    today = datetime.now().strftime("%d %b %Y")
    subject = f"Hansard Monitor — BETA — {today} ({total_matches} match{'es' if total_matches != 1 else ''})"

    # Recipients/credentials from env
    user = os.environ.get("EMAIL_USER")
    pwd = os.environ.get("EMAIL_PASS")
    to_raw = os.environ.get("EMAIL_TO", "")
    if not user or not pwd or not to_raw:
        raise SystemExit("EMAIL_USER, EMAIL_PASS, and EMAIL_TO must be set in env.")

    to_list = [addr.strip() for addr in re.split(r"[;,]", to_raw) if addr.strip()]

    # Send with yagmail; DO NOT pass custom 'encoding' (avoids 3.13 set_charset issues)
    yag = yagmail.SMTP(user, pwd)
    yag.send(
        to=to_list,
        subject=subject,
        contents=html_out,  # HTML body
        attachments=files,  # attach source transcripts if desired
    )

    # Log
    LOG_PATH.write_text(
        f"{datetime.utcnow().isoformat()}Z - sent to {', '.join(to_list)} - files {len(files)} - matches {total_matches}\n",
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()
