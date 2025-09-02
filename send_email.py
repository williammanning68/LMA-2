import os
import re
import glob
from pathlib import Path
from datetime import datetime
import html

import yagmail
import subprocess  # optional: only used if ATTRIB_WITH_LLM=1


# File that records which transcripts have already been emailed
LOG_FILE = Path("sent.log")

# --- Tunables ----------------------------------------------------------------
WINDOW_MERGE_GAP = 4        # merge windows if the gap between sentence windows <= 4 sentences
CONTEXT_BACK_LINES = 80     # lines to look back for LLM QC (if enabled)


# --- Helpers -----------------------------------------------------------------

def load_keywords():
    """Load keywords from keywords.txt or KEYWORDS env var, ignoring comments and stripping quotes."""
    if os.path.exists("keywords.txt"):
        kws = []
        with open("keywords.txt", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if s.startswith('"') and s.endswith('"') and len(s) >= 2:
                    s = s[1:-1]
                kws.append(s)
        return kws
    if "KEYWORDS" in os.environ:
        return [kw.strip().strip('"') for kw in os.environ["KEYWORDS"].split(",") if kw.strip()]
    return []


# --- Speaker header detection & guards ---------------------------------------

# Accept titles (case-insensitive) or ALL-CAPS surnames; require ":" or a dash after header
SPEAKER_HEADER_RE = re.compile(
    r"""
^
(?:
  # (A) Title + optional ALL-CAPS surname(s)
  (?P<title>(?i:Mr|Ms|Mrs|Miss|Hon|Dr|Prof|Professor|Premier|Madam\s+SPEAKER|The\s+SPEAKER|The\s+PRESIDENT|The\s+CLERK|Deputy\s+Speaker|Deputy\s+President))
  (?:[\s.]+(?P<name>[A-Z][A-Z'’\-]+(?:\s+[A-Z][A-Z'’\-]+){0,3}))?
 |
  # (B) ALL-CAPS name-only (e.g., DOW, WOODRUFF, O'BYRNE)
  (?P<name_only>[A-Z][A-Z'’\-]+(?:\s+[A-Z][A-Z'’\-]+){0,3})
)
(?:\s*\([^)]*\))?        # optional (Electorate—Portfolio)
\s*(?::|[-–—]\s)         # ":" OR " - " / "– " / "— "
""",
    re.VERBOSE,
)

# Lines that look like prose-with-colon; never treat them as headers (belt-and-braces)
CONTENT_COLON_RE = re.compile(r"^(Then|There|And|But|So|If|When|Now|Finally|First|Second|Third)\b", re.IGNORECASE)


def _canonicalize(s: str) -> str:
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(
        r"\b(Mr|Ms|Mrs|Miss|Hon|Dr|Prof|Professor|Premier|Madam SPEAKER|The SPEAKER|The PRESIDENT|The CLERK|Deputy Speaker|Deputy President)\b\.?",
        "",
        s,
        flags=re.I,
    )
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()


def _speaker_map_by_line(text: str):
    """
    Map each line index -> current speaker (persisting until the next real header).
    We treat only genuine headers as speaker boundaries; everything else inherits.
    """
    lines = text.splitlines()
    curr = None
    mapping = []
    for ln in lines:
        s = ln.strip()
        m = SPEAKER_HEADER_RE.match(s)
        # Guard: e.g., "There are the projects:" must NOT be a header
        if m and m.group("name_only") and CONTENT_COLON_RE.match(s):
            m = None

        if m:
            title = (m.group("title") or "").strip()
            name = (m.group("name") or m.group("name_only") or "").strip()
            curr = " ".join(x for x in (title, name) if x) or curr
            mapping.append(curr)
            continue

        mapping.append(curr)
    return lines, mapping


def _nearest_speaker_above(mapping, idx):
    """Walk upward to the closest previous non-None speaker."""
    for i in range(idx, -1, -1):
        if mapping[i]:
            return mapping[i]
    return None


# --- Sentence tools -----------------------------------------------------------

_SENT_SPLIT_RE = re.compile(r"(?<=[\.\?\!])\s+")

def _split_sentences_with_spans(text: str):
    """
    Return list of (start, end) spans for sentences in text.
    We split on ., ?, ! followed by whitespace; any trailing text is a sentence.
    """
    spans = []
    pos = 0
    for m in _SENT_SPLIT_RE.finditer(text):
        end = m.start() + 1  # include the punctuation
        spans.append((pos, end))
        pos = m.end()
    if pos < len(text):
        spans.append((pos, len(text)))
    return spans

def _charpos_to_line(block_lines, line_offsets, pos):
    """
    Map a character position in block_text back to the (1-based) original file line number.
    """
    # binary search over line_offsets
    lo, hi = 0, len(line_offsets) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        start = line_offsets[mid]
        if mid + 1 < len(line_offsets):
            nxt = line_offsets[mid + 1]
        else:
            nxt = line_offsets[mid] + len(block_lines[mid]) + 1  # include trailing newline
        if start <= pos < nxt:
            return mid  # index within block_lines
        if pos < start:
            hi = mid - 1
        else:
            lo = mid + 1
    return max(0, min(len(block_lines) - 1, lo))


def _build_blocks(lines, line_speaker):
    """
    Yield blocks: dict with keys:
      speaker, start_line_idx (global), end_line_idx (inclusive), block_lines, block_text, line_offsets
    """
    blocks = []
    if not lines:
        return blocks
    cur_spk = line_speaker[0]
    start = 0
    for i in range(1, len(lines) + 1):
        if i == len(lines) or line_speaker[i] != cur_spk:
            block_lines = lines[start:i]
            block_text = "\n".join(block_lines)
            # offsets of each line's start (in chars) within block_text
