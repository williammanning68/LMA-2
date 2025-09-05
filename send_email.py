#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
from pathlib import Path
from datetime import datetime
from bisect import bisect_right
import yagmail

# Silence cssutils warnings that arise during premailer CSS parsing.  Without
# this the logs become extremely noisy when Word‑generated styles contain
# line breaks or other unexpected characters.  If cssutils is not available
# (e.g. when using a pure smtplib sender), the import will fail silently.
try:
    import logging as _logging  # type: ignore
    import cssutils as _cssutils  # type: ignore
    _cssutils.log.setLevel(_logging.FATAL)
except Exception:
    pass

# =============================================================================
# Template resolution (robust)
# =============================================================================

def _resolve_template_path() -> Path:
    # 1) Allow env override
    env_path = os.environ.get("TEMPLATE_HTML_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    # 2) Common names (what you've used)
    script_dir = Path(__file__).resolve().parent
    candidates = [
        "email_template.html",
        "email_template.htm",
        "email_template (1).html",
        "email_template (1).htm",
        "Hansard Monitor - Email Format - Version 3.htm",
        "Hansard Monitor - Email Format - Version 3.html",
        "templates/email_template.html",
        "templates/email_template.htm",
        "templates/email_template (1).html",
        "templates/email_template (1).htm",
        "templates/Hansard Monitor - Email Format - Version 3.htm",
        "templates/Hansard Monitor - Email Format - Version 3.html",
    ]
    for name in candidates:
        p = script_dir / name
        if p.exists():
            return p

    # 3) Fallback: find any plausible .htm(l)
    for pat in ("**/*.htm", "**/*.html"):
        for fp in script_dir.glob(pat):
            if any(k in fp.name.lower() for k in ("email", "template", "hansard", "format")):
                return fp
    for pat in ("**/*.htm", "**/*.html"):
        for fp in script_dir.glob(pat):
            return fp

    raise FileNotFoundError(
        "HTML template not found. Set TEMPLATE_HTML_PATH or place the template "
        "next to send_email.py (e.g., 'Hansard Monitor - Email Format - Version 3.htm')."
    )

TEMPLATE_HTML_PATH = _resolve_template_path()

# =============================================================================
# Config
# =============================================================================

LOG_FILE = Path("sent.log")
DEFAULT_TITLE = "Hansard Monitor – BETA Version 18.3"

# excerpt/windowing
MAX_SNIPPET_CHARS = 800
WINDOW_PAD_SENTENCES = 1
FIRST_SENT_FOLLOWING = 2
MERGE_IF_GAP_GT = 2

# =============================================================================
# Keyword loading
# =============================================================================

def load_keywords():
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

# =============================================================================
# Transcript segmentation (utterances)
# =============================================================================

SPEAKER_HEADER_RE = re.compile(
    r"""
    ^
    (?:
        (?P<title>(?i:Mr|Ms|Mrs|Miss|Hon|Dr|Prof|Professor|Premier|Madam\s+SPEAKER|The\s+SPEAKER|The\s+PRESIDENT|The\s+CLERK|Deputy\s+Speaker|Deputy\s+President))
        (?:[\s.]+(?P<name>[A-Z][A-Z'’\-]+(?:\s+[A-Z][A-Z'’\-]+){0,3}))?
      |
        (?P<name_only>[A-Z][A-Z'’\-]+(?:\s+[A-Z][A-Z'’\-]+){0,3})
    )
    (?:\s*\([^)]*\))?
    \s*(?::|[-–—]\s)
    """,
    re.VERBOSE,
)
CONTENT_COLON_RE   = re.compile(r"^(Then|There|And|But|So|If|When|Now|Finally|First|Second|Third)\b", re.I)
TIME_STAMP_RE      = re.compile(r"^\[\d{1,2}\.\d{2}\s*(a|p)\.m\.\]$", re.I)
UPPER_HEADING_RE   = re.compile(r"^[A-Z][A-Z\s’'—\-&,;:.()]+$")
INTERJECTION_RE    = re.compile(r"^(Members interjecting\.|The House suspended .+)$", re.I)

def _build_utterances(text: str):
    all_lines = text.splitlines()
    utterances = []
    curr = {"speaker": None, "lines": [], "line_nums": []}

    def flush():
        if not curr["lines"]:
            return
        joined = "\n".join(curr["lines"])
        offs, total = [], 0
        for i, ln in enumerate(curr["lines"]):
            offs.append(total)
            total += len(ln) + (1 if i < len(curr["lines"]) - 1 else 0)

        # very simple sentence segmentation: split on punctuation + whitespace
        sents, start = [], 0
        for m in re.finditer(r"(?<=[\.!\?])\s+", joined):
            end = m.start()
            if end > start:
                sents.append((start, end))
            start = m.end()
        if start < len(joined):
            sents.append((start, len(joined)))

        utterances.append({
            "speaker": curr["speaker"],
            "lines": curr["lines"][:],
            "line_nums": curr["line_nums"][:],
            "joined": joined,
            "line_offsets": offs,
            "sents": sents
        })

    for idx, raw in enumerate(all_lines):
        s = raw.strip()
        if not s or TIME_STAMP_RE.match(s) or UPPER_HEADING_RE.match(s) or INTERJECTION_RE.match(s):
            continue
        m = SPEAKER_HEADER_RE.match(s)
        if m and not (m.group("name_only") and CONTENT_COLON_RE.match(s)):
            flush()
            title = (m.group("title") or "").strip()
            name  = (m.group("name") or m.group("name_only") or "").strip()
            curr  = {"speaker": " ".join(x for x in (title, name) if x).strip(), "lines": [], "line_nums": []}
            continue
        curr["lines"].append(raw.rstrip())
        curr["line_nums"].append(idx + 1)

    flush()
    return utterances, all_lines

def _line_for_char_offset(line_offsets, line_nums, pos):
    i = bisect_right(line_offsets, pos) - 1
    i = max(0, min(i, len(line_nums) - 1))
    return line_nums[i]

# =============================================================================
# Matching and excerpt building
# =============================================================================

def _compile_kw_patterns(keywords):
    pats = []
    for kw in sorted(keywords, key=len, reverse=True):
        pats.append((kw, re.compile(re.escape(kw) if " " in kw else rf"\b{re.escape(kw)}\b", re.I)))
    return pats

def _collect_hits_in_utterance(utt, kw_pats):
    hits, joined = [], utt["joined"]
    for si, (a, b) in enumerate(utt["sents"]):
        seg = joined[a:b]
        for kw, pat in kw_pats:
            m = pat.search(seg)
            if not m:
                continue
            char_pos = a + m.start()
            line_no  = _line_for_char_offset(utt["line_offsets"], utt["line_nums"], char_pos)
            hits.append({"kw": kw, "sent_idx": si, "line_no": line_no})
    return hits

def _windows_for_hits(hits, sent_count):
    wins = []
    for h in hits:
        j = h["sent_idx"]
        if j == 0:
            start = 0
            end   = min(sent_count - 1, FIRST_SENT_FOLLOWING)
        else:
            start = max(0, j - WINDOW_PAD_SENTENCES)
            end   = min(sent_count - 1, j + WINDOW_PAD_SENTENCES)
        wins.append([start, end, {h["kw"]}, {h["line_no"]}])
    wins.sort(key=lambda w: (w[0], w[1]))
    return wins

def _dedup_windows(wins):
    if not wins:
        return []
    bucket = {}
    for s, e, kws, lines in wins:
        key = (s, e)
        if key in bucket:
            bucket[key][0] |= kws
            bucket[key][1] |= lines
        else:
            bucket[key] = [set(kws), set(lines)]
    return [[s, e, bucket[(s, e)][0], bucket[(s, e)][1]] for (s, e) in sorted(bucket.keys())]

def _merge_windows_far_only(wins, gap_gt=MERGE_IF_GAP_GT):
    if not wins:
        return []
    merged = [wins[0]]
    for s, e, kws, lines in wins[1:]:
        ps, pe, pk, pl = merged[-1]
        gap = s - pe
        if gap > gap_gt:
            merged[-1] = [ps, max(pe, e), pk | kws, pl | lines]
        else:
            merged.append([s, e, kws, lines])
    return merged

def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def _highlight_keywords_html(text_html: str, keywords: list[str]) -> str:
    out = text_html
    for kw in sorted(keywords, key=len, reverse=True):
        pat = re.compile(re.escape(_html_escape(kw)) if " " in kw else rf"\b{re.escape(_html_escape(kw))}\b", re.I)
        out = pat.sub(lambda m: "<b><span style='background:lightgrey;mso-highlight:lightgrey'>" +
                                 m.group(0) + "</span></b>", out)
    return out

def _excerpt_from_window_html(utt, win, keywords):
    sents  = utt["sents"]
    joined = utt["joined"]
    start, end, kws, lines = win
    a = sents[start][0]
    b = sents[end][1]

    # Normalize newlines to avoid stacked <br>
    raw = joined[a:b]
    raw = re.sub(r"\r\n?", "\n", raw)   # unify CRLF/CR
    raw = re.sub(r"\n{2,}", "\n", raw)  # collapse blank lines
    raw = raw.strip()

    if len(raw) > MAX_SNIPPET_CHARS:
        raw = raw[:MAX_SNIPPET_CHARS].rstrip() + "…"

    html = _html_escape(raw)
    html = _highlight_keywords_html(html, keywords)
    html = html.replace("\n", "<br>")
    html = re.sub(r"(?:<br\s*/?>\s*){2,}", "<br>", html)                 # collapse runs
    html = re.sub(r"^(?:<br\s*/?>\s*)+|(?:<br\s*/?>\s*)+$", "", html)    # trim leading/trailing

    start_line = _line_for_char_offset(utt["line_offsets"], utt["line_nums"], a)
    end_line   = _line_for_char_offset(utt["line_offsets"], utt["line_nums"], max(a, b - 1))
    return html, sorted(lines), sorted(kws, key=str.lower), start_line, end_line

def extract_matches(text: str, keywords):
    utts, _all_lines = _build_utterances(text)
    kw_pats = _compile_kw_patterns(keywords)
    results = []
    for utt in utts:
        speaker = utt["speaker"]
        if not utt["lines"]:
            continue
        hits = _collect_hits_in_utterance(utt, kw_pats)
        if not hits:
            continue
        wins   = _windows_for_hits(hits, sent_count=len(utt["sents"]))
        wins   = _dedup_windows(wins)
        merged = _merge_windows_far_only(wins, gap_gt=MERGE_IF_GAP_GT)
        for win in merged:
            excerpt_html, line_list, kws_in_excerpt, win_start, win_end = _excerpt_from_window_html(utt, win, keywords)
            results.append((set(kws_in_excerpt), excerpt_html, speaker, line_list, win_start, win_end))
    return results

# =============================================================================
# Outlook/Gmail whitespace fixes
# =============================================================================

_EMPTY_MSOP_RE = re.compile(
    r"<p\b[^>]*>(?:\s|&nbsp;|<br[^>]*>|"
    r"<o:p>\s*&nbsp;\s*</o:p>|"
    r"<span\b[^>]*>(?:\s|&nbsp;|<br[^>]*>)*</span>)*</p>",
    re.I,
)

def _tighten_outlook_whitespace(html: str) -> str:
    # 1) Remove empty Word/Outlook paragraphs (even if wrapped)
    html = _EMPTY_MSOP_RE.sub("", html)
    # 2) Collapse runs of <br> to a single <br>
    html = re.sub(r"(?:\s*<br[^>]*>\s*){2,}", "<br>", html, flags=re.I)
    # 3) Remove whitespace/comments between adjacent tables
    html = re.sub(r"(</table>)\s+(?=(?:<!--.*?-->\s*)*<table\b)", r"\1", html, flags=re.I | re.S)
    # 4) Trim blank space just inside table cells
    html = re.sub(r">\s*(?:&nbsp;|<br[^>]*>|\s)+</td>", "></td>", html, flags=re.I)
    return html

def _minify_inter_tag_whitespace(html: str) -> str:
    # Critical for Outlook: remove inter-tag newlines/indentation
    return re.sub(r">\s+<", "><", html)

def _inject_mso_css_reset(html: str) -> str:
    # MSO conditional CSS to kill default MsoNormal margins/line-height
    mso_block = (
        "<!--[if mso]>"
        "<style>"
        "p.MsoNormal,div.MsoNormal,li.MsoNormal{margin:0 !important;line-height:normal !important;}"
        "table,td{mso-table-lspace:0pt !important;mso-table-rspace:0pt !important;mso-line-height-rule:exactly !important;}"
        "</style>"
        "<![endif]-->"
    )
    # Insert before </head> if possible; else prepend
    if re.search(r"</head\s*>", html, re.I):
        return re.sub(r"</head\s*>", mso_block + "</head>", html, flags=re.I, count=1)
    return mso_block + html

# =============================================================================
# Template operations (summary & sections)
# =============================================================================

def _build_detection_row(kw, hoa, lc, tot) -> str:
    # Use pixel paddings; margin:0 paragraphs
    return (
        "<tr>"
        "<td width=\"28%\" style='border-top:none;border-left:solid #D8DCE0 1px;border-bottom:solid #ECF0F1 1px;border-right:none;padding:8px 10px;'>"
        f"<p class=MsoNormal style='margin:0;'><b><span style='font-size:10pt;font-family:\"Segoe UI\",sans-serif;color:black'>{_html_escape(kw)}</span></b></p></td>"
        "<td width=\"28%\" style='border-bottom:solid #ECF0F1 1px;padding:8px 10px;'>"
        f"<p class=MsoNormal align=center style='text-align:center;margin:0;'><b><span style='font-size:10pt;font-family:\"Segoe UI\",sans-serif;color:black'>{hoa}</span></b></p></td>"
        "<td width=\"28%\" style='border-bottom:solid #ECF0F1 1px;padding:8px 10px;'>"
        f"<p class=MsoNormal align=center style='text-align:center;margin:0;'><b><span style='font-size:10pt;font-family:\"Segoe UI\",sans-serif;color:black'>{lc}</span></b></p></td>"
        "<td width=\"15%\" style='border-bottom:solid #ECF0F1 1px;border-right:solid #D8DCE0 1px;padding:8px 10px;'>"
        f"<p class=MsoNormal align=center style='text-align:center;margin:0;'><b><span style='font-size:10pt;font-family:\"Segoe UI\",sans-serif;color:black'>{tot}</span></b></p></td>"
        "</tr>"
    )

def _replace_detection_rows_in_template(html, row_html):
    # Find "Detection Match by Chamber" then the next inner table, keep header row, replace the rest.
    hdr = re.search(r"Detection\s+Match\s+by\s+Chamber", html, flags=re.I)
    if not hdr:
        return html
    m_table_start = re.search(r"<table[^>]*>", html[hdr.end():], flags=re.I | re.S)
    if not m_table_start:
        return html
    tbl_start = hdr.end() + m_table_start.start()
    m_table_end = re.search(r"</table\s*>", html[tbl_start:], flags=re.I | re.S)
    if not m_table_end:
        return html
    tbl_end = tbl_start + m_table_end.end()

    table_html = html[tbl_start:tbl_end]
    m_header_row = re.search(r"<tr[^>]*>.*?Keyword.*?</tr\s*>", table_html, flags=re.I | re.S)
    if not m_header_row:
        return html
    before = table_html[:m_header_row.end()]
    new_table = before + row_html + "</table>"
    return html[:tbl_start] + new_table + html[tbl_end:]

def _strip_sample_section(html):
    # Remove the sample block marked in the template
    pattern = re.compile(r"<!--\s*Sample section to be replaced\s*-->.*?<!--\s*End sample section\s*-->", re.I | re.S)
    return re.sub(pattern, "", html)

def _inject_sections_after_detection(html, sections_html):
    hdr = re.search(r"Detection\s+Match\s+by\s+Chamber", html, flags=re.I)
    if not hdr:
        return html + sections_html
    m_table_start = re.search(r"<table[^>]*>", html[hdr.end():], flags=re.I | re.S)
    if not m_table_start:
        return html + sections_html
    tbl_start = hdr.end() + m_table_start.start()
    m_table_end = re.search(r"</table\s*>", html[tbl_start:], flags=re.I | re.S)
    if not m_table_end:
        return html + sections_html
    insert_at = tbl_start + m_table_end.end()
    return html[:insert_at] + sections_html + html[insert_at:]

# =============================================================================
# Per-file sections (“cards”) — Outlook-safe, tight
# =============================================================================

def _build_file_section_html(filename: str, matches):
    esc = _html_escape
    cards = []

    for idx, (_kw_set, excerpt_html, speaker, line_list, _s, _e) in enumerate(matches, 1):
        line_txt = f"line {line_list[0]}" if len(line_list) == 1 else "lines " + ", ".join(str(n) for n in line_list)

        # Card header + body (top-aligned, pixel line-heights)
        card = (
            "<table role='presentation' width='100%' cellpadding='0' cellspacing='0' border='0' "
            "style='border-collapse:collapse;border:1px solid #D8DCE0;'>"
            "<tr>"
            "<td valign='top' style='background:#ECF0F1;border-bottom:1px solid #D8DCE0;padding:4px 8px;"
            "font-size:0;line-height:0;mso-line-height-rule:exactly;vertical-align:top;'>"
              "<table role='presentation' width='100%' cellpadding='0' cellspacing='0' border='0' style='border-collapse:collapse;'>"
              "<tr>"
                "<td width='32' align='center' valign='top' style='background:#4A5A6A;border:0;height:18px;vertical-align:top;'>"
                  "<div style=\"font:bold 10pt 'Segoe UI',sans-serif;color:#FFFFFF;line-height:18px;mso-line-height-rule:exactly;display:block;\">"
                  f"{idx}</div>"
                "</td>"
                "<td width='8' style='font-size:0;line-height:0;vertical-align:top;'>&nbsp;</td>"
                "<td valign='top' style='vertical-align:top;'>"
                  "<div style=\"font:bold 10pt 'Segoe UI',sans-serif;color:#24313F;text-transform:uppercase;"
                  "line-height:15px;mso-line-height-rule:exactly;display:block;\">"
                  f"{esc(speaker) if speaker else 'UNKNOWN'}</div>"
                "</td>"
                "<td align='right' valign='top' style='vertical-align:top;'>"
                  "<div style=\"font:10pt 'Segoe UI',sans-serif;color:#6A7682;line-height:15px;mso-line-height-rule:exactly;display:block;\">"
                  f"{line_txt}</div>"
                "</td>"
              "</tr>"
              "</table>"
            "</td>"
            "</tr>"
            "<tr>"
            "<td valign='top' style='padding:6px 8px;vertical-align:top;'>"
              "<div style=\"font:10pt 'Segoe UI',sans-serif;color:#1F2A36;line-height:16px;mso-line-height-rule:exactly;display:block;\">"
              f"{excerpt_html}</div>"
            "</td>"
            "</tr>"
            "</table>"
        )
        cards.append(card)

    # 2px spacer BETWEEN cards (none after the last)
    spacer = ("<table role='presentation' width='100%' cellpadding='0' cellspacing='0' border='0'>"
              "<tr><td style='height:2px;line-height:2px;font-size:0;'>&nbsp;</td></tr></table>")
    cards_html = spacer.join(cards)

    section = (
        "<table role='presentation' width='100%' cellpadding='0' cellspacing='0' border='0' style='border-collapse:collapse;'>"
        "<tr>"
        "<td style='border-left:3px solid #C5A572;background:#F7F9FA;padding:6px 10px;'>"
        f"<div style=\"font:bold 10pt 'Segoe UI',sans-serif;color:#000;line-height:15px;mso-line-height-rule:exactly;display:block;\">{esc(filename)}</div>"
        f"<div style=\"font:10pt 'Segoe UI',sans-serif;color:#000;line-height:15px;mso-line-height-rule:exactly;display:block;\">{len(matches)} match(es)</div>"
        "</td>"
        "</tr>"
        "<tr>"
        "<td style='border:1px solid #D8DCE0;border-top:none;background:#FFFFFF;padding:6px 8px;'>"
        f"{cards_html}"
        "</td>"
        "</tr>"
        "</table>"
    )
    return section

# =============================================================================
# Build the full HTML
#
# The goal of this implementation is to avoid embedding any visual HTML
# directly in Python.  Instead we treat the shipped Word/Outlook template
# (``email_template.htm`` or another override) as the single source of
# presentation markup.  The placeholders defined in the template (e.g.
# ``[Keyword]``, ``[House of Assembly count]``, ``[Transcript filename]``,
# etc.) are replaced and cloned at runtime to produce the final digest.
#
# This function performs the following high level steps:
#
# 1. Read the template into a string, falling back from Windows‑1252 to
#    UTF‑8 as necessary.  Substitute the current date into the ``[DATE]``
#    placeholder.
# 2. Parse the provided transcript files and compute keyword hit counts
#    for each chamber as well as detailed match records for each file.
# 3. Locate the keyword row template (the table row containing
#    ``[Keyword]`` and the count placeholders) and duplicate it once per
#    keyword, substituting each of the count values.
# 4. Locate the transcript card template (the outermost table
#    containing ``[Transcript filename]``) and, within that table,
#    locate the two row match template (the rows containing
#    ``[Match #]`` and the snippet placeholder).  For each transcript
#    file we produce a copy of the card with the appropriate filename
#    and match count, duplicating the match rows for each match in that
#    file.
# 5. Replace the single keyword row and transcript card in the template
#    with all of the generated rows/cards.  Finally apply minor
#    whitespace fixups to accommodate Outlook quirks.

def _parse_chamber_from_filename(filename: str) -> str:
    """
    Infer the parliamentary chamber from the transcript filename.  The
    repository convention prefixes files with ``House_of_Assembly`` or
    ``Legislative_Council``; fall back to "Unknown" otherwise.
    """
    lower = filename.lower()
    if "house_of_assembly" in lower:
        return "House of Assembly"
    if "legislative_council" in lower:
        return "Legislative Council"
    return "Unknown"


def build_digest_html(files: list[str], keywords: list[str]) -> tuple[str, int, dict[str, dict[str, int]]]:
    """
    Build the HTML digest by reading the email template once and
    injecting dynamic data into placeholder slots.  This function does
    **not** construct any presentation markup itself; it simply clones
    structures from the template.

    Parameters
    ----------
    files : list[str]
        Paths to transcript text files to include in the digest.
    keywords : list[str]
        List of keywords to search for within the transcripts.

    Returns
    -------
    html : str
        The fully rendered HTML email body.
    total_matches : int
        The total number of keyword hits across all provided transcripts.
    counts : dict[str, dict[str, int]]
        A mapping of each keyword to per‑chamber hit counts.
    """
    # ------------------------------------------------------------------
    # 1. Load the template and insert the current date
    # ------------------------------------------------------------------
    try:
        raw_html = TEMPLATE_HTML_PATH.read_text(encoding="windows-1252", errors="ignore")
    except Exception:
        raw_html = TEMPLATE_HTML_PATH.read_text(encoding="utf-8", errors="ignore")

    # Extract only the content between <body> and </body> to avoid sending
    # Word/Outlook <head> and style definitions that often leak into the
    # rendered email.  If no <body> tags are found we fall back to the
    # entire document.
    m_body = re.search(r"<body\b[^>]*>(?P<body>[\s\S]*?)</body\s*>", raw_html, flags=re.I)
    template_html = m_body.group("body") if m_body else raw_html

    # Remove any <style> blocks that contain '<br>' or '&lt;br' which are
    # often introduced by Word and cause cssutils to throw parse errors.
    def _remove_broken_style_blocks(html: str) -> str:
        out, pos = [], 0
        while True:
            start = html.find("<style", pos)
            if start == -1:
                out.append(html[pos:])
                break
            out.append(html[pos:start])
            end = html.find("</style>", start)
            if end == -1:
                # unmatched style: drop remainder
                break
            block = html[start:end + len("</style>")]
            # remove block if it contains <br> or &lt;br
            if "<br" in block.lower() or "&lt;br" in block.lower():
                # skip this block
                pass
            else:
                out.append(block)
            pos = end + len("</style>")
        return "".join(out)

    template_html = _remove_broken_style_blocks(template_html)

    # Substitute the run date into the [DATE] placeholder.
    run_date = datetime.now().strftime("%d %B %Y")
    template_html = re.sub(r"\[\s*DATE\s*\]", _html_escape(run_date), template_html, flags=re.I)

    # ------------------------------------------------------------------
    # 2. Collect matches and per‑keyword/chamber counts
    # ------------------------------------------------------------------
    counts: dict[str, dict[str, int]] = {kw: {"House of Assembly": 0, "Legislative Council": 0} for kw in keywords}
    file_matches: list[tuple[str, list[tuple[set[str], str, str, list[int], int, int]]]] = []
    total_matches = 0

    for fpath in files:
        text = Path(fpath).read_text(encoding="utf-8", errors="ignore")
        matches = extract_matches(text, keywords)
        if not matches:
            continue
        total_matches += len(matches)
        chamber = _parse_chamber_from_filename(Path(fpath).name)
        for kw_set, _excerpt_html, _speaker, _lines, _s, _e in matches:
            for kw in kw_set:
                # ensure counts mapping exists even if keyword not originally provided
                counts.setdefault(kw, {"House of Assembly": 0, "Legislative Council": 0})
                if chamber in counts[kw]:
                    counts[kw][chamber] += 1
        file_matches.append((Path(fpath).name, matches))

    # ------------------------------------------------------------------
    # 3. Duplicate the detection match row (one row per keyword)
    # ------------------------------------------------------------------
    # Find the <tr> element in the template that contains [Keyword]; treat
    # this row as the pattern for keyword counts.  We use a simple
    # substring search to locate the placeholder and then scan outward
    # to encompass the entire <tr>...</tr> block.
    m_kw = re.search(r"\[\s*Keyword\s*\]", template_html, flags=re.I)
    if m_kw:
        # Locate the start of the containing <tr>
        tr_start = template_html.rfind("<tr", 0, m_kw.start())
        # Locate the end of the </tr>
        tr_end = template_html.find("</tr", m_kw.end())
        if tr_start != -1 and tr_end != -1:
            tr_end = template_html.find(">", tr_end)  # end of closing tag
            if tr_end != -1:
                # Extract the original row template
                row_template = template_html[tr_start:tr_end + 1]
                # Build new rows by replacing placeholders
                rows_html_parts: list[str] = []
                for kw in keywords:
                    hoa = counts.get(kw, {}).get("House of Assembly", 0)
                    lc = counts.get(kw, {}).get("Legislative Council", 0)
                    total = hoa + lc
                    row_html = row_template
                    row_html = re.sub(r"\[\s*Keyword\s*\]", _html_escape(kw), row_html, flags=re.I)
                    row_html = re.sub(r"\[\s*House\s+of\s+Assembly\s+count\s*\]", str(hoa), row_html, flags=re.I)
                    row_html = re.sub(r"\[\s*Legislative\s+Council\s+count\s*\]", str(lc), row_html, flags=re.I)
                    row_html = re.sub(r"\[\s*Total\s+count\s*\]", str(total), row_html, flags=re.I)
                    rows_html_parts.append(row_html)
                detection_rows_html = "".join(rows_html_parts)
                # Replace the first occurrence of the template row with all generated rows
                template_html = template_html.replace(row_template, detection_rows_html, 1)

    # ------------------------------------------------------------------
    # 4. Duplicate transcript sections (cards)
    # ------------------------------------------------------------------
    # Find the first [Transcript filename] placeholder
    m_file = re.search(r"\[\s*Transcript\s+filename\s*\]", template_html, flags=re.I)
    if m_file:
        idx = m_file.start()
        # Scan backwards to find the start of the enclosing <table>
        table_start = template_html.rfind("<table", 0, idx)
        # Scan forward to find the matching closing </table> tag using a simple stack
        pos = table_start
        depth = 0
        end_idx = None
        while pos < len(template_html):
            if template_html.startswith("<table", pos):
                depth += 1
                pos += 6
                continue
            if template_html.startswith("</table", pos):
                depth -= 1
                pos += 7  # skip "</table"
                # find the end of this closing tag
                close_gt = template_html.find(">", pos)
                if close_gt != -1:
                    pos = close_gt + 1
                if depth == 0:
                    end_idx = pos
                    break
                continue
            pos += 1

        if table_start != -1 and end_idx is not None:
            card_template = template_html[table_start:end_idx]
            # Within card template find the row containing [Match #]
            match_header_match = re.search(r"\[\s*Match\s*#\s*\]", card_template, flags=re.I)
            # Find the snippet placeholder (it might contain markup, but always within brackets containing the word 'snippet')
            snippet_match = re.search(r"\[[^\]]*snippet[^\]]*\]", card_template, flags=re.I)
            if match_header_match and snippet_match:
                # Find the <tr> boundaries for the two rows that make up one match entry
                r1_start = card_template.rfind("<tr", 0, match_header_match.start())
                r1_end   = card_template.find("</tr", match_header_match.end())
                if r1_end != -1:
                    # end index of closing tag
                    r1_end = card_template.find(">", r1_end) + 1
                r2_start = card_template.rfind("<tr", 0, snippet_match.start())
                r2_end   = card_template.find("</tr", snippet_match.end())
                if r2_end != -1:
                    r2_end = card_template.find(">", r2_end) + 1
                # Extract the match rows template
                match_rows_template = card_template[r1_start:r2_end]
                # Split card template into prefix, match rows placeholder, and suffix
                card_before = card_template[:r1_start]
                card_after  = card_template[r2_end:]
                # Build all cards
                cards_html_parts: list[str] = []
                for fname, matches in file_matches:
                    if not matches:
                        continue
                    match_rows_html_parts: list[str] = []
                    for i, (_kw_set, excerpt_html, speaker, line_list, _s, _e) in enumerate(matches, 1):
                        row_html = match_rows_template
                        row_html = re.sub(r"\[\s*Match\s*#\s*\]", str(i), row_html, flags=re.I)
                        row_html = re.sub(r"\[\s*SPEAKER\s+NAME\s*\]", _html_escape(speaker) if speaker else "UNKNOWN", row_html, flags=re.I)
                        # Line number(s) placeholder; support singular or plural text
                        if line_list:
                            line_txt = f"line {line_list[0]}" if len(line_list) == 1 else "lines " + ", ".join(str(n) for n in line_list)
                        else:
                            line_txt = ""
                        row_html = re.sub(r"\[\s*Line\s+number\(s\)\s*\]", _html_escape(line_txt), row_html, flags=re.I)
                        # Replace the snippet/excerpt placeholder; Word sometimes introduces additional markup inside []
                        row_html = re.sub(r"\[[^\]]*snippet[^\]]*\]", excerpt_html, row_html, flags=re.I)
                        match_rows_html_parts.append(row_html)
                    # Assemble card
                    card_html = card_before + "".join(match_rows_html_parts) + card_after
                    card_html = re.sub(r"\[\s*Transcript\s+filename\s*\]", _html_escape(fname), card_html, flags=re.I)
                    card_html = re.sub(r"\[\s*Match\s+count\s*\]", str(len(matches)), card_html, flags=re.I)
                    cards_html_parts.append(card_html)
                # Replace the single card in the original template with all generated cards
                template_html = template_html.replace(card_template, "".join(cards_html_parts), 1)

    # ------------------------------------------------------------------
    # 5. Final whitespace cleanup for Outlook
    # ------------------------------------------------------------------
    template_html = _tighten_outlook_whitespace(template_html)
    template_html = _minify_inter_tag_whitespace(template_html)
    # Wrap in minimal HTML shell for sending; Outlook requires a root element
    full_html = f"<!DOCTYPE html><html><body>{template_html}</body></html>"
    return full_html, total_matches, counts

# =============================================================================
# Sent-log helpers
# =============================================================================

def load_sent_log() -> set[str]:
    if LOG_FILE.exists():
        with open(LOG_FILE, encoding="utf-8") as f:
            return {ln.strip() for ln in f if ln.strip()}
    return set()

def update_sent_log(files: list[str]):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        for fp in files:
            f.write(Path(fp).name + "\n")

# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """
    Entry point for sending the digest email.  This function loads
    keywords and unsent transcripts, renders the HTML using
    ``build_digest_html``, assembles a multipart message with
    attachments and sends it using the standard library ``smtplib``.
    Using ``smtplib`` avoids any implicit HTML processing (e.g.
    CSS inlining) that third‑party mailers perform, preserving the
    Word/Outlook template verbatim and preventing stray CSS parse
    warnings.  Sent transcripts are appended to the sent log to
    prevent repeat emails.
    """
    EMAIL_USER = os.environ.get("EMAIL_USER")
    EMAIL_PASS = os.environ.get("EMAIL_PASS")
    EMAIL_TO   = os.environ.get("EMAIL_TO")
    if not (EMAIL_USER and EMAIL_PASS and EMAIL_TO):
        raise SystemExit("EMAIL_USER, EMAIL_PASS and EMAIL_TO environment variables must be set.")

    SMTP_HOST      = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT      = int(os.environ.get("SMTP_PORT", "587"))
    SMTP_STARTTLS  = os.environ.get("SMTP_STARTTLS", "1").lower() in ("1", "true", "yes")
    SMTP_SSL       = os.environ.get("SMTP_SSL", "0").lower() in ("1", "true", "yes")

    # Load keywords from file or environment
    keywords = load_keywords()
    if not keywords:
        raise SystemExit("No keywords found (keywords.txt or KEYWORDS env var).")

    # Discover transcript files
    all_files = sorted(glob.glob("transcripts/*.txt"))
    if not all_files:
        raise SystemExit("No transcripts found in transcripts/")

    # Filter out previously sent transcripts
    sent = load_sent_log()
    files = [f for f in all_files if Path(f).name not in sent]
    if not files:
        print("No new transcripts to email.")
        return

    # Render HTML and collect counts
    body_html, total_hits, _counts = build_digest_html(files, keywords)
    subject = f"{DEFAULT_TITLE} — {datetime.now().strftime('%d %b %Y')}"

    # Build email message
    from email.message import EmailMessage
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = EMAIL_USER
    # The To header should be a comma‑separated string
    msg['To'] = ", ".join([addr.strip() for addr in re.split(r"[,\s]+", EMAIL_TO) if addr.strip()])
    # Plain text fallback can be empty or a short note
    msg.set_content("This email requires an HTML capable client.")
    msg.add_alternative(body_html, subtype='html')
    # Attach transcripts as plain text
    for fp in files:
        with open(fp, 'rb') as f:
            data = f.read()
            # Use plain text maintype/subtype for transcripts
            msg.add_attachment(data, maintype='text', subtype='plain', filename=Path(fp).name)

    # Send via smtplib
    import smtplib
    if SMTP_SSL:
        with smtplib.SMTP_SSL(host=SMTP_HOST, port=SMTP_PORT) as server:
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
    else:
        with smtplib.SMTP(host=SMTP_HOST, port=SMTP_PORT) as server:
            if SMTP_STARTTLS:
                server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)

    # Update sent log to avoid resending
    update_sent_log(files)
    print(f"✅ Email sent to {EMAIL_TO} with {len(files)} file(s), {total_hits} match(es).")

if __name__ == "__main__":
    main()
