#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
from pathlib import Path
from datetime import datetime
from bisect import bisect_right
from zoneinfo import ZoneInfo
import yagmail

# =============================================================================
# Template resolution (robust)
# =============================================================================

def _resolve_template_path() -> Path:
    env_path = os.environ.get("TEMPLATE_HTML_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    script_dir = Path(__file__).resolve().parent
    candidates = [
        "email_template.html",
        "email_template.htm",
    ]
    for name in candidates:
        p = script_dir / name
        if p.exists():
            return p

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

# Brand colours (centralised)
SLATE = "#4A5A6A"
GOLD  = "#C5A572"
GREY  = "#D8DCE0"

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
        # MERGE when the next window overlaps or is within <= gap_gt sentences
        if gap <= gap_gt:
            merged[-1] = [ps, max(pe, e), pk | kws, pl | lines]
        else:
            merged.append([s, e, kws, lines])
    return merged

def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def _highlight_keywords_html(text_html: str, keywords: list[str]) -> str:
    """
    GOLD highlight to match brand. Outlook honours inline background colour.
    """
    out = text_html
    for kw in sorted(keywords, key=len, reverse=True):
        pat = re.compile(re.escape(_html_escape(kw)) if " " in kw else rf"\b{re.escape(_html_escape(kw))}\b", re.I)
        out = pat.sub(lambda m: "<b><span style='background:%s;color:#ffffff'>" % GOLD + m.group(0) + "</span></b>", out)
    return out

def _excerpt_from_window_html(utt, win, keywords):
    sents  = utt["sents"]
    joined = utt["joined"]
    start, end, kws, lines = win
    a = sents[start][0]
    b = sents[end][1]

    raw = joined[a:b]
    raw = re.sub(r"\r\n?", "\n", raw)   # normalise
    raw = re.sub(r"\n{2,}", "\n", raw)  # collapse blank lines
    raw = raw.strip()

    if len(raw) > MAX_SNIPPET_CHARS:
        raw = raw[:MAX_SNIPPET_CHARS].rstrip() + "…"

    # Outlook-friendly: present as ONE block of text (no paragraph spacing)
    html = _html_escape(raw)
    html = _highlight_keywords_html(html, keywords)
    html = html.replace("\n", " ")
    html = re.sub(r"\s{2,}", " ", html).strip()

    start_line = _line_for_char_offset(utt["line_offsets"], utt["line_nums"], a)
    end_line   = _line_for_char_offset(utt["line_offsets"], utt["line_nums"], max(a, b - 1))
    visible_lines = [n for n in sorted(lines) if start_line <= n <= end_line]
    return html, visible_lines, sorted(kws, key=str.lower), start_line, end_line

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
    html = _EMPTY_MSOP_RE.sub("", html)
    html = re.sub(r"(?:\s*<br[^>]*>\s*){2,}", "<br>", html, flags=re.I)
    html = re.sub(r"(</table>)\s+(?=(?:<!--.*?-->\s*)*<table\b)", r"\1", html, flags=re.I | re.S)
    html = re.sub(r">\s*(?:&nbsp;|<br[^>]*>|\s)+</td>", "></td>", html, flags=re.I)
    html = re.sub(
        r"(?:\s*<p\b[^>]*>(?:\s|&nbsp;|<br[^>]*>|"
        r"<span\b[^>]*>(?:\s|&nbsp;|<br[^>]*>)*</span>|<o:p>.*?</o:p>)*</p>)+\s*(?=</body>)",
        "",
        html,
        flags=re.I | re.S,
    )
    return html

def _minify_inter_tag_whitespace(html: str) -> str:
    return re.sub(r">\s+<", "><", html)

def _inject_mso_css_reset(html: str) -> str:
    mso_block = (
        "<!--[if mso]>"
        "<style>"
        "p.MsoNormal,div.MsoNormal,li.MsoNormal{margin:0 !important;line-height:normal !important;}"
        "table,td{mso-table-lspace:0pt !important;mso-table-rspace:0pt !important;mso-line-height-rule:exactly !important;}"
        "</style>"
        "<![endif]-->"
    )
    if re.search(r"</head\s*>", html, re.I):
        return re.sub(r"</head\s*>", mso_block + "</head>", html, flags=re.I, count=1)
    return mso_block + html

# Targeted sanitizer: remove the spacer table that sits immediately before the FOOTER block
def _remove_bottom_spacer_before_footer(html: str) -> str:
    return re.sub(
        r"(<!--\s*SPACER\s*-->\s*)"
        r"<table[^>]*>\s*<tr>\s*<td[^>]*>(?:&nbsp;|\s*)</td>\s*</tr>\s*</table>\s*"
        r"(?=\s*<!--\s*FOOTER\s*-->)",
        "",
        html,
        flags=re.I | re.S,
    )

# NEW: Ensure no trailing whitespace/line after the footer (before </body>)
def _strip_trailing_after_footer(html: str) -> str:
    """
    Remove any trailing whitespace or <br> just before </body>, so nothing
    renders as a 'line' below the footer in picky clients.
    """
    return re.sub(r"(?:\s|&nbsp;|<br[^>]*>)+(?=\s*</body>)", "", html, flags=re.I | re.S)

# =============================================================================
# Template operations (summary & sections)
# =============================================================================

def _build_detection_row(kw, hoa, lc, tot) -> str:
    """
    Build one row for the summary table with consistent GREY borders
    (internal + external) so nothing drops back to black in Outlook.
    """
    return (
        "<tr>"
        f"<td width=\"28%\" style='border-top:none;border-left:solid {GREY} 1px;border-bottom:solid {GREY} 1px;border-right:none;padding:8px 10px;'>"
        f"<p class=MsoNormal style='margin:0;'><b><span style='font-size:10pt;font-family:\"Segoe UI\",sans-serif;color:black'>{_html_escape(kw)}</span></b></p></td>"
        f"<td width=\"28%\" style='border-left:solid {GREY} 1px;border-bottom:solid {GREY} 1px;padding:8px 10px;'>"
        f"<p class=MsoNormal align=center style='text-align:center;margin:0;'><b><span style='font-size:10pt;font-family:\"Segoe UI\",sans-serif;color:black'>{hoa}</span></b></p></td>"
        f"<td width=\"28%\" style='border-left:solid {GREY} 1px;border-bottom:solid {GREY} 1px;padding:8px 10px;'>"
        f"<p class=MsoNormal align=center style='text-align:center;margin:0;'><b><span style='font-size:10pt;font-family:\"Segoe UI\",sans-serif;color:black'>{lc}</span></b></p></td>"
        f"<td width=\"15%\" style='border-left:solid {GREY} 1px;border-bottom:solid {GREY} 1px;border-right:solid {GREY} 1px;padding:8px 10px;'>"
        f"<p class=MsoNormal align=center style='text-align:center;margin:0;'><b><span style='font-size:10pt;font-family:\"Segoe UI\",sans-serif;color:black'>{tot}</span></b></p></td>"
        "</tr>"
    )

def _replace_detection_rows_in_template(html, row_html):
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
    return html[:tbl_start] + new_table + html[tpl_end:] if False else html[:tbl_start] + new_table + html[tbl_end:]

def _strip_sample_section(html):
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
# Per-file sections (“cards”) — Outlook-safe
# =============================================================================

def _build_file_section_html(filename: str, matches):
    esc = _html_escape
    cards = []

    for idx, (_kw_set, excerpt_html, speaker, line_list, _s, _e) in enumerate(matches, 1):
        line_txt = f"line {line_list[0]}" if len(line_list) == 1 else "lines " + ", ".join(str(n) for n in line_list)

        card = (
            "<table role='presentation' width='100%' cellpadding='0' cellspacing='0' border='0' "
            f"style='border-collapse:collapse;border:1px solid {GREY};'>"
            "<tr>"
            "<td valign='middle' style='background:#ECF0F1;border-bottom:1px solid #D8DCE0;padding:9px 12px;"
            "font-size:0;line-height:0;mso-line-height-rule:exactly;vertical-align:middle;'>"
              "<table role='presentation' width='100%' cellpadding='0' cellspacing='0' border='0' style='border-collapse:collapse;'>"
              "<tr style='height:24px;'>"
    "<td width='24' align='center' valign='middle' "
    "    style='background:#4A5A6A;width:24px;height:24px;vertical-align:middle;mso-line-height-rule:exactly;'>"
    "<p class='MsoNormal' style=\"margin:0;padding:0;line-height:24px;mso-line-height-rule:exactly;\">"
    f"<b><span style=\"font-size:10pt;font-family:'Segoe UI',sans-serif;color:#FFFFFF\">{idx}</span></b>"
    "</p>"
    "</td>"
                "<td width='8' style='font-size:0;line-height:0;height:24px;vertical-align:middle;'>&nbsp;</td>"
                "<td valign='middle' style='vertical-align:middle;height:24px;'>"
                  "<div style=\"font:bold 10pt 'Segoe UI',sans-serif;color:#24313F;text-transform:uppercase;"
                  "line-height:16px;mso-line-height-rule:exactly;display:block;\">"
                  f"{esc(speaker) if speaker else 'UNKNOWN'}</div>"
                "</td>"
                "<td align='right' valign='middle' style='vertical-align:middle;height:24px;'>"
                  "<div style=\"font:10pt 'Segoe UI',sans-serif;color:#6A7682;line-height:16px;mso-line-height-rule:exactly;display:block;\">"
                  f"{line_txt}</div>"
                "</td>"
              "</tr>"
              "</table>"
            "</td>"
            "</tr>"
            "<tr>"
            "<td valign='middle' style='padding:8px 10px 9px 10px;vertical-align:middle;'>"
              "<div style=\"font:10pt 'Segoe UI',sans-serif;color:#1F2A36;margin:0;line-height:17px;mso-line-height-rule:exactly;display:block;\">"
              f"{excerpt_html}</div>"
            "</td>"
            "</tr>"
            "</table>"
        )
        cards.append(card)

    spacer = ("<table role='presentation' width='100%' cellpadding='0' cellspacing='0' border='0'>"
              "<tr><td style='height:10px;line-height:10px;font-size:0;'>&nbsp;</td></tr></table>")
    cards_html = spacer.join(cards)

    section = (
        "<table role='presentation' width='100%' cellpadding='0' cellspacing='0' border='0' style='border-collapse:collapse;'>"
        "<tr>"
        f"<td style='border-left:3px solid {GOLD};background:#F7F9FA;padding:6px 10px;'>"
        f"<div style=\"font:bold 10pt 'Segoe UI',sans-serif;color:#000;line-height:15px;mso-line-height-rule:exactly;display:block;\">{esc(filename)}</div>"
        f"<div style=\"font:10pt 'Segoe UI',sans-serif;color:#000;line-height:15px;mso-line-height-rule:exactly;display:block;\">{len(matches)} match(es)</div>"
        "</td>"
        "</tr>"
        "<tr>"
        f"<td style='border:1px solid {GREY};border-top:none;background:#FFFFFF;padding:6px 8px;'>"
        f"{cards_html}"
        "</td>"
        "</tr>"
        "</table>"
    )
    return section

# =============================================================================
# Build the full HTML
# =============================================================================

def _parse_chamber_from_filename(filename: str) -> str:
    name = filename.lower()
    if "house_of_assembly" in name:
        return "House of Assembly"
    if "legislative_council" in name:
        return "Legislative Council"
    return "Unknown"

def build_digest_html(files: list[str], keywords: list[str]):
    # Load template (Word/Outlook often uses Windows-1252)
    try:
        template_html = TEMPLATE_HTML_PATH.read_text(encoding="windows-1252", errors="ignore")
    except Exception:
        template_html = TEMPLATE_HTML_PATH.read_text(encoding="utf-8", errors="ignore")

    # Program run date in AU (Tasmania)
    au_now = datetime.now(ZoneInfo("Australia/Hobart"))
    run_date = au_now.strftime("%d %B %Y")

    template_html = template_html.replace("[DATE]", run_date)
    template_html = template_html.replace(
        '<span style="font-size:12.0pt;color:black">Detection Match by Chamber</span>',
        '<span style="font-size:12.0pt;font-family:\'Segoe UI\',sans-serif;color:black">Detection Match by Chamber</span>',
    )

    template_html = _inject_mso_css_reset(template_html)

    # Collect matches + counts
    counts = {kw: {"House of Assembly": 0, "Legislative Council": 0} for kw in keywords}
    sections, total_matches = [], 0

    for f in files:
        text = Path(f).read_text(encoding="utf-8", errors="ignore")
        matches = extract_matches(text, keywords)
        if not matches:
            continue
        total_matches += len(matches)
        chamber = _parse_chamber_from_filename(Path(f).name)
        for kw_set, _, _, _, _, _ in matches:
            for kw in kw_set:
                counts.setdefault(kw, {"House of Assembly": 0, "Legislative Council": 0})
                if chamber in counts[kw]:
                    counts[kw][chamber] += 1
        sections.append(_build_file_section_html(Path(f).name, matches))

    # Detection rows
    det_rows = []
    for kw in keywords:
        hoa = counts.get(kw, {}).get("House of Assembly", 0)
        lc  = counts.get(kw, {}).get("Legislative Council", 0)
        total = hoa + lc
        if total > 0:  # only include if there was at least one hit
            det_rows.append(_build_detection_row(kw, hoa, lc, total))
    detection_rows_html = "".join(det_rows)

    template_html = _replace_detection_rows_in_template(template_html, detection_rows_html)

    template_html = _strip_sample_section(template_html)

    # Spacer between detection table and first file section — ONLY if sections exist
    if sections:
        spacer_before_sections = (
            "<table role='presentation' width='100%' cellpadding='0' cellspacing='0' border='0' style='border-collapse:collapse;'>"
            "<tr><td style='height:16px;line-height:16px;font-size:0;'>&nbsp;</td></tr></table>"
        )
        sections_html = spacer_before_sections + "".join(sections)
        template_html = _inject_sections_after_detection(template_html, sections_html)

    # Remove the spacer immediately above the footer if the template still has one
    template_html = _remove_bottom_spacer_before_footer(template_html)

    # NEW: ensure absolutely nothing trails after the footer (no <br> or spaces)
    template_html = _strip_trailing_after_footer(template_html)

    # Final tidy/minify
    template_html = _tighten_outlook_whitespace(template_html)
    template_html = _minify_inter_tag_whitespace(template_html)

    return template_html, total_matches, counts

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

def main():
    EMAIL_USER = os.environ["EMAIL_USER"]
    EMAIL_PASS = os.environ["EMAIL_PASS"]
    EMAIL_TO   = os.environ["EMAIL_TO"]

    SMTP_HOST      = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT      = int(os.environ.get("SMTP_PORT", "587"))
    SMTP_STARTTLS  = os.environ.get("SMTP_STARTTLS", "1").lower() in ("1", "true", "yes")
    SMTP_SSL       = os.environ.get("SMTP_SSL", "0").lower() in ("1", "true", "yes")

    keywords = load_keywords()
    if not keywords:
        raise SystemExit("No keywords found (keywords.txt or KEYWORDS env var).")

    all_files = sorted(glob.glob("transcripts/*.txt"))
    if not all_files:
        raise SystemExit("No transcripts found in transcripts/")

    sent = load_sent_log()
    files = [f for f in all_files if Path(f).name not in sent]
    if not files:
        print("No new transcripts to email.")
        return

    body_html, total_hits, _counts = build_digest_html(files, keywords)
    if total_hits == 0:
    print("No keyword matches found — no email will be sent.")
    return

    au_now = datetime.now(ZoneInfo("Australia/Hobart"))
    subject = f"{DEFAULT_TITLE} — {au_now.strftime('%d %b %Y')}"

    to_list = [addr.strip() for addr in re.split(r"[,\s]+", EMAIL_TO) if addr.strip()]

    yag = yagmail.SMTP(
        user=EMAIL_USER,
        password=EMAIL_PASS,
        host=SMTP_HOST,
        port=SMTP_PORT,
        smtp_starttls=SMTP_STARTTLS,
        smtp_ssl=SMTP_SSL,
    )

    yag.send(
        to=to_list,
        subject=subject,
        contents=body_html,
        attachments=files,
    )

    update_sent_log(files)
    print(f"✅ Email sent to {EMAIL_TO} with {len(files)} file(s), {total_hits} match(es).")

if __name__ == "__main__":
    main()
