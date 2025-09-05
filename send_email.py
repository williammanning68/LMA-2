#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
from pathlib import Path
from datetime import datetime
from bisect import bisect_right
import yagmail

# -----------------------------------------------------------------------------
# Configuration (no visual formatting here)
# -----------------------------------------------------------------------------

LOG_FILE = Path("sent.log")
DEFAULT_TITLE = "Hansard Monitor – BETA Version 18.3"

# excerpt/windowing – unchanged extraction behavior
MAX_SNIPPET_CHARS = 800
WINDOW_PAD_SENTENCES = 1
FIRST_SENT_FOLLOWING = 2
MERGE_IF_GAP_GT = 2

# Template file resolution
def resolve_template_path() -> Path:
    env_path = os.environ.get("TEMPLATE_HTML_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    here = Path(__file__).resolve().parent
    candidates = [
        "email_template.html",
        "email_template.htm",
        "Hansard Monitor - Email Format - Version 3.htm",
        "templates/email_template.html",
        "templates/email_template.htm",
        "templates/Hansard Monitor - Email Format - Version 3.htm",
    ]
    for name in candidates:
        p = here / name
        if p.exists():
            return p

    # Fallback: any plausible template
    for pat in ("**/*.htm", "**/*.html"):
        for fp in here.glob(pat):
            if any(k in fp.name.lower() for k in ("email", "template", "hansard", "format")):
                return fp
    raise FileNotFoundError("HTML template not found. Set TEMPLATE_HTML_PATH or place the template next to send_email.py")

TEMPLATE_HTML_PATH = resolve_template_path()


# -----------------------------------------------------------------------------
# Keywords
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Transcript segmentation and matching (unchanged logic)
# -----------------------------------------------------------------------------

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

        # simple sentence segmentation
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
    from bisect import bisect_right
    i = bisect_right(line_offsets, pos) - 1
    i = max(0, min(i, len(line_nums) - 1))
    return line_nums[i]

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
        out = pat.sub(lambda m: "<b><span style='background:#fff0a6;mso-highlight:yellow'>" + m.group(0) + "</span></b>", out)
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
    utts, _ = _build_utterances(text)
    kw_pats = _compile_kw_patterns(keywords)
    results = []
    for utt in utts:
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
            results.append((set(kws_in_excerpt), excerpt_html, utt["speaker"], line_list, win_start, win_end))
    return results


# -----------------------------------------------------------------------------
# Tiny templating helpers (extract sub-templates and fill placeholders)
# -----------------------------------------------------------------------------

def _read_template_html(path: Path) -> str:
    # Word/Outlook templates often use cp1252
    try:
        return path.read_text(encoding="windows-1252", errors="ignore")
    except Exception:
        return path.read_text(encoding="utf-8", errors="ignore")

def _extract_block(html: str, begin_marker: str, end_marker: str) -> tuple[str, str]:
    """
    Returns (block_html, html_without_block) for the first block between markers.
    Markers are literal HTML comments in the template.
    """
    m_start = re.search(re.escape(begin_marker), html, flags=re.I)
    m_end   = re.search(re.escape(end_marker), html, flags=re.I)
    if not m_start or not m_end or m_end.start() <= m_start.end():
        return "", html
    block = html[m_start.end():m_end.start()]
    # Remove the whole marker block from the template (including markers)
    html2 = html[:m_start.start()] + html[m_end.end():]
    return block, html2

def _render_detection_rows(row_tpl: str, keywords: list[str], counts: dict) -> str:
    parts = []
    for kw in keywords:
        hoa = counts.get(kw, {}).get("House of Assembly", 0)
        lc  = counts.get(kw, {}).get("Legislative Council", 0)
        tot = hoa + lc
        row = (row_tpl
               .replace("{{KW}}", _html_escape(kw))
               .replace("{{HOA}}", str(hoa))
               .replace("{{LC}}", str(lc))
               .replace("{{TOTAL}}", str(tot)))
        parts.append(row)
    return "".join(parts)

def _render_file_sections(section_tpl: str, card_tpl: str, file_to_matches: list[tuple[str, list[tuple]]]) -> str:
    out_sections = []
    for filename, matches in file_to_matches:
        # build cards for this file
        cards = []
        for idx, (_kw_set, excerpt_html, speaker, line_list, _s, _e) in enumerate(matches, 1):
            line_txt = f"line {line_list[0]}" if len(line_list) == 1 else "lines " + ", ".join(str(n) for n in line_list)
            card = (card_tpl
                    .replace("{{INDEX}}", str(idx))
                    .replace("{{SPEAKER}}", _html_escape(speaker or "UNKNOWN"))
                    .replace("{{LINETEXT}}", _html_escape(line_txt))
                    .replace("{{EXCERPT_HTML}}", excerpt_html))
            cards.append(card)

        sect = (section_tpl
                .replace("{{FILENAME}}", _html_escape(filename))
                .replace("{{MATCHCOUNT}}", str(len(matches)))
                .replace("{{CARDS}}", "".join(cards)))
        out_sections.append(sect)
    return "".join(out_sections)

def _parse_chamber_from_filename(filename: str) -> str:
    name = filename.lower()
    if "house_of_assembly" in name:
        return "House of Assembly"
    if "legislative_council" in name:
        return "Legislative Council"
    return "Unknown"


def build_body_from_template(files: list[str], keywords: list[str]) -> tuple[str, int]:
    # Read template HTML
    html = _read_template_html(TEMPLATE_HTML_PATH)

    # Extract row/section/card sub-templates (these live in HTML comments)
    det_row_tpl, html = _extract_block(html, "<!-- BEGIN:DETECTION_ROW_TEMPLATE -->", "<!-- END:DETECTION_ROW_TEMPLATE -->")
    section_tpl, html = _extract_block(html, "<!-- BEGIN:FILE_SECTION_TEMPLATE -->", "<!-- END:FILE_SECTION_TEMPLATE -->")
    card_tpl, html    = _extract_block(html, "<!-- BEGIN:MATCH_CARD_TEMPLATE -->",    "<!-- END:MATCH_CARD_TEMPLATE -->")

    if not det_row_tpl or not section_tpl or not card_tpl:
        raise SystemExit("Template missing one or more required sub-templates. Please keep the three comment-delimited blocks in email_template.html.")

    # Replace [DATE]
    run_date = datetime.now().strftime("%d %B %Y")
    html = html.replace("[DATE]", run_date)

    # Build matches + counts
    counts = {kw: {"House of Assembly": 0, "Legislative Council": 0} for kw in keywords}
    file_sections = []
    total_matches = 0

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
        file_sections.append((Path(f).name, matches))

    # Detection rows
    detection_rows_html = _render_detection_rows(det_row_tpl, keywords, counts)

    # File sections
    sections_html = _render_file_sections(section_tpl, card_tpl, file_sections)

    # Inject into main placeholders
    html = html.replace("{{DETECTION_ROWS}}", detection_rows_html)
    html = html.replace("{{SECTIONS}}", sections_html)

    return html, total_matches


# -----------------------------------------------------------------------------
# Sent-log helpers
# -----------------------------------------------------------------------------

def load_sent_log() -> set[str]:
    if LOG_FILE.exists():
        with open(LOG_FILE, encoding="utf-8") as f:
            return {ln.strip() for ln in f if ln.strip()}
    return set()

def update_sent_log(files: list[str]):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        for fp in files:
            f.write(Path(fp).name + "\n")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

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

    body_html, total_hits = build_body_from_template(files, keywords)
    subject = f"{DEFAULT_TITLE} — {datetime.now().strftime('%d %b %Y')}"

    to_list = [addr.strip() for addr in re.split(r"[,\s]+", EMAIL_TO) if addr.strip()]

    yag = yagmail.SMTP(
        user=EMAIL_USER,
        password=EMAIL_PASS,
        host=SMTP_HOST,
        port=SMTP_PORT,
        smtp_starttls=SMTP_STARTTLS,
        smtp_ssl=SMTP_SSL,
    )

    # Important: one big HTML string – avoids yagmail splitting/adding <br>
    yag.send(
        to=to_list,
        subject=subject,
        contents=body_html,
        attachments=files,  # optional
    )

    update_sent_log(files)
    print(f"✅ Email sent to {EMAIL_TO} with {len(files)} file(s), {total_hits} match(es).")

if __name__ == "__main__":
    main()
