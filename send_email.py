#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
from pathlib import Path
from datetime import datetime
from bisect import bisect_right
import yagmail

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
MERGE_IF_GAP_LE = 2  # merge windows when the gap in sentences is <= this

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

        # simple sentence segmentation: split on punctuation + whitespace
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

def _merge_windows_nearby_only(wins, gap_le=MERGE_IF_GAP_LE):
    if not wins:
        return []
    merged = [wins[0]]
    for s, e, kws, lines in wins[1:]:
        ps, pe, pk, pl = merged[-1]
        if s - pe <= gap_le:
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
        merged = _merge_windows_nearby_only(wins, gap_le=MERGE_IF_GAP_LE)
        for win in merged:
            excerpt_html, line_list, kws_in_excerpt, win_start, win_end = _excerpt_from_window_html(utt, win, keywords)
            results.append((set(kws_in_excerpt), excerpt_html, speaker, line_list, win_start, win_end))
    return results

# =============================================================================
# Token-based template extraction
# =============================================================================

def _extract_detection_row_template(html: str) -> str:
    """Locate the <tr> used for keyword counts (row with [Keyword])."""
    m = re.search(r"(<tr\b[^>]*>.*?\[Keyword\].*?</tr\s*>)", html, flags=re.I | re.S)
    if not m:
        raise ValueError("Detection row placeholder not found")
    return m.group(1)

def _extract_section_template(html: str) -> str:
    """Locate the outer table used for each transcript section (block with [Transcript filename])."""
    pattern = re.compile(
        r"(<table\b[^>]*>.*?\[Transcript filename\].*?</table>"
        r"(?:\s*</td>\s*</tr>\s*</table>)?)",
        re.I | re.S,
    )
    m = pattern.search(html)
    if not m:
        raise ValueError("Transcript section placeholder not found")
    return m.group(1)

def _extract_match_template(section_html: str) -> str:
    """
    Grab the whole match "card" block: a header table containing [Match #]
    followed by another table containing the Excerpt/snippet token.
    Supports Word SpellE wrapper and both Excerpt/Exerpt spellings.
    """
    pattern = re.compile(
        r"("  # capture the whole card block
        r"<table\b[^>]*>.*?\[Match\s*#\].*?</table>"        # header table
        r".*?"                                              # anything between
        r"<table\b[^>]*>.*?\["                              # start snippet token
        r"(?:<span\b[^>]*class=['\"]?SpellE['\"]?[^>]*>\s*)?"  # optional SpellE open
        r"Excer?p?t"                                        # Excerpt/Exerpt
        r"(?:\s*</span>)?\s*/snippet\]"                     # optional SpellE close + /snippet
        r".*?</table>"
        r")",
        re.I | re.S,
    )
    m = pattern.search(section_html)
    if not m:
        raise ValueError("Match template not found")
    return m.group(1)

_SNIPPET_TOKEN_RE = re.compile(
    r"\[(?:<span\b[^>]*class=['\"]?SpellE['\"]?[^>]*>\s*)?Excer?p?t(?:\s*</span>)?\s*/snippet\]",
    re.I,
)

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
    try:
        template_html = TEMPLATE_HTML_PATH.read_text(encoding="windows-1252", errors="ignore")
    except Exception:
        template_html = TEMPLATE_HTML_PATH.read_text(encoding="utf-8", errors="ignore")

    run_date = datetime.now().strftime("%d %B %Y")
    template_html = template_html.replace("[DATE]", run_date)

    det_row_tpl = _extract_detection_row_template(template_html)
    section_tpl = _extract_section_template(template_html)
    match_tpl   = _extract_match_template(section_tpl)

    counts = {kw: {"House of Assembly": 0, "Legislative Council": 0} for kw in keywords}
    sections_html = []
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

        match_blocks = []
        for idx, (_kw_set, excerpt_html, speaker, line_list, _s, _e) in enumerate(matches, 1):
            line_txt = f"line {line_list[0]}" if len(line_list) == 1 else "lines " + ', '.join(str(n) for n in line_list)
            mhtml = match_tpl.replace("[Match #]", str(idx))
            mhtml = mhtml.replace("[SPEAKER NAME]", _html_escape(speaker) if speaker else "UNKNOWN")
            mhtml = mhtml.replace("[Line number(s)]", line_txt)
            mhtml = _SNIPPET_TOKEN_RE.sub(excerpt_html, mhtml, count=1)
            match_blocks.append(mhtml)

        sect_html = section_tpl.replace(match_tpl, "".join(match_blocks))
        sect_html = sect_html.replace("[Transcript filename]", _html_escape(Path(f).name))
        sect_html = sect_html.replace("[Match count]", str(len(matches)))
        sections_html.append(sect_html)

    det_rows = []
    for kw in keywords:
        hoa = counts.get(kw, {}).get("House of Assembly", 0)
        lc  = counts.get(kw, {}).get("Legislative Council", 0)
        tot = hoa + lc
        row_html = det_row_tpl.replace("[Keyword]", _html_escape(kw))
        row_html = row_html.replace("[House of Assembly count]", str(hoa))
        row_html = row_html.replace("[Legislative Council count]", str(lc))
        row_html = row_html.replace("[Total count]", str(tot))
        det_rows.append(row_html)

    html = template_html.replace(det_row_tpl, "".join(det_rows))
    html = html.replace(section_tpl, "".join(sections_html))
    return html, total_matches, counts

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

    # IMPORTANT: pass ONE HTML string to avoid yagmail inserting extra <br>
    yag.send(
        to=to_list,
        subject=subject,
        contents=body_html,
        attachments=files,  # optional: attach transcripts
    )

    update_sent_log(files)
    print(f"✅ Email sent to {EMAIL_TO} with {len(files)} file(s), {total_hits} match(es).")

if __name__ == "__main__":
    main()
