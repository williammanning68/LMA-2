import os
import re
import glob
from bisect import bisect_right
from pathlib import Path
from datetime import datetime, UTC
import yagmail
import subprocess  # optional: only used if ATTRIB_WITH_LLM=1

# File that records which transcripts have already been emailed
LOG_FILE = Path("sent.log")

# --- Tunables (extraction behaviour stays the same) --------------------------
MAX_SNIPPET_CHARS = 800   # upper bound after merging windows; keep readable but compact
WINDOW_PAD_SENTENCES = 1  # for non-first-sentence hits: one sentence either side
FIRST_SENT_FOLLOWING = 2  # for first-sentence hits: include next two sentences
MERGE_IF_GAP_GT = 2       # Only merge windows if the gap (in sentences) is > this value

# --- Template inputs ---------------------------------------------------------
# Point to the bundled Outlook/Word HTML template (can be overridden via env)
TEMPLATE_HTML_PATH = Path(
    os.environ.get("TEMPLATE_HTML_PATH", "email_template.html")
)
DEFAULT_TITLE = "Hansard Monitor – BETA Version 18.3"

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
CONTENT_COLON_RE = re.compile(r"^(Then|There|And|But|So|If|When|Now|Finally|First|Second|Third)\b", re.IGNORECASE)
TIME_STAMP_RE = re.compile(r"^\[\d{1,2}\.\d{2}\s*(a|p)\.m\.\]$", re.IGNORECASE)
UPPER_HEADING_RE = re.compile(r"^[A-Z][A-Z\s’'—\-&,;:.()]+$")
INTERJECTION_RE = re.compile(r"^(Members interjecting\.|The House suspended .+)$", re.IGNORECASE)

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

# --- Utterance segmentation with spans ---------------------------------------
def _build_utterances(text: str):
    all_lines = text.splitlines()
    utterances = []

    curr = {"speaker": None, "lines": [], "line_nums": []}

    def flush():
        if curr["lines"]:
            joined = "\n".join(curr["lines"])
            offs = []
            total = 0
            for i, ln in enumerate(curr["lines"]):
                offs.append(total)
                total += len(ln) + (1 if i < len(curr["lines"]) - 1 else 0)

            sents = []
            start = 0
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
            name = (m.group("name") or m.group("name_only") or "").strip()
            curr = {"speaker": " ".join(x for x in (title, name) if x).strip(), "lines": [], "line_nums": []}
            continue

        curr["lines"].append(raw.rstrip())
        curr["line_nums"].append(idx + 1)

    flush()
    return utterances, all_lines

def _line_for_char_offset(line_offsets, line_nums, pos):
    i = bisect_right(line_offsets, pos) - 1
    if i < 0:
        i = 0
    if i >= len(line_nums):
        i = len(line_nums) - 1
    return line_nums[i]

# --- Matching, windows & merging ---------------------------------------------
def _compile_kw_patterns(keywords):
    pats = []
    for kw in sorted(keywords, key=len, reverse=True):
        if " " in kw:
            pats.append((kw, re.compile(re.escape(kw), re.IGNORECASE)))
        else:
            pats.append((kw, re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)))
    return pats

def _collect_hits_in_utterance(utt, kw_pats):
    hits = []
    joined = utt["joined"]
    sents = utt["sents"]
    for si, (a, b) in enumerate(sents):
        seg = joined[a:b]
        for kw, pat in kw_pats:
            m = pat.search(seg)
            if not m:
                continue
            char_pos = a + m.start()
            line_no = _line_for_char_offset(utt["line_offsets"], utt["line_nums"], char_pos)
            hits.append({"kw": kw, "sent_idx": si, "line_no": line_no})
    return hits

def _windows_for_hits(hits, sent_count):
    wins = []
    for h in hits:
        j = h["sent_idx"]
        if j == 0:
            start = 0
            end = min(sent_count - 1, FIRST_SENT_FOLLOWING)
        else:
            start = max(0, j - WINDOW_PAD_SENTENCES)
            end = min(sent_count - 1, j + WINDOW_PAD_SENTENCES)
        wins.append([start, end, {h["kw"]}, {h["line_no"]}])
    wins.sort(key=lambda w: (w[0], w[1]))
    return wins

def _dedup_windows(wins):
    """Collapse windows with identical (start,end); union keywords and line numbers."""
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
    deduped = [[s, e, bucket[(s, e)][0], bucket[(s, e)][1]] for (s, e) in sorted(bucket.keys())]
    return deduped

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
        if " " in kw:
            pat = re.compile(re.escape(_html_escape(kw)), re.IGNORECASE)
        else:
            pat = re.compile(rf"\b{re.escape(_html_escape(kw))}\b", re.IGNORECASE)
        out = pat.sub(lambda m: f"<strong>{m.group(0)}</strong>", out)
    return out

def _excerpt_from_window_html(utt, win, keywords):
    sents = utt["sents"]
    joined = utt["joined"]
    start, end, kws, lines = win
    a = sents[start][0]
    b = sents[end][1]
    raw = joined[a:b].strip()
    if len(raw) > MAX_SNIPPET_CHARS:
        raw = raw[:MAX_SNIPPET_CHARS].rstrip() + "…"
    html = _html_escape(raw)
    html = _highlight_keywords_html(html, keywords).replace("\n", "<br>")
    start_line = _line_for_char_offset(utt["line_offsets"], utt["line_nums"], a)
    end_line = _line_for_char_offset(utt["line_offsets"], utt["line_nums"], max(a, b - 1))
    return html, sorted(lines), sorted(kws, key=str.lower), start_line, end_line

def _looks_suspicious(s: str | None) -> bool:
    if not s:
        return True
    s = s.strip()
    if re.match(
        r"(?i)^(Mr|Ms|Mrs|Miss|Hon|Dr|Prof|Professor|Premier|Madam SPEAKER|The SPEAKER|The PRESIDENT|The CLERK|Deputy Speaker|Deputy President)"
        r"(?:\s+[A-Z][A-Z'’\-]+(?:\s+[A-Z][A-Z'’\-]+){0,3})?$",
        s,
    ):
        return False
    if re.match(r"^[A-Z][A-Z'’\-]+(?:\s+[A-Z][A-Z'’\-]+){0,3}$", s):
        return False
    return True

def _llm_qc_speaker(full_lines, hit_line_no, candidates, model=None, timeout=30):
    model = model or os.environ.get("ATTRIB_LLM_MODEL", "llama3.2:3b")
    i = max(0, hit_line_no - 1)
    start = max(0, i - 80)
    context = "\n".join(full_lines[start: i + 5])[:3000]
    options = "\n".join(f"- {c}" for c in candidates[:50]) or "- UNKNOWN"
    prompt = f"""Choose the most likely speaker from the list (or 'UNKNOWN'):
{options}

Use the transcript context (previous lines first, then nearby):
{context}
"""
    try:
        out = subprocess.check_output(["ollama", "run", model, prompt], text=True, timeout=timeout)
        ans = out.strip().splitlines()[-1].strip()
        if not ans or ans.upper().startswith("UNKNOWN"):
            return None
        return ans
    except Exception:
        return None

def extract_matches(text: str, keywords):
    """Return list of: (kw_set, excerpt_html, speaker, line_numbers_list, win_start_line, win_end_line)"""
    use_llm = os.environ.get("ATTRIB_WITH_LLM", "").lower() in ("1", "true", "yes")
    llm_timeout = int(os.environ.get("ATTRIB_LLM_TIMEOUT", "30"))

    utts, all_lines = _build_utterances(text)
    kw_pats = _compile_kw_patterns(keywords)

    results = []
    for utt in utts:
        speaker = utt["speaker"]
        if not utt["lines"]:
            continue

        hits = _collect_hits_in_utterance(utt, kw_pats)
        if not hits:
            continue

        wins = _windows_for_hits(hits, sent_count=len(utt["sents"]))
        wins = _dedup_windows(wins)
        merged = _merge_windows_far_only(wins, gap_gt=MERGE_IF_GAP_GT)

        if use_llm and _looks_suspicious(speaker):
            candidates = sorted({u["speaker"] for u in utts if u["speaker"]})
            earliest_line = min(min(w[3]) for w in merged)
            guess = _llm_qc_speaker(all_lines, earliest_line, candidates, timeout=llm_timeout)
            if guess:
                speaker = guess

        for win in merged:
            excerpt_html, line_list, kws_in_excerpt, win_start, win_end = _excerpt_from_window_html(utt, win, keywords)
            results.append((set(kws_in_excerpt), excerpt_html, speaker, line_list, win_start, win_end))

    return results

# --- Digest / visual assembly (injecting into your HTML) ---------------------
def parse_date_from_filename(filename: str):
    m = re.search(r"(\d{1,2} \w+ \d{4})", filename)
    if m:
        try:
            return datetime.strptime(m.group(1), "%d %B %Y")
        except ValueError:
            return datetime.min
    return datetime.min

def parse_chamber_from_filename(filename: str) -> str:
    name = filename.lower()
    if "house_of_assembly" in name:
        return "House of Assembly"
    if "legislative_council" in name:
        return "Legislative Council"
    return "Unknown"

def _build_detection_row(kw, hoa, lc, tot):
    # Mirrors the Word/Outlook styling of the provided HTML
    return (
        "<tr>"
        "<td width=\"28%\" style='width:28.12%;border-top:none;border-left:solid #D8DCE0 1.0pt;"
        "border-bottom:solid #ECF0F1 1.0pt;border-right:none;mso-border-left-alt:solid #D8DCE0 .75pt;"
        "mso-border-bottom-alt:solid #ECF0F1 .75pt;padding:6.0pt 7.5pt 6.0pt 7.5pt'>"
        f"<p class=MsoNormal><b><span style='font-size:10.0pt;font-family:\"Segoe UI\",sans-serif;"
        "mso-fareast-font-family:Aptos;color:black;mso-color-alt:windowtext'>"
        f"{_html_escape(kw)}</span></b></p></td>"
        "<td width=\"28%\" style='width:28.12%;border:none;border-bottom:solid #ECF0F1 1.0pt;"
        "mso-border-bottom-alt:solid #ECF0F1 .75pt;padding:6.0pt 7.5pt 6.0pt 7.5pt'>"
        f"<p class=MsoNormal align=center style='text-align:center'><b>"
        f"<span style='font-size:10.0pt;font-family:\"Segoe UI\",sans-serif;color:black;mso-color-alt:windowtext'>{hoa}</span>"
        f"</b></p></td>"
        "<td width=\"28%\" style='width:28.14%;border:none;border-bottom:solid #ECF0F1 1.0pt;"
        "mso-border-bottom-alt:solid #ECF0F1 .75pt;padding:6.0pt 7.5pt 6.0pt 7.5pt'>"
        f"<p class=MsoNormal align=center style='text-align:center'><b>"
        f"<span style='font-size:10.0pt;font-family:\"Segoe UI\",sans-serif;color:black;mso-color-alt:windowtext'>{lc}</span>"
        f"</b></p></td>"
        "<td width=\"15%\" style='width:15.62%;border-top:none;border-left:none;border-bottom:solid #ECF0F1 1.0pt;"
        "border-right:solid #D8DCE0 1.0pt;mso-border-bottom-alt:solid #ECF0F1 .75pt;"
        "mso-border-right-alt:solid #D8DCE0 .75pt;padding:6.0pt 7.5pt 6.0pt 7.5pt'>"
        f"<p class=MsoNormal align=center style='text-align:center'><b>"
        f"<span style='font-size:10.0pt;font-family:\"Segoe UI\",sans-serif;color:black;mso-color-alt:windowtext'>{tot}</span>"
        f"</b></p></td>"
        "</tr>"
    )

def _compose_sections(matches_by_file):
    """
    Build the per-file 'cards' block using Outlook-friendly nested tables,
    styled to match your template palette.
    """
    out = []
    for fpath, items in matches_by_file:
        fname = Path(fpath).name
        out.append(
            "<tr><td style='padding:12.0pt 0 6.0pt 0;'>"
            "<table role=presentation width='100%' cellpadding=0 cellspacing=0 border=0 "
            "style='border-collapse:collapse;mso-table-lspace:0pt;mso-table-rspace:0pt;'>"
            "<tr>"
            f"<td style='font:bold 14px/18px \"Segoe UI\",Arial,sans-serif;color:#24313F;'>{_html_escape(fname)}</td>"
            f"<td align='right' style='font:12px/16px \"Segoe UI\",Arial,sans-serif;color:#6A7682;'>{len(items)} match(es)</td>"
            "</tr></table>"
            "</td></tr>"
        )
        for i, (kw_set, excerpt_html, speaker, line_list, win_start, win_end) in enumerate(items, 1):
            # line labels
            if line_list:
                lines_str = ", ".join(str(n) for n in sorted(set(line_list)))
                line_label = "lines" if len(set(line_list)) > 1 else "line"
            else:
                lines_str = str(win_start)
                line_label = "line"
            speaker_display = speaker if (speaker and not _looks_suspicious(speaker)) else "Unknown"
            out.append(
                "<tr><td style='padding:0 0 8.0pt 0;'>"
                "<table role=presentation width='100%' cellpadding=0 cellspacing=0 border=0 "
                "style='border-collapse:collapse;mso-table-lspace:0pt;mso-table-rspace:0pt;'>"
                "<tr>"
                "<td width='36' align='center' valign='top' "
                "style='background:#E8ECF1;border:1px solid #D4D9E0;"
                "font:bold 13px/32px \"Segoe UI\",Arial,sans-serif;color:#24313F;height:32px;'>"
                f"{i}</td>"
                "<td style='width:10px;font-size:0;line-height:0;'>&nbsp;</td>"
                "<td valign='top' style='background:#FFFFFF;border:1px solid #D4D9E0;padding:10px 12px;'>"
                f"<div style='font:bold 13px/18px \"Segoe UI\",Arial,sans-serif;color:#24313F;'>{_html_escape(speaker_display)} "
                f"<span style='font-weight:normal;color:#6A7682;'>— {line_label} {lines_str}</span></div>"
                f"<div style='font:13px/20px \"Segoe UI\",Arial,sans-serif;color:#1F2A36;mso-line-height-rule:exactly;'>{excerpt_html}</div>"
                "</td></tr></table>"
                "</td></tr>"
            )
        # divider
        out.append(
            "<tr><td style='padding:4px 0 0 0;'>"
            "<table role=presentation width='100%' cellpadding=0 cellspacing=0 border=0>"
            "<tr><td style='border-bottom:1px solid #E0E5EB;height:1px;line-height:1px;font-size:0;'>&nbsp;</td></tr>"
            "</table>"
            "</td></tr>"
        )
    return "".join(out)

def _replace_detection_rows_in_template(html, row_html):
    """
    Find the inner detection table (immediately after 'Detection Match by Chamber')
    and replace all data <tr> rows after its header row with our compiled rows.
    """
    # Find the 'Detection Match by Chamber' label
    hdr = re.search(r"Detection Match by Chamber", html, flags=re.I)
    if not hdr:
        return html  # can't find; leave template untouched
    # Find the next <table ...> after this label
    m_table_start = re.search(r"<table[^>]*>", html[hdr.end():], flags=re.I | re.S)
    if not m_table_start:
        return html
    tbl_start = hdr.end() + m_table_start.start()
    # Find the end of this table
    m_table_end = re.search(r"</table\s*>", html[tbl_start:], flags=re.I | re.S)
    if not m_table_end:
        return html
    tbl_end = tbl_start + m_table_end.end()

    table_html = html[tbl_start:tbl_end]
    # Find the header row containing 'Keyword'
    m_header_row = re.search(r"<tr[^>]*>.*?Keyword.*?</tr\s*>", table_html, flags=re.I | re.S)
    if not m_header_row:
        return html

    before = table_html[:m_header_row.end()]
    after = "</table>"
    new_table = before + row_html + after
    return html[:tbl_start] + new_table + html[tbl_end:]

def _strip_existing_file_sections(html):
    """
    Best-effort: remove any sample 'match(es)' blocks that may be in the template
    so we don't duplicate when we inject our live sections.
    """
    # Match repeating tables that show a filename and 'match(es)'
    pattern = re.compile(
        r"(?:<tr[^>]*>\s*<td[^>]*>\s*<table[^>]*>.*?match\(es\).*?</table>\s*</td>\s*</tr>\s*)+",
        flags=re.I | re.S
    )
    return re.sub(pattern, "", html)

def _inject_sections_after_detection(html, sections_html):
    """
    Insert our per-file sections just after the detection table block.
    """
    # After we rebuilt the detection inner table, put sections right after it.
    # Locate the detection inner table again and insert following its </table>.
    hdr = re.search(r"Detection Match by Chamber", html, flags=re.I)
    if not hdr:
        return html + sections_html  # fallback: append
    m_table_start = re.search(r"<table[^>]*>", html[hdr.end():], flags=re.I | re.S)
    if not m_table_start:
        return html + sections_html
    tbl_start = hdr.end() + m_table_start.start()
    m_table_end = re.search(r"</table\s*>", html[tbl_start:], flags=re.I | re.S)
    if not m_table_end:
        return html + sections_html
    insert_at = tbl_start + m_table_end.end()
    return html[:insert_at] + sections_html + html[insert_at:]

def build_digest_html(files, keywords):
    """
    Build final HTML by injecting data into the provided Outlook/Word HTML template.
    Falls back to a compact Outlook-safe layout if the template cannot be read.
    """
    program_date = datetime.now().strftime("%d %B %Y")

    # --- collect matches per file & tallies for the detection table -----------
    chambers = ["House of Assembly", "Legislative Council"]
    counts = {ch: {kw: 0 for kw in keywords} for ch in chambers}
    totals = {kw: 0 for kw in keywords}
    total_matches = 0
    matches_by_file = []

    for f in sorted(files, key=lambda x: (parse_date_from_filename(Path(x).name), Path(x).name)):
        text = Path(f).read_text(encoding="utf-8", errors="ignore")
        chamber = parse_chamber_from_filename(Path(f).name)
        matches = extract_matches(text, keywords)
        matches.sort(key=lambda item: min(item[3]) if item[3] else 10**9)
        total_matches += len(matches)
        matches_by_file.append((f, matches))
        for kw_set, _excerpt_html, _speaker, _lines, _s, _e in matches:
            for kw in kw_set:
                if chamber in counts:
                    counts[chamber][kw] += 1
                totals[kw] += 1

    # Build detection rows HTML (matching the template's styles)
    det_rows = []
    for kw in keywords:
        hoa = counts["House of Assembly"].get(kw, 0)
        lc = counts["Legislative Council"].get(kw, 0)
        tot = hoa + lc
        det_rows.append(_build_detection_row(kw, hoa, lc, tot))
    detection_rows_html = "".join(det_rows)

    # Build per-file sections
    sections_html = _compose_sections(matches_by_file)

    # --- try to load and inject into your HTML template -----------------------
    tpl_html = None
    if TEMPLATE_HTML_PATH.exists():
        try:
            tpl_html = TEMPLATE_HTML_PATH.read_text(encoding="windows-1252", errors="ignore")
        except Exception:
            tpl_html = None

    if tpl_html:
        # Replace Program Run date
        tpl_html = re.sub(r"Program Run:\s*\[DATE\]", f"Program Run: {program_date}", tpl_html, flags=re.I)

        # Replace detection table body with our rows
        tpl_html = _replace_detection_rows_in_template(tpl_html, detection_rows_html)

        # Drop any sample file sections and inject live sections
        tpl_html = _strip_existing_file_sections(tpl_html)
        tpl_html = _inject_sections_after_detection(tpl_html, sections_html)

        return tpl_html, total_matches, counts

    # --- Fallback: minimal Outlook-safe layout (if template missing) ----------
    # (Keeps colours & fonts close to your design)
    fallback_html = (
        "<!DOCTYPE html><html><head><meta http-equiv='Content-Type' content='text/html; charset=utf-8'>"
        "<meta name='x-apple-disable-message-reformatting'></head>"
        "<body style='margin:0;padding:0;background:#F3F6F9;'>"
        "<table role='presentation' width='100%' cellspacing='0' cellpadding='0' border='0' style='background:#F3F6F9;'>"
        "<tr><td align='center' style='padding:20px 10px;'>"
        "<table role='presentation' width='768' cellspacing='0' cellpadding='0' border='0' "
        "style='width:768px;max-width:768px;background:#FFFFFF;border:1px solid #D4D9E0;'>"
        "<tr><td style='background:#475560;color:#FFFFFF;padding:18px 20px;'>"
        f"<div style='font:bold 22px/26px Segoe UI,Arial,sans-serif;'>{_html_escape(DEFAULT_TITLE)}</div>"
        f"<div style='font:12px/16px Segoe UI,Arial,sans-serif;margin-top:4px;'>Program Run: {program_date}</div>"
        "</td></tr>"
        "<tr><td style='padding:14px 20px 8px 20px;'><div style='font:bold 14px/18px Segoe UI,Arial,sans-serif;color:#24313F;'>Detection Match by Chamber</div></td></tr>"
        "<tr><td style='padding:0 20px 14px 20px;'>"
        "<table role='presentation' width='100%' cellpadding='0' cellspacing='0' border='0' "
        "style='border-collapse:collapse;'>"
        "<tr>"
        "<td style='border:1px solid #D8DCE0;background:#4A5A6A;padding:8px 10px;font:bold 12px/16px Segoe UI,Arial,sans-serif;color:#fff;'>Keyword</td>"
        "<td align='center' style='border:1px solid #D8DCE0;background:#4A5A6A;padding:8px 10px;font:bold 12px/16px Segoe UI,Arial,sans-serif;color:#fff;'>House of Assembly</td>"
        "<td align='center' style='border:1px solid #D8DCE0;background:#4A5A6A;padding:8px 10px;font:bold 12px/16px Segoe UI,Arial,sans-serif;color:#fff;'>Legislative Council</td>"
        "<td align='center' style='border:1px solid #D8DCE0;background:#4A5A6A;padding:8px 10px;font:bold 12px/16px Segoe UI,Arial,sans-serif;color:#fff;'>Total</td>"
        "</tr>"
        f"{detection_rows_html}"
        "</table></td></tr>"
        f"<tr><td style='padding:6px 20px 18px 20px;'><table role='presentation' width='100%' cellpadding='0' cellspacing='0' border='0'>{sections_html}</table></td></tr>"
        "<tr><td style='padding:10px 20px 18px 20px;'>"
        "<table role='presentation' width='100%' cellpadding='0' cellspacing='0' border='0'>"
        "<tr><td style='text-align:center;font:bold 11px/16px Segoe UI,Arial,sans-serif;color:#24313F;'>**THIS PROGRAM IS IN BETA TESTING – DO NOT FORWARD**</td></tr>"
        "<tr><td style='text-align:center;font:11px/16px Segoe UI,Arial,sans-serif;color:#6A7682;padding-top:4px;'>"
        "Contact developer with any issues, queries, or suggestions: William.Manning@FederalGroup.com.au</td></tr>"
        "</table></td></tr>"
        "</table></td></tr></table></body></html>"
    )
    return fallback_html, total_matches, counts

def load_sent_log():
    if LOG_FILE.exists():
        return {line.strip() for line in LOG_FILE.read_text().splitlines() if line.strip()}
    return set()

def update_sent_log(files):
    with LOG_FILE.open("a", encoding="utf-8") as f:
        for file in files:
            f.write(f"{Path(file).name}\n")

# --- Main --------------------------------------------------------------------
def main():
    EMAIL_USER = os.environ["EMAIL_USER"]
    EMAIL_PASS = os.environ["EMAIL_PASS"]
    EMAIL_TO = os.environ["EMAIL_TO"]

    # Gmail defaults, overridable via env
    SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
    SMTP_STARTTLS = os.environ.get("SMTP_STARTTLS", "1").lower() in ("1", "true", "yes")
    SMTP_SSL = os.environ.get("SMTP_SSL", "0").lower() in ("1", "true", "yes")

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

    # Pass HTML string directly so yagmail doesn't inject <br> separators.
    yag.send(
        to=to_list,
        subject=subject,
        contents=body_html,  # HTML string
        attachments=files,   # also attach the transcripts
    )

    update_sent_log(files)
    print(f"✅ Email sent to {EMAIL_TO} with {len(files)} file(s), {total_hits} match(es).")

if __name__ == "__main__":
    main()
