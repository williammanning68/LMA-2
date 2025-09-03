import os
import re
import glob
from bisect import bisect_right
from pathlib import Path
from datetime import datetime
import yagmail
import subprocess  # only used if ATTRIB_WITH_LLM=1

# ---------------------------- CONFIG / PATHS ---------------------------------

# Robust template resolver: checks env, common names, then searches repo
def _resolve_template_path() -> Path:
    # 1) environment override
    env_path = os.environ.get("TEMPLATE_HTML_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    # 2) common filenames used in this project
    common_names = [
        "email_template.html",
        "email_template.htm",
        "email_template (1).html",
        "email_template (1).htm",
        "Hansard Monitor - Email Format - Version 3.htm",
        "Hansard Monitor - Email Format - Version 3.html",
        # in a templates/ subfolder
        "templates/email_template.html",
        "templates/email_template.htm",
        "templates/email_template (1).html",
        "templates/email_template (1).htm",
        "templates/Hansard Monitor - Email Format - Version 3.htm",
        "templates/Hansard Monitor - Email Format - Version 3.html",
    ]
    script_dir = Path(__file__).resolve().parent
    for name in common_names:
        cand = script_dir / name
        if cand.exists():
            return cand

    # 3) fallback: search repo (script directory) for any plausible .htm(l)
    patterns = ["**/*.htm", "**/*.html"]
    keywords = ("email_template", "Email Format", "Hansard Monitor")
    for pat in patterns:
        for fp in script_dir.glob(pat):
            # prefer matches with likely keywords in filename
            if any(k.lower() in fp.name.lower() for k in keywords):
                return fp
    # last resort: first .htm(l) we find
    for pat in patterns:
        for fp in script_dir.glob(pat):
            return fp

    raise FileNotFoundError(
        "Could not locate an HTML template. Provide TEMPLATE_HTML_PATH env var, "
        "or place one of the common files next to send_email.py "
        "(e.g., 'Hansard Monitor - Email Format - Version 3.htm')."
    )

TEMPLATE_HTML_PATH = _resolve_template_path()

# Sent-log to avoid re-sending the same transcripts
LOG_FILE = Path("sent.log")

# Subject prefix
DEFAULT_TITLE = "Hansard Monitor – BETA Version 18.3"

# Transcript parsing & excerpt settings
MAX_SNIPPET_CHARS = 800
WINDOW_PAD_SENTENCES = 1
FIRST_SENT_FOLLOWING = 2
MERGE_IF_GAP_GT = 2

# ------------------------------- KEYWORDS ------------------------------------

def load_keywords():
    """
    Load keywords from keywords.txt or KEYWORDS env var.
    """
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

# ------------------------ TRANSCRIPT SEGMENTATION -----------------------------

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

            # naive sentence splitter on punctuation + whitespace
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
    i = max(0, min(i, len(line_nums) - 1))
    return line_nums[i]

# ---------------------------- MATCH COLLECTION -------------------------------

def _compile_kw_patterns(keywords):
    pats = []
    for kw in sorted(keywords, key=len, reverse=True):
        pats.append((kw, re.compile(re.escape(kw) if " " in kw else rf"\b{re.escape(kw)}\b", re.I)))
    return pats

def _collect_hits_in_utterance(utt, kw_pats):
    hits = []
    joined = utt["joined"]
    for si, (a, b) in enumerate(utt["sents"]):
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

# ----------------------------- HTML HELPERS ----------------------------------

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
    # Extract + normalize newlines so we don't spam <br>
    sents = utt["sents"]
    joined = utt["joined"]
    start, end, kws, lines = win
    a = sents[start][0]
    b = sents[end][1]

    raw = joined[a:b]
    raw = re.sub(r"\r\n?", "\n", raw)      # unify CRLF/CR
    raw = re.sub(r"\n{2,}", "\n", raw)     # collapse blank lines
    raw = raw.strip()

    if len(raw) > MAX_SNIPPET_CHARS:
        raw = raw[:MAX_SNIPPET_CHARS].rstrip() + "…"

    html = _html_escape(raw)
    html = _highlight_keywords_html(html, keywords)
    html = html.replace("\n", "<br>")
    html = re.sub(r"(?:<br\s*/?>\s*){2,}", "<br>", html)                 # collapse runs
    html = re.sub(r"^(?:<br\s*/?>\s*)+|(?:<br\s*/?>\s*)+$", "", html)    # trim ends

    start_line = _line_for_char_offset(utt["line_offsets"], utt["line_nums"], a)
    end_line = _line_for_char_offset(utt["line_offsets"], utt["line_nums"], max(a, b - 1))
    return html, sorted(lines), sorted(kws, key=str.lower), start_line, end_line

def _looks_suspicious(s: str | None) -> bool:
    if not s:
        return True
    s2 = s.strip()
    if re.match(
        r"(?i)^(Mr|Ms|Mrs|Miss|Hon|Dr|Prof|Professor|Premier|Madam SPEAKER|The SPEAKER|The PRESIDENT|The CLERK|Deputy Speaker|Deputy President)"
        r"(?:\s+[A-Z][A-Z'’\-]+(?:\s+[A-Z][A-Z'’\-]+){0,3})?$",
        s2,
    ):
        return False
    if re.match(r"^[A-Z][A-Z'’\-]+(?:\s+[A-Z][A-Z'’\-]+){0,3}$", s2):
        return False
    return True

def _llm_qc_speaker(full_lines, hit_line_no, candidates, model=None, timeout=30):
    # Optional — only if you opt in with ATTRIB_WITH_LLM=1
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
        if os.environ.get("ATTRIB_WITH_LLM", "").lower() in ("1", "true", "yes") and _looks_suspicious(speaker):
            candidates = sorted({u["speaker"] for u in utts if u["speaker"]})
            earliest_line = min(min(w[3]) for w in merged)
            guess = _llm_qc_speaker(all_lines, earliest_line, candidates, timeout=int(os.environ.get("ATTRIB_LLM_TIMEOUT", "30")))
            if guess:
                speaker = guess
        for win in merged:
            excerpt_html, line_list, kws_in_excerpt, win_start, win_end = _excerpt_from_window_html(utt, win, keywords)
            results.append((set(kws_in_excerpt), excerpt_html, speaker, line_list, win_start, win_end))
    return results

# ----------------------- OUTLOOK/GMAIL WHITESPACE FIX ------------------------

_EMPTY_MSOP_RE = re.compile(
    r"<p\b[^>]*>(?:\s|&nbsp;|<br[^>]*>|"
    r"<o:p>\s*&nbsp;\s*</o:p>|"
    r"<span\b[^>]*>(?:\s|&nbsp;|<br[^>]*>)*</span>)*</p>",
    flags=re.I,
)

def _tighten_outlook_whitespace(html: str) -> str:
    # 1) Remove empty Word/Outlook paragraphs (even wrapped)
    html = _EMPTY_MSOP_RE.sub("", html)
    # 2) Collapse runs of <br> to a single <br> (keep one if present)
    html = re.sub(r"(?:\s*<br[^>]*>\s*){2,}", "<br>", html, flags=re.I)
    # 3) Remove whitespace/comments between adjacent tables
    html = re.sub(r"(</table>)\s+(?=(?:<!--.*?-->\s*)*<table\b)", r"\1", html, flags=re.I | re.S)
    # 4) Trim blank space just inside table cells
    html = re.sub(r">\s*(?:&nbsp;|<br[^>]*>|\s)+</td>", "></td>", html, flags=re.I)
    return html

# --------------------------- TEMPLATE INJECTION ------------------------------

def _build_detection_row(kw, hoa, lc, tot):
    # Keep template's fonts; use px paddings for reliability
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
    """
    Find the 'Detection Match by Chamber' block -> next inner table -> keep header row, replace following rows.
    """
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

def _strip_existing_file_sections(html):
    # Remove any sample “match(es)” block that shipped in the template
    pattern = re.compile(
        r"<!--\s*Sample section to be replaced\s*-->.*?<!--\s*End sample section\s*-->",
        flags=re.I | re.S
    )
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

# ------------------------ PER-FILE “CARD” SECTIONS ---------------------------

def _build_file_section_html(filename: str, matches):
    """
    Outlook-safe, tight cards:
    - Top-aligned cells
    - Pixel line-heights + mso-line-height-rule:exactly
    - No trailing spacer after the last card
    """
    def _esc(s): return _html_escape(s) if s else ""

    card_rows = []
    for idx, (_kw_set, excerpt_html, speaker, line_list, _s, _e) in enumerate(matches, 1):
        line_txt = f"line {line_list[0]}" if len(line_list) == 1 else "lines " + ", ".join(str(n) for n in line_list)
        card_rows.append(f"""
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="border-collapse:collapse;border:1px solid #D8DCE0;">
  <tr>
    <td valign="top" style="background:#ECF0F1;border-bottom:1px solid #D8DCE0;padding:6px 10px;font-size:0;line-height:0;mso-line-height-rule:exactly;vertical-align:top;">
      <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="border-collapse:collapse;">
        <tr>
          <td width="36" align="center" valign="top" style="background:#4A5A6A;border:0;height:24px;vertical-align:top;">
            <div style="font:bold 10pt 'Segoe UI',sans-serif;color:#FFFFFF;line-height:24px;mso-line-height-rule:exactly;display:block;">{idx}</div>
          </td>
          <td width="10" style="font-size:0;line-height:0;vertical-align:top;">&nbsp;</td>
          <td valign="top" style="vertical-align:top;">
            <div style="font:bold 10pt 'Segoe UI',sans-serif;color:#24313F;text-transform:uppercase;line-height:16px;mso-line-height-rule:exactly;display:block;">{_esc(speaker) or 'UNKNOWN'}</div>
          </td>
          <td align="right" valign="top" style="vertical-align:top;">
            <div style="font:10pt 'Segoe UI',sans-serif;color:#6A7682;line-height:16px;mso-line-height-rule:exactly;display:block;">{line_txt}</div>
          </td>
        </tr>
      </table>
    </td>
  </tr>
  <tr>
    <td valign="top" style="padding:10px 12px;vertical-align:top;">
      <div style="font:10pt 'Segoe UI',sans-serif;color:#1F2A36;line-height:19px;mso-line-height-rule:exactly;display:block;">{excerpt_html}</div>
    </td>
  </tr>
</table>
""")

    # spacer BETWEEN cards (not after last)
    spacer = "<table role='presentation' width='100%' cellpadding='0' cellspacing='0' border='0'><tr><td style='height:4px;line-height:4px;font-size:0;'>&nbsp;</td></tr></table>"
    cards_html = spacer.join(card_rows)

    return f"""
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="border-collapse:collapse;">
  <tr>
    <td style="border-left:3px solid #C5A572;background:#F7F9FA;padding:8px 12px;">
      <div style="font:bold 10pt 'Segoe UI',sans-serif;color:#000;line-height:16px;mso-line-height-rule:exactly;">{_esc(filename)}</div>
      <div style="font:10pt 'Segoe UI',sans-serif;color:#000;line-height:16px;mso-line-height-rule:exactly;">{len(matches)} match(es)</div>
    </td>
  </tr>
  <tr>
    <td style="border:1px solid #D8DCE0;border-top:none;background:#FFFFFF;padding:8px 9px;">
      {cards_html}
    </td>
  </tr>
</table>
"""

# ------------------------------ BUILD DIGEST ---------------------------------

def _parse_chamber_from_filename(filename: str) -> str:
    name = filename.lower()
    if "house_of_assembly" in name:
        return "House of Assembly"
    if "legislative_council" in name:
        return "Legislative Council"
    return "Unknown"

def build_digest_html(files: list[str], keywords: list[str]):
    # Read template (prefer Windows-1252; fallback to UTF-8)
    try:
        template_html = TEMPLATE_HTML_PATH.read_text(encoding="windows-1252", errors="ignore")
    except Exception:
        template_html = TEMPLATE_HTML_PATH.read_text(encoding="utf-8", errors="ignore")

    # Program date
    run_date = datetime.now().strftime("%d %B %Y")
    template_html = template_html.replace("[DATE]", run_date)

    # Ensure the section title uses Segoe UI even if missing in template
    template_html = template_html.replace(
        '<span style="font-size:12.0pt;color:black">Detection Match by Chamber</span>',
        '<span style="font-size:12.0pt;font-family:\'Segoe UI\',sans-serif;color:black">Detection Match by Chamber</span>',
    )

    # Collect matches and tallies
    counts = {kw: {"House of Assembly": 0, "Legislative Council": 0} for kw in keywords}
    sections = []
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
        sections.append(_build_file_section_html(Path(f).name, matches))

    # Build detection summary table rows
    det_rows = []
    for kw in keywords:
        hoa = counts.get(kw, {}).get("House of Assembly", 0)
        lc = counts.get(kw, {}).get("Legislative Council", 0)
        det_rows.append(_build_detection_row(kw, hoa, lc, hoa + lc))
    detection_rows_html = "".join(det_rows)

    # Replace detection rows robustly
    template_html = _replace_detection_rows_in_template(template_html, detection_rows_html)

    # Remove any sample section block in template, then inject our sections after the detection table
    template_html = _strip_existing_file_sections(template_html)
    sections_html = "".join(sections)
    template_html = _inject_sections_after_detection(template_html, sections_html)

    # Final scrub: remove Outlook/Word ghost spacing & stray <br> runs
    template_html = _tighten_outlook_whitespace(template_html)

    return template_html, total_matches, counts

# ------------------------------ SEND / LOG -----------------------------------

def load_sent_log() -> set[str]:
    if LOG_FILE.exists():
        with open(LOG_FILE, encoding="utf-8") as f:
            return {ln.strip() for ln in f if ln.strip()}
    return set()

def update_sent_log(files: list[str]):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        for fp in files:
            f.write(Path(fp).name + "\n")

# ---------------------------------- MAIN -------------------------------------

def main():
    EMAIL_USER = os.environ["EMAIL_USER"]
    EMAIL_PASS = os.environ["EMAIL_PASS"]
    EMAIL_TO = os.environ["EMAIL_TO"]

    SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
    SMTP_STARTTLS = os.environ.get("SMTP_STARTTLS", "1").lower() in ("1", "true", "yes")
    SMTP_SSL = os.environ.get("SMTP_SSL", "0").lower() in ("1", "true", "yes")

    keywords = load_keywords()
    if not keywords:
        raise SystemExit("No keywords found (keywords.txt or KEYWORDS env var).")

    # Gather transcripts (avoid re-sending same file names)
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

    # Send the HTML body directly (yagmail won’t inject <br> if given a single HTML string)
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
