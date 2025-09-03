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

# --- Tunables ---------------------------------------------------------------
MAX_SNIPPET_CHARS = 800
WINDOW_PAD_SENTENCES = 1
FIRST_SENT_FOLLOWING = 2
MERGE_IF_GAP_GT = 2

# --- Template inputs ---------------------------------------------------------
TEMPLATE_HTML_PATH = Path(os.environ.get("TEMPLATE_HTML_PATH", "email_template (1).html"))
DEFAULT_TITLE = "Hansard Monitor – BETA Version 18.3"

# --- Helpers -----------------------------------------------------------------
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
        if " " in kw:
            pat = re.compile(re.escape(_html_escape(kw)), re.IGNORECASE)
        else:
            pat = re.compile(rf"\b{re.escape(_html_escape(kw))}\b", re.IGNORECASE)
        out = pat.sub(lambda m: "<b><span style='background:lightgrey;mso-highlight:lightgrey'>"
                                 + m.group(0) + "</span></b>", out)
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

        use_llm = os.environ.get("ATTRIB_WITH_LLM", "").lower() in ("1", "true", "yes")
        if use_llm and _looks_suspicious(speaker):
            candidates = sorted({u["speaker"] for u in utts if u["speaker"]})
            earliest_line = min(min(w[3]) for w in merged)
            guess = _llm_qc_speaker(all_lines, earliest_line, candidates, timeout=int(os.environ.get("ATTRIB_LLM_TIMEOUT", "30")))
            if guess:
                speaker = guess

        for win in merged:
            excerpt_html, line_list, kws_in_excerpt, win_start, win_end = _excerpt_from_window_html(utt, win, keywords)
            results.append((set(kws_in_excerpt), excerpt_html, speaker, line_list, win_start, win_end))

    return results

# --- Aggressive Outlook whitespace cleaner -----------------------------------
_EMPTY_MSOP_RE = re.compile(
    r"<p\b[^>]*>(?:\s|&nbsp;|<br[^>]*>|"
    r"<o:p>\s*&nbsp;\s*</o:p>|"
    r"<span\b[^>]*>(?:\s|&nbsp;|<br[^>]*>)*</span>)*</p>",
    flags=re.I,
)
def _tighten_outlook_whitespace(html: str) -> str:
    html = _EMPTY_MSOP_RE.sub("", html)  # remove empty paras (even wrapped)
    html = re.sub(r"(?:\s*<br[^>]*>\s*){1,}", "", html, flags=re.I)  # kill stray <br>
    html = re.sub(r"(</table>)\s+(?=(?:<!--.*?-->\s*)*<table\b)", r"\1", html, flags=re.I | re.S)  # tighten table-to-table
    html = re.sub(r">\s*(?:&nbsp;|<br[^>]*>|\s)+</td>", "></td>", html, flags=re.I)  # trim inside <td>
    return html

# --- Visual assembly ----------------------------------------------------------
SUMMARY_ROW_TMPL = """
<tr>
 <td width="28%" style='width:28.12%;border-top:none;border-left:solid #D8DCE0 1.0pt;border-bottom:solid #ECF0F1 1.0pt;border-right:none;mso-border-left-alt:solid #D8DCE0 .75pt;mso-border-bottom-alt:solid #ECF0F1 .75pt;padding:6.0pt 7.5pt 6.0pt 7.5pt'><p class=MsoNormal style='margin:0;'><b><span style='font-size:10.0pt;font-family:"Segoe UI",sans-serif;color:black'>{kw}</span></b></p></td>
 <td width="28%" style='width:28.12%;border:none;border-bottom:solid #ECF0F1 1.0pt;mso-border-bottom-alt:solid #ECF0F1 .75pt;padding:6.0pt 7.5pt 6.0pt 7.5pt'><p class=MsoNormal align=center style='text-align:center;margin:0;'><b><span style='font-size:10.0pt;font-family:"Segoe UI",sans-serif;color:black'>{ha}</span></b></p></td>
 <td width="28%" style='width:28.14%;border:none;border-bottom:solid #ECF0F1 1.0pt;mso-border-bottom-alt:solid #ECF0F1 .75pt;padding:6.0pt 7.5pt 6.0pt 7.5pt'><p class=MsoNormal align=center style='text-align:center;margin:0;'><b><span style='font-size:10.0pt;font-family:"Segoe UI",sans-serif;color:black'>{lc}</span></b></p></td>
 <td width="15%" style='width:15.62%;border-top:none;border-left:none;border-bottom:solid #ECF0F1 1.0pt;border-right:solid #D8DCE0 1.0pt;mso-border-bottom-alt:solid #ECF0F1 .75pt;mso-border-right-alt:solid #D8DCE0 .75pt;padding:6.0pt 7.5pt 6.0pt 7.5pt'><p class=MsoNormal align=center style='text-align:center;margin:0;'><b><span style='font-size:10.0pt;font-family:"Segoe UI",sans-serif;color:black'>{total}</span></b></p></td>
</tr>
"""

def _build_file_section_html(filename: str, matches):
    """Tight, Outlook-safe section with fixed line-heights and no MsoNormal paras."""
    def _esc(s): return _html_escape(s) if s else ""

    rows = []
    for idx, (_kw_set, excerpt_html, speaker, line_list, _s, _e) in enumerate(matches, 1):
        line_txt = f"line {line_list[0]}" if len(line_list) == 1 else "lines " + ", ".join(str(n) for n in line_list)

        rows.append(f"""
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="border-collapse:collapse;border:1px solid #D8DCE0;">
  <tr>
    <td style="background:#ECF0F1;border-bottom:1px solid #D8DCE0;padding:6px 10px;font-size:0;line-height:0;mso-line-height-rule:exactly;">
      <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="border-collapse:collapse;">
        <tr>
          <td width="36" align="center" valign="middle" style="background:#4A5A6A;border:0;height:24px;">
            <div style="font:bold 10pt 'Segoe UI',sans-serif;color:#FFFFFF;line-height:24px;mso-line-height-rule:exactly;">{idx}</div>
          </td>
          <td width="10" style="font-size:0;line-height:0;">&nbsp;</td>
          <td valign="middle">
            <div style="font:bold 10pt 'Segoe UI',sans-serif;color:#24313F;text-transform:uppercase;line-height:16px;mso-line-height-rule:exactly;">{_esc(speaker) or 'UNKNOWN'}</div>
          </td>
          <td align="right" valign="middle">
            <div style="font:10pt 'Segoe UI',sans-serif;color:#6A7682;line-height:16px;mso-line-height-rule:exactly;">{line_txt}</div>
          </td>
        </tr>
      </table>
    </td>
  </tr>
  <tr>
    <td style="padding:10px 12px;">
      <div style="font:10pt 'Segoe UI',sans-serif;color:#1F2A36;line-height:19px;mso-line-height-rule:exactly;">{excerpt_html}</div>
    </td>
  </tr>
</table>
<table role='presentation' width='100%' cellpadding='0' cellspacing='0' border='0'><tr><td style='height:4px;line-height:4px;font-size:0;'>&nbsp;</td></tr></table>
""")

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
      {''.join(rows)}
    </td>
  </tr>
</table>
"""

def build_digest_html(files: list[str], keywords: list[str]):
    template_html = TEMPLATE_HTML_PATH.read_text(encoding="windows-1252", errors="ignore")
    run_date = datetime.now().strftime("%d %B %Y")
    template_html = template_html.replace("[DATE]", run_date)

    # Ensure section title font family is Segoe UI (belt-and-suspenders)
    template_html = template_html.replace(
        '<span style="font-size:12.0pt;color:black">Detection Match by Chamber</span>',
        '<span style="font-size:12.0pt;font-family:\'Segoe UI\',sans-serif;color:black">Detection Match by Chamber</span>',
    )

    counts = {kw: {"House of Assembly": 0, "Legislative Council": 0} for kw in keywords}
    sections = []
    total_hits = 0

    def parse_chamber_from_filename(filename: str) -> str:
        name = filename.lower()
        if "house_of_assembly" in name: return "House of Assembly"
        if "legislative_council" in name: return "Legislative Council"
        return "Unknown"

    for f in files:
        text = Path(f).read_text(encoding="utf-8", errors="ignore")
        matches = extract_matches(text, keywords)
        if not matches:
            continue
        total_hits += len(matches)
        chamber = parse_chamber_from_filename(Path(f).name)
        for kw_set, _, _, _, _, _ in matches:
            for kw in kw_set:
                counts.setdefault(kw, {"House of Assembly": 0, "Legislative Council": 0})
                if chamber in counts[kw]:
                    counts[kw][chamber] += 1
        sections.append(_build_file_section_html(Path(f).name, matches))

    # Build summary table rows with tight paragraph margins
    summary_rows = []
    for kw in keywords:
        ha = counts.get(kw, {}).get("House of Assembly", 0)
        lc = counts.get(kw, {}).get("Legislative Council", 0)
        summary_rows.append(SUMMARY_ROW_TMPL.format(kw=_html_escape(kw), ha=ha, lc=lc, total=ha + lc))
    summary_html = "".join(summary_rows)

    # Replace the sample summary row (first data row after the header)
    template_html = re.sub(
        r"<tr>\s*<td[^>]*>\s*<p[^>]*>\s*<b>\s*<span[^>]*>\s*pokies\s*</span>.*?</tr>",
        summary_html,
        template_html,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Replace the sample section block
    sections_html = "".join(sections)
    template_html = re.sub(
        r"<!--\s*Sample section to be replaced\s*-->.*?<!--\s*End sample section\s*-->",
        sections_html,
        template_html,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Final pass: strip Outlook/Word ghost spacing
    template_html = _tighten_outlook_whitespace(template_html)
    return template_html, total_hits, counts

# --- Sent log helpers --------------------------------------------------------
def load_sent_log() -> set[str]:
    if LOG_FILE.exists():
        with open(LOG_FILE, encoding="utf-8") as f:
            return {ln.strip() for ln in f if ln.strip()}
    return set()

def update_sent_log(files: list[str]):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        for fp in files:
            f.write(Path(fp).name + "\n")

# --- Main --------------------------------------------------------------------
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

    # Pass the HTML string directly so yagmail doesn't insert <br> separators.
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
