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

# --- Tunables ----------------------------------------------------------------
MAX_SNIPPET_CHARS = 800  # upper bound after merging windows; keep readable but compact
WINDOW_PAD_SENTENCES = 1  # for non-first-sentence hits: one sentence either side
FIRST_SENT_FOLLOWING = 2  # for first-sentence hits: include next two sentences
MERGE_IF_GAP_GT = 2  # Only merge windows if the gap (in sentences) is > this value

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
        (?:[\s.]+(?P<name>[A-Z][A-Z''\-]+(?:\s+[A-Z][A-Z''\-]+){0,3}))?
        |
        (?P<name_only>[A-Z][A-Z''\-]+(?:\s+[A-Z][A-Z''\-]+){0,3})
    )
    (?:\s*\([^)]*\))?
    \s*(?::|[-–—]\s)
    """,
    re.VERBOSE,
)
CONTENT_COLON_RE = re.compile(r"^(Then|There|And|But|So|If|When|Now|Finally|First|Second|Third)\b", re.IGNORECASE)
TIME_STAMP_RE = re.compile(r"^\[\d{1,2}\.\d{2}\s*(a|p)\.m\.\]$", re.IGNORECASE)
UPPER_HEADING_RE = re.compile(r"^[A-Z][A-Z\s''—\-&,;:.()]+$")
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
        r"(?:\s+[A-Z][A-Z''\-]+(?:\s+[A-Z][A-Z''\-]+){0,3})?$",
        s,
    ):
        return False
    if re.match(r"^[A-Z][A-Z''\-]+(?:\s+[A-Z][A-Z''\-]+){0,3}$", s):
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
    """
    Return list of: (kw_set, excerpt_html, speaker, line_numbers_list, win_start_line, win_end_line)
    """
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
        # Remove identical (start,end) windows to avoid duplicate excerpts
        wins = _dedup_windows(wins)
        # Keep separate unless far apart; if far (>2), merge into a longer excerpt
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

# --- Digest / email pipeline (HTML) ------------------------------------------
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

def build_digest_html(files, keywords):
    """Build HTML in the Hansard Monitor - BETA Version 18.3 layout."""
    now_utc = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    chambers = ["House of Assembly", "Legislative Council"]
    counts = {ch: {kw: 0 for kw in keywords} for ch in chambers}
    totals = {kw: 0 for kw in keywords}
    doc_sections = []
    total_matches = 0

    # Build doc sections and accumulate counts
    for f in sorted(files, key=lambda x: (parse_date_from_filename(Path(x).name), Path(x).name)):
        text = Path(f).read_text(encoding="utf-8", errors="ignore")
        chamber = parse_chamber_from_filename(Path(f).name)
        matches = extract_matches(text, keywords)
        
        if not matches:
            # Still show the file title with "0 match(es)"
            file_section = f'''
            <div style="margin-bottom:24px;">
                <div style="background:#F0F4F7;border-left:4px solid #5C6F7F;padding:12px;margin-bottom:12px;">
                    <div style="font:bold 16px/22px Arial,Helvetica,sans-serif;color:#2C3E50;margin-bottom:4px;">
                        {_html_escape(Path(f).name)}
                    </div>
                    <div style="font:14px/20px Arial,Helvetica,sans-serif;color:#7F8C8D;">
                        0 match(es)
                    </div>
                </div>
            </div>
            '''
            doc_sections.append(file_section)
            continue

        matches.sort(key=lambda item: min(item[3]) if item[3] else 10**9)
        total_matches += len(matches)

        # Build file section with matches
        match_cards = []
        for i, (kw_set, excerpt_html, speaker, line_list, win_start, win_end) in enumerate(matches, 1):
            for kw in kw_set:
                if chamber in counts:
                    counts[chamber][kw] += 1
                totals[kw] += 1
            
            # Format line numbers
            if line_list and len(line_list) > 1:
                lines_str = "lines " + ", ".join(str(n) for n in sorted(set(line_list)))
            elif line_list:
                lines_str = "line " + str(line_list[0])
            else:
                lines_str = "line " + str(win_start)
            
            # Show Unknown if missing or looks suspicious
            speaker_display = speaker if (speaker and not _looks_suspicious(speaker)) else "Unknown"
            
            match_cards.append(f'''
            <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="margin-bottom:12px;">
                <tr>
                    <td width="40" align="center" valign="top" style="background:#5C6F7F;color:#FFFFFF;font:bold 16px/40px Arial,Helvetica,sans-serif;border-radius:4px;">
                        {i}
                    </td>
                    <td width="12">&nbsp;</td>
                    <td valign="top" style="background:#FFFFFF;border:1px solid #D5DBDF;padding:12px;border-radius:4px;">
                        <div style="font:bold 14px/20px Arial,Helvetica,sans-serif;color:#2C3E50;margin-bottom:8px;">
                            {_html_escape(speaker_display)}
                            <span style="font-weight:normal;color:#7F8C8D;margin-left:16px;">{lines_str}</span>
                        </div>
                        <div style="font:14px/22px Arial,Helvetica,sans-serif;color:#34495E;">
                            {excerpt_html}
                        </div>
                    </td>
                </tr>
            </table>
            ''')

        file_section = f'''
        <div style="margin-bottom:24px;">
            <div style="background:#F0F4F7;border-left:4px solid #5C6F7F;padding:12px;margin-bottom:12px;">
                <div style="font:bold 16px/22px Arial,Helvetica,sans-serif;color:#2C3E50;margin-bottom:4px;">
                    {_html_escape(Path(f).name)}
                </div>
                <div style="font:14px/20px Arial,Helvetica,sans-serif;color:#7F8C8D;">
                    {len(matches)} match(es)
                </div>
            </div>
            <div style="padding-left:12px;">
                {''.join(match_cards)}
            </div>
        </div>
        '''
        doc_sections.append(file_section)

    # Build summary table rows
    table_rows = []
    for kw in keywords:
        hoa = counts["House of Assembly"][kw] if "House of Assembly" in counts else 0
        lc = counts["Legislative Council"][kw] if "Legislative Council" in counts else 0
        tot = totals[kw]
        table_rows.append(f'''
        <tr>
            <td style="padding:10px;border:1px solid #D5DBDF;font:14px/20px Arial,Helvetica,sans-serif;color:#2C3E50;">
                {_html_escape(kw)}
            </td>
            <td align="center" style="padding:10px;border:1px solid #D5DBDF;font:bold 14px/20px Arial,Helvetica,sans-serif;color:#2C3E50;">
                {hoa}
            </td>
            <td align="center" style="padding:10px;border:1px solid #D5DBDF;font:bold 14px/20px Arial,Helvetica,sans-serif;color:#2C3E50;">
                {lc}
            </td>
            <td align="center" style="padding:10px;border:1px solid #D5DBDF;font:bold 14px/20px Arial,Helvetica,sans-serif;color:#2C3E50;">
                {tot}
            </td>
        </tr>
        ''')

    # Assemble final HTML
    html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
        <style>
            @media only screen and (max-width: 600px) {{
                .container {{ width: 100% !important; }}
            }}
        </style>
    </head>
    <body style="margin:0;padding:0;background:#ECF0F1;font-family:Arial,Helvetica,sans-serif;">
        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background:#ECF0F1;">
            <tr>
                <td align="center" style="padding:20px;">
                    <table class="container" role="presentation" width="700" cellspacing="0" cellpadding="0" border="0" style="max-width:700px;background:#FFFFFF;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.1);">
                        <!-- Header -->
                        <tr>
                            <td style="background:#5C6F7F;color:#FFFFFF;padding:20px;border-radius:8px 8px 0 0;">
                                <h1 style="margin:0;font:bold 24px/30px Arial,Helvetica,sans-serif;text-align:center;">
                                    Hansard Monitor – BETA Version 18.3
                                </h1>
                                <p style="margin:8px 0 0 0;font:14px/20px Arial,Helvetica,sans-serif;text-align:center;">
                                    Program Run: {now_utc}
                                </p>
                            </td>
                        </tr>
                        
                        <!-- Content -->
                        <tr>
                            <td style="padding:24px;">
                                <!-- Detection Match by Chamber Table -->
                                <div style="margin-bottom:32px;">
                                    <h2 style="font:bold 18px/24px Arial,Helvetica,sans-serif;color:#2C3E50;margin:0 0 16px 0;padding-bottom:8px;border-bottom:2px solid #5C6F7F;">
                                        Detection Match by Chamber
                                    </h2>
                                    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="border-collapse:collapse;">
                                        <thead>
                                            <tr>
                                                <th align="left" style="padding:10px;background:#5C6F7F;color:#FFFFFF;font:bold 14px/20px Arial,Helvetica,sans-serif;border:1px solid #5C6F7F;">
                                                    Keyword
                                                </th>
                                                <th align="center" style="padding:10px;background:#5C6F7F;color:#FFFFFF;font:bold 14px/20px Arial,Helvetica,sans-serif;border:1px solid #5C6F7F;">
                                                    House of Assembly
                                                </th>
                                                <th align="center" style="padding:10px;background:#5C6F7F;color:#FFFFFF;font:bold 14px/20px Arial,Helvetica,sans-serif;border:1px solid #5C6F7F;">
                                                    Legislative Council
                                                </th>
                                                <th align="center" style="padding:10px;background:#5C6F7F;color:#FFFFFF;font:bold 14px/20px Arial,Helvetica,sans-serif;border:1px solid #5C6F7F;">
                                                    Total
                                                </th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {''.join(table_rows)}
                                        </tbody>
                                    </table>
                                </div>
                                
                                <!-- Document Sections -->
                                <div style="border-top:2px solid #D5DBDF;padding-top:24px;">
                                    {''.join(doc_sections)}
                                </div>
                            </td>
                        </tr>
                        
                        <!-- Footer -->
                        <tr>
                            <td style="background:#2C3E50;color:#FFFFFF;padding:20px;text-align:center;border-radius:0 0 8px 8px;">
                                <p style="margin:0 0 8px 0;font:bold 14px/20px Arial,Helvetica,sans-serif;">
                                    **THIS PROGRAM IS IN BETA TESTING – DO NOT FORWARD**
                                </p>
                                <p style="margin:0;font:12px/18px Arial,Helvetica,sans-serif;">
                                    Contact developer with any issues, queries, or suggestions:<br>
                                    William.Manning@FederalGroup.com.au
                                </p>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
    </body>
    </html>
    '''
    
    return html, total_matches, counts

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

    # Keep Gmail defaults, allow override via env if needed
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
    subject = f"Hansard Monitor Report — {datetime.now().strftime('%d %b %Y')}"

    to_list = [addr.strip() for addr in re.split(r"[,\s]+", EMAIL_TO) if addr.strip()]

    yag = yagmail.SMTP(
        user=EMAIL_USER,
        password=EMAIL_PASS,
        host=SMTP_HOST,
        port=SMTP_PORT,
        smtp_starttls=SMTP_STARTTLS,
        smtp_ssl=SMTP_SSL,
    )

    # IMPORTANT: pass the HTML string directly (NOT a list), so yagmail
    # doesn't inject <br> between lines (which can break tables).
    yag.send(
        to=to_list,
        subject=subject,
        contents=body_html,  # HTML string
        attachments=files,
    )

    update_sent_log(files)
    print(f"✅ Email sent to {EMAIL_TO} with {len(files)} file(s), {total_hits} match(es).")

if __name__ == "__main__":
    main()
