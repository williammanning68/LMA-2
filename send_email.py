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
MAX_SNIPPET_CHARS = 800     # upper bound after merging windows; keep readable but compact
WINDOW_PAD_SENTENCES = 1    # for non-first-sentence hits: one sentence either side
FIRST_SENT_FOLLOWING = 2    # for first-sentence hits: include next two sentences
MERGE_IF_GAP_GT = 2         # Only merge windows if the gap (in sentences) is > this value


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
    """
    Return list of:
      (kw_set, excerpt_html, speaker, line_numbers_list, win_start_line, win_end_line)
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

        # ✅ NEW: Remove identical (start,end) windows to avoid duplicate excerpts
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
    now_utc = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    chambers = ["House of Assembly", "Legislative Council"]
    counts = {ch: {kw: 0 for kw in keywords} for ch in chambers}
    totals = {kw: 0 for kw in keywords}

    doc_sections = []
    total_matches = 0

    for f in sorted(files, key=lambda x: (parse_date_from_filename(Path(x).name), Path(x).name)):
        text = Path(f).read_text(encoding="utf-8", errors="ignore")
        chamber = parse_chamber_from_filename(Path(f).name)

        matches = extract_matches(text, keywords)
        if not matches:
            continue

        matches.sort(key=lambda item: min(item[3]) if item[3] else 10**9)
        total_matches += len(matches)

        # Extract date for better display
        date_obj = parse_date_from_filename(Path(f).name)
        date_str = date_obj.strftime("%d %B %Y") if date_obj != datetime.min else ""
        
        # Chamber icon
        chamber_icon = "🏛️" if chamber == "House of Assembly" else "⚖️" if chamber == "Legislative Council" else "📋"
        
        sec_lines = [
            f'<div class="document-section" role="region" aria-label="Document: {_html_escape(Path(f).name)}">'
            f'  <div class="doc-header">'
            f'    <div class="doc-title-wrapper">'
            f'      <span class="chamber-icon" aria-hidden="true">{chamber_icon}</span>'
            f'      <div class="doc-info">'
            f'        <h3 class="doc-title">{_html_escape(Path(f).name)}</h3>'
            f'        <div class="doc-meta">'
            f'          <span class="doc-chamber">{chamber}</span>'
            f'          {f"<span class='doc-date'>{date_str}</span>" if date_str else ""}'
            f'          <span class="match-count">{len(matches)} matches</span>'
            f'        </div>'
            f'      </div>'
            f'    </div>'
            f'  </div>'
            f'  <div class="matches-container">'
        ]
        
        for i, (kw_set, excerpt_html, speaker, line_list, win_start, win_end) in enumerate(matches, 1):
            for kw in kw_set:
                if chamber in counts:
                    counts[chamber][kw] += 1
                totals[kw] += 1

            first_line = min(line_list) if line_list else win_start
            speaker_html = _html_escape(speaker) if speaker else "UNKNOWN"
            line_label = "line" if len(line_list) == 1 else "lines"
            lines_str = ", ".join(str(n) for n in sorted(set(line_list))) if line_list else str(first_line)
            
            # Keywords as tags
            kw_tags = "".join([f'<span class="keyword-tag">{_html_escape(kw)}</span>' for kw in sorted(kw_set, key=str.lower)])

            sec_lines.append(
                f'<article class="match-card" role="article" aria-label="Match {i}">'
                f'  <div class="match-header">'
                f'    <div class="match-header-left">'
                f'      <span class="match-number" aria-label="Match number">{i}</span>'
                f'      <div class="speaker-info">'
                f'        <span class="speaker-icon" aria-hidden="true">👤</span>'
                f'        <span class="speaker-name">{speaker_html}</span>'
                f'      </div>'
                f'    </div>'
                f'    <div class="match-header-right">'
                f'      <span class="line-info" aria-label="{line_label}">'
                f'        <span class="line-icon" aria-hidden="true">📍</span>'
                f'        {lines_str}'
                f'      </span>'
                f'    </div>'
                f'  </div>'
                f'  <div class="keyword-tags" aria-label="Keywords found">{kw_tags}</div>'
                f'  <div class="excerpt" aria-label="Excerpt">{excerpt_html}</div>'
                f'</article>'
            )
        sec_lines.append('  </div>')
        sec_lines.append('</div>')
        doc_sections.append("\n".join(sec_lines))

    # Build summary statistics
    total_docs = len([f for f in files if extract_matches(Path(f).read_text(encoding="utf-8", errors="ignore"), keywords)])
    
    # Build enhanced summary table
    header_cols = "".join([
        "<th scope='col' class='th-keyword'>Keyword</th>",
        "<th scope='col' class='th-chamber'><span class='chamber-icon-small' aria-hidden='true'>🏛️</span> House of Assembly</th>",
        "<th scope='col' class='th-chamber'><span class='chamber-icon-small' aria-hidden='true'>⚖️</span> Legislative Council</th>",
        "<th scope='col' class='th-total'>Total</th>",
    ])
    
    row_html = []
    for kw in keywords:
        hoa = counts["House of Assembly"][kw] if "House of Assembly" in counts else 0
        lc  = counts["Legislative Council"][kw] if "Legislative Council" in counts else 0
        tot = totals[kw]
        
        # Add visual indicators for high counts
        hoa_class = "high-count" if hoa > 10 else "medium-count" if hoa > 5 else ""
        lc_class = "high-count" if lc > 10 else "medium-count" if lc > 5 else ""
        tot_class = "high-count" if tot > 20 else "medium-count" if tot > 10 else ""
        
        row_html.append(
            f"<tr>"
            f"<td class='keyword-cell'><strong>{_html_escape(kw)}</strong></td>"
            f"<td class='num {hoa_class}'>{hoa}</td>"
            f"<td class='num {lc_class}'>{lc}</td>"
            f"<td class='num total {tot_class}'>{tot}</td>"
            f"</tr>"
        )
    
    summary_table = (
        f'<table class="summary-table" role="table" aria-label="Keyword summary statistics">'
        f'  <thead><tr>{header_cols}</tr></thead>'
        f'  <tbody>{"".join(row_html)}</tbody>'
        f'</table>'
    )

    # Enhanced CSS with better accessibility and modern design
    style = """
    <style>
      * { box-sizing: border-box; }
      
      /* Base styles */
      body { 
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', sans-serif; 
        line-height: 1.6; 
        color: #2c3e50; 
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        margin: 0; 
        padding: 20px;
        min-height: 100vh;
      }
      
      .container { 
        max-width: 900px; 
        margin: 0 auto; 
        background: white; 
        border-radius: 16px; 
        box-shadow: 0 20px 60px rgba(0,0,0,0.15);
        overflow: hidden;
      }
      
      /* Header section */
      .header { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; 
        padding: 32px;
        position: relative;
        overflow: hidden;
      }
      
      .header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
      }
      
      @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.3; }
      }
      
      .header h1 { 
        margin: 0 0 24px 0; 
        font-size: 32px; 
        font-weight: 700;
        letter-spacing: -0.5px;
        position: relative;
        z-index: 1;
      }
      
      .header-info { 
        background: rgba(255,255,255,0.2); 
        backdrop-filter: blur(10px);
        border-radius: 12px; 
        padding: 20px; 
        position: relative;
        z-index: 1;
      }
      
      .header-stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 16px;
        margin-top: 16px;
      }
      
      .stat-item {
        display: flex;
        flex-direction: column;
      }
      
      .stat-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        opacity: 0.9;
        margin-bottom: 4px;
      }
      
      .stat-value {
        font-size: 24px;
        font-weight: 700;
      }
      
      /* Content section */
      .content { 
        padding: 32px;
      }
      
      /* Summary section */
      .summary-section {
        margin-bottom: 40px;
      }
      
      .summary-section h2 { 
        color: #2c3e50; 
        margin: 0 0 24px 0; 
        font-size: 24px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 12px;
      }
      
      .summary-section h2::before {
        content: '📊';
        font-size: 28px;
      }
      
      /* Summary table */
      .summary-table { 
        width: 100%; 
        border-collapse: separate;
        border-spacing: 0;
        margin: 0 0 32px 0; 
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        border-radius: 12px; 
        overflow: hidden;
      }
      
      .summary-table th { 
        background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%);
        color: white; 
        padding: 16px; 
        text-align: left; 
        font-weight: 600; 
        font-size: 14px;
        letter-spacing: 0.3px;
      }
      
      .summary-table th.th-total {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
      }
      
      .chamber-icon-small {
        font-size: 16px;
        margin-right: 4px;
      }
      
      .summary-table td { 
        padding: 14px 16px; 
        border-bottom: 1px solid #e2e8f0;
        transition: all 0.2s ease;
      }
      
      .summary-table tbody tr:hover { 
        background: #f7fafc;
        transform: translateX(4px);
      }
      
      .summary-table tbody tr:last-child td {
        border-bottom: none;
      }
      
      .keyword-cell {
        color: #4a5568;
      }
      
      .summary-table td.num { 
        text-align: center; 
        font-weight: 600; 
        font-variant-numeric: tabular-nums;
        font-size: 16px;
      }
      
      .summary-table td.total { 
        background: linear-gradient(135deg, #e6fffa 0%, #c6f6d5 100%);
        font-weight: 700; 
        color: #22543d;
      }
      
      .summary-table td.high-count {
        color: #9f1239;
        background: #fef2f2;
      }
      
      .summary-table td.medium-count {
        color: #92400e;
        background: #fef3c7;
      }
      
      /* Document sections */
      .document-section { 
        margin: 40px 0;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        overflow: hidden;
        transition: box-shadow 0.3s ease;
      }
      
      .document-section:hover {
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
      }
      
      .doc-header {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        padding: 20px;
        border-bottom: 2px solid #e2e8f0;
      }
      
      .doc-title-wrapper {
        display: flex;
        align-items: flex-start;
        gap: 16px;
      }
      
      .chamber-icon {
        font-size: 32px;
        margin-top: 4px;
      }
      
      .doc-info {
        flex: 1;
      }
      
      .doc-title { 
        color: #2d3748;
        margin: 0 0 8px 0; 
        font-size: 18px; 
        font-weight: 600;
        line-height: 1.3;
      }
      
      .doc-meta {
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
        font-size: 14px;
        color: #718096;
      }
      
      .doc-chamber {
        background: #edf2f7;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 500;
      }
      
      .doc-date {
        display: flex;
        align-items: center;
        gap: 4px;
      }
      
      .doc-date::before {
        content: '📅';
        font-size: 14px;
      }
      
      .match-count {
        background: #f0fff4;
        color: #22543d;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
      }
      
      .matches-container {
        padding: 20px;
      }
      
      /* Match cards */
      .match-card { 
        margin: 16px 0;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        overflow: hidden;
        transition: all 0.3s ease;
        background: white;
      }
      
      .match-card:hover { 
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        border-color: #cbd5e0;
      }
      
      .match-header { 
        background: linear-gradient(135deg, #f7fafc 0%, #ffffff 100%);
        padding: 16px 20px;
        border-bottom: 1px solid #e2e8f0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
      }
      
      .match-header-left {
        display: flex;
        align-items: center;
        gap: 16px;
      }
      
      .match-number { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
        font-weight: 700;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.4);
      }
      
      .speaker-info {
        display: flex;
        align-items: center;
        gap: 8px;
      }
      
      .speaker-icon {
        font-size: 20px;
        opacity: 0.8;
      }
      
      .speaker-name { 
        font-weight: 600;
        color: #2d3748;
        font-size: 16px;
      }
      
      .match-header-right {
        display: flex;
        align-items: center;
        gap: 16px;
      }
      
      .line-info { 
        color: #718096;
        font-size: 14px;
        display: flex;
        align-items: center;
        gap: 6px;
        background: #f7fafc;
        padding: 4px 12px;
        border-radius: 20px;
      }
      
      .line-icon {
        font-size: 14px;
      }
      
      .keyword-tags {
        padding: 12px 20px;
        background: #faf5ff;
        border-bottom: 1px solid #e9d5ff;
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
      }
      
      .keyword-tag {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.3px;
        box-shadow: 0 2px 4px rgba(139, 92, 246, 0.3);
      }
      
      .excerpt { 
        padding: 20px;
        background: white;
        line-height: 1.8;
        color: #4a5568;
        font-size: 15px;
        position: relative;
      }
      
      .excerpt strong { 
        background: linear-gradient(135deg, #fef3c7 0%, #fed7aa 100%);
        color: #92400e;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
        box-decoration-break: clone;
        -webkit-box-decoration-break: clone;
        box-shadow: 0 1px 3px rgba(251, 191, 36, 0.3);
      }
      
      /* No matches message */
      .no-matches { 
        text-align: center;
        padding: 60px 20px;
        color: #718096;
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border-radius: 12px;
        margin: 20px 0;
        font-size: 16px;
      }
      
      .no-matches::before {
        content: '🔍';
        display: block;
        font-size: 48px;
        margin-bottom: 16px;
        opacity: 0.5;
      }
      
      /* Accessibility improvements */
      @media (prefers-reduced-motion: reduce) {
        * {
          animation: none !important;
          transition: none !important;
        }
      }
      
      /* High contrast mode support */
      @media (prefers-contrast: high) {
        .match-card, .document-section {
          border-width: 2px;
        }
        
        .excerpt strong {
          text-decoration: underline;
          text-decoration-thickness: 2px;
        }
      }
      
      /* Dark mode support */
      @media (prefers-color-scheme: dark) {
        body {
          background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        }
        
        .container {
          background: #2d3748;
          color: #e2e8f0;
        }
        
        .header {
          background: linear-gradient(135deg, #4c1d95 0%, #5b21b6 100%);
        }
        
        .content {
          background: #2d3748;
        }
        
        .match-card, .document-section {
          background: #374151;
          border-color: #4b5563;
        }
        
        .doc-header, .match-header {
          background: linear-gradient(135deg, #374151 0%, #4b5563 100%);
        }
        
        .excerpt {
          background: #374151;
          color: #e5e7eb;
        }
        
        .excerpt strong {
          background: linear-gradient(135deg, #7c2d12 0%, #991b1b 100%);
          color: #fef3c7;
        }
        
        .doc-title, .speaker-name {
          color: #f3f4f6;
        }
        
        .summary-table tbody tr:hover {
          background: #374151;
        }
      }
      
      /* Mobile responsiveness */
      @media (max-width: 768px) {
        body { 
          padding: 0;
        }
        
        .container { 
          border-radius: 0;
          box-shadow: none;
        }
        
        .header { 
          padding: 24px 16px;
          border-radius: 0;
        }
        
        .header h1 {
          font-size: 24px;
        }
        
        .content { 
          padding: 20px 16px;
        }
        
        .header-stats {
          grid-template-columns: 1fr 1fr;
        }
        
        .summary-table {
          font-size: 14px;
        }
        
        .summary-table th, 
        .summary-table td {
          padding: 10px 8px;
        }
        
        .doc-title-wrapper {
          flex-direction: column;
        }
        
        .chamber-icon {
          font-size: 24px;
        }
        
        .match-header {
          flex-direction: column;
          align-items: flex-start;
        }
        
        .match-header-left,
        .match-header-right {
          width: 100%;
        }
        
        .excerpt {
          padding: 16px;
          font-size: 14px;
        }
      }
      
      @media (max-width: 480px) {
        .header-stats {
          grid-template-columns: 1fr;
        }
        
        .doc-meta {
          flex-direction: column;
          gap: 8px;
        }
        
        .summary-table {
          font-size: 12px;
        }
      }
      
      /* Print styles */
      @media print {
        body {
          background: white;
          padding: 0;
        }
        
        .container {
          box-shadow: none;
          border: none;
        }
        
        .header {
          background: none;
          color: black;
          border-bottom: 2px solid black;
        }
        
        .header-info {
          background: none;
          border: 1px solid black;
        }
        
        .match-card {
          page-break-inside: avoid;
          border: 1px solid black;
        }
        
        .excerpt strong {
          background: none;
          text-decoration: underline;
          font-weight: 900;
        }
      }
    </style>
    """

    # Build header with enhanced statistics
    header_html = (
        f'<div class="header">'
        f'  <h1>📜 Hansard Keyword Digest</h1>'
        f'  <div class="header-info">'
        f'    <div><strong>Generated:</strong> {now_utc}</div>'
        f'    <div><strong>Keywords monitored:</strong> {_html_escape(", ".join(keywords))}</div>'
        f'    <div class="header-stats">'
        f'      <div class="stat-item">'
        f'        <span class="stat-label">Documents</span>'
        f'        <span class="stat-value">{len(files)}</span>'
        f'      </div>'
        f'      <div class="stat-item">'
        f'        <span class="stat-label">Total Matches</span>'
        f'        <span class="stat-value">{total_matches}</span>'
        f'      </div>'
        f'      <div class="stat-item">'
        f'        <span class="stat-label">Active Keywords</span>'
        f'        <span class="stat-value">{len([k for k in keywords if totals[k] > 0])}/{len(keywords)}</span>'
        f'      </div>'
        f'    </div>'
        f'  </div>'
        f'</div>'
    )

    summary_section = (
        f'<div class="summary-section">'
        f'  <h2>Summary Statistics</h2>'
        f'  {summary_table}'
        f'</div>'
    )

    doc_html = "\n".join(doc_sections) if doc_sections else (
        '<div class="no-matches">'
        '  No keyword matches found in the processed documents.'
        '</div>'
    )

    html = (
        f'<!DOCTYPE html>'
        f'<html lang="en">'
        f'<head>'
        f'  <meta charset="UTF-8">'
        f'  <meta name="viewport" content="width=device-width, initial-scale=1">'
        f'  <meta name="description" content="Hansard Keyword Digest - Parliamentary transcript analysis">'
        f'  <title>Hansard Keyword Digest - {datetime.now().strftime("%d %B %Y")}</title>'
        f'  {style}'
        f'</head>'
        f'<body>'
        f'  <div class="container">'
        f'    {header_html}'
        f'    <main class="content" role="main">'
        f'      {summary_section}'
        f'      <section aria-label="Document matches">'
        f'        <h2 style="color: #2c3e50; margin: 32px 0 24px 0; font-size: 24px; font-weight: 600; display: flex; align-items: center; gap: 12px;">'
        f'          <span aria-hidden="true">📄</span> Document Matches'
        f'        </h2>'
        f'        {doc_html}'
        f'      </section>'
        f'    </main>'
        f'  </div>'
        f'</body>'
        f'</html>'
    )
    
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

    subject = f"Hansard keyword digest — {datetime.now().strftime('%d %b %Y')}"
    to_list = [addr.strip() for addr in re.split(r"[,\s]+", EMAIL_TO) if addr.strip()]

    yag = yagmail.SMTP(
        user=EMAIL_USER,
        password=EMAIL_PASS,
        host="smtp.gmail.com",
        port=587,
        smtp_starttls=True,
        smtp_ssl=False,
    )

    yag.send(
        to=to_list,
        subject=subject,
        contents=[body_html],   # HTML string (no links)
        attachments=files,
    )

    update_sent_log(files)

    print(f"✅ Email sent to {EMAIL_TO} with {len(files)} file(s), {total_hits} match(es).")


if __name__ == "__main__":
    main()
