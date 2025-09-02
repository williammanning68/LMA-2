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
    """
    Returns: (html_string, total_matches, counts_by_chamber_and_kw)

    Layout mirrors the provided inspiration:
      - Navy header with gold underline and a four-cell meta grid
      - "Keyword Summary by Chamber" table
      - One "document section" per transcript, with numbered match cards
      - No emojis, no hyperlinks; line numbers shown as plain text
    """
    import re
    from pathlib import Path
    from datetime import datetime, UTC

    # ---------- helpers ----------
    def esc(s: str) -> str:
        try:
            return _html_escape(s)
        except NameError:  # fallback if helper not in scope
            import html
            return html.escape(s or "")

    def parse_date_from_filename(filename: str):
        m = re.search(r"(\d{1,2} \w+ \d{4})", filename)
        if m:
            try:
                return datetime.strptime(m.group(1), "%d %B %Y")
            except ValueError:
                return datetime.min
        return datetime.min

    def parse_chamber_from_filename(filename: str) -> str:
        low = filename.lower()
        if "house_of_assembly" in low:
            return "House of Assembly"
        if "legislative_council" in low:
            return "Legislative Council"
        return "Unknown"

    # ---------- palette (email-safe hexes) ----------
    FEDERAL_GOLD   = "#C5A572"
    FEDERAL_NAVY   = "#4A5A6A"
    FEDERAL_DARK   = "#475560"
    FEDERAL_LIGHT  = "#ECF0F1"
    FEDERAL_ACCENT = "#D4AF37"
    BORDER_LIGHT   = "#D8DCE0"
    TEXT_PRIMARY   = "#2C3440"
    TEXT_SECONDARY = "#6B7684"
    WHITE          = "#FFFFFF"

    # ---------- counters ----------
    chambers = ["House of Assembly", "Legislative Council"]
    counts = {ch: {kw: 0 for kw in keywords} for ch in chambers}
    totals = {kw: 0 for kw in keywords}
    total_matches = 0
    now_utc = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    # ---------- per-document sections ----------
    doc_sections = []
    files_sorted = sorted(files, key=lambda x: (parse_date_from_filename(Path(x).name), Path(x).name))
    docs_analyzed = len(files_sorted)
    docs_with_matches = 0

    for fpath in files_sorted:
        name = Path(fpath).name
        text = Path(fpath).read_text(encoding="utf-8", errors="ignore")
        chamber = parse_chamber_from_filename(name)

        matches = extract_matches(text, keywords)
        if not matches:
            continue

        docs_with_matches += 1
        matches.sort(key=lambda item: min(item[3]) if item[3] else 10**9)
        total_matches += len(matches)

        # Document "card" (header + match list)
        sec = []
        sec.append(
            f"""
            <div class='document-section'>
              <h3 class='doc-title'>{esc(name)}</h3>
            """
        )

        # Match cards
        for i, (kw_set, excerpt_html, speaker, line_list, _w0, _w1) in enumerate(matches, 1):
            # Update summary counts
            for kw in kw_set:
                if chamber in counts:
                    counts[chamber][kw] += 1
                totals[kw] += 1

            line_label = "line" if len(set(line_list)) == 1 else "lines"
            lines_str = ", ".join(str(n) for n in sorted(set(line_list))) if line_list else "—"
            speaker_html = esc(speaker) if speaker else "UNKNOWN"

            sec.append(
                f"""
                <div class='match-card'>
                  <div class='match-header'>
                    <span class='match-index'>{i}</span>
                    <span class='speaker-info'>
                      <span class='speaker-name'>{speaker_html}</span>
                      <span class='line-ref'>{line_label} {esc(lines_str)}</span>
                    </span>
                  </div>
                  <div class='excerpt'>
                    {excerpt_html}
                  </div>
                </div>
                """
            )

        sec.append("</div>")  # close .document-section
        doc_sections.append("".join(sec))

    # ---------- summary table rows ----------
    def summary_rows() -> str:
        rows = []
        for kw in keywords:
            hoa = counts["House of Assembly"][kw] if "House of Assembly" in counts else 0
            lc  = counts["Legislative Council"][kw] if "Legislative Council" in counts else 0
            tot = totals[kw]
            rows.append(
                f"""
                <tr>
                  <td class='keyword-cell'>{esc(kw)}</td>
                  <td class='count-cell'>{hoa}</td>
                  <td class='count-cell'>{lc}</td>
                  <td class='count-cell total-cell'>{tot}</td>
                </tr>
                """
            )
        return "".join(rows)

    summary_table_html = f"""
      <table class='summary-table' role='table' cellpadding='0' cellspacing='0'>
        <thead>
          <tr>
            <th scope='col'>Keyword</th>
            <th scope='col'>House of Assembly</th>
            <th scope='col'>Legislative Council</th>
            <th scope='col'>Total</th>
          </tr>
        </thead>
        <tbody>
          {summary_rows()}
        </tbody>
      </table>
    """

    # ---------- assembled HTML (email-safe) ----------
    style = f"""
    <style>
      :root {{
        --federal-gold: {FEDERAL_GOLD};
        --federal-navy: {FEDERAL_NAVY};
        --federal-dark: {FEDERAL_DARK};
        --federal-light: {FEDERAL_LIGHT};
        --federal-accent: {FEDERAL_ACCENT};
        --white: {WHITE};
        --border-light: {BORDER_LIGHT};
        --text-primary: {TEXT_PRIMARY};
        --text-secondary: {TEXT_SECONDARY};
      }}
      * {{ box-sizing: border-box; margin:0; padding:0; }}
      body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
        line-height: 1.6;
        color: var(--text-primary);
        background: var(--federal-light);
        padding: 24px;
      }}
      .container {{
        max-width: 900px; margin: 0 auto; background: var(--white);
        border-radius: 4px; overflow: hidden;
        box-shadow: 0 1px 3px rgba(71,85,96,0.1);
      }}
      .header {{
        background: var(--federal-navy); color: var(--white);
        padding: 32px 40px; border-bottom: 4px solid var(--federal-gold);
      }}
      .header h1 {{ font-size: 28px; font-weight: 300; letter-spacing: -0.5px; margin-bottom: 24px; }}
      .header-meta {{
        display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px; padding: 20px; background: rgba(255,255,255,0.08); border-radius: 4px;
      }}
      .meta-item {{ display:flex; flex-direction:column; }}
      .meta-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: var(--federal-gold); margin-bottom: 4px; }}
      .meta-value {{ font-size: 14px; color: var(--white); }}
      .content {{ padding: 32px 40px; }}
      .section-title {{
        font-size: 18px; font-weight: 600; color: var(--federal-dark);
        margin-bottom: 20px; padding-bottom: 12px; border-bottom: 2px solid var(--federal-gold);
      }}
      .summary-table {{
        width: 100%; border-collapse: collapse; margin-bottom: 40px; background: var(--white); border: 1px solid var(--border-light);
      }}
      .summary-table th {{
        background: var(--federal-dark); color: var(--white); padding: 14px 16px; text-align: left; font-weight: 500;
        font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;
      }}
      .summary-table td {{ padding: 12px 16px; border-bottom: 1px solid var(--border-light); font-size: 14px; }}
      .summary-table tbody tr:hover {{ background: rgba(236,240,241,0.5); }}
      .keyword-cell {{ font-weight: 500; color: var(--federal-navy); }}
      .count-cell {{ text-align: center; font-variant-numeric: tabular-nums; color: var(--text-secondary); }}
      .total-cell {{ background: rgba(212,175,55,0.1); font-weight: 600; color: var(--federal-dark); }}
      .document-section {{ margin: 40px 0; }}
      .doc-title {{
        font-size: 16px; font-weight: 500; color: var(--federal-navy); margin-bottom: 20px;
        padding: 12px 16px; background: var(--federal-light); border-left: 3px solid var(--federal-gold);
      }}
      .match-card {{ margin: 16px 0; border: 1px solid var(--border-light); border-radius: 4px; overflow: hidden; }}
      .match-header {{
        background: rgba(236,240,241,0.6); padding: 12px 16px; display:flex; align-items:center; gap:16px; border-bottom:1px solid var(--border-light);
      }}
      .match-index {{
        display:inline-flex; align-items:center; justify-content:center; min-width:28px; height:28px;
        background: var(--federal-gold); color: var(--white); border-radius:50%; font-size:12px; font-weight:600;
      }}
      .speaker-info {{ display:flex; flex-direction:column; gap:2px; }}
      .speaker-name {{ font-weight: 600; color: var(--federal-dark); font-size: 14px; }}
      .line-ref {{ color: var(--text-secondary); font-size: 12px; }}
      .excerpt {{ padding: 16px 20px; background: var(--white); line-height: 1.7; font-size: 14px; color: var(--text-primary); }}
      .excerpt strong {{
        background: rgba(212,175,55,0.2); color: var(--federal-dark); padding: 2px 4px; border-radius: 2px; font-weight: 600; box-decoration-break: clone;
      }}
      @media (max-width: 520px) {{
        body {{ padding: 16px; }}
        .content {{ padding: 20px; }}
        .header {{ padding: 24px; }}
      }}
    </style>
    """

    header_html = f"""
      <div class='header'>
        <h1>Hansard Keyword Digest</h1>
        <div class='header-meta'>
          <div class='meta-item'>
            <span class='meta-label'>Generated</span>
            <span class='meta-value'>{esc(now_utc)}</span>
          </div>
          <div class='meta-item'>
            <span class='meta-label'>Keywords Tracked</span>
            <span class='meta-value'>{esc(", ".join(keywords))}</span>
          </div>
          <div class='meta-item'>
            <span class='meta-label'>Documents Analyzed</span>
            <span class='meta-value'>{docs_analyzed}</span>
          </div>
          <div class='meta-item'>
            <span class='meta-label'>Total Matches</span>
            <span class='meta-value'>{total_matches}</span>
          </div>
        </div>
      </div>
    """

    # Summary section
    summary_section = f"""
      <div class='summary-section'>
        <h2 class='section-title'>Keyword Summary by Chamber</h2>
        {summary_table_html}
      </div>
    """

    # Documents section (or empty state)
    docs_html = "\n".join(doc_sections) if doc_sections else """
      <div class='document-section'>
        <h3 class='doc-title'>No transcripts with matches</h3>
      </div>
    """

    # Outer wrapper
    html = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <meta name="color-scheme" content="light">
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <title>Hansard Keyword Digest</title>
    {style}
  </head>
  <body>
    <div class='container'>
      {header_html}
      <div class='content'>
        {summary_section}
        {docs_html}
      </div>
    </div>
  </body>
</html>"""

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

    print(f"Email sent to {EMAIL_TO} with {len(files)} file(s), {total_hits} match(es).")


if __name__ == "__main__":
    main()
