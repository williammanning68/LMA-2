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
    """Build the HTML body and return (html_string, total_matches, counts_by_chamber_and_kw)."""
    from datetime import datetime, UTC
    now_utc = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    # Summary counters
    chambers = ["House of Assembly", "Legislative Council"]
    counts = {ch: {kw: 0 for kw in keywords} for ch in chambers}
    totals = {kw: 0 for kw in keywords}

    # Collect per-document sections
    doc_sections = []
    total_matches = 0

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

    # Order documents by date (parsed from filename), then by name
    for f in sorted(files, key=lambda x: (parse_date_from_filename(Path(x).name), Path(x).name)):
        text = Path(f).read_text(encoding="utf-8", errors="ignore")
        chamber = parse_chamber_from_filename(Path(f).name)

        matches = extract_matches(text, keywords)
        if not matches:
            continue

        # Order matches by earliest line number
        matches.sort(key=lambda item: min(item[3]) if item[3] else 10**9)
        total_matches += len(matches)

        # Build section HTML for this document
        sec = []
        sec.append(
            f'<section class="doc">'
            f'  <header class="doc__header">'
            f'    <div class="doc__title">{_html_escape(Path(f).name)}</div>'
            f'    <div class="doc__meta">{_html_escape(chamber)}</div>'
            f'  </header>'
        )

        for i, (kw_set, excerpt_html, speaker, line_list, _win_start, _win_end) in enumerate(matches, 1):
            # Update counts
            for kw in kw_set:
                if chamber in counts:
                    counts[chamber][kw] += 1
                totals[kw] += 1

            first_line = min(line_list) if line_list else None
            speaker_html = _html_escape(speaker) if speaker else "UNKNOWN"
            line_label = "line" if len(line_list) == 1 else "lines"
            lines_str = ", ".join(str(n) for n in sorted(set(line_list))) if line_list else "—"

            sec.append(
                f'  <article class="match">'
                f'    <div class="match__meta">'
                f'      <span class="match__index">Match #{i}</span>'
                f'      <span class="match__speaker">{speaker_html}</span>'
                f'      <span class="match__lines">{line_label} {lines_str}</span>'
                f'    </div>'
                f'    <div class="match__excerpt">{excerpt_html}</div>'
                f'  </article>'
            )

        sec.append('</section>')
        doc_sections.append("\n".join(sec))

    # Build summary table
    header_cols = "".join([
        "<th>Keyword</th>",
        "<th>House of Assembly</th>",
        "<th>Legislative Council</th>",
        "<th>Total</th>",
    ])
    row_html = []
    for kw in keywords:
        hoa = counts["House of Assembly"][kw] if "House of Assembly" in counts else 0
        lc  = counts["Legislative Council"][kw] if "Legislative Council" in counts else 0
        tot = totals[kw]
        row_html.append(
            f"<tr>"
            f"  <td class='kw'><span class='kw__pill'>{_html_escape(kw)}</span></td>"
            f"  <td class='num'>{hoa}</td>"
            f"  <td class='num'>{lc}</td>"
            f"  <td class='num total'>{tot}</td>"
            f"</tr>"
        )
    summary_table = (
        f'<table class="summary">'
        f'  <thead><tr>{header_cols}</tr></thead>'
        f'  <tbody>{"".join(row_html)}</tbody>'
        f'</table>'
    )

    # Assemble full HTML
    style = """
    <style>
      :root{
        --federal-gold:#C5A572;
        --federal-navy:#4A5A6A;
        --federal-dark:#475560;
        --federal-light:#ECF0F1;
        --federal-accent:#D4AF37;
        --bg: var(--federal-light);
        --text: var(--federal-dark);
        --card-bg: #ffffff;
        --muted: #6b7a89;
        --rule: rgba(71,85,96,0.12);
      }
      body{
        margin:0; padding:24px;
        background:var(--bg);
        color:var(--text);
        font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
      }
      .wrap{
        max-width: 860px; margin: 0 auto;
      }
      .brand{
        background: var(--card-bg);
        border: 1px solid var(--rule);
        border-left: 6px solid var(--federal-gold);
        border-radius: 8px;
        padding: 16px 18px;
        margin-bottom: 16px;
      }
      .brand__title{
        font-size: 18px; font-weight: 700; color: var(--federal-navy);
        margin: 0 0 2px 0;
      }
      .brand__sub{ margin:0; color: var(--muted); }
      .summary-card{
        background: var(--card-bg);
        border: 1px solid var(--rule);
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 18px;
      }
      .kvs{ display:flex; flex-wrap:wrap; gap:12px 24px; margin: 6px 0 0 0; padding:0; list-style:none;}
      .kvs li{ margin:0; }
      .kvs .k{ color: var(--muted); margin-right:6px; }
      .summary{
        width:100%;
        border-collapse: collapse;
        background: var(--card-bg);
        border: 1px solid var(--rule);
        border-radius: 8px;
        overflow: hidden;
        margin: 6px 0 22px 0;
      }
      .summary thead th{
        text-align:left;
        padding:10px 12px;
        background: linear-gradient(0deg, var(--federal-navy), var(--federal-dark));
        color: #fff;
        font-weight:600;
        font-size:13px;
        border-bottom: 1px solid var(--rule);
      }
      .summary tbody td{
        padding:10px 12px; border-bottom:1px solid var(--rule);
        vertical-align: top; background: #fff;
      }
      .summary tbody tr:nth-child(even) td{ background:#fafbfc; }
      .summary td.num{ text-align:right; width:110px; }
      .summary td.total{ font-weight:700; }
      .kw__pill{
        display:inline-block; background: rgba(197,165,114,0.15);
        color: var(--federal-dark); border:1px solid rgba(197,165,114,0.35);
        border-radius: 999px; padding:2px 8px; font-size:12px;
      }

      section.doc{
        background: var(--card-bg);
        border: 1px solid var(--rule);
        border-radius: 10px;
        margin: 14px 0;
        overflow: hidden;
      }
      .doc__header{
        display:flex; justify-content:space-between; align-items:center;
        padding: 12px 14px;
        background: #fff;
        border-bottom: 1px solid var(--rule);
      }
      .doc__title{
        font-weight:700; color: var(--federal-navy);
      }
      .doc__meta{
        color: var(--muted); font-size: 12px;
      }

      article.match{
        padding: 12px 14px;
        border-top: 1px solid var(--rule);
      }
      .match:first-of-type{ border-top:none; }
      .match__meta{
        display:flex; flex-wrap:wrap; gap:10px 16px; align-items:baseline;
        color: var(--muted); font-size: 12px; margin-bottom: 6px;
      }
      .match__index{
        color: var(--federal-dark); font-weight:700; font-size: 12px;
        background: rgba(74,90,106,0.08);
        border: 1px solid rgba(74,90,106,0.18);
        border-radius: 6px; padding: 2px 6px;
      }
      .match__speaker{ font-weight:600; color: var(--federal-dark); }
      .match__lines{ }
      .match__excerpt{
        background: #fbfbfb;
        border-left: 3px solid var(--federal-accent);
        padding: 10px 12px;
        border-radius: 4px;
      }
      strong{ font-weight:700; color: #222; }
      @media (max-width: 520px){
        body{ padding: 16px; }
        .doc__header{ flex-direction: column; align-items: flex-start; gap: 4px; }
      }
    </style>
    """

    header_html = (
        f'<div class="brand">'
        f'  <h1 class="brand__title">Hansard Keyword Digest</h1>'
        f'  <p class="brand__sub">Program Runtime: {now_utc}</p>'
        f'</div>'
        f'<div class="summary-card">'
        f'  <ul class="kvs">'
        f'    <li><span class="k">Keywords:</span><span>{_html_escape(", ".join(keywords))}</span></li>'
        f'    <li><span class="k">Total matches:</span><span>{total_matches}</span></li>'
        f'  </ul>'
        f'</div>'
        f'<table class="summary">'
        f'  <thead><tr>'
        f'    <th>Keyword</th><th>House of Assembly</th><th>Legislative Council</th><th>Total</th>'
        f'  </tr></thead>'
        f'  <tbody>'
        f'    {summary_table.split("<tbody>")[1].split("</tbody>")[0]}'
        f'  </tbody>'
        f'</table>'
    )

    doc_html = "\n".join(doc_sections) if doc_sections else (
        '<section class="doc">'
        '  <header class="doc__header"><div class="doc__title">No transcripts with matches</div>'
        '  <div class="doc__meta">—</div></header>'
        '  <article class="match"><div class="match__excerpt">No keyword matches found.</div></article>'
        '</section>'
    )

    html = f"<!doctype html><html><head>{style}</head><body><div class='wrap'>{header_html}{doc_html}</div></body></html>"
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
