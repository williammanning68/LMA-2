import os
import re
import glob
from bisect import bisect_right
from pathlib import Path
from datetime import datetime, UTC  # ✅ use UTC constant (Py 3.11+)

import yagmail
import subprocess  # optional: only used if ATTRIB_WITH_LLM=1


# File that records which transcripts have already been emailed
LOG_FILE = Path("sent.log")

# --- Tunables ----------------------------------------------------------------
MAX_SNIPPET_CHARS = 800     # upper bound after merging windows; keep readable but compact
WINDOW_PAD_SENTENCES = 1    # for non-first-sentence hits: one sentence either side
FIRST_SENT_FOLLOWING = 2    # for first-sentence hits: include next two sentences
WIDER_CONTEXT_PAD = 10      # lines to include on either side for the "wider context" link


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

# Accept titles (case-insensitive) or ALL-CAPS surnames; require ":" or a dash after header
SPEAKER_HEADER_RE = re.compile(
    r"""
^
(?:
  # (A) Title + optional ALL-CAPS surname(s)
  (?P<title>(?i:Mr|Ms|Mrs|Miss|Hon|Dr|Prof|Professor|Premier|Madam\s+SPEAKER|The\s+SPEAKER|The\s+PRESIDENT|The\s+CLERK|Deputy\s+Speaker|Deputy\s+President))
  (?:[\s.]+(?P<name>[A-Z][A-Z'’\-]+(?:\s+[A-Z][A-Z'’\-]+){0,3}))?
 |
  # (B) ALL-CAPS name-only (e.g., DOW, WOODRUFF, O'BYRNE)
  (?P<name_only>[A-Z][A-Z'’\-]+(?:\s+[A-Z][A-Z'’\-]+){0,3})
)
(?:\s*\([^)]*\))?        # optional (Electorate—Portfolio)
\s*(?::|[-–—]\s)         # ":" OR " - " / "– " / "— "
""",
    re.VERBOSE,
)

# Lines that look like prose-with-colon; never treat them as headers (belt-and-braces)
CONTENT_COLON_RE = re.compile(r"^(Then|There|And|But|So|If|When|Now|Finally|First|Second|Third)\b", re.IGNORECASE)

# Timestamps/headings to ignore in body
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
    """
    Return a list of utterances with:
      speaker: str | None
      lines: [str]         # content lines (header line excluded)
      line_nums: [int]     # 1-based file line numbers for 'lines'
      joined: str          # "\n".join(lines)
      line_offsets: [int]  # starting char offset in 'joined' for each content line
      sents: [(start,end)] # sentence spans in 'joined'
    """
    all_lines = text.splitlines()
    utterances = []
    curr = {"speaker": None, "lines": [], "line_nums": []}

    def flush():
        if curr["lines"]:
            joined = "\n".join(curr["lines"])
            # compute line start offsets within joined
            offs = []
            total = 0
            for i, ln in enumerate(curr["lines"]):
                offs.append(total)
                total += len(ln) + (1 if i < len(curr["lines"]) - 1 else 0)

            # sentence spans in joined
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
        # treat obvious non-speech as boundaries (do not include)
        if not s or TIME_STAMP_RE.match(s) or UPPER_HEADING_RE.match(s) or INTERJECTION_RE.match(s):
            continue

        m = SPEAKER_HEADER_RE.match(s)
        if m and not (m.group("name_only") and CONTENT_COLON_RE.match(s)):
            # new speaker header: flush previous content and start new utterance
            flush()
            title = (m.group("title") or "").strip()
            name = (m.group("name") or m.group("name_only") or "").strip()
            curr = {"speaker": " ".join(x for x in (title, name) if x).strip(), "lines": [], "line_nums": []}
            # header line itself not added to content
            continue

        # normal content line
        curr["lines"].append(raw.rstrip())
        curr["line_nums"].append(idx + 1)

    # tail
    flush()
    return utterances, all_lines


def _line_for_char_offset(line_offsets, line_nums, pos):
    """Map a char offset in 'joined' back to the 1-based original file line number."""
    i = bisect_right(line_offsets, pos) - 1
    if i < 0:
        i = 0
    if i >= len(line_nums):
        i = len(line_nums) - 1
    return line_nums[i]


# --- Matching, windows & merging ---------------------------------------------

def _compile_kw_patterns(keywords):
    pats = []
    for kw in sorted(keywords, key=len, reverse=True):  # phrases first
        if " " in kw:
            pats.append((kw, re.compile(re.escape(kw), re.IGNORECASE)))
        else:
            pats.append((kw, re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)))
    return pats


def _collect_hits_in_utterance(utt, kw_pats):
    """
    For one utterance, return list of hits:
      { 'kw': str, 'sent_idx': int, 'line_no': int }
    A sentence can yield multiple hits (for different keywords).
    """
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
    """
    For each hit -> a sentence window:
      - if sent_idx == 0: [0 .. 0+FIRST_SENT_FOLLOWING]
      - else: [sent_idx-1 .. sent_idx+1]
    Return sorted list of (start,end, kws_set, line_set).
    """
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


def _merge_windows(wins, gap_allow=4):
    """
    Merge windows when the next window starts within <= gap_allow sentences of the previous.
    Merge also unions keyword sets and line numbers.
    """
    if not wins:
        return []
    merged = [wins[0]]
    for s, e, kws, lines in wins[1:]:
        ps, pe, pk, pl = merged[-1]
        if s - pe <= gap_allow:
            merged[-1] = [ps, max(pe, e), pk | kws, pl | lines]
        else:
            merged.append([s, e, kws, lines])
    return merged


def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _highlight_keywords_html(text_html: str, keywords: list[str]) -> str:
    # Longest phrases first to avoid partial overlap
    out = text_html
    for kw in sorted(keywords, key=len, reverse=True):
        if " " in kw:
            pat = re.compile(re.escape(_html_escape(kw)), re.IGNORECASE)
        else:
            pat = re.compile(rf"\b{re.escape(_html_escape(kw))}\b", re.IGNORECASE)
        out = pat.sub(lambda m: f"<strong>{m.group(0)}</strong>", out)
    return out


def _excerpt_from_window_html(utt, win, keywords):
    """
    Build HTML excerpt string for utterance window [start,end] and highlight keywords.
    Also return (win_start_line, win_end_line) in original file.
    """
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

    # Map window char bounds back to line numbers in original file
    start_line = _line_for_char_offset(utt["line_offsets"], utt["line_nums"], a)
    end_line = _line_for_char_offset(utt["line_offsets"], utt["line_nums"], max(a, b - 1))

    return html, sorted(lines), sorted(kws, key=str.lower), start_line, end_line


def _looks_suspicious(s: str | None) -> bool:
    """Decide if an attributed speaker string looks untrustworthy (to trigger QC)."""
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
    """
    Ask local Ollama to choose ONE candidate speaker using a large backward context.
    Used only as a last-resort QC. Returns None if uncertain.
    """
    model = model or os.environ.get("ATTRIB_LLM_MODEL", "llama3.2:3b")
    i = max(0, hit_line_no - 1)
    start = max(0, i - 80)  # ~80 lines preceding
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
    Build excerpts per rules and return:
      [(kw_set, excerpt_html, speaker, line_numbers_list, win_start_line, win_end_line)]
    kw_set is the set of keywords included in the excerpt.
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

        # Build windows per hit according to first/other sentence rule
        wins = _windows_for_hits(hits, sent_count=len(utt["sents"]))

        # Merge windows when hits are within 4 sentences
        merged = _merge_windows(wins, gap_allow=4)

        # Optional LLM QC if we don't trust the speaker string
        if use_llm and _looks_suspicious(speaker):
            candidates = sorted({u["speaker"] for u in utts if u["speaker"]})
            earliest_line = min(min(w[3]) for w in merged)
            guess = _llm_qc_speaker(all_lines, earliest_line, candidates, timeout=llm_timeout)
            if guess:
                speaker = guess

        # Emit excerpts
        for win in merged:
            excerpt_html, line_list, kws_in_excerpt, win_start, win_end = _excerpt_from_window_html(utt, win, keywords)
            results.append((set(kws_in_excerpt), excerpt_html, speaker, line_list, win_start, win_end))

    return results


# --- Repo link helpers --------------------------------------------------------

def _repo_blob_base():
    """
    Build the base URL to the repo blob view:
      https://github.com/<org>/<repo>/blob/<ref>
    Choose <ref> as GITHUB_SHA (preferred) or REPO_REF env (fallback, default 'main').
    """
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com").rstrip("/")
    repo = os.environ.get("GITHUB_REPOSITORY", "").strip()
    if not repo:
        return None  # no links
    ref = os.environ.get("GITHUB_SHA") or os.environ.get("REPO_REF", "main")
    return f"{server}/{repo}/blob/{ref}"


def _github_line_link(blob_base, relpath, line):
    return f"{blob_base}/{relpath}#L{line}"


def _github_range_link(blob_base, relpath, start_line, end_line):
    return f"{blob_base}/{relpath}#L{start_line}-L{end_line}"


# --- Digest / email pipeline (HTML) ------------------------------------------

def parse_date_from_filename(filename: str):
    """Extract datetime from Hansard filename."""
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
    now_utc = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")  # ✅ no deprecation
    blob_base = _repo_blob_base()

    # Summary counters
    chambers = ["House of Assembly", "Legislative Council"]
    counts = {ch: {kw: 0 for kw in keywords} for ch in chambers}
    totals = {kw: 0 for kw in keywords}

    doc_sections = []
    total_matches = 0

    # Order documents by date (parsed from filename), then by name
    for f in sorted(files, key=lambda x: (parse_date_from_filename(Path(x).name), Path(x).name)):
        text = Path(f).read_text(encoding="utf-8", errors="ignore")
        total_lines = len(text.splitlines())
        chamber = parse_chamber_from_filename(Path(f).name)
        relpath = f"transcripts/{Path(f).name}"

        matches = extract_matches(text, keywords)
        if not matches:
            continue

        # Order matches by earliest line number
        matches.sort(key=lambda item: min(item[3]) if item[3] else 10**9)

        total_matches += len(matches)

        # Build section HTML for this document
        sec_lines = [f'<h3 class="doc-title">{_html_escape(Path(f).name)}</h3>']
        for i, (kw_set, excerpt_html, speaker, line_list, win_start, win_end) in enumerate(matches, 1):
            # update counts
            for kw in kw_set:
                if chamber in counts:
                    counts[chamber][kw] += 1
                totals[kw] += 1

            first_line = min(line_list) if line_list else win_start
            # wider context range
            ctx_start = max(1, win_start - WIDER_CONTEXT_PAD)
            ctx_end = min(total_lines, win_end + WIDER_CONTEXT_PAD)

            speaker_html = _html_escape(speaker) if speaker else "UNKNOWN"
            line_label = "line" if len(line_list) == 1 else "lines"
            # each line can be clicked individually
            if blob_base:
                line_links = ", ".join(
                    f'<a href="{_github_line_link(blob_base, relpath, n)}" title="Open on GitHub at line {n}">{n}</a>'
                    for n in sorted(set(line_list))
                ) if line_list else str(first_line)
                header_link = f'<a href="{_github_line_link(blob_base, relpath, first_line)}" title="Open on GitHub at line {first_line}">Match #{i}</a>'
                ctx_link = f' &nbsp;·&nbsp; <a href="{_github_range_link(blob_base, relpath, ctx_start, ctx_end)}" title="Show wider context: lines {ctx_start}–{ctx_end}">wider context (±{WIDER_CONTEXT_PAD})</a>'
            else:
                line_links = ", ".join(str(n) for n in sorted(set(line_list))) if line_list else str(first_line)
                header_link = f"Match #{i}"
                ctx_link = ""

            sec_lines.append(
                f'<div class="match">'
                f'  <div class="meta">{header_link} (<strong>{speaker_html}</strong>) — {line_label} {line_links}{ctx_link}</div>'
                f'  <div class="excerpt">{excerpt_html}</div>'
                f'</div>'
            )
        doc_sections.append("\n".join(sec_lines))

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
            f"<tr><td>{_html_escape(kw)}</td>"
            f"<td class='num'>{hoa}</td>"
            f"<td class='num'>{lc}</td>"
            f"<td class='num total'>{tot}</td></tr>"
        )
    summary_table = (
        f'<table class="summary-table">'
        f'  <thead><tr>{header_cols}</tr></thead>'
        f'  <tbody>{"".join(row_html)}</tbody>'
        f'</table>'
    )

    # Assemble full HTML
    style = """
    <style>
      body { font-family: Arial, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; color:#111; }
      a { color: #0b57d0; text-decoration: none; }
      a:hover { text-decoration: underline; }
      .hdr { margin: 0 0 12px 0; }
      .small { color:#444; }
      .summary-table { border-collapse: collapse; margin: 8px 0 18px 0; width: 100%; }
      .summary-table th, .summary-table td { border: 1px solid #e0e0e0; padding: 6px 8px; text-align: left; }
      .summary-table th { background: #fafafa; }
      .summary-table td.num { text-align: right; }
      .summary-table td.total { font-weight: 600; }
      .doc-title { margin: 18px 0 6px 0; }
      .match { margin: 10px 0 14px 0; }
      .meta { color:#333; font-size: 0.95em; margin-bottom: 6px; }
      .excerpt { background:#fbfbfb; border-left:3px solid #d0d0d0; padding:8px 10px; line-height:1.4; }
      strong { font-weight: 700; }
    </style>
    """

    header_html = (
        f'<p class="hdr"><strong>Program Runtime:</strong> {now_utc}</p>'
        f'<p class="hdr"><strong>Keywords:</strong> {_html_escape(", ".join(keywords))}</p>'
        f'<h2 class="hdr">Keywords Triggered</h2>{summary_table}'
    )

    doc_html = "\n".join(doc_sections) if doc_sections else "<p>No keyword matches found.</p>"

    html = f"<!doctype html><html><head>{style}</head><body>{header_html}{doc_html}</body></html>"
    return html, total_matches, counts


def load_sent_log():
    """Return set of transcript filenames that have already been emailed."""
    if LOG_FILE.exists():
        return {line.strip() for line in LOG_FILE.read_text().splitlines() if line.strip()}
    return set()


def update_sent_log(files):
    """Append newly emailed filenames to the log."""
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

    # ✅ Send HTML body as a plain string — let yagmail build the MIME parts
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
        contents=[body_html],   # <-- pass HTML string, NOT MIMEText
        attachments=files,
    )

    update_sent_log(files)

    print(
        f"✅ Email sent to {EMAIL_TO} with {len(files)} file(s), {total_hits} match(es)."
    )


if __name__ == "__main__":
    main()
