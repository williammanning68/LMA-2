import os
import re
import glob
from pathlib import Path
from datetime import datetime
from bisect import bisect_right

import yagmail
import subprocess  # optional: only used if ATTRIB_WITH_LLM=1


# File that records which transcripts have already been emailed
LOG_FILE = Path("sent.log")

# --- Tunables ----------------------------------------------------------------
MAX_SNIPPET_CHARS = 800     # upper bound after merging windows; keep readable but compact
WINDOW_PAD_SENTENCES = 1    # for non-first-sentence hits: one sentence either side
FIRST_SENT_FOLLOWING = 2    # for first-sentence hits: include next two sentences


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
            # keep collecting within the same utterance (no flush), but skip adding line
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
    # line_offsets is sorted; find rightmost offset <= pos
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
            # map to original line number
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


def _excerpt_from_window(utt, win, keywords):
    """
    Build excerpt string for utterance window [start,end] and highlight keywords.
    """
    sents = utt["sents"]
    joined = utt["joined"]
    start, end, kws, lines = win
    a = sents[start][0]
    b = sents[end][1]
    raw = joined[a:b].strip()
    # soft cap length without chopping meaningfully
    if len(raw) > MAX_SNIPPET_CHARS:
        raw = raw[:MAX_SNIPPET_CHARS].rstrip() + "…"
    return _highlight_keywords(raw, keywords), sorted(lines), sorted(kws, key=str.lower)


def _highlight_keywords(text: str, keywords: list[str]) -> str:
    # Longest phrases first to avoid partial overlaps
    for kw in sorted(keywords, key=len, reverse=True):
        if " " in kw:
            pat = re.compile(re.escape(kw), re.IGNORECASE)
        else:
            pat = re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)
        text = pat.sub(lambda m: f"**{m.group(0)}**", text)
    return text


def _looks_suspicious(s: str | None) -> bool:
    """Decide if an attributed speaker string looks untrustworthy (to trigger QC)."""
    if not s:
        return True
    s = s.strip()
    # Accept known title+ALLCAPS surname or role-only
    if re.match(
        r"(?i)^(Mr|Ms|Mrs|Miss|Hon|Dr|Prof|Professor|Premier|Madam SPEAKER|The SPEAKER|The PRESIDENT|The CLERK|Deputy Speaker|Deputy President)"
        r"(?:\s+[A-Z][A-Z'’\-]+(?:\s+[A-Z][A-Z'’\-]+){0,3})?$",
        s,
    ):
        return False
    # Accept pure ALL-CAPS name(s)
    if re.match(r"^[A-Z][A-Z'’\-]+(?:\s+[A-Z][A-Z'’\-]+){0,3}$", s):
        return False
    # Otherwise, suspicious
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
    Build excerpts per your rules and return:
      [(kw_label, snippet, speaker, line_numbers_list)]
    kw_label is a comma-separated label of keywords in the excerpt (not used in email header).
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
        candidates = []
        if use_llm and _looks_suspicious(speaker):
            # Gather all speakers seen in the file (cheap pass across utterances)
            candidates = sorted({u["speaker"] for u in utts if u["speaker"]})
            # Use the earliest hit line for context
            earliest_line = min(min(w[3]) for w in merged)
            guess = _llm_qc_speaker(all_lines, earliest_line, candidates, timeout=llm_timeout)
            if guess:
                speaker = guess

        # Emit excerpts
        for win in merged:
            excerpt, line_list, kws_in_excerpt = _excerpt_from_window(utt, win, keywords)
            kw_label = ", ".join(kws_in_excerpt)
            results.append((kw_label, excerpt, speaker, line_list))

    return results


# --- Digest / email pipeline --------------------------------------------------

def parse_date_from_filename(filename: str):
    """Extract datetime from Hansard filename."""
    m = re.search(r"(\d{1,2} \w+ \d{4})", filename)
    if m:
        try:
            return datetime.strptime(m.group(1), "%d %B %Y")
        except ValueError:
            return datetime.min
    return datetime.min


def build_digest(files, keywords):
    """Build the digest body text for email."""
    body_lines = []

    # Header
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    body_lines.append(f"Time: {now}")
    body_lines.append("Keywords: " + ", ".join(keywords))

    # Process each transcript file
    total_matches = 0
    for f in sorted(files, key=lambda x: parse_date_from_filename(Path(x).name)):
        text = Path(f).read_text(encoding="utf-8", errors="ignore")
        matches = extract_matches(text, keywords)
        if not matches:
            continue
        total_matches += len(matches)

        body_lines.append(f"\n=== {Path(f).name} ===")
        for i, (kw_label, snippet, speaker, line_list) in enumerate(matches, 1):
            line_label = "line" if len(line_list) == 1 else "lines"
            lines_str = ", ".join(str(n) for n in sorted(set(line_list)))
            if speaker:
                body_lines.append(f"Match #{i} ({speaker}) — {line_label} {lines_str}")
            else:
                body_lines.append(f"Match #{i} — {line_label} {lines_str}")
            body_lines.append(snippet)
            body_lines.append("")

    body_lines.insert(2, f"Matches found: {total_matches}\n")

    if total_matches == 0:
        body_lines.append("\n(No keyword matches found.)")
    else:
        body_lines.append("(Full transcript(s) attached.)")

    return "\n".join(body_lines), total_matches


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

    body, total_hits = build_digest(files, keywords)

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
        contents=body,
        attachments=files,
    )

    update_sent_log(files)

    print(
        f"✅ Email sent to {EMAIL_TO} with {len(files)} file(s), {total_hits} match(es)."
    )


if __name__ == "__main__":
    main()
