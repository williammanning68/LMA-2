import os
import re
import glob
from pathlib import Path
from datetime import datetime

import yagmail
import subprocess  # optional: only used if ATTRIB_WITH_LLM=1


# File that records which transcripts have already been emailed
LOG_FILE = Path("sent.log")


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


def _speaker_map_by_line(text: str):
    """
    Map each line index -> current speaker (persisting until the next real header).
    We treat only genuine headers as speaker boundaries; everything else inherits.
    """
    lines = text.splitlines()
    curr = None
    mapping = []
    for ln in lines:
        s = ln.strip()
        m = SPEAKER_HEADER_RE.match(s)
        # Guard: e.g., "There are the projects:" must NOT be a header
        if m and m.group("name_only") and CONTENT_COLON_RE.match(s):
            m = None

        if m:
            title = (m.group("title") or "").strip()
            name = (m.group("name") or m.group("name_only") or "").strip()
            curr = " ".join(x for x in (title, name) if x) or curr
            mapping.append(curr)
            continue

        mapping.append(curr)
    return lines, mapping


def _nearest_speaker_above(mapping, idx):
    """Walk upward to the closest previous non-None speaker."""
    for i in range(idx, -1, -1):
        if mapping[i]:
            return mapping[i]
    return None


def _kw_hit(text: str, kw: str):
    """Word boundary for single words; substring for phrases."""
    if " " in kw:
        return re.search(re.escape(kw), text, re.IGNORECASE)
    return re.search(rf"\b{re.escape(kw)}\b", text, re.IGNORECASE)


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


def _llm_qc_speaker(full_lines, hit_index, candidates, model=None, timeout=30):
    """
    Ask local Ollama to choose ONE candidate speaker using a large backward context.
    Used only as a last-resort QC. Returns None if uncertain.
    """
    model = model or os.environ.get("ATTRIB_LLM_MODEL", "llama3.2:3b")

    # Give the model substantial context ABOVE the hit so it can see the last header.
    start = max(0, hit_index - 80)  # ~80 lines preceding
    context = "\n".join(full_lines[start: hit_index + 5])[:3000]

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
    Find keyword matches line-by-line, assign speaker by walking back to the last header.
    Optionally call LLM as a final QC if the result looks suspicious or missing.
    """
    use_llm = os.environ.get("ATTRIB_WITH_LLM", "").lower() in ("1", "true", "yes")
    llm_timeout = int(os.environ.get("ATTRIB_LLM_TIMEOUT", "30"))

    # Precompute per-line speaker map from real headers
    lines, line_speaker = _speaker_map_by_line(text)
    # Candidate list for optional QC (unique speakers seen today)
    candidates = sorted({s for s in line_speaker if s})
    norm_candidates = {_canonicalize(c): c for c in candidates}

    results = []
    for i, raw in enumerate(lines):
        row = raw.rstrip()
        if not row:
            continue

        for kw in keywords:
            # match: word boundary for single words, substring for phrases
            if not ((" " in kw and re.search(re.escape(kw), row, re.IGNORECASE)) or
                    re.search(rf"\b{re.escape(kw)}\b", row, re.IGNORECASE)):
                continue

            # Build a concise snippet from nearby lines (≈2–3 sentences / a few lines)
            window = "\n".join(lines[max(0, i-3): min(len(lines), i+6)])
            snippet = re.sub(r"\s+\n", "\n", window).strip()

            # Deterministic attribution by walking backwards to the last header
            speaker = _nearest_speaker_above(line_speaker, i)

            # OPTIONAL: final QC via LLM if speaker missing or looks fishy
            if use_llm and _looks_suspicious(speaker):
                guess = _llm_qc_speaker(lines, i, candidates, timeout=llm_timeout)
                if guess and _canonicalize(guess) in norm_candidates:
                    speaker = norm_candidates[_canonicalize(guess)]

            results.append((kw, snippet, speaker))
            break  # one snippet per line per keyword window

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
        for i, (kw, snippet, speaker) in enumerate(matches, 1):
            if speaker:
                body_lines.append(f"🔹 Match #{i} ({speaker})")
            else:
                body_lines.append(f"🔹 Match #{i}")
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
