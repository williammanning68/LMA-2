import os
import re
import glob
from pathlib import Path
from datetime import datetime

import yagmail
import subprocess 


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

# --- Speaker-aware segmentation tuned for Tas Hansard ---------------------------------
# Accept colon OR dash after the header; allow title-only (e.g., "The SPEAKER")
SPEAKER_HEADER_RE = re.compile(
    r"""
^
(?:
  # (A) Title + optional name
  (?P<title>
      Mr|Ms|Mrs|Miss|Hon|Dr|Prof|Professor|
      Premier|Madam\s+SPEAKER|The\s+SPEAKER|The\s+PRESIDENT|The\s+CLERK|
      Deputy\s+Speaker|Deputy\s+President
  )
  (?:[\s.]+(?P<name>[A-Z][A-Za-z'’\-]+(?:\s+[A-Z][A-Za-z'’\-]+){0,3}))?
 |
  # (B) Name-only (covers things like "Prof RAZAY" or "Jane HOWLETT")
  (?P<name_only>[A-Z][A-Za-z'’\-]+(?:\s+[A-Z][A-Za-z'’\-]+){0,3})
)
(?:\s*\([^)]*\))?        # optional (Electorate—Portfolio)
\s*
(?::|[-–—]\s)            # delimiter: ":" OR " - " / "– " / "— "
""",
    re.IGNORECASE | re.VERBOSE,
)

# Timestamps and headings to skip/flush on
TIME_STAMP_RE = re.compile(r"^\[\d{1,2}\.\d{2}\s*(a|p)\.m\.\]$", re.IGNORECASE)
UPPER_HEADING_RE = re.compile(r"^[A-Z][A-Z\s’'—\-&,;:.()]+$")  # e.g., MOTION, ADJOURNMENT
INTERJECTION_RE = re.compile(r"^(Members interjecting\.|The House suspended .+)$", re.IGNORECASE)

def _segment_utterances(text: str):
    """
    Yield (speaker, utterance_text). Speaker persists until a new header appears.
    Skip timestamps, headings, and editorial interjections.
    """
    current_speaker = None
    buff = []

    def flush():
        nonlocal buff, current_speaker
        body = "\n".join(buff).strip()
        if body:
            yield (current_speaker, body)
        buff = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        # boundaries to skip (do not attach to any speaker)
        if not line or TIME_STAMP_RE.match(line) or UPPER_HEADING_RE.match(line) or INTERJECTION_RE.match(line):
            if line == "" and buff:
                buff.append("")  # keep paragraph separation inside same speaker
            else:
                yield from flush()
            continue

        m = SPEAKER_HEADER_RE.match(line)
        if m:
            yield from flush()
            title = (m.group("title") or "").strip()
            name = (m.group("name") or m.group("name_only") or "").strip()
            current_speaker = " ".join(x for x in (title, name) if x).strip()
            continue

        buff.append(raw_line.rstrip())

    yield from flush()

def _canonicalize(s: str) -> str:
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"\b(Mr|Ms|Mrs|Miss|Hon|Dr|Prof|Professor|Premier|Madam SPEAKER|The SPEAKER|The PRESIDENT|The CLERK|Deputy Speaker|Deputy President)\b\.?", "", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

def _known_speakers_from(text: str) -> list[str]:
    seen = []
    for m in SPEAKER_HEADER_RE.finditer(text):
        title = (m.group("title") or "").strip()
        name = (m.group("name") or m.group("name_only") or "").strip()
        spk = " ".join(x for x in (title, name) if x).strip()
        if spk and spk not in seen:
            seen.append(spk)
    for r in ["The SPEAKER", "Madam SPEAKER", "The CLERK"]:
        if r not in seen:
            seen.append(r)
    return seen

def _llm_guess_speaker(snippet: str, context: str, candidates: list[str], model: str | None = None, timeout: int = 30) -> str | None:
    """
    Ask local Ollama (e.g., llama3.2:3b) to choose ONE candidate or UNKNOWN.
    """
    model = model or os.environ.get("ATTRIB_LLM_MODEL", "llama3.2:3b")
    options = "\n".join(f"- {c}" for c in candidates[:40])  # keep prompt small
    prompt = f"""Link this Hansard excerpt to the most likely speaker.
Return EXACTLY one item from the candidate list below. If none fits, return "UNKNOWN".

Candidates:
{options}

Context (nearby lines):
{context[:1500]}

Excerpt:
{snippet}
"""
    try:
        out = subprocess.check_output(["ollama", "run", model, prompt], text=True, timeout=timeout)
        ans = out.strip().splitlines()[-1].strip()
        if ans.upper().startswith("UNKNOWN") or not ans:
            return None
        return ans
    except Exception:
        return None

def _kw_hit(text: str, kw: str):
    # Word boundary for single words; tolerant substring for phrases
    if " " in kw:
        return re.search(re.escape(kw), text, re.IGNORECASE)
    return re.search(rf"\b{re.escape(kw)}\b", text, re.IGNORECASE)

def extract_matches(text: str, keywords):
    """
    Find keyword matches across speaker-anchored utterances.
    If the speaker isn't known from headers, optionally call LLM to infer it.
    """
    use_llm = os.environ.get("ATTRIB_WITH_LLM", "").lower() in ("1", "true", "yes")
    llm_timeout = int(os.environ.get("ATTRIB_LLM_TIMEOUT", "30"))
    candidates = _known_speakers_from(text)
    norm_candidates = { _canonicalize(c): c for c in candidates }
    results = []

    for speaker, utt in _segment_utterances(text):
        if not utt.strip():
            continue

        for kw in keywords:
            if not _kw_hit(utt, kw):
                continue

            # 2–3 sentence snippet around first occurrence
            sentences = re.split(r"(?<=[.!?])\s+", utt.strip())
            where = next((i for i, s in enumerate(sentences) if _kw_hit(s, kw)), 0)
            start, end = max(0, where - 1), min(len(sentences), where + 2)
            snippet = " ".join(sentences[start:end]).strip()

            linked = speaker
            if (not linked) and use_llm:
                guess = _llm_guess_speaker(snippet, context=utt, candidates=candidates, timeout=llm_timeout)
                if guess and _canonicalize(guess) in norm_candidates:
                    linked = norm_candidates[_canonicalize(guess)]
                else:
                    linked = None  # prefer UNKNOWN over wrong

            results.append((kw, snippet, linked))
            break  # one snippet per utterance per keyword

    return results


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

