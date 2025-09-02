import os
import re
import glob
from bisect import bisect_right
from pathlib import Path
from datetime import datetime, UTC

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

from premailer import transform

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

def _bake_css_vars(html: str) -> str:
    mapping = {
        "--federal-gold":   "#C5A572",
        "--federal-navy":   "#4A5A6A",
        "--federal-dark":   "#475560",
        "--federal-light":  "#ECF0F1",
        "--federal-accent": "#D4AF37",
        "--white":          "#FFFFFF",
        "--border-light":   "#D8DCE0",
        "--text-primary":   "#2C3440",
        "--text-secondary": "#6B7684",
    }
    for var, val in mapping.items():
        html = html.replace(f"var({var})", val)
    html = re.sub(r":root\s*\{[^}]*\}", "", html, flags=re.S)
    return html

def _inline_css(html: str) -> str:
    return transform(
        html,
        remove_classes=False,
        keep_style_tags=False,
        disable_leftover_css=True,
        strip_important=False,
    )


def build_digest_html(files, keywords):
    """
    Returns: (html_string, total_matches, counts_by_chamber_and_kw)

    Email-safe HTML:
      - Single column
      - Inline styles only (no <style> blocks, no CSS classes)
      - Table-based structure (works in Gmail/Outlook/Apple Mail)
      - Palette: federal gold/navy/dark/light/accent
    """
    import re
    from pathlib import Path
    from datetime import datetime, UTC

    # --- palette (inline hex) ---
    FEDERAL_GOLD   = "#C5A572"
    FEDERAL_NAVY   = "#4A5A6A"
    FEDERAL_DARK   = "#475560"
    FEDERAL_LIGHT  = "#ECF0F1"
    FEDERAL_ACCENT = "#D4AF37"
    BORDER_LIGHT   = "#D8DCE0"
    TEXT_PRIMARY   = "#2C3440"
    TEXT_SECONDARY = "#6B7684"
    WHITE          = "#FFFFFF"

    # helpers
    def esc(s: str) -> str:
        try:
            return _html_escape(s)
        except NameError:
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

    # counters
    chambers = ["House of Assembly", "Legislative Council"]
    counts = {ch: {kw: 0 for kw in keywords} for ch in chambers}
    totals = {kw: 0 for kw in keywords}
    total_matches = 0
    now_utc = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    # per-document sections
    doc_sections = []
    files_sorted = sorted(files, key=lambda x: (parse_date_from_filename(Path(x).name), Path(x).name))
    docs_analyzed = len(files_sorted)

    for fpath in files_sorted:
        name = Path(fpath).name
        text = Path(fpath).read_text(encoding="utf-8", errors="ignore")
        chamber = parse_chamber_from_filename(name)

        matches = extract_matches(text, keywords)
        if not matches:
            continue

        matches.sort(key=lambda item: min(item[3]) if item[3] else 10**9)
        total_matches += len(matches)

        # Document header row
        sec = []
        sec.append(f"""
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse; margin:0 0 18px 0;">
  <tr>
    <td style="background:{WHITE}; border:1px solid {BORDER_LIGHT}; border-left:6px solid {FEDERAL_GOLD}; border-radius:8px; overflow:hidden;">
      <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">
        <tr>
          <td style="padding:14px 16px; border-bottom:1px solid {BORDER_LIGHT};">
            <div style="font-weight:700; color:{FEDERAL_NAVY}; font-size:16px; line-height:1.3;">{esc(name)}</div>
            <div style="color:{TEXT_SECONDARY}; font-size:12px; margin-top:2px;">{esc(chamber)}</div>
          </td>
        </tr>
""")

        # Matches
        for i, (kw_set, excerpt_html, speaker, line_list, _w0, _w1) in enumerate(matches, 1):
            for kw in kw_set:
                if chamber in counts:
                    counts[chamber][kw] += 1
                totals[kw] += 1

            line_label = "line" if len(set(line_list)) == 1 else "lines"
            lines_str = ", ".join(str(n) for n in sorted(set(line_list))) if line_list else "—"
            speaker_html = esc(speaker) if speaker else "UNKNOWN"

            sec.append(f"""
        <tr>
          <td style="padding:0; border-top:1px solid {BORDER_LIGHT};">
            <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">
              <tr>
                <td style="padding:10px 16px;">
                  <div style="font-size:12px; color:{TEXT_SECONDARY}; margin:0 0 6px 0;">
                    <span style="display:inline-block; padding:1px 6px; background:rgba(74,90,106,.08); border:1px solid rgba(74,90,106,.18); border-radius:6px; color:{FEDERAL_DARK}; font-weight:700; margin-right:8px;">Match #{i}</span>
                    <span style="font-weight:600; color:{FEDERAL_DARK};">{speaker_html}</span>
                    <span style="margin-left:10px;">{line_label} {esc(lines_str)}</span>
                  </div>
                  <div style="background:#fbfbfb; border-left:3px solid {FEDERAL_ACCENT}; padding:10px 12px; border-radius:4px; color:{TEXT_PRIMARY}; font-size:14px; line-height:1.5;">
                    {excerpt_html}
                  </div>
                </td>
              </tr>
            </table>
          </td>
        </tr>
""")

        # Close card
        sec.append("""
      </table>
    </td>
  </tr>
</table>
""")
        doc_sections.append("".join(sec))

    # Summary rows
    def summary_rows():
        rows = []
        for kw in keywords:
            hoa = counts["House of Assembly"][kw] if "House of Assembly" in counts else 0
            lc  = counts["Legislative Council"][kw] if "Legislative Council" in counts else 0
            tot = totals[kw]
            rows.append(f"""
<tr>
  <td style="padding:10px 12px; border-bottom:1px solid {BORDER_LIGHT};">
    <span style="display:inline-block; background:rgba(197,165,114,.15); color:{FEDERAL_DARK}; border:1px solid rgba(197,165,114,.35); border-radius:999px; padding:2px 8px; font-size:12px;">{esc(kw)}</span>
  </td>
  <td align="right" style="padding:10px 12px; border-bottom:1px solid {BORDER_LIGHT}; color:{TEXT_SECONDARY};">{hoa}</td>
  <td align="right" style="padding:10px 12px; border-bottom:1px solid {BORDER_LIGHT}; color:{TEXT_SECONDARY};">{lc}</td>
  <td align="right" style="padding:10px 12px; border-bottom:1px solid {BORDER_LIGHT}; font-weight:700; color:{FEDERAL_DARK}; background:rgba(212,175,55,0.08);">{tot}</td>
</tr>
""")
        return "".join(rows)

    summary_table = f"""
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse; background:{WHITE}; border:1px solid {BORDER_LIGHT}; border-radius:8px; overflow:hidden; margin:0 0 18px 0;">
  <thead>
    <tr>
      <th align="left" style="padding:12px 14px; background:{FEDERAL_DARK}; color:{WHITE}; font-weight:600; font-size:13px;">Keyword</th>
      <th align="right" style="padding:12px 14px; background:{FEDERAL_DARK}; color:{WHITE}; font-weight:600; font-size:13px;">House of Assembly</th>
      <th align="right" style="padding:12px 14px; background:{FEDERAL_DARK}; color:{WHITE}; font-weight:600; font-size:13px;">Legislative Council</th>
      <th align="right" style="padding:12px 14px; background:{FEDERAL_DARK}; color:{WHITE}; font-weight:600; font-size:13px;">Total</th>
    </tr>
  </thead>
  <tbody>
    {summary_rows()}
  </tbody>
</table>
"""

    # Header block
    header_block = f"""
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse; margin:0 0 18px 0;">
  <tr>
    <td style="background:{FEDERAL_NAVY}; color:{WHITE}; padding:18px; border-radius:10px; border-bottom:4px solid {FEDERAL_GOLD};">
      <div style="font-size:18px; font-weight:700; margin:0 0 4px 0;">Hansard Keyword Digest</div>
      <div style="opacity:.95; font-size:12px;">Program Runtime: {esc(now_utc)}</div>
    </td>
  </tr>
</table>
"""

    # Summary card
    summary_card = f"""
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse; margin:0 0 18px 0;">
  <tr>
    <td style="background:{WHITE}; border:1px solid {BORDER_LIGHT}; border-radius:8px; padding:12px 14px;">
      <div style="font-size:14px; color:{TEXT_PRIMARY};">
        <span style="color:{TEXT_SECONDARY};">Keywords:</span> {esc(", ".join(keywords))}
        <span style="margin-left:12px; color:{TEXT_SECONDARY};">Documents analyzed:</span> {docs_analyzed}
        <span style="margin-left:12px; color:{TEXT_SECONDARY};">Total matches:</span> {total_matches}
      </div>
    </td>
  </tr>
</table>
"""

    docs_html = "\n".join(doc_sections) if doc_sections else f"""
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse; margin-top:16px;">
  <tr>
    <td style="background:{WHITE}; border:1px solid {BORDER_LIGHT}; border-radius:8px; padding:14px 16px;">No transcripts with matches.</td>
  </tr>
</table>
"""

    # Outer wrapper: use tables + inline styles only
    html = f"""<!doctype html>
<html>
  <head><meta charset="utf-8"><meta name="viewport" content="width=device-width"/></head>
  <body style="margin:0; padding:24px; background:{FEDERAL_LIGHT}; color:{TEXT_PRIMARY}; font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif;">
    <center>
      <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse; max-width:900px; margin:0 auto;">
        <tr><td>
          {header_block}
          {summary_card}
          {summary_table}
          {docs_html}
        </td></tr>
      </table>
    </center>
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
    EMAIL_TO   = os.environ["EMAIL_TO"]

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

    # Build the HTML body with your existing renderer
    body_html, total_hits, _counts = build_digest_html(files, keywords)

    # ✅ STEP 3: Bake CSS variables and inline styles before sending
    body_html = _bake_css_vars(body_html)   # replace var(--color) with real hex
    body_html = _inline_css(body_html)      # inline CSS so Outlook/etc. render it

    subject = f"Hansard keyword digest — {datetime.now().strftime('%d %b %Y')}"
    to_list = [addr.strip() for addr in re.split(r"[,\s]+", EMAIL_TO) if addr.strip()]

    # Plain-text fallback (compact)
    def html_to_text_min(html: str) -> str:
        text = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
        text = re.sub(r"</(p|div|tr|table|section|article|h\d)>", "\n", text, flags=re.I)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text

    # Build multipart/alternative email with attachments
    msg = MIMEMultipart("mixed")
    msg["Subject"] = subject
    msg["From"] = EMAIL_USER
    msg["To"] = ", ".join(to_list)

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(html_to_text_min(body_html), "plain", "utf-8"))
    alt.attach(MIMEText(body_html, "html", "utf-8"))
    msg.attach(alt)

    for fpath in files:
        with open(fpath, "rb") as fp:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(fp.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{Path(fpath).name}"')
        msg.attach(part)

    with smtplib.SMTP("smtp.gmail.com", 587) as s:
        s.ehlo()
        s.starttls()
        s.login(EMAIL_USER, EMAIL_PASS)
        s.sendmail(EMAIL_USER, to_list, msg.as_string())

    update_sent_log(files)
    print(f"✅ Email sent to {EMAIL_TO} with {len(files)} file(s), {total_hits} match(es).")




if __name__ == "__main__":
    main()
