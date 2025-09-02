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

# --- Keyword matching helper -----------------------------------------------
def _kw_hit(text: str, kw: str):
    """
    Return a regex match if the keyword is found in text.
    - For multi-word phrases, do a case-insensitive substring match.
    - For single words, enforce word boundaries.
    """
    if not text or not kw:
        return None
    if " " in kw:
        return re.search(re.escape(kw), text, re.IGNORECASE)
    return re.search(rf"\b{re.escape(kw)}\b", text, re.IGNORECASE)


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
    Build the HTML email body with Outlook-safe rounded corners (VML) and
    a clean, modern layout.  Returns (html, total_matches, counts_by_keyword).

    - Does NOT call _kw_hit (avoids NameError).
    - If a function extract_matches(text, keywords) exists, it uses that to
      obtain (kw, snippet, speaker) tuples. Otherwise, it falls back to a
      basic in-function matcher.
    - Adds <b>…</b> highlighting for all configured keywords.
    - Computes a per-chamber summary table from occurrences.
    """
    import re, html
    from datetime import datetime, timezone
    from pathlib import Path

    # ----- helpers ----------------------------------------------------------

    def program_runtime_str():
        try:
            now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            # Fallback for very old Python if timezone is missing
            now_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        return now_utc

    def is_house_of_assembly(name: str) -> bool:
        name_l = name.lower()
        return ("house_of_assembly" in name_l) or ("house of assembly" in name_l)

    def is_legislative_council(name: str) -> bool:
        name_l = name.lower()
        return ("legislative_council" in name_l) or ("legislative council" in name_l)

    def kw_regex(kw: str):
        # Word-bound for single tokens; gap-tolerant for phrases
        parts = kw.strip().split()
        if len(parts) == 1:
            return re.compile(rf"\b{re.escape(parts[0])}\b", re.IGNORECASE)
        return re.compile(r"\b" + r"\s+".join(re.escape(p) for p in parts) + r"\b", re.IGNORECASE)

    kw_patterns = {kw: kw_regex(kw) for kw in keywords}

    def count_occurrences_per_kw(text: str) -> dict[str, int]:
        counts = {kw: 0 for kw in keywords}
        for kw, pat in kw_patterns.items():
            counts[kw] = len(pat.findall(text))
        return counts

    def highlight_all_keywords(s: str) -> str:
        # Replace longer keywords first to avoid nested/overlap issues
        ordered = sorted(keywords, key=len, reverse=True)
        out = s
        for kw in ordered:
            pat = kw_patterns[kw]
            out = pat.sub(lambda m: f"<b>{html.escape(m.group(0))}</b>", out)
        return out

    def first_line_for(text: str, needle_regex: re.Pattern) -> int | None:
        m = needle_regex.search(text)
        if not m:
            return None
        upto = text[: m.start()]
        return upto.count("\n") + 1  # 1-based line no.

    # Fallback extractor if your enhanced one is not available
    def fallback_extract_matches(text: str, kws: list[str]):
        """Returns a list of dicts: {kw, speaker, snippet, line} (speaker may be None)."""
        results = []
        # naive: take first hit per keyword; try to find a nearby block of text
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        for kw in kws:
            pat = kw_patterns[kw]
            for sent in sentences:
                if pat.search(sent):
                    # Make a tiny excerpt: 1 sentence around the hit
                    snippet = sent.strip()
                    # Try to guess a line number by searching whole text
                    line = first_line_for(text, pat)
                    results.append({
                        "kw": kw,
                        "speaker": None,
                        "snippet": snippet,
                        "line": line
                    })
                    break
        return results

    def normalize_from_extract_matches(text: str, raw):
        """
        Turn your extract_matches() output into a list of dicts with:
        {kw, speaker, snippet, line}
        Compatible with earlier tuples (kw, snippet, speaker) or an extended
        structure if you’ve already added line numbers.
        """
        norm = []
        for item in raw:
            # Common original form: (kw, snippet, speaker)
            if isinstance(item, tuple) and len(item) == 3:
                kw, snippet, speaker = item
                pat = kw_patterns.get(kw)
                line = first_line_for(text, pat) if pat else None
            elif isinstance(item, dict):
                # If you already return dicts, just map what we can
                kw = item.get("kw") or item.get("keyword")
                snippet = item.get("snippet") or item.get("text") or ""
                speaker = item.get("speaker")
                line = item.get("line")
                if line is None and kw in kw_patterns:
                    line = first_line_for(text, kw_patterns[kw])
            else:
                # Unknown shape; skip
                continue
            norm.append({
                "kw": kw,
                "speaker": speaker,
                "snippet": snippet,
                "line": line
            })
        return norm

    # ----- compute data -----------------------------------------------------

    runtime = program_runtime_str()

    # Totals for summary table
    by_kw = {kw: {"hoa": 0, "lc": 0, "total": 0} for kw in keywords}

    # Per-document matches (for rendering below)
    documents = []  # [{name, matches: [{idx, speaker, line_label, excerpt_html}], match_count}]
    grand_total_matches = 0

    for fpath in files:
        name = Path(fpath).name
        try:
            text = Path(fpath).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = ""

        # Count occurrences per chamber for summary
        occ = count_occurrences_per_kw(text)
        for kw, c in occ.items():
            if is_house_of_assembly(name):
                by_kw[kw]["hoa"] += c
            elif is_legislative_council(name):
                by_kw[kw]["lc"] += c
            else:
                # Unknown chamber -> count only in total
                pass
            by_kw[kw]["total"] += c

        # Collect matches for this doc (prefer your extractor if available)
        matches_raw = []
        if "extract_matches" in globals() and callable(globals()["extract_matches"]):
            try:
                matches_raw = globals()["extract_matches"](text, keywords)
            except Exception:
                matches_raw = []
        if not matches_raw:
            matches_raw = fallback_extract_matches(text, keywords)

        matches = normalize_from_extract_matches(text, matches_raw)

        # Build renderable entries
        rendered = []
        for i, m in enumerate(matches, 1):
            speaker = m.get("speaker") or ""
            # Line label
            line = m.get("line")
            if isinstance(line, int):
                line_label = f"line {line}"
            elif isinstance(line, str):
                line_label = line
            else:
                line_label = ""

            # Escape + highlight
            snippet_html = highlight_all_keywords(html.escape(m.get("snippet", "")))

            rendered.append({
                "idx": i,
                "speaker": html.escape(speaker) if speaker else "",
                "line_label": html.escape(line_label) if line_label else "",
                "excerpt_html": snippet_html
            })

        documents.append({
            "name": name,
            "matches": rendered,
            "match_count": len(rendered),
        })
        grand_total_matches += len(rendered)

    # Inline keyword "pills"
    if keywords:
        pills = []
        for kw in keywords:
            pills.append(
                f'<span style="display:inline-block;margin:4px 6px 0 0;padding:6px 10px;'
                f'font:600 12px/1.2 Segoe UI,Arial,sans-serif;color:#4A5A6A;'
                f'background:#C5A57214;border:1px solid #C5A572;border-radius:12px;">'
                f'{html.escape(kw)}</span>'
            )
        keywords_inline_html = "".join(pills)
    else:
        keywords_inline_html = '<span style="font:14px Segoe UI,Arial,sans-serif;color:#475560;">(none)</span>'

    # Summary table rows
    summary_rows = []
    for kw in keywords:
        r = by_kw[kw]
        summary_rows.append(
            f"""<tr>
<td style="padding:10px 12px;border-bottom:1px solid #D9DFE3;">{html.escape(kw)}</td>
<td align="right" style="padding:10px 12px;border-bottom:1px solid #D9DFE3;">{r['hoa']}</td>
<td align="right" style="padding:10px 12px;border-bottom:1px solid #D9DFE3;">{r['lc']}</td>
<td align="right" style="padding:10px 12px;border-bottom:1px solid #D9DFE3;">
  <span style="display:inline-block;min-width:28px;padding:2px 8px;border-radius:10px;background:#ECF0F1;border:1px solid #D9DFE3;color:#475560;font-weight:700;">{r['total']}</span>
</td>
</tr>"""
        )
    summary_rows_html = "".join(summary_rows)

    # ----- HTML (with VML rounded wrappers) ---------------------------------

    html_parts = []
    html_parts.append(f"""\
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="x-apple-disable-message-reformatting">
  <meta name="format-detection" content="telephone=no,address=no,email=no,date=no,url=no">
  <meta name="color-scheme" content="light only">
  <title>Hansard Keyword Digest</title>
</head>
<body style="margin:0;padding:0;background:#f4f6f8;">
  <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%" style="background:#f4f6f8;">
    <tr>
      <td align="center" style="padding:24px 8px;">
        <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="640" style="width:640px;background:#ffffff;border-collapse:separate;border-spacing:0;">
""")

    # HEADER block (VML rounded)
    html_parts.append(f"""\
<tr>
  <td align="center" style="padding:16px 12px;">
    <!--[if mso]>
    <v:roundrect xmlns:v="urn:schemas-microsoft-com:vml"
      arcsize="8%" fillcolor="#4A5A6A" strokecolor="#4A5A6A" strokeweight="1px"
      style="width:640px;mso-fit-shape-to-text:true;">
      <v:textbox inset="0,0,0,0">
    <![endif]-->
      <div style="background:#4A5A6A;border:1px solid #4A5A6A;border-radius:14px;color:#ECF0F1;">
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">
          <tr>
            <td style="padding:20px 24px;font-family:Segoe UI,Arial,sans-serif;">
              <h1 style="margin:0 0 4px 0;font-size:24px;line-height:1.25;font-weight:700;color:#ECF0F1;">
                Hansard Keyword Digest
              </h1>
              <p style="margin:0;font-size:13px;line-height:1.4;color:#ECF0F1;opacity:.9;">
                Comprehensive parliamentary transcript analysis — {runtime}
              </p>
            </td>
          </tr>
        </table>
      </div>
    <!--[if mso]></v:textbox></v:roundrect><![endif]-->
  </td>
</tr>
""")

    # KEYWORDS PANEL (VML rounded)
    html_parts.append(f"""\
<tr>
  <td align="center" style="padding:12px 12px 0;">
    <!--[if mso]>
    <v:roundrect xmlns:v="urn:schemas-microsoft-com:vml"
      arcsize="12%" fillcolor="#ECF0F1" strokecolor="#D9DFE3" strokeweight="1px"
      style="width:640px;mso-fit-shape-to-text:true;">
      <v:textbox inset="0,0,0,0">
    <![endif]-->
      <div style="background:#ECF0F1;border:1px solid #D9DFE3;border-radius:16px;">
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">
          <tr>
            <td style="padding:16px 20px;font-family:Segoe UI,Arial,sans-serif;">
              <h2 style="margin:0 0 8px 0;font-size:16px;line-height:1.3;color:#475560;">Keywords Being Tracked</h2>
              <div style="font-size:13px;line-height:1.6;color:#475560;">
                {keywords_inline_html}
              </div>
            </td>
          </tr>
        </table>
      </div>
    <!--[if mso]></v:textbox></v:roundrect><![endif]-->
  </td>
</tr>
""")

    # SUMMARY PANEL (VML rounded)
    html_parts.append(f"""\
<tr>
  <td align="center" style="padding:12px 12px 0;">
    <!--[if mso]>
    <v:roundrect xmlns:v="urn:schemas-microsoft-com:vml"
      arcsize="12%" fillcolor="#ECF0F1" strokecolor="#D9DFE3" strokeweight="1px"
      style="width:640px;mso-fit-shape-to-text:true;">
      <v:textbox inset="0,0,0,0">
    <![endif]-->
      <div style="background:#ECF0F1;border:1px solid #D9DFE3;border-radius:16px;">
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">
          <tr>
            <td style="padding:16px 20px;font-family:Segoe UI,Arial,sans-serif;">
              <h2 style="margin:0 0 12px 0;font-size:16px;line-height:1.3;color:#475560;">Summary by Chamber</h2>
              <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="border-collapse:collapse;font-size:13px;color:#475560;">
                <tr>
                  <td style="padding:10px 12px;border-bottom:1px solid #D9DFE3;font-weight:700;">Keyword</td>
                  <td align="right" style="padding:10px 12px;border-bottom:1px solid #D9DFE3;font-weight:700;">House of Assembly</td>
                  <td align="right" style="padding:10px 12px;border-bottom:1px solid #D9DFE3;font-weight:700;">Legislative Council</td>
                  <td align="right" style="padding:10px 12px;border-bottom:1px solid #D9DFE3;font-weight:700;">Total</td>
                </tr>
                {summary_rows_html}
              </table>
            </td>
          </tr>
        </table>
      </div>
    <!--[if mso]></v:textbox></v:roundrect><![endif]-->
  </td>
</tr>
""")

    # PER-DOCUMENT sections + MATCH CARDS (VML rounded)
    for doc in documents:
        doc_name = html.escape(doc["name"])
        match_count = doc["match_count"]
        html_parts.append(f"""\
<tr>
  <td style="padding:18px 12px 8px 12px;font-family:Segoe UI,Arial,sans-serif;color:#475560;">
    <div style="font-size:14px;font-weight:700;margin-bottom:4px;">{doc_name}</div>
    <div style="font-size:12px;opacity:.75;margin-bottom:2px;">{match_count} match(es)</div>
  </td>
</tr>
""")
        for m in doc["matches"]:
            idx = m["idx"]
            speaker = m["speaker"]
            line_label = m["line_label"]
            excerpt_html = m["excerpt_html"]

            html_parts.append(f"""\
<tr>
  <td align="center" style="padding:12px 12px 0;">
    <!--[if mso]>
    <v:roundrect xmlns:v="urn:schemas-microsoft-com:vml"
      arcsize="10%" fillcolor="#FFFFFF" strokecolor="#E3E8EB" strokeweight="1px"
      style="width:640px;mso-fit-shape-to-text:true;">
      <v:textbox inset="0,0,0,0">
    <![endif]-->
      <div style="background:#FFFFFF;border:1px solid #E3E8EB;border-radius:12px;">
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">
          <tr>
            <td style="padding:16px 16px 0 16px;font-family:Segoe UI,Arial,sans-serif;">
              <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%">
                <tr>
                  <td style="font-size:12px;font-weight:700;color:#4A5A6A;background:#ECF0F1;border-radius:6px;padding:6px 10px;width:32px;text-align:center;">{idx}</td>
                  <td style="padding-left:10px;font-size:14px;font-weight:700;color:#475560;">{speaker}</td>
                  <td align="right" style="font-size:12px;color:#475560;opacity:.8;">{line_label}</td>
                </tr>
              </table>
            </td>
          </tr>
          <tr>
            <td style="padding:12px 16px 16px 16px;font-family:Segoe UI,Arial,sans-serif;font-size:14px;line-height:1.6;color:#475560;">
              {excerpt_html}
            </td>
          </tr>
        </table>
      </div>
    <!--[if mso]></v:textbox></v:roundrect><![endif]-->
  </td>
</tr>
""")

    # close containers
    html_parts.append("""\
        </table>
      </td>
    </tr>
  </table>
</body>
</html>
""")

    return "".join(html_parts), grand_total_matches, by_kw


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
