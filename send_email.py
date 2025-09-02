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
    Scans transcripts and returns (html, total_hits, counts_by_keyword)

    Email-safe HTML:
      - Table-based layout
      - All styles inlined
      - No CSS variables
      - Rounded corners supported in Outlook via VML on the header band
    """
    import re
    from datetime import datetime, timezone
    from pathlib import Path

    # --------------------------- helpers ---------------------------
    def compile_keyword_pattern(kw: str) -> re.Pattern:
        # Full-word/phrase match, case-insensitive
        return re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)

    kw_patterns = {kw: compile_keyword_pattern(kw) for kw in keywords}

    def highlight(text: str, kw_list):
        # Bold each keyword occurrence (no links)
        out = text
        for kw in sorted(kw_list, key=len, reverse=True):
            pat = re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
            out = pat.sub(lambda m: f"<strong>{m.group(0)}</strong>", out)
        return out

    def find_speaker(prev_lines):
        # Heuristic: look back up to 5 lines for “NAME” patterns
        sp_pat = re.compile(r"^\s*([A-Z][A-Z\s\.\-']+)\s*[:\-–]\s*$")
        for line in reversed(prev_lines[-5:]):
            m = sp_pat.match(line.strip())
            if m:
                return m.group(1).strip()
        # Another common pattern: "Mr WINTER —"
        sp_pat2 = re.compile(r"^\s*(Mr|Ms|Mrs|Dr|Hon|Madam|Sir)\s+[A-Z][A-Z\-']+\b")
        for line in reversed(prev_lines[-5:]):
            if sp_pat2.search(line):
                return sp_pat2.search(line).group(0).strip()
        return "Unknown"

    def sentence_spans(s):
        # crude sentence splitter for snippets
        spans, start = [], 0
        for m in re.finditer(r"([\.!?])\s+", s):
            spans.append((start, m.end()))
            start = m.end()
        spans.append((start, len(s)))
        return spans

    # ------------------------- scan files --------------------------
    counts = {kw: {"hoa": 0, "lc": 0, "total": 0} for kw in keywords}
    file_matches = []  # [{file, display, matches:[{n,speaker,line,excerpt, kws}]}]
    total_hits = 0

    for fp in files:
        name = Path(fp).name
        chamber = "hoa" if "house_of_assembly" in name.lower() else ("lc" if "legislative_council" in name.lower() else "hoa")
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()

        lines = raw.splitlines()
        text = raw
        # Count per keyword
        for kw, pat in kw_patterns.items():
            n = len(list(pat.finditer(text)))
            if n:
                counts[kw][chamber] += n
                counts[kw]["total"] += n
                total_hits += n

        # Build excerpts
        matches = []
        # Map index -> keywords hit at that index
        hit_positions = []
        for kw, pat in kw_patterns.items():
            for m in pat.finditer(text):
                hit_positions.append((m.start(), m.end(), kw))
        if hit_positions:
            # group nearby hits into excerpts (<= 2 sentences apart & same speaker heuristic)
            hit_positions.sort(key=lambda x: x[0])
            # precompute line numbers for fast lookup
            nl_positions = [i for i, ch in enumerate(text) if ch == "\n"]

            def pos_to_line(pos):
                # 1-based line number
                import bisect
                return bisect.bisect_left(nl_positions, pos) + 1

            # Build snippets around each first occurrence in a group
            used_idxs = set()
            group_id = 0
            for i, (start, end, kw0) in enumerate(hit_positions):
                if i in used_idxs:
                    continue
                # speaker near the hit
                line_no = pos_to_line(start)
                prev_lines = lines[max(0, line_no - 6):line_no - 1]
                speaker = find_speaker(prev_lines)

                # sentence window
                # Take surrounding 1 sentence on each side (expand later if needed)
                para_start = text.rfind("\n\n", 0, start)
                para_start = 0 if para_start == -1 else para_start + 2
                para_end = text.find("\n\n", end)
                para_end = len(text) if para_end == -1 else para_end
                para = text[para_start:para_end]

                spans = sentence_spans(para)
                # locate sentence index containing hit
                s_idx = next((si for si, (a, b) in enumerate(spans) if para_start + a <= start <= para_start + b), 0)
                # base window [s_idx-1 .. s_idx+1]
                left = max(0, s_idx - 1)
                right = min(len(spans) - 1, s_idx + 1)

                # look ahead for nearby hits by same speaker (<= 2 sentences apart)
                kws_in_snippet = {kw0}
                for j in range(i + 1, len(hit_positions)):
                    s2, e2, kw2 = hit_positions[j]
                    if s2 - end > 3000:  # far in text; stop early
                        break
                    line2 = pos_to_line(s2)
                    prev_lines2 = lines[max(0, line2 - 6):line2 - 1]
                    speaker2 = find_speaker(prev_lines2)
                    if speaker2 != speaker:
                        continue
                    # if within 2 sentences, merge into same snippet window
                    if para_start <= s2 <= para_end:
                        s2_idx = next((si for si, (a, b) in enumerate(spans) if para_start + a <= s2 <= para_start + b), s_idx)
                        if abs(s2_idx - s_idx) <= 2:
                            left = min(left, min(s_idx, s2_idx) - 1 if min(s_idx, s2_idx) > 0 else 0)
                            right = max(right, max(s_idx, s2_idx) + 1 if max(s_idx, s2_idx) + 1 < len(spans) else len(spans) - 1)
                            kws_in_snippet.add(kw2)
                            used_idxs.add(j)

                a, _ = spans[left]
                _, b = spans[right]
                excerpt = para[a:b].strip()
                excerpt = highlight(excerpt, list(kws_in_snippet))

                group_id += 1
                matches.append({
                    "n": group_id,
                    "speaker": speaker,
                    "line": line_no,
                    "kws": sorted(kws_in_snippet),
                    "excerpt": excerpt
                })

        file_matches.append({
            "file": fp,
            "display": name,
            "matches": matches
        })

    # ---------------------- render email (tables) -------------------
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Summaries
    def td_num(n):
        return f'''<td align="right" style="font:14px/20px Arial,Helvetica,sans-serif;color:#475560;padding:8px 12px;border-bottom:1px solid #ECF0F1;">{n}</td>'''

    rows_summary = []
    for kw in keywords:
        rows_summary.append(
            f'''
            <tr>
              <td style="font:14px/20px Arial,Helvetica,sans-serif;color:#475560;padding:8px 12px;border-bottom:1px solid #ECF0F1;">{kw}</td>
              {td_num(counts[kw]["hoa"])}{td_num(counts[kw]["lc"])}{td_num(counts[kw]["total"])}
            </tr>
            '''
        )
    summary_table_html = "\n".join(rows_summary) if rows_summary else ""

    # Keyword chips fallback (comma separated for Outlook)
    if keywords:
        kw_list_html = ", ".join(keywords)
    else:
        kw_list_html = "—"

    # Per-file sections
    file_sections = []
    for doc in file_matches:
        matches_html = []
        if doc["matches"]:
            for m in doc["matches"]:
                matches_html.append(f"""
                  <tr>
                    <td style="padding:0 0 12px 0;">
                      <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="border-collapse:separate;">
                        <tr>
                          <td width="40" align="center" valign="top" style="background:#ECF0F1;border-radius:8px;font:bold 14px/40px Arial,Helvetica,sans-serif;color:#4A5A6A;height:40px;">{m["n"]}</td>
                          <td style="width:12px;">&nbsp;</td>
                          <td valign="top" style="background:#FFFFFF;border:1px solid #ECF0F1;border-radius:12px;padding:12px;">
                            <div style="font:bold 14px/20px Arial,Helvetica,sans-serif;color:#4A5A6A;margin:0 0 6px 0;">{m["speaker"]} <span style="font-weight:normal;color:#8795A1;">— line {m["line"]}</span></div>
                            <div style="font:14px/22px Arial,Helvetica,sans-serif;color:#475560;">{m["excerpt"]}</div>
                          </td>
                        </tr>
                      </table>
                    </td>
                  </tr>
                """)
        else:
            matches_html.append(f'''
              <tr>
                <td style="font:14px/20px Arial,Helvetica,sans-serif;color:#8795A1;padding:8px 0;">No keyword matches found in this transcript.</td>
              </tr>
            ''')

        section = f"""
        <tr>
          <td style="padding:16px 0 8px 0;font:bold 16px/20px Arial,Helvetica,sans-serif;color:#4A5A6A;">
            {doc["display"]}
          </td>
        </tr>
        <tr><td style="font:12px/18px Arial,Helvetica,sans-serif;color:#8795A1;padding:0 0 8px 0;">{len(doc["matches"])} match(es)</td></tr>
        {''.join(matches_html)}
        """
        file_sections.append(section)
    files_block_html = "\n".join(file_sections)

    # Small KPI cards
    k_card = lambda title, value: f"""
      <td width="150" valign="top" style="padding:0 8px 0 0;">
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="border-collapse:separate;background:#FFFFFF;border:1px solid #ECF0F1;border-radius:14px;">
          <tr><td align="center" style="font:bold 26px/34px Arial,Helvetica,sans-serif;color:#C5A572;padding:16px 12px 4px 12px;">{value}</td></tr>
          <tr><td align="center" style="font:12px/16px Arial,Helvetica,sans-serif;color:#475560;padding:0 12px 14px 12px;">{title}</td></tr>
        </table>
      </td>
    """

    # Header band with rounded corners (VML for Outlook)
    header_band = f"""
    <!--[if mso]>
    <v:roundrect xmlns:v="urn:schemas-microsoft-com:vml" arcsize="8%" fillcolor="#4A5A6A" stroke="f" style="width:560px;height:60px;">
      <v:textbox inset="0,0,0,0">
        <div style="text-align:left;color:#FFFFFF;font:bold 22px Arial,Helvetica,sans-serif;line-height:60px;padding-left:20px;">Hansard Keyword Digest</div>
      </v:textbox>
    </v:roundrect>
    <![endif]-->
    <!--[if !mso]><!-- -->
    <div style="background:#4A5A6A;border-radius:14px;color:#FFFFFF;font:bold 22px/26px Arial,Helvetica,sans-serif;padding:18px 20px;">Hansard Keyword Digest</div>
    <!--<![endif]-->
    """

    # Outer shell
    html = f"""
<!DOCTYPE html>
<html>
  <body style="margin:0;padding:0;background:#F5F7F9;">
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="background:#F5F7F9;">
      <tr>
        <td align="center" style="padding:20px 12px;">
          <table role="presentation" width="600" cellspacing="0" cellpadding="0" border="0" style="width:600px;max-width:600px;background:#F5F7F9;border-collapse:separate;">
            <tr><td>{header_band}</td></tr>

            <tr><td height="10" style="font-size:0;line-height:0;">&nbsp;</td></tr>

            <!-- KPI row -->
            <tr>
              <td>
                <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="border-collapse:separate;">
                  <tr>
                    {k_card("New transcripts", len(files))}
                    {k_card("Keywords", len(keywords))}
                    {k_card("Total matches", total_hits)}
                    <td valign="top" style="padding:0;">
                      <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="border-collapse:separate;background:#FFFFFF;border:1px solid #ECF0F1;border-radius:14px;">
                        <tr><td align="center" style="font:bold 20px/34px Arial,Helvetica,sans-serif;color:#4A5A6A;padding:16px 12px 4px 12px;">Now</td></tr>
                        <tr><td align="center" style="font:12px/16px Arial,Helvetica,sans-serif;color:#475560;padding:0 12px 14px 12px;">{now_utc}</td></tr>
                      </table>
                    </td>
                  </tr>
                </table>
              </td>
            </tr>

            <tr><td height="16" style="font-size:0;line-height:0;">&nbsp;</td></tr>

            <!-- Keywords Being Tracked -->
            <tr>
              <td>
                <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="border-collapse:separate;background:#FFFFFF;border:1px solid #ECF0F1;border-radius:14px;">
                  <tr>
                    <td style="padding:14px 16px;font:bold 16px/20px Arial,Helvetica,sans-serif;color:#4A5A6A;">Keywords Being Tracked</td>
                  </tr>
                  <tr>
                    <td style="padding:0 16px 16px 16px;font:14px/20px Arial,Helvetica,sans-serif;color:#475560;">{kw_list_html}</td>
                  </tr>
                </table>
              </td>
            </tr>

            <tr><td height="16" style="font-size:0;line-height:0;">&nbsp;</td></tr>

            <!-- Summary by Chamber -->
            <tr>
              <td>
                <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="border-collapse:separate;background:#FFFFFF;border:1px solid #ECF0F1;border-radius:14px;">
                  <tr>
                    <td style="padding:14px 16px 6px 16px;font:bold 16px/20px Arial,Helvetica,sans-serif;color:#4A5A6A;">Summary by Chamber</td>
                  </tr>
                  <tr>
                    <td style="padding:0 16px 16px 16px;">
                      <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="border-collapse:collapse;">
                        <tr>
                          <th align="left" style="font:bold 12px/16px Arial,Helvetica,sans-serif;color:#8795A1;padding:8px 12px;border-bottom:2px solid #ECF0F1;">Keyword</th>
                          <th align="right" style="font:bold 12px/16px Arial,Helvetica,sans-serif;color:#8795A1;padding:8px 12px;border-bottom:2px solid #ECF0F1;">House of Assembly</th>
                          <th align="right" style="font:bold 12px/16px Arial,Helvetica,sans-serif;color:#8795A1;padding:8px 12px;border-bottom:2px solid #ECF0F1;">Legislative Council</th>
                          <th align="right" style="font:bold 12px/16px Arial,Helvetica,sans-serif;color:#8795A1;padding:8px 12px;border-bottom:2px solid #ECF0F1;">Total</th>
                        </tr>
                        {summary_table_html}
                      </table>
                    </td>
                  </tr>
                </table>
              </td>
            </tr>

            <tr><td height="16" style="font-size:0;line-height:0;">&nbsp;</td></tr>

            <!-- Files & matches -->
            <tr>
              <td>
                <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="border-collapse:separate;background:#FFFFFF;border:1px solid #ECF0F1;border-radius:14px;padding:0 16px 12px 16px;">
                  {files_block_html if files_block_html else '<tr><td style="font:14px/20px Arial,Helvetica,sans-serif;color:#8795A1;padding:16px 0;">No transcripts contained keyword matches.</td></tr>'}
                </table>
              </td>
            </tr>

            <tr><td height="22" style="font-size:0;line-height:0;">&nbsp;</td></tr>

          </table>
        </td>
      </tr>
    </table>
  </body>
</html>
    """

    return html, total_hits, counts



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
