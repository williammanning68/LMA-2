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
    Outlook-safe HTML digest:
      - Table-based layout (no flex/grid)
      - Inline styles only (works in Outlook/Word engine)
      - Matches grouped by document, with speaker + line refs
      - Keywords bolded in excerpts
      - Summary table per keyword (HoA / LC / Total)
    Returns: (html, total_matches, counts_dict)
    """
    # ----- helpers (local, digest-scoped) ------------------------------------
    def is_legislative_council(name: str) -> bool:
        n = name.lower()
        return "legislative" in n or "council" in n

    def chamber_of(name: str) -> str:
        return "Legislative Council" if is_legislative_council(name) else "House of Assembly"

    # sentence split (simple, email-safe)
    SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
    def split_sentences(text: str) -> list[str]:
        return SENT_SPLIT_RE.split(text.strip())

    # bold a keyword (case-insensitive) inside an HTML-safe string
    def bold_kw(html_text: str, kw: str) -> str:
        return re.sub(rf"({re.escape(kw)})", r"<strong>\1</strong>", html_text, flags=re.I)

    # find first line number (1-based) for kw inside a range of original lines
    def first_kw_line_for(block_text: str, block_start_line: int, kw: str) -> int | None:
        # scan within block by lines for first occurrence
        lines = block_text.splitlines()
        for i, ln in enumerate(lines):
            if re.search(rf"(?i){re.escape(kw)}", ln):
                return block_start_line + i
        return None

    # Use your existing robust Tas Hansard header regex to segment speakers.
    # We also track the starting source line index for each utterance so we
    # can compute accurate "line N" references.
    def segment_with_line_start(raw: str):
        current_speaker = None
        buff = []
        start_line = 1
        line_idx = 0

        def flush():
            nonlocal buff, current_speaker, start_line
            body = "\n".join(buff).strip()
            if body:
                yield (current_speaker, body, start_line)
            buff = []

        lines = raw.splitlines()
        while line_idx < len(lines):
            raw_line = lines[line_idx]
            line = raw_line.strip()

            # boundaries to skip/flush on (reuse your compiled regexes)
            if (not line) or TIME_STAMP_RE.match(line) or UPPER_HEADING_RE.match(line) or INTERJECTION_RE.match(line):
                if line == "" and buff:
                    buff.append("")  # keep paragraph gap
                else:
                    yield from flush()
                line_idx += 1
                start_line = line_idx + 1
                continue

            m = SPEAKER_HEADER_RE.match(line)
            if m:
                # new speaker — flush previous
                yield from flush()
                title = (m.group("title") or "").strip()
                name = (m.group("name") or m.group("name_only") or "").strip()
                current_speaker = " ".join(x for x in (title, name) if x).strip()
                line_idx += 1
                start_line = line_idx + 1
                continue

            # accumulate
            if not buff:
                start_line = line_idx + 1
            buff.append(raw_line.rstrip())
            line_idx += 1

        yield from flush()

    # For combining nearby keyword mentions within the same speaker’s block
    def combine_if_near(sentences: list[str], hit_idxs: list[int]) -> tuple[int, int]:
        """
        If multiple keywords for a speaker occur within 2 sentences of each other,
        keep the original single-window logic (do not extend).
        Only extend the window if another matched keyword is > 2 sentences away.
        Returns a (start_idx, end_idx_exclusive) to slice sentences.
        """
        if not hit_idxs:
            return (0, min(2, len(sentences)))

        hit_idxs = sorted(set(hit_idxs))
        # Start with a window around the first hit per your rules
        first = hit_idxs[0]
        # rule: if hit in first sentence -> next 2 sentences, else one on either side
        if first == 0:
            start, end = 0, min(3, len(sentences))
        else:
            start, end = max(0, first - 1), min(len(sentences), first + 2)

        # Now check for other hits by the same speaker
        for hi in hit_idxs[1:]:
            if abs(hi - first) > 2:
                # spread out enough -> extend to include that far hit with same rule
                if hi == 0:
                    s2, e2 = 0, min(3, len(sentences))
                else:
                    s2, e2 = max(0, hi - 1), min(len(sentences), hi + 2)
                start = min(start, s2)
                end = max(end, e2)

        return (start, end)

    # ----- build content ------------------------------------------------------
    now_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    total_matches = 0

    # counts[keyword] = {"House of Assembly": n, "Legislative Council": n, "Total": n}
    counts = {kw: {"House of Assembly": 0, "Legislative Council": 0, "Total": 0} for kw in keywords}

    documents_render = []

    for f in sorted(files, key=lambda x: parse_date_from_filename(Path(x).name)):
        name = Path(f).name
        chamber = chamber_of(name)
        raw = Path(f).read_text(encoding="utf-8", errors="ignore")

        # Collect matches for this document: [(match_no, speaker, lines_text, excerpt_html)]
        match_cards = []
        match_no = 0

        for speaker, utt, start_line in segment_with_line_start(raw):
            if not utt.strip():
                continue

            # Which keywords hit in this utterance?
            hits = []
            for kw in keywords:
                if _kw_hit(utt, kw):
                    hits.append(kw)

            if not hits:
                continue

            # sentence window logic
            sentences = split_sentences(utt)
            hit_idxs = []
            for i, s in enumerate(sentences):
                for kw in hits:
                    if _kw_hit(s, kw):
                        hit_idxs.append(i)
                        break
            s_idx, e_idx = combine_if_near(sentences, hit_idxs)
            snippet = " ".join(sentences[s_idx:e_idx]).strip()

            # line reference: first matching kw within the utterance window
            line_refs = []
            for kw in hits:
                ln = first_kw_line_for(utt, start_line, kw)
                if ln is not None:
                    line_refs.append(ln)

            # highlight keywords
            for kw in keywords:
                if _kw_hit(snippet, kw):
                    snippet = bold_kw(snippet, kw)

            # Count each keyword once per utterance block
            for kw in set(hits):
                counts[kw][chamber] += 1
                counts[kw]["Total"] += 1
                total_matches += 1

            match_no += 1
            line_label = (
                f"line {line_refs[0]}" if len(line_refs) == 1
                else f"lines {', '.join(str(x) for x in sorted(set(line_refs)))}"
            ) if line_refs else "line —"

            # Render a match card (table-based)
            match_cards.append(f"""
              <!-- match card -->
              <table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="border:1px solid #D8DCE0;border-radius:6px;margin:12px 0;">
                <tr>
                  <td style="padding:10px 12px;background:#ECF0F1;border-bottom:1px solid #D8DCE0;">
                    <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
                      <tr>
                        <td width="36" align="center" valign="middle"
                            style="background:#4A5A6A;color:#FFFFFF;font:600 12px Segoe UI,Arial,sans-serif;border-radius:18px;height:32px;width:32px;">
                          {match_no}
                        </td>
                        <td style="padding-left:12px;font:600 14px Segoe UI,Arial,sans-serif;color:#475560;">
                          {speaker if speaker else "UNKNOWN"}
                        </td>
                        <td align="right" style="font:12px Segoe UI,Arial,sans-serif;color:#6B7684;">
                          {line_label}
                        </td>
                      </tr>
                    </table>
                  </td>
                </tr>
                <tr>
                  <td style="padding:14px 16px;font:14px/1.6 Segoe UI,Arial,sans-serif;color:#2C3440;">
                    {snippet}
                  </td>
                </tr>
              </table>
            """)

        if match_cards:
            documents_render.append(f"""
              <!-- document block -->
              <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 20px 0;">
                <tr>
                  <td style="padding:12px 14px;border-left:4px solid #C5A572;background:#F7F9FA;border-radius:8px 8px 0 0;">
                    <div style="font:600 15px Segoe UI,Arial,sans-serif;color:#4A5A6A;">{name}</div>
                    <div style="font:13px Segoe UI,Arial,sans-serif;color:#6B7684;margin-top:2px;">{len(match_cards)} match(es)</div>
                  </td>
                </tr>
                <tr>
                  <td style="border:1px solid #D8DCE0;border-top:none;border-radius:0 0 8px 8px;background:#FFFFFF;padding:10px 12px;">
                    {''.join(match_cards)}
                  </td>
                </tr>
              </table>
            """)

    # Build keyword summary table
    def td(s, style=""): return f"<td style=\"padding:8px 10px;border-bottom:1px solid #ECF0F1;{style}\">{s}</td>"
    rows = []
    for kw in keywords:
        hoa = counts[kw]["House of Assembly"]
        lc  = counts[kw]["Legislative Council"]
        tot = counts[kw]["Total"]
        rows.append(
            "<tr>" +
            td(f"<span style='font-weight:600;color:#4A5A6A'>{kw}</span>") +
            td(f"<div style='text-align:center;color:#6B7684;font-weight:600'>{hoa}</div>") +
            td(f"<div style='text-align:center;color:#6B7684;font-weight:600'>{lc}</div>") +
            td(f"<div style='text-align:center;background:#F4E9D3;color:#475560;font-weight:700;border-radius:4px;padding:4px 0'>{tot}</div>") +
            "</tr>"
        )
    summary_table_html = f"""
      <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;background:#FFFFFF;border-radius:8px;overflow:hidden;border:1px solid #D8DCE0;">
        <tr>
          <th align="left" style="background:#4A5A6A;color:#FFFFFF;padding:12px 10px;font:600 12px Segoe UI,Arial,sans-serif;text-transform:uppercase;letter-spacing:.5px;">Keyword</th>
          <th align="left" style="background:#4A5A6A;color:#FFFFFF;padding:12px 10px;font:600 12px Segoe UI,Arial,sans-serif;text-transform:uppercase;letter-spacing:.5px;">House of Assembly</th>
          <th align="left" style="background:#4A5A6A;color:#FFFFFF;padding:12px 10px;font:600 12px Segoe UI,Arial,sans-serif;text-transform:uppercase;letter-spacing:.5px;">Legislative Council</th>
          <th align="left" style="background:#4A5A6A;color:#FFFFFF;padding:12px 10px;font:600 12px Segoe UI,Arial,sans-serif;text-transform:uppercase;letter-spacing:.5px;">Total</th>
        </tr>
        {''.join(rows) if rows else '<tr>' + td('—')*4 + '</tr>'}
      </table>
    """

    # Header stat cards (table layout)
    total_docs = sum(1 for _ in documents_render)
    total_terms = len(keywords)
    header_stats_html = f"""
      <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="margin-top:10px;">
        <tr>
          <td width="25%" valign="top" style="padding:8px 6px;">
            <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#FFFFFF;border-radius:8px;border-left:3px solid #C5A572;">
              <tr><td style="padding:10px 12px;">
                <div style="font:500 12px Segoe UI,Arial,sans-serif;color:#FFFFFF;opacity:.9">Documents</div>
                <div style="font:600 22px Segoe UI,Arial,sans-serif;color:#C5A572"> {total_docs} </div>
                <div style="font:13px Segoe UI,Arial,sans-serif;color:#FFFFFF;opacity:.9">transcripts analyzed</div>
              </td></tr>
            </table>
          </td>
          <td width="25%" valign="top" style="padding:8px 6px;">
            <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#FFFFFF;border-radius:8px;border-left:3px solid #C5A572;">
              <tr><td style="padding:10px 12px;">
                <div style="font:500 12px Segoe UI,Arial,sans-serif;color:#FFFFFF;opacity:.9">Keywords</div>
                <div style="font:600 22px Segoe UI,Arial,sans-serif;color:#C5A572"> {total_terms} </div>
                <div style="font:13px Segoe UI,Arial,sans-serif;color:#FFFFFF;opacity:.9">terms tracked</div>
              </td></tr>
            </table>
          </td>
          <td width="25%" valign="top" style="padding:8px 6px;">
            <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#FFFFFF;border-radius:8px;border-left:3px solid #C5A572;">
              <tr><td style="padding:10px 12px;">
                <div style="font:500 12px Segoe UI,Arial,sans-serif;color:#FFFFFF;opacity:.9">Matches</div>
                <div style="font:600 22px Segoe UI,Arial,sans-serif;color:#C5A572"> {total_matches} </div>
                <div style="font:13px Segoe UI,Arial,sans-serif;color:#FFFFFF;opacity:.9">excerpts found</div>
              </td></tr>
            </table>
          </td>
          <td width="25%" valign="top" style="padding:8px 6px;">
            <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#FFFFFF;border-radius:8px;border-left:3px solid #C5A572;">
              <tr><td style="padding:10px 12px;">
                <div style="font:500 12px Segoe UI,Arial,sans-serif;color:#FFFFFF;opacity:.9">Generated</div>
                <div style="font:600 22px Segoe UI,Arial,sans-serif;color:#C5A572"> Now </div>
                <div style="font:13px Segoe UI,Arial,sans-serif;color:#FFFFFF;opacity:.9">{now_utc}</div>
              </td></tr>
            </table>
          </td>
        </tr>
      </table>
    """

    # Full Outlook-safe shell (no external CSS; all inline)
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="x-apple-disable-message-reformatting">
  <meta http-equiv="x-ua-compatible" content="ie=edge">
  <title>Hansard Keyword Digest</title>
</head>
<body style="margin:0;padding:0;background:#ECF0F1;">
  <center style="width:100%;background:#ECF0F1;">
    <!--[if mso]><table role="presentation" width="600" align="center" cellpadding="0" cellspacing="0"><tr><td><![endif]-->
    <table role="presentation" align="center" width="100%" cellpadding="0" cellspacing="0" style="max-width:900px;margin:0 auto;background:#FFFFFF;">
      <tr>
        <td style="padding:0;">
          <!-- Header -->
          <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#475560;">
            <tr>
              <td style="padding:24px 28px;">
                <div style="font:400 24px Segoe UI,Arial,sans-serif;color:#FFFFFF;">Hansard Keyword Digest</div>
                <div style="font:14px Segoe UI,Arial,sans-serif;color:#FFFFFF;opacity:.9;margin-top:6px;">Comprehensive parliamentary transcript analysis — {now_utc}</div>
                {header_stats_html}
              </td>
            </tr>
          </table>

          <!-- Content -->
          <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#ECF0F1;">
            <tr><td style="height:16px;line-height:16px;">&nbsp;</td></tr>
            <tr>
              <td style="padding:0 16px;">
                <!-- Panel: Keywords Being Tracked -->
                <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#FFFFFF;border-radius:12px;padding:0;border:1px solid #D8DCE0;">
                  <tr>
                    <td style="padding:16px 18px;border-bottom:3px solid #C5A572;">
                      <div style="font:600 18px Segoe UI,Arial,sans-serif;color:#4A5A6A;">Keywords Being Tracked</div>
                      <div style="margin-top:8px;">
                        {"".join(f"<span style='display:inline-block;padding:4px 10px;background:#C5A572;color:#FFFFFF;border-radius:4px;font:600 13px Segoe UI,Arial,sans-serif;margin:4px 6px 0 0;'>{kw}</span>" for kw in keywords)}
                      </div>
                    </td>
                  </tr>
                </table>

                <div style="height:16px;line-height:16px;">&nbsp;</div>

                <!-- Panel: Summary by Chamber -->
                <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#FFFFFF;border-radius:12px;border:1px solid #D8DCE0;">
                  <tr>
                    <td style="padding:16px 18px;border-bottom:3px solid #C5A572;">
                      <div style="font:600 18px Segoe UI,Arial,sans-serif;color:#4A5A6A;">Summary by Chamber</div>
                    </td>
                  </tr>
                  <tr><td style="padding:12px 16px;">{summary_table_html}</td></tr>
                </table>

                <div style="height:16px;line-height:16px;">&nbsp;</div>

                <!-- Documents & Matches -->
                {"".join(documents_render) if documents_render else
                 "<table role='presentation' width='100%'><tr><td style='font:14px Segoe UI,Arial,sans-serif;color:#6B7684;padding:12px;'>No keyword matches found.</td></tr></table>"}
              </td>
            </tr>
            <tr><td style="height:16px;line-height:16px;">&nbsp;</td></tr>
          </table>

          <!-- Footer -->
          <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#4A5A6A;">
            <tr>
              <td style="padding:16px;text-align:center;font:12px Segoe UI,Arial,sans-serif;color:#FFFFFF;opacity:.9;">
                Automated Hansard Digest System — Generated {now_utc}
              </td>
            </tr>
          </table>
        </td>
      </tr>
    </table>
    <!--[if mso]></td></tr></table><![endif]-->
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
