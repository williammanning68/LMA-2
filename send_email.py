import os
import re
import glob
from bisect import bisect_right
from pathlib import Path
from datetime import datetime, UTC
from zoneinfo import ZoneInfo

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
    """
    Visual-only change: match your template's silver highlight while preserving the same
    extraction method — we just wrap hits in a styled span.
    """
    out = text_html
    for kw in sorted(keywords, key=len, reverse=True):
        if " " in kw:
            pat = re.compile(re.escape(_html_escape(kw)), re.IGNORECASE)
        else:
            pat = re.compile(rf"\b{re.escape(_html_escape(kw))}\b", re.IGNORECASE)
        out = pat.sub(lambda m: f"<b><span style=\"background:silver;mso-highlight:silver\">{m.group(0)}</span></b>", out)
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

        # Remove identical (start,end) windows to avoid duplicate excerpts
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


def _mso_td(content: str, style: str) -> str:
    return f"<td style='{style}'>{content}</td>"


def build_digest_html(files, keywords):
    """
    Build HTML using your exact Outlook/Word layout:
      - Hero bar (dark slate) with title and Program Run date
      - Grey section containing 'Detection Match by Chamber' with gold underline and a 4-column summary
      - Per-file match blocks in white tables styled to match the doc
      - Bottom dark beta banner
    """
    # Program run time in Hobart local time (as your doc is AU-centric)
    hobart = ZoneInfo("Australia/Hobart")
    now_local = datetime.now(hobart).strftime("%d %B %Y %I:%M %p %Z")

    chambers = ["House of Assembly", "Legislative Council"]
    counts = {ch: {kw: 0 for kw in keywords} for ch in chambers}
    totals = {kw: 0 for kw in keywords}

    doc_sections = []
    total_matches = 0

    # Build per-file blocks and accumulate counts
    for f in sorted(files, key=lambda x: (parse_date_from_filename(Path(x).name), Path(x).name)):
        text = Path(f).read_text(encoding="utf-8", errors="ignore")
        chamber = parse_chamber_from_filename(Path(f).name)
        matches = extract_matches(text, keywords)

        # Per-file header (white box with border like doc)
        header = (
            "<table class='MsoNormalTable' border='1' cellspacing='0' cellpadding='0' width='100%' "
            "style='width:100.0%;mso-cellspacing:0cm;background:white;border:solid #D8DCE0 1.0pt;"
            "mso-border-alt:solid #D8DCE0 .75pt;mso-yfti-tbllook:1184;mso-padding-alt:0cm 0cm 0cm 0cm'>"
            "<tr>"
            "<td style='border:none;border-bottom:solid #C5A572 2.25pt;padding:12.0pt 13.5pt 12.0pt 13.5pt'>"
            f"<p class='MsoNormal' style='margin-bottom:0cm;line-height:normal'>"
            f"<b><span style='font-family:\"Segoe UI\",sans-serif;color:black;mso-color-alt:windowtext'>{_html_escape(Path(f).name)}</span></b>"
            f"<span style='font-family:\"Segoe UI\",sans-serif;color:#8795A1'> — {len(matches)} match(es)</span>"
            "</p>"
            "</td>"
            "</tr>"
        )

        if not matches:
            doc_sections.append(header + "</table>")
            continue

        matches.sort(key=lambda item: min(item[3]) if item[3] else 10**9)
        total_matches += len(matches)

        # Match blocks
        block_rows = []
        for i, (kw_set, excerpt_html, speaker, line_list, win_start, win_end) in enumerate(matches, 1):
            for kw in kw_set:
                if chamber in counts:
                    counts[chamber][kw] += 1
                totals[kw] += 1

            first_line = min(line_list) if line_list else win_start
            line_label = "line" if len(line_list) <= 1 else "lines"
            lines_str = ", ".join(str(n) for n in sorted(set(line_list))) if line_list else str(first_line)

            # Speaker label; if suspicious/missing → "Unknown"
            speaker_display = speaker if (speaker and not _looks_suspicious(speaker)) else "Unknown"

            block_rows.append(
                "<tr><td style='border:none;padding:9.0pt 12.0pt 9.0pt 12.0pt'>"
                "<table class='MsoNormalTable' border='0' cellspacing='0' cellpadding='0' width='100%' "
                "style='width:100.0%;background:white;border-collapse:collapse;border:solid #D8DCE0 1.0pt;"
                "mso-border-alt:solid #D8DCE0 .75pt;'>"
                "<tr>"
                "<td style='border-left:solid #C5A572 3.0pt;border-top:none;border-bottom:none;border-right:none;"
                "padding:7.5pt 9.0pt 7.5pt 9.0pt'>"
                f"<p class='MsoNormal' style='margin-bottom:6.0pt;line-height:normal'>"
                f"<b><span style='font-family:\"Segoe UI\",sans-serif;color:#475560'>{_html_escape(speaker_display)}</span></b>"
                f"<span style='font-family:\"Segoe UI\",sans-serif;color:#8795A1'> — {line_label} {lines_str}</span>"
                f"</p>"
                f"<p class='MsoNormal' style='margin:0;line-height:22px'><span style='font-family:\"Segoe UI\",sans-serif;color:#475560'>{excerpt_html}</span></p>"
                "</td>"
                "</tr>"
                "</table>"
                "</td></tr>"
            )

        doc_sections.append(header + "".join(block_rows) + "</table>")

    # Summary table rows (exact widths/borders and header colors per your doc)
    def _summary_rows():
        out = []
        for kw in keywords:
            hoa = counts["House of Assembly"][kw] if "House of Assembly" in counts else 0
            lc = counts["Legislative Council"][kw] if "Legislative Council" in counts else 0
            tot = totals[kw]
            out.append(
                "<tr>"
                f"<td width='28%' style='width:28.12%;border-left:solid #D8DCE0 1.0pt;border-right:none;border-top:none;"
                f"border-bottom:solid #D8DCE0 1.0pt;padding:7.5pt 9.0pt 7.5pt 9.0pt'>"
                f"<p class='MsoNormal' style='margin:0;line-height:normal'><span style='font-family:\"Segoe UI\",sans-serif;color:#475560'>{_html_escape(kw)}</span></p></td>"
                f"<td width='28%' align='right' style='width:28.12%;border-top:none;border-bottom:solid #D8DCE0 1.0pt;"
                f"border-left:none;border-right:none;padding:7.5pt 9.0pt 7.5pt 9.0pt'>"
                f"<p class='MsoNormal' style='margin:0;line-height:normal'><span style='font-family:\"Segoe UI\",sans-serif;color:#475560'>{hoa}</span></p></td>"
                f"<td width='28%' align='right' style='width:28.14%;border-top:none;border-bottom:solid #D8DCE0 1.0pt;"
                f"border-left:none;border-right:none;padding:7.5pt 9.0pt 7.5pt 9.0pt'>"
                f"<p class='MsoNormal' style='margin:0;line-height:normal'><span style='font-family:\"Segoe UI\",sans-serif;color:#475560'>{lc}</span></p></td>"
                f"<td width='16%' align='right' style='width:15.62%;border-right:solid #D8DCE0 1.0pt;border-left:none;border-top:none;"
                f"border-bottom:solid #D8DCE0 1.0pt;padding:7.5pt 9.0pt 7.5pt 9.0pt'>"
                f"<p class='MsoNormal' style='margin:0;line-height:normal'><span style='font-family:\"Segoe UI\",sans-serif;color:#475560'>{tot}</span></p></td>"
                "</tr>"
            )
        return "".join(out)

    # Build the 'Detection Match by Chamber' block exactly like the doc
    summary_table = (
        "<table class='MsoNormalTable' border='1' cellspacing='0' cellpadding='0' width='100%' "
        "style='width:100.0%;mso-cellspacing:0cm;background:white;border:solid #D8DCE0 1.0pt;"
        "mso-border-alt:solid #D8DCE0 .75pt;mso-yfti-tbllook:1184;mso-padding-alt:0cm 0cm 0cm 0cm'>"
        "<tr>"
        "<td style='border:none;border-bottom:solid #C5A572 2.25pt;padding:12.0pt 13.5pt 12.0pt 13.5pt'>"
        "<p class='MsoNormal' align='center' style='margin-bottom:0cm;text-align:center;line-height:normal'>"
        "<b><span style='font-size:16.0pt;mso-ascii-font-family:Aptos;mso-hansi-font-family:Aptos;"
        "mso-bidi-font-family:\"Segoe UI\";color:black;mso-color-alt:windowtext'>Detection Match by Chamber</span></b>"
        "</p>"
        "</td>"
        "</tr>"
        "<tr>"
        "<td style='border:none;padding:9.0pt 12.0pt 9.0pt 12.0pt'>"
        "<table class='MsoNormalTable' border='1' cellspacing='0' cellpadding='0' width='100%' "
        "style='width:100.0%;background:white;border-collapse:collapse;border:none;mso-border-alt:solid #D8DCE0 .75pt;"
        "mso-yfti-tbllook:1184;mso-padding-alt:0cm 0cm 0cm 0cm'>"
        # Header row
        "<tr>"
        "<td width='28%' style='width:28.12%;border-top:solid #D8DCE0 1.0pt;border-left:solid #D8DCE0 1.0pt;"
        "border-bottom:none;border-right:none;mso-border-top-alt:solid #D8DCE0 .75pt;mso-border-left-alt:solid #D8DCE0 .75pt;"
        "padding:9.0pt 7.5pt 9.0pt 7.5pt;background:#4A5A6A;'>"
        "<p class='MsoNormal' align='center' style='margin:0;text-align:center;line-height:normal'>"
        "<b><span style='font-family:\"Segoe UI\",sans-serif;color:white;mso-themecolor:background1'>Keyword</span></b></p></td>"
        "<td width='28%' style='width:28.12%;border-top:solid #D8DCE0 1.0pt;border-left:none;border-bottom:none;border-right:none;"
        "mso-border-top-alt:solid #D8DCE0 .75pt;padding:9.0pt 7.5pt 9.0pt 7.5pt;background:#4A5A6A;'>"
        "<p class='MsoNormal' align='right' style='margin:0;text-align:right;line-height:normal'>"
        "<b><span style='font-family:\"Segoe UI\",sans-serif;color:white;mso-themecolor:background1'>House of Assembly</span></b></p></td>"
        "<td width='28%' style='width:28.14%;border-top:solid #D8DCE0 1.0pt;border-left:none;border-bottom:none;border-right:none;"
        "mso-border-top-alt:solid #D8DCE0 .75pt;padding:9.0pt 7.5pt 9.0pt 7.5pt;background:#4A5A6A;'>"
        "<p class='MsoNormal' align='right' style='margin:0;text-align:right;line-height:normal'>"
        "<b><span style='font-family:\"Segoe UI\",sans-serif;color:white;mso-themecolor:background1'>Legislative Council</span></b></p></td>"
        "<td width='16%' style='width:15.62%;border-top:solid #D8DCE0 1.0pt;border-right:solid #D8DCE0 1.0pt;"
        "border-left:none;border-bottom:none;mso-border-top-alt:solid #D8DCE0 .75pt;mso-border-right-alt:solid #D8DCE0 .75pt;"
        "padding:9.0pt 7.5pt 9.0pt 7.5pt;background:#4A5A6A;'>"
        "<p class='MsoNormal' align='right' style='margin:0;text-align:right;line-height:normal'>"
        "<b><span style='font-family:\"Segoe UI\",sans-serif;color:white;mso-themecolor:background1'>Total</span></b></p></td>"
        "</tr>"
        f"{_summary_rows()}"
        "</table>"
        "</td>"
        "</tr>"
        "</table>"
    )

    # Compose full HTML using your wrapper tables and footer banner
    # (We avoid external <link> references that Word inserts; they aren’t needed for rendering.)
    html = (
        "<!DOCTYPE html><html xmlns:v='urn:schemas-microsoft-com:vml' "
        "xmlns:o='urn:schemas-microsoft-com:office:office'>"
        "<head><meta http-equiv='Content-Type' content='text/html; charset=utf-8'></head>"
        "<body lang='EN-AU' link='#467886' vlink='#96607D' style='tab-interval:36.0pt;word-wrap:break-word'>"
        "<div class='WordSection1'>"
        "<div align='center'>"
        "<table class='MsoNormalTable' border='0' cellspacing='0' cellpadding='0' width='600' "
        "style='width:450.0pt;mso-cellspacing:0cm;mso-yfti-tbllook:1184;mso-padding-alt:0cm 0cm 0cm 0cm'>"
        "<tr><td style='padding:0'>"
        "<div align='center'>"
        "<table class='MsoNormalTable' border='0' cellspacing='0' cellpadding='0' width='900' "
        "style='width:675.0pt;mso-cellspacing:0cm;background:white;mso-yfti-tbllook:1184;mso-padding-alt:0cm 0cm 0cm 0cm'>"
        "<tr><td style='padding:0'>"

        # HERO
        "<table class='MsoNormalTable' border='0' cellspacing='0' cellpadding='0' width='100%' "
        "style='width:100.0%;mso-cellspacing:0cm;background:#475560;mso-yfti-tbllook:1184;'>"
        "<tr><td style='padding:18.0pt 21.0pt'>"
        "<p class='MsoNormal' align='center' style='margin-bottom:0cm;text-align:center;line-height:normal'>"
        "<b><span style='font-size:28.0pt;font-family:\"Segoe UI\",sans-serif;color:white;mso-themecolor:background1'>"
        "Hansard Monitor   BETA Version 18.3</span></b></p>"
        "<p class='MsoNormal' align='center' style='margin-bottom:0cm;text-align:center;line-height:normal'>"
        f"<span style='font-size:16.0pt;font-family:\"Segoe UI\",sans-serif;color:white;mso-themecolor:background1'>"
        f"Program Run: {now_local}</span></p>"
        "</td></tr></table>"

        # GREY SECTION
        "<p class='MsoNormal' style='margin-bottom:0cm;line-height:0'><span style='display:none;mso-hide:all'>&nbsp;</span></p>"
        "<table class='MsoNormalTable' border='0' cellspacing='0' cellpadding='0' width='100%' "
        "style='width:100.0%;mso-cellspacing:0cm;background:#ECF0F1;mso-yfti-tbllook:1184;'>"
        "<tr><td style='padding:0cm 12.0pt'>"
        "<p class='MsoNormal' style='margin-bottom:0cm;line-height:0'>&nbsp;</p>"

        # SUMMARY BLOCK
        f"{summary_table}"

        # FILES & MATCHES (each as white box with gold heading rule)
        f"{''.join('<p class=\"MsoNormal\" style=\"margin:0;line-height:0\">&nbsp;</p>'+s for s in doc_sections)}"

        "</td></tr>"
        "<tr style='height:12.0pt'><td style='height:12.0pt;padding:0'><p class='MsoNormal' style='margin:0;line-height:0'>&nbsp;</p></td></tr>"
        "</table>"

        # FOOTER BETA BANNER
        "<p class='MsoNormal' style='margin-bottom:0cm;line-height:0'><span style='display:none;mso-hide:all'>&nbsp;</span></p>"
        "<table class='MsoNormalTable' border='0' cellspacing='0' cellpadding='0' width='100%' "
        "style='width:100.0%;mso-cellspacing:0cm;background:#4A5A6A;mso-yfti-tbllook:1184;'>"
        "<tr><td style='padding:12.0pt'>"
        "<p class='MsoNormal' align='center' style='margin-bottom:0cm;text-align:center;line-height:normal'>"
        "<b><span style='font-family:\"Segoe UI\",sans-serif;color:white;mso-themecolor:background1'>"
        "**THIS PROGRAM IS IN BETA TESTING DO NOT FORWARD**</span></b></p>"
        "<p class='MsoNormal' align='center' style='margin-bottom:0cm;text-align:center;line-height:normal'>"
        "<span style='font-size:11.0pt;font-family:\"Segoe UI\",sans-serif;color:white;mso-themecolor:background1'>"
        "Contact developer with any issues, queries, or suggestions: William.Manning@FederalGroup.com.au</span></p>"
        "</td></tr></table>"

        "</td></tr></table>"
        "</div>"
        "</td></tr></table>"
        "</div>"
        "<p class='MsoNormal' style='margin-bottom:0cm;line-height:normal'><span>&nbsp;</span></p>"
        "</div>"
        "</body></html>"
    )

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

    # Keep Gmail defaults, allow override via env if needed
    SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
    SMTP_STARTTLS = os.environ.get("SMTP_STARTTLS", "1").lower() in ("1", "true", "yes")
    SMTP_SSL = os.environ.get("SMTP_SSL", "0").lower() in ("1", "true", "yes")

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
        host=SMTP_HOST,
        port=SMTP_PORT,
        smtp_starttls=SMTP_STARTTLS,
        smtp_ssl=SMTP_SSL,
    )

    # IMPORTANT: pass the HTML string directly (NOT a list), so yagmail
    # doesn’t inject <br> between lines (which can break Outlook conditionals).
    yag.send(
        to=to_list,
        subject=subject,
        contents=body_html,  # HTML string
        attachments=files,
    )

    update_sent_log(files)

    print(f"✅ Email sent to {EMAIL_TO} with {len(files)} file(s), {total_hits} match(es).")


if __name__ == "__main__":
    main()
