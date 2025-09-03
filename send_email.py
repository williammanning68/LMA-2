# send_email.py
import os
import re
import glob
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

from bisect import bisect_right
from pathlib import Path
from datetime import datetime, UTC
from zoneinfo import ZoneInfo
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
    Visual-only change to match your Word look: silver highlight + bold.
    (Extraction method is unchanged.)
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


# ---------------------- TEMPLATE & INJECTION ---------------------------------

# Fallback HTML that mirrors your Word/Outlook layout:
# - Hero: dark slate banner "Hansard Monitor – BETA Version 18.3" + "Program Run: [DATE]"
# - Grey section with "Detection Match by Chamber" and a gold underline
# - Summary table header row (Keyword / House of Assembly / Legislative Council / Total)
# - Per-file blocks with a gold left border and silver keyword highlights
# - Dark BETA footer with contact line
#
# NOTE: If you export your .docx as "Web Page, Filtered (*.htm; *.html)" and drop it next to this script
# (e.g., template.html), the loader below will send it verbatim instead of this fallback string.
TEMPLATE_FALLBACK_HTML = """\
<!DOCTYPE html>
<html xmlns:v="urn:schemas-microsoft-com:vml" xmlns:o="urn:schemas-microsoft-com:office:office">
<head><meta http-equiv="Content-Type" content="text/html; charset=utf-8"></head>
<body lang="EN-AU" link="#467886" vlink="#96607D" style="word-wrap:break-word">
<div align="center">
<table class="MsoNormalTable" border="0" cellspacing="0" cellpadding="0" width="700"
       style="width:525.0pt;mso-cellspacing:0cm;background:white;">
<tr><td style="padding:0">

  <!-- HERO -->
  <table class="MsoNormalTable" border="0" cellspacing="0" cellpadding="0" width="100%"
         style="width:100.0%;background:#475560;">
    <tr>
      <td style="padding:18.0pt 21.0pt">
        <p class="MsoNormal" align="center" style="margin:0;text-align:center;line-height:normal">
          <b><span style="font-size:28.0pt;font-family:'Segoe UI',sans-serif;color:white">
            Hansard Monitor – BETA Version 18.3
          </span></b>
        </p>
        <p class="MsoNormal" align="center" style="margin:0;text-align:center;line-height:normal">
          <span style="font-size:16.0pt;font-family:'Segoe UI',sans-serif;color:white">
            Program Run: [DATE]
          </span>
        </p>
      </td>
    </tr>
  </table>

  <!-- GREY WRAP -->
  <table class="MsoNormalTable" border="0" cellspacing="0" cellpadding="0" width="100%"
         style="width:100.0%;background:#ECF0F1;">
    <tr><td style="padding:12.0pt">

      <!-- SUMMARY WHITE BOX -->
      <table class="MsoNormalTable" border="1" cellspacing="0" cellpadding="0" width="100%"
             style="width:100.0%;background:white;border:solid #D8DCE0 1.0pt;">
        <tr>
          <td style="border:none;border-bottom:solid #C5A572 2.25pt;padding:12.0pt 13.5pt">
            <p class="MsoNormal" align="center" style="margin:0;text-align:center;line-height:normal">
              <b><span style="font-size:16.0pt;font-family:'Segoe UI',sans-serif;color:black">
                Detection Match by Chamber
              </span></b>
            </p>
          </td>
        </tr>
        <tr>
          <td style="border:none;padding:9.0pt 12.0pt">
            <table class="MsoNormalTable" border="1" cellspacing="0" cellpadding="0" width="100%"
                   style="width:100.0%;background:white;border:none;border-collapse:collapse;">
              <tr>
                <td width="28%" style="width:28.12%;padding:9.0pt 7.5pt;background:#4A5A6A;">
                  <p class="MsoNormal" align="center" style="margin:0;text-align:center">
                    <b><span style="font-family:'Segoe UI',sans-serif;color:white">Keyword</span></b>
                  </p>
                </td>
                <td width="28%" style="width:28.12%;padding:9.0pt 7.5pt;background:#4A5A6A;">
                  <p class="MsoNormal" align="center" style="margin:0;text-align:center">
                    <b><span style="font-family:'Segoe UI',sans-serif;color:white">House of Assembly</span></b>
                  </p>
                </td>
                <td width="28%" style="width:28.14%;padding:9.0pt 7.5pt;background:#4A5A6A;">
                  <p class="MsoNormal" align="center" style="margin:0;text-align:center">
                    <b><span style="font-family:'Segoe UI',sans-serif;color:white">Legislative Council</span></b>
                  </p>
                </td>
                <td width="16%" style="width:15.62%;padding:9.0pt 7.5pt;background:#4A5A6A;">
                  <p class="MsoNormal" align="center" style="margin:0;text-align:center">
                    <b><span style="font-family:'Segoe UI',sans-serif;color:white">Total</span></b>
                  </p>
                </td>
              </tr>
              <!-- [SUMMARY_ROWS_ANCHOR] : rows inserted here -->
            </table>
          </td>
        </tr>
      </table>

      <!-- [FILE_BLOCKS_ANCHOR] : file blocks inserted here -->

    </td></tr>
  </table>

  <!-- FOOTER -->
  <table class="MsoNormalTable" border="0" cellspacing="0" cellpadding="0" width="100%"
         style="width:100.0%;background:#4A5A6A;">
    <tr>
      <td style="padding:12.0pt">
        <p class="MsoNormal" align="center" style="margin:0;text-align:center;line-height:normal">
          <b><span style="font-family:'Segoe UI',sans-serif;color:white">**THIS PROGRAM IS IN BETA TESTING – DO NOT FORWARD**</span></b>
        </p>
        <p class="MsoNormal" align="center" style="margin:0;text-align:center;line-height:normal">
          <span style="font-size:11.0pt;font-family:'Segoe UI',sans-serif;color:white">
            Contact developer with any issues, queries, or suggestions: William.Manning@FederalGroup.com.au
          </span>
        </p>
      </td>
    </tr>
  </table>

</td></tr></table>
</div>
</body>
</html>
"""

TEMPLATE_CANDIDATES = [
    # If you export your .docx as "Web Page, Filtered" and place it here, it will be used verbatim:
    "Hansard Monitor - Email Format - Version 2 - 03092025.html",
    "Hansard Monitor - Email Format - Version 2 - 03092025.htm",
    # A previous text/HTML dump:
    "Hansard Monitor - Email Format - Version 2 - 03092025.txt",
]


def _load_template_html() -> str:
    """
    Load a verbatim HTML file if present; otherwise fall back to the inline template
    that mirrors the docx/pdf visuals for Outlook/Word.
    """
    for p in TEMPLATE_CANDIDATES:
        if Path(p).exists():
            return Path(p).read_text(encoding="utf-8", errors="ignore")
    return TEMPLATE_FALLBACK_HTML


def _inject_program_run_date(html: str, dt_hobart: datetime) -> str:
    return html.replace("Program Run: [DATE]", f"Program Run: {dt_hobart.strftime('%d %B %Y %I:%M %p %Z')}")


def _summary_row_html(kw: str, hoa: int, lc: int, tot: int) -> str:
    return (
        "<tr>"
        f"<td width='28%' style='width:28.12%;border-left:solid #D8DCE0 1.0pt;border-right:none;border-top:none;"
        f"border-bottom:solid #D8DCE0 1.0pt;padding:7.5pt 9.0pt 7.5pt 9.0pt'>"
        f"<p class='MsoNormal' style='margin:0;line-height:normal'><span style='font-family:\"Segoe UI\",sans-serif;color:#475560'>{_html_escape(kw)}</span></p></td>"
        f"<td width='28%' align='right' style='width:28.12%;border-top:none;border-bottom:solid #D8DCE0 1.0pt;border-left:none;border-right:none;padding:7.5pt 9.0pt'>"
        f"<p class='MsoNormal' style='margin:0;line-height:normal'><span style='font-family:\"Segoe UI\",sans-serif;color:#475560'>{hoa}</span></p></td>"
        f"<td width='28%' align='right' style='width:28.14%;border-top:none;border-bottom:solid #D8DCE0 1.0pt;border-left:none;border-right:none;padding:7.5pt 9.0pt'>"
        f"<p class='MsoNormal' style='margin:0;line-height:normal'><span style='font-family:\"Segoe UI\",sans-serif;color:#475560'>{lc}</span></p></td>"
        f"<td width='16%' align='right' style='width:15.62%;border-right:solid #D8DCE0 1.0pt;border-left:none;border-top:none;border-bottom:solid #D8DCE0 1.0pt;padding:7.5pt 9.0pt'>"
        f"<p class='MsoNormal' style='margin:0;line-height:normal'><span style='font-family:\"Segoe UI\",sans-serif;color:#475560'>{tot}</span></p></td>"
        "</tr>"
    )


def _match_card_html(speaker: str, lines_str: str, excerpt_html: str) -> str:
    return (
        "<tr><td style='border:none;padding:9.0pt 12.0pt'>"
        "<table class='MsoNormalTable' border='0' cellspacing='0' cellpadding='0' width='100%' "
        "style='width:100.0%;background:white;border-collapse:collapse;border:solid #D8DCE0 1.0pt;'>"
        "<tr>"
        "<td style='border-left:solid #C5A572 3.0pt;border-top:none;border-bottom:none;border-right:none;padding:7.5pt 9.0pt'>"
        f"<p class='MsoNormal' style='margin:0 0 6.0pt 0;line-height:normal'><b><span style='font-family:\"Segoe UI\",sans-serif;color:#475560'>{_html_escape(speaker)}</span></b>"
        f"<span style='font-family:\"Segoe UI\",sans-serif;color:#8795A1'> — {lines_str}</span></p>"
        f"<p class='MsoNormal' style='margin:0;line-height:22px'><span style='font-family:\"Segoe UI\",sans-serif;color:#475560'>{excerpt_html}</span></p>"
        "</td></tr></table>"
        "</td></tr>"
    )


def _file_block_html(filename: str, match_count: int, match_cards_html: str) -> str:
    return (
        "<table class='MsoNormalTable' border='1' cellspacing='0' cellpadding='0' width='100%' "
        "style='width:100.0%;background:white;border:solid #D8DCE0 1.0pt;'>"
        "<tr><td style='border:none;border-bottom:solid #C5A572 2.25pt;padding:12.0pt 13.5pt'>"
        f"<p class='MsoNormal' style='margin:0;line-height:normal'><b><span style='font-family:\"Segoe UI\",sans-serif;color:black'>{_html_escape(filename)}</span></b>"
        f"<span style='font-family:\"Segoe UI\",sans-serif;color:#8795A1'> — {match_count} match(es)</span></p>"
        "</td></tr>"
        f"{match_cards_html}"
        "</table>"
    )


def build_digest_html(files, keywords):
    """
    Build final HTML by:
      1) Loading the template (verbatim HTML if supplied; otherwise fallback HTML that mirrors the docx/pdf),
      2) Replacing the Program Run date token,
      3) Injecting summary rows after the summary header,
      4) Appending per-file blocks below the summary.
    """
    hobart = ZoneInfo("Australia/Hobart")
    now_local = datetime.now(hobart)

    chambers = ["House of Assembly", "Legislative Council"]
    counts = {ch: {kw: 0 for kw in keywords} for ch in chambers}
    totals = {kw: 0 for kw in keywords}
    total_matches = 0

    # Per-file blocks
    file_blocks = []
    for f in sorted(files, key=lambda x: (parse_date_from_filename(Path(x).name), Path(x).name)):
        text = Path(f).read_text(encoding="utf-8", errors="ignore")
        chamber = parse_chamber_from_filename(Path(f).name)
        matches = extract_matches(text, keywords)

        match_cards = []
        for (kw_set, excerpt_html, speaker, line_list, win_start, _win_end) in matches:
            # counts
            for kw in kw_set:
                if chamber in counts:
                    counts[chamber][kw] += 1
                totals[kw] += 1

            first_line = min(line_list) if line_list else win_start
            lines_str = ("line " if len(line_list) <= 1 else "lines ") + (
                ", ".join(str(n) for n in sorted(set(line_list))) if line_list else str(first_line)
            )
            speaker_display = speaker if speaker and not _looks_suspicious(speaker) else "Unknown"
            match_cards.append(_match_card_html(speaker_display, lines_str, excerpt_html))

        total_matches += len(matches)
        file_blocks.append(_file_block_html(Path(f).name, len(matches), "".join(match_cards)))

    # Summary rows for the “Detection Match by Chamber” table
    summary_rows_html = "".join(
        _summary_row_html(kw, counts["House of Assembly"].get(kw, 0),
                          counts["Legislative Council"].get(kw, 0),
                          totals.get(kw, 0))
        for kw in keywords
    )

    # Load + inject into template
    html = _load_template_html()
    html = _inject_program_run_date(html, now_local)

    # Insert summary rows after header row (identified by the 4 header cells)
    header_regex = re.compile(
        r"(<tr>\s*"
        r"(?:.|\s)*?Keyword(?:.|\s)*?</tr>)",  # the header <tr> that contains 'Keyword' ... 'Total'
        re.IGNORECASE
    )
    def _after_header(m):
        return m.group(1) + summary_rows_html

    html = header_regex.sub(_after_header, html, count=1)

    # Insert file blocks at our anchor
    if "[FILE_BLOCKS_ANCHOR]" in html:
        html = html.replace("<!-- [FILE_BLOCKS_ANCHOR] : file blocks inserted here -->", "".join(file_blocks))
    else:
        # fallback: append to end
        html = html.replace("</body>", "".join(file_blocks) + "</body>")

    return html, total_matches, counts


def load_sent_log():
    if LOG_FILE.exists():
        return {line.strip() for line in LOG_FILE.read_text().splitlines() if line.strip()}
    return set()


def update_sent_log(files):
    with LOG_FILE.open("a", encoding="utf-8") as f:
        for file in files:
            f.write(f"{Path(file).name}\n")


# --- Send email (raw MIME to preserve formatting) ----------------------------

def main():
    EMAIL_USER = os.environ["EMAIL_USER"]
    EMAIL_PASS = os.environ["EMAIL_PASS"]
    EMAIL_TO = os.environ["EMAIL_TO"]

    # You can switch to Office 365 for best Outlook fidelity:
    SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")     # e.g., "smtp.office365.com"
    SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
    SMTP_STARTTLS = os.environ.get("SMTP_STARTTLS", "1").lower() in ("1", "true", "yes")

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

    # Build a mixed MIME container so we can attach the transcript files too
    msg = MIMEMultipart("mixed")
    msg["Subject"] = f"Hansard keyword digest — {datetime.now().strftime('%d %b %Y')}"
    msg["From"] = EMAIL_USER
    msg["To"] = EMAIL_TO

    # Alternative part (just HTML)
    alt = MIMEMultipart("alternative")
    msg.attach(alt)

    # HTML part (UTF-8)
    part_html = MIMEText(body_html, "html", "utf-8")
    alt.attach(part_html)

    # Attach transcripts
    for path in files:
        with open(path, "rb") as fh:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(fh.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{Path(path).name}"')
        msg.attach(part)

    # Send
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        if SMTP_STARTTLS:
            s.starttls()
        s.login(EMAIL_USER, EMAIL_PASS)
        s.sendmail(EMAIL_USER, [a.strip() for a in re.split(r"[,\s]+", EMAIL_TO) if a.strip()], msg.as_string())

    update_sent_log(files)
    print(f"✅ Email sent to {EMAIL_TO} with {len(files)} file(s), {total_hits} match(es).")


if __name__ == "__main__":
    main()
