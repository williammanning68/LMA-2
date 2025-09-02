import os
import re
import glob
from bisect import bisect_right
from pathlib import Path
from datetime import datetime, UTC

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


# --- Outlook-safe rounded card wrapper (VML) ---------------------------------

def vml_card(inner_html: str, bg="#FFFFFF", border="#E4E9EE", arc="6%", pad_pt=(14, 14, 14, 14), width=640) -> str:
    """
    Wraps a content block in a VML rounded-rect for Outlook, with a <div> fallback
    for other clients. Crucial bits:
      - xmlns:v at <html> level (added below in html_open)
      - mso-fit-shape-to-text:t so Outlook grows to content height
    """
    inset = ",".join(f"{p}pt" for p in pad_pt)
    return (
        f'<!--[if mso | IE]>'
        f'<v:roundrect xmlns:v="urn:schemas-microsoft-com:vml" '
        f'arcsize="{arc}" fillcolor="{bg}" strokecolor="{border}" strokeweight="1px" '
        f'style="width:{width}px;">'
        f'<v:textbox inset="{inset}" style="mso-fit-shape-to-text:t;word-wrap:break-word">'
        f'<![endif]-->'
        f'<div style="background:{bg};border:1px solid {border};border-radius:12px;padding:18px;">'
        f'{inner_html}'
        f'</div>'
        f'<!--[if mso | IE]></v:textbox></v:roundrect><![endif]-->'
    )


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


def build_digest_html(files, keywords):
    now_utc = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    chambers = ["House of Assembly", "Legislative Council"]
    counts = {ch: {kw: 0 for kw in keywords} for ch in chambers}
    totals = {kw: 0 for kw in keywords}

    doc_sections = []
    total_matches = 0

    # Build summary table data as before
    for f in sorted(files, key=lambda x: (parse_date_from_filename(Path(x).name), Path(x).name)):
        text = Path(f).read_text(encoding="utf-8", errors="ignore")
        chamber = parse_chamber_from_filename(Path(f).name)

        matches = extract_matches(text, keywords)
        if not matches:
            continue

        matches.sort(key=lambda item: min(item[3]) if item[3] else 10**9)
        total_matches += len(matches)

        # Document title card
        title_inner = (
            '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">'
            f'<tr><td style="font-size:18px;color:#475560;font-weight:700;">{_html_escape(Path(f).name)}</td></tr>'
            '</table>'
        )
        doc_html_parts = [vml_card(title_inner, bg="#FFFFFF", border="#E4E9EE", arc="6%", pad_pt=(10, 10, 10, 10))]

        # Each match card
        for i, (kw_set, excerpt_html, speaker, line_list, win_start, win_end) in enumerate(matches, 1):
            for kw in kw_set:
                if chamber in counts:
                    counts[chamber][kw] += 1
                totals[kw] += 1

            first_line = min(line_list) if line_list else win_start
            speaker_html = _html_escape(speaker) if speaker else "UNKNOWN"
            line_label = "line" if len(line_list) == 1 else "lines"
            lines_str = ", ".join(str(n) for n in sorted(set(line_list))) if line_list else str(first_line)

            match_inner = (
                f'<div style="color:#475560;font-size:14px;font-weight:700;">'
                f'Match #{i} ({speaker_html}) — {line_label} {lines_str}'
                f'</div>'
                f'<div style="margin-top:8px;font-family:Georgia,\'Times New Roman\',serif;'
                f'font-size:15px;line-height:1.55;color:#222;">{excerpt_html}</div>'
            )
            doc_html_parts.append(vml_card(match_inner, bg="#FFFFFF", border="#E4E9EE", arc="6%"))

            # spacer between match cards (table row is added later)
            doc_html_parts.append(
                '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">'
                '<tr><td height="12" style="line-height:12px;font-size:12px;">&nbsp;</td></tr>'
                '</table>'
            )

        # wrap doc block rows
        doc_sections.append(''.join([
            '<tr><td>',
            ''.join(doc_html_parts),
            '</td></tr>',
        ]))

    # Summary table HTML (unchanged content, improved styles)
    header_cols = "".join([
        "<th align=\"left\" style=\"padding:12px;border-bottom:2px solid #E4E9EE;color:#475560;font-size:13px;\">Keyword</th>",
        "<th align=\"center\" style=\"padding:12px;border-bottom:2px solid #E4E9EE;color:#475560;font-size:13px;\">House of Assembly</th>",
        "<th align=\"center\" style=\"padding:12px;border-bottom:2px solid #E4E9EE;color:#475560;font-size:13px;\">Legislative Council</th>",
        "<th align=\"center\" style=\"padding:12px;border-bottom:2px solid #E4E9EE;color:#475560;font-size:13px;\">Total</th>",
    ])
    row_html = []
    for kw in keywords:
        hoa = counts["House of Assembly"][kw] if "House of Assembly" in counts else 0
        lc  = counts["Legislative Council"][kw] if "Legislative Council" in counts else 0
        tot = totals[kw]
        row_html.append(
            f"<tr>"
            f"<td style='padding:10px 12px;border-bottom:1px solid #E4E9EE;font-size:14px;'>{_html_escape(kw)}</td>"
            f"<td align='center' style='padding:10px 12px;border-bottom:1px solid #E4E9EE;font-size:14px;'>{hoa}</td>"
            f"<td align='center' style='padding:10px 12px;border-bottom:1px solid #E4E9EE;font-size:14px;'>{lc}</td>"
            f"<td align='center' style='padding:10px 12px;border-bottom:1px solid #E4E9EE;font-size:14px;font-weight:700;'>{tot}</td>"
            f"</tr>"
        )
    summary_table = (
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
        f'style="border-collapse:collapse;background:#FFFFFF;">'
        f'  <thead><tr style="background:#ECF0F1;">{header_cols}</tr></thead>'
        f'  <tbody>{"".join(row_html)}</tbody>'
        f'</table>'
    )

    # Build the hero + keyword pills + summary using VML cards
    # hero
    hero_inner = (
        '<div style="color:#FFFFFF">'
        '<div style="font-size:22px;font-weight:700;margin:0 0 6px 0;">Hansard Keyword Digest</div>'
        f'<div style="font-size:13px;opacity:.9;">Comprehensive parliamentary transcript analysis — {now_utc}</div>'
        '</div>'
    )
    hero = vml_card(hero_inner, bg="#4A5A6A", border="#4A5A6A", arc="6%", pad_pt=(14, 14, 14, 14))

    # keyword pills
    pills = ''.join(
        f'<span style="display:inline-block;margin:0 6px 6px 0;'
        f'padding:6px 8px;border:1px solid #C5A572;border-radius:8px;'
        f'background:#fff4de;color:#475560;font-size:13px;">{_html_escape(kw)}</span>'
        for kw in keywords
    )
    kw_block = vml_card(
        f'<div style="font-size:16px;color:#475560;font-weight:700;margin-bottom:10px;">Keywords Being Tracked</div>'
        f'<div>{pills}</div>',
        bg="#ECF0F1", border="#E4E9EE", arc="6%"
    )

    # summary table block
    summary_block = vml_card(
        '<div style="font-size:16px;color:#475560;font-weight:700;margin-bottom:10px;">Summary by Chamber</div>'
        f'{summary_table}',
        bg="#ECF0F1", border="#E4E9EE", arc="6%"
    )

    # Minimal, client-safe HEAD + scaffolding
    head = (
        '<meta http-equiv="Content-Type" content="text/html; charset=utf-8">'
        '<!--[if mso]>'
        '<xml><o:OfficeDocumentSettings xmlns:o="urn:schemas-microsoft-com:office:office">'
        '<o:AllowPNG/>'
        '</o:OfficeDocumentSettings></xml>'
        '<style>*,body,table,td,div,p,a{font-family:Arial,Helvetica,sans-serif!important}</style>'
        '<![endif]-->'
        '<style>table{border-collapse:collapse}strong{font-weight:700}</style>'
    )
    html_open = (
        '<!doctype html>'
        '<html xmlns:v="urn:schemas-microsoft-com:vml" '
        'xmlns:o="urn:schemas-microsoft-com:office:office">'
        f'<head>{head}<title>Hansard Keyword Digest</title></head>'
        '<body style="margin:0;padding:0;background:#ECF0F1;">'
    )
    html_close = '</body></html>'

    # Outer table scaffold
    outer_open = (
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background:#ECF0F1;">'
        '<tr><td align="center">'
        '<table role="presentation" width="640" cellpadding="0" cellspacing="0" border="0" style="width:640px;">'
        '<tr><td height="16" style="line-height:16px;font-size:16px;">&nbsp;</td></tr>'
    )
    outer_close = (
        '<tr><td height="24" style="line-height:24px;font-size:24px;">&nbsp;</td></tr>'
        '</table></td></tr></table>'
    )

    # Assemble page
    parts = [
        html_open,
        outer_open,
        '<tr><td>', hero, '</td></tr>',
        '<tr><td height="16" style="line-height:16px;font-size:16px;">&nbsp;</td></tr>',
        '<tr><td>', kw_block, '</td></tr>',
        '<tr><td height="16" style="line-height:16px;font-size:16px;">&nbsp;</td></tr>',
        '<tr><td>', summary_block, '</td></tr>',
    ]

    if doc_sections:
        # spacer
        parts.append('<tr><td height="16" style="line-height:16px;font-size:16px;">&nbsp;</td></tr>')
        parts.extend(doc_sections)
    else:
        empty_block = vml_card('<div>No keyword matches found.</div>', bg="#FFFFFF", border="#E4E9EE", arc="6%")
        parts.extend([
            '<tr><td height="16" style="line-height:16px;font-size:16px;">&nbsp;</td></tr>',
            f'<tr><td>{empty_block}</td></tr>',
        ])

    parts.extend([outer_close, html_close])

    html = ''.join(parts)
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

    # Optional SMTP overrides (keeps Gmail defaults)
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

    # IMPORTANT: pass the HTML string directly (NOT as a single-item list),
    # so yagmail does NOT inject <br> between lines (which breaks MSO conditionals).
    yag.send(
        to=to_list,
        subject=subject,
        contents=body_html,   # HTML string, not a list
        attachments=files,
    )

    update_sent_log(files)

    print(f"✅ Email sent to {EMAIL_TO} with {len(files)} file(s), {total_hits} match(es).")


if __name__ == "__main__":
    main()
