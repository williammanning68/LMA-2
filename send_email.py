#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import smtplib
import ssl
import subprocess
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders
from pathlib import Path
from datetime import datetime, timezone

# -----------------------------------------------------------------------------
# Config / constants
# -----------------------------------------------------------------------------

LOG_FILE = Path("sent.log")

EMAIL_WIDTH = 640  # fixed width for the email body; used by VML wrapper

C = {
    "gold": "#C5A572",      # --federal-gold
    "navy": "#4A5A6A",      # --federal-navy
    "dark": "#475560",      # --federal-dark
    "light": "#ECF0F1",     # --federal-light
    "accent": "#D4AF37",    # --federal-accent
    "border": "#E4E9EE"     # neutral border for boxes
}

# -----------------------------------------------------------------------------
# Helpers: keywords
# -----------------------------------------------------------------------------

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

def _kw_hit(text: str, kw: str):
    """Word boundary for single words; tolerant substring for phrases (case-insensitive)."""
    if " " in kw:
        return re.search(re.escape(kw), text, re.IGNORECASE)
    return re.search(rf"\b{re.escape(kw)}\b", text, re.IGNORECASE)

def _sort_keywords_longest_first(keywords):
    """Return keywords sorted by length desc to prefer phrase highlighting first."""
    return sorted(keywords, key=lambda s: len(s), reverse=True)

def _bold_keywords(text: str, keywords):
    """Bold matched keywords in HTML (no links). Handles words & phrases, case-insensitive."""
    out = text
    for kw in _sort_keywords_longest_first(keywords):
        if not kw:
            continue
        # For phrases, simple case-insensitive replace using regex
        if " " in kw:
            pattern = re.compile(re.escape(kw), re.IGNORECASE)
        else:
            pattern = re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)
        out = pattern.sub(lambda m: f"<b>{m.group(0)}</b>", out)
    return out

# -----------------------------------------------------------------------------
# Speaker-aware segmentation tuned for Tas Hansard
# -----------------------------------------------------------------------------

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

TIME_STAMP_RE = re.compile(r"^\[\d{1,2}\.\d{2}\s*(a|p)\.m\.\]$", re.IGNORECASE)
UPPER_HEADING_RE = re.compile(r"^[A-Z][A-Z\s’'—\-&,;:.()]+$")  # e.g., MOTION, ADJOURNMENT
INTERJECTION_RE = re.compile(r"^(Members interjecting\.|The House suspended .+)$", re.IGNORECASE)

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

def _speaker_from_prior_lines(lines, start_idx):
    """Scan backward from start_idx to find the most recent speaker header; return display name or None."""
    for i in range(start_idx, -1, -1):
        m = SPEAKER_HEADER_RE.match(lines[i].strip())
        if m:
            title = (m.group("title") or "").strip()
            name = (m.group("name") or m.group("name_only") or "").strip()
            spk = " ".join(x for x in (title, name) if x).strip()
            if spk:
                return spk
    return None

def _segment_utterances_with_lines(text: str):
    """
    Yield tuples: (speaker, utterance_text, utter_lines) where
      utter_lines = list of (line_no_1based, raw_line).
    Speaker persists until a new header appears. Skip timestamps and headings.
    """
    current_speaker = None
    buff = []
    buff_line_nums = []
    all_lines = text.splitlines()

    def flush():
        nonlocal buff, buff_line_nums, current_speaker
        body = "\n".join(buff).strip()
        if body:
            yield (current_speaker, body, list(buff_line_nums))
        buff = []
        buff_line_nums = []

    for idx, raw_line in enumerate(all_lines):
        line_no = idx + 1
        line = raw_line.strip()

        # boundaries to skip (do not attach to any speaker)
        if not line or TIME_STAMP_RE.match(line) or UPPER_HEADING_RE.match(line) or INTERJECTION_RE.match(line):
            if line == "" and buff:
                # keep paragraph break inside same speaker
                buff.append("")
                buff_line_nums.append(line_no)
            else:
                # flush any accumulated paragraph, then skip this line
                yield from flush()
            continue

        m = SPEAKER_HEADER_RE.match(line)
        if m:
            # header line - start a new utterance
            yield from flush()
            title = (m.group("title") or "").strip()
            name = (m.group("name") or m.group("name_only") or "").strip()
            current_speaker = " ".join(x for x in (title, name) if x).strip()
            continue

        # normal text line
        buff.append(raw_line.rstrip())
        buff_line_nums.append(line_no)

    # final flush
    yield from flush()

# -----------------------------------------------------------------------------
# Optional LLM attribution (local Ollama)
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# Match extraction with sentence windows, dedupe, line numbers
# -----------------------------------------------------------------------------

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

def _sentences(s: str):
    parts = [p.strip() for p in SENT_SPLIT_RE.split(s.strip()) if p.strip()]
    return parts

def _window_for_index(idx: int, n: int):
    """Per rules: first sentence -> next two; otherwise one before + one after."""
    if idx <= 0:
        return (0, min(2, n - 1))  # inclusive indices
    return (max(0, idx - 1), min(n - 1, idx + 1))

def _merge_windows_by_gap(base_window, new_window, idx_gap, allow_if_within=4, require_gap_gt=2):
    """
    Expand base_window to also cover new_window only if:
      - sentence index gap <= allow_if_within, AND
      - gap > require_gap_gt (avoid tiny overlaps that cause duplication)
    Windows are inclusive (start,end).
    """
    if idx_gap <= allow_if_within and idx_gap > require_gap_gt:
        return (min(base_window[0], new_window[0]), max(base_window[1], new_window[1]))
    return base_window

def _chamber_from_filename(name: str) -> str:
    if "House_of_Assembly" in name:
        return "House of Assembly"
    if "Legislative_Council" in name or "Legislative Council" in name:
        return "Legislative Council"
    return "Unknown"

def extract_matches(text: str, keywords):
    """
    Returns list of dicts:
      {
        "keyword": kw,
        "speaker": speaker or None,
        "excerpt_html": "...",
        "line_no": int,                 # first line where any keyword in group occurs
        "file_chamber": "House of Assembly" | "Legislative Council" | "Unknown",
      }
    """
    use_llm = os.environ.get("ATTRIB_WITH_LLM", "").lower() in ("1", "true", "yes")
    llm_timeout = int(os.environ.get("ATTRIB_LLM_TIMEOUT", "30"))

    candidates = _known_speakers_from(text)
    norm_candidates = { _canonicalize(c): c for c in candidates }

    results = []
    all_lines = text.splitlines()

    for speaker, utt, utter_lines in _segment_utterances_with_lines(text):
        if not utt.strip():
            continue

        # Per-utterance: compute sentence list and map sentence index to line numbers
        sents = _sentences(utt)
        if not sents:
            continue

        # Find which sentences contain any keywords; collect (sent_idx, kw, line_no)
        hit_sent_indices = []
        hit_details = []  # each: (sent_idx, kw, first_line_no_for_kw)
        # map sentence to first line containing the keyword (approx)
        # We scan the utter_lines in order to find first matching line for the sentence keyword
        for i, sent in enumerate(sents):
            for kw in keywords:
                if _kw_hit(sent, kw):
                    # find a line number within this utterance that contains the kw
                    line_no = None
                    for ln, raw in utter_lines:
                        if _kw_hit(raw, kw):
                            line_no = ln
                            break
                    if line_no is None and utter_lines:
                        line_no = utter_lines[0][0]
                    hit_sent_indices.append(i)
                    hit_details.append((i, kw, line_no))
                    # NOTE: keep multiple keywords in same sentence; we'll handle dedupe later

        if not hit_details:
            continue

        # Build a combined window per the rules
        hit_sent_indices = sorted(set(hit_sent_indices))
        n = len(sents)
        # Start from first hit
        base_i = hit_sent_indices[0]
        window = _window_for_index(base_i, n)

        for j in hit_sent_indices[1:]:
            new_window = _window_for_index(j, n)
            window = _merge_windows_by_gap(window, new_window, j - base_i, allow_if_within=4, require_gap_gt=2)
            base_i = j  # advance for next gap computation

        start_idx, end_idx = window

        # Compose excerpt sentences in window; bold keywords
        excerpt_text = " ".join(sents[start_idx:end_idx+1]).strip()
        excerpt_html = _bold_keywords(excerpt_text, keywords)

        # Determine the representative keyword & first line in the window
        # Pick the earliest line number among hits that fall inside [start_idx, end_idx]
        in_window_hits = [h for h in hit_details if start_idx <= h[0] <= end_idx]
        if in_window_hits:
            first_line = min(h[2] for h in in_window_hits if h[2] is not None)
            rep_kw = in_window_hits[0][1]
        else:
            first_line = utter_lines[0][0] if utter_lines else 1
            rep_kw = hit_details[0][1]

        linked = speaker
        if not linked:
            # Try to infer by scanning backward for last seen header
            start_line_idx = utter_lines[0][0] - 2 if utter_lines else 0
            back_guess = _speaker_from_prior_lines(all_lines, start_line_idx)
            if back_guess:
                linked = back_guess

        if (not linked) and use_llm:
            # fallback: ask LLM
            guess = _llm_guess_speaker(excerpt_text, context=utt, candidates=candidates, timeout=llm_timeout)
            if guess and _canonicalize(guess) in norm_candidates:
                linked = norm_candidates[_canonicalize(guess)]
            else:
                linked = None

        results.append({
            "keyword": rep_kw,
            "speaker": linked,
            "excerpt_html": excerpt_html,
            "line_no": first_line,
        })

    return results

# -----------------------------------------------------------------------------
# Digest + HTML building with bulletproof rounded corners (VML)
# -----------------------------------------------------------------------------

def _px_to_pt(px: int) -> int:
    """Convert pixels to points for VML text insets (~0.75pt per px at 96dpi)."""
    return max(0, int(round(px * 0.75)))

def vml_rounded_container(inner_html: str,
                          width_px: int = EMAIL_WIDTH,
                          bg: str = "#FFFFFF",
                          border: str = C["border"],
                          radius_px: int = 12,
                          pad_px: int = 20) -> str:
    """
    Bulletproof rounded box:
      • Modern clients use the DIV (border-radius).
      • Outlook desktop (Word engine) uses the VML <v:roundrect> fallback.
    """
    arc_percent = max(1, min(50, int(round(radius_px * 100.0 / max(1, width_px)))))  # corner %
    inset_pt = _px_to_pt(pad_px)
    return f"""
    <!--[if mso]>
    <v:roundrect xmlns:v="urn:schemas-microsoft-com:vml"
        arcsize="{arc_percent}%"
        fillcolor="{bg}"
        strokecolor="{border}"
        strokeweight="1px"
        style="width:{width_px}px; mso-wrap-style:none; mso-position-horizontal:center;">
      <v:textbox inset="{inset_pt}pt,{inset_pt}pt,{inset_pt}pt,{inset_pt}pt" style="mso-fit-shape-to-text:t">
    <![endif]-->
    <div style="background:{bg}; border:1px solid {border}; border-radius:{radius_px}px; padding:{pad_px}px;">
      {inner_html}
    </div>
    <!--[if mso]></v:textbox></v:roundrect><![endif]-->
    """

def parse_date_from_filename(filename: str):
    """Extract datetime from Hansard filename."""
    m = re.search(r"(\d{1,2} \w+ \d{4})", filename)
    if m:
        try:
            return datetime.strptime(m.group(1), "%d %B %Y")
        except ValueError:
            return datetime.min
    return datetime.min

def _counts_table_html(counts_by_keyword):
    # Build a simple table with headers
    rows = []
    # sort keywords alphabetically for stable order
    for kw in sorted(counts_by_keyword.keys(), key=str.lower):
        ha = counts_by_keyword[kw].get("House of Assembly", 0)
        lc = counts_by_keyword[kw].get("Legislative Council", 0)
        total = ha + lc
        rows.append(f"""
          <tr>
            <td style="padding:10px 12px; border-bottom:1px solid {C['border']};">{kw}</td>
            <td style="padding:10px 12px; border-bottom:1px solid {C['border']}; text-align:center;">{ha}</td>
            <td style="padding:10px 12px; border-bottom:1px solid {C['border']}; text-align:center;">{lc}</td>
            <td style="padding:10px 12px; border-bottom:1px solid {C['border']}; text-align:center; font-weight:600;">{total}</td>
          </tr>
        """)
    table = f"""
      <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="border-collapse:collapse; background:#FFFFFF;">
        <thead>
          <tr style="background:{C['light']};">
            <th align="left" style="padding:12px; border-bottom:2px solid {C['border']}; font-weight:700; color:{C['dark']};">Keyword</th>
            <th align="center" style="padding:12px; border-bottom:2px solid {C['border']}; font-weight:700; color:{C['dark']};">House of Assembly</th>
            <th align="center" style="padding:12px; border-bottom:2px solid {C['border']}; font-weight:700; color:{C['dark']};">Legislative Council</th>
            <th align="center" style="padding:12px; border-bottom:2px solid {C['border']}; font-weight:700; color:{C['dark']};">Total</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows) if rows else f'<tr><td colspan="4" style="padding:12px;">No keywords triggered.</td></tr>'}
        </tbody>
      </table>
    """
    return table

def build_digest_html(files, keywords):
    """Build the full HTML email; return (html, total_matches, counts_by_keyword)."""
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Counts structure: { kw: {"House of Assembly": n, "Legislative Council": n} }
    counts_by_keyword = {kw: {"House of Assembly": 0, "Legislative Council": 0} for kw in keywords}
    total_matches = 0

    # Collect per-file matches (ordered by filename date, then by line number)
    file_sections = []

    for f in sorted(files, key=lambda x: parse_date_from_filename(Path(x).name)):
        name = Path(f).name
        chamber = _chamber_from_filename(name)
        text = Path(f).read_text(encoding="utf-8", errors="ignore")

        matches = extract_matches(text, keywords)

        if not matches:
            continue

        # sort by first occurrence line number
        matches.sort(key=lambda m: (m["line_no"], (m["speaker"] or "ZZZ")))

        # Update counts
        for m in matches:
            kw = m["keyword"]
            if chamber in ("House of Assembly", "Legislative Council"):
                counts_by_keyword.setdefault(kw, {"House of Assembly": 0, "Legislative Council": 0})
                counts_by_keyword[kw][chamber] = counts_by_keyword[kw].get(chamber, 0) + 1

        total_matches += len(matches)

        # Build per-file heading
        file_heading_inner = f"""
          <h3 style="margin:0 0 4px 0; font-family:Arial,Helvetica,sans-serif; font-size:18px; color:{C['dark']};">
            {name}
          </h3>
          <div style="font-family:Arial,Helvetica,sans-serif; font-size:12px; color:{C['navy']};">
            {len(matches)} match(es)
          </div>
        """
        file_heading_html = vml_rounded_container(
            inner_html=file_heading_inner,
            width_px=EMAIL_WIDTH,
            bg="#FFFFFF",
            border=C["border"],
            radius_px=12,
            pad_px=16
        )

        # Build each match card
        match_cards = []
        for i, m in enumerate(matches, 1):
            spk = m["speaker"] or "UNKNOWN"
            ln = m["line_no"]
            title_row = f"""
              <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">
                <tr>
                  <td style="font-family:Arial,Helvetica,sans-serif; font-size:14px; color:{C['dark']}; font-weight:700;">
                    Match #{i} ({spk}) — line {ln}
                  </td>
                </tr>
              </table>
            """
            body_row = f"""
              <div style="margin-top:8px; font-family:Georgia, 'Times New Roman', serif; font-size:15px; line-height:1.5; color:#222;">
                {m["excerpt_html"]}
              </div>
            """
            match_inner = title_row + body_row
            card = vml_rounded_container(
                inner_html=match_inner,
                width_px=EMAIL_WIDTH,
                bg="#FFFFFF",
                border=C["border"],
                radius_px=12,
                pad_px=20
            )
            # Spacer table (Outlook-safe)
            card = f"""
            <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">
              <tr><td height="12" style="line-height:12px;font-size:12px;">&nbsp;</td></tr>
              <tr><td>{card}</td></tr>
            </table>
            """
            match_cards.append(card)

        file_section_html = f"""
          <!-- File section -->
          {file_heading_html}
          {''.join(match_cards)}
          <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">
            <tr><td height="16" style="line-height:16px;font-size:16px;">&nbsp;</td></tr>
          </table>
        """
        file_sections.append(file_section_html)

    # Header block with title + small summary tiles
    # You can tailor the tiles; here we show files count, distinct keywords, total matches, runtime
    tiles = []
    # Compute a few quick tiles
    files_count = sum(1 for _ in file_sections)
    distinct_kw = sum(1 for kw, c in counts_by_keyword.items() if (c.get("House of Assembly", 0) + c.get("Legislative Council", 0)) > 0)
    tiles_data = [
        ("Files", files_count),
        ("Keywords", distinct_kw),
        ("Matches", total_matches),
        ("Runtime", "Now"),
    ]
    # Lay out 4 tiles with fixed widths
    tile_w = 148  # (640 - 3*16)/4
    tile_cells = []
    for label, value in tiles_data:
        tile_inner = f"""
          <div style="font-family:Arial,Helvetica,sans-serif; font-size:28px; font-weight:700; color:{C['gold']};">{value}</div>
          <div style="font-family:Arial,Helvetica,sans-serif; font-size:12px; color:#FFFFFF; opacity:.85; margin-top:2px;">{label}</div>
        """
        tile_html = vml_rounded_container(
            inner_html=tile_inner,
            width_px=tile_w,
            bg="rgba(255,255,255,0.08)".replace("rgba", "rgb"),  # Outlook ignores alpha; fine as light box
            border=C["navy"],
            radius_px=10,
            pad_px=14
        )
        tile_cells.append(f'<td width="{tile_w}" valign="top">{tile_html}</td>')

    tiles_row = f"""
      <table role="presentation" width="{EMAIL_WIDTH}" cellpadding="0" cellspacing="0" border="0">
        <tr>
          {tile_cells[0]}
          <td width="16">&nbsp;</td>
          {tile_cells[1]}
          <td width="16">&nbsp;</td>
          {tile_cells[2]}
          <td width="16">&nbsp;</td>
          {tile_cells[3]}
        </tr>
      </table>
    """

    header_inner = f"""
      <div style="font-family:Arial,Helvetica,sans-serif; color:#FFFFFF;">
        <h1 style="margin:0 0 6px 0; font-size:28px; font-weight:700;">Hansard Keyword Digest</h1>
        <div style="font-size:13px; opacity:.9;">Comprehensive parliamentary transcript analysis — {now_utc}</div>
        <div style="height:16px; line-height:16px; font-size:16px;">&nbsp;</div>
        {tiles_row}
      </div>
    """

    header_html = vml_rounded_container(
        inner_html=header_inner,
        width_px=EMAIL_WIDTH,
        bg=C["navy"],
        border=C["navy"],
        radius_px=12,
        pad_px=20
    )

    # Keywords being tracked block
    if keywords:
        kw_badges = " ".join(
            f'<span style="display:inline-block; padding:6px 8px; border:1px solid {C["gold"]}; border-radius:8px; margin:0 6px 6px 0; color:{C["dark"]}; background:#fff4de;">{k}</span>'
            for k in keywords
        )
    else:
        kw_badges = '<span style="color:#555;">(none)</span>'

    keywords_inner_html = f"""
      <div style="font-family:Arial,Helvetica,sans-serif; font-size:16px; color:{C['dark']}; font-weight:700; margin-bottom:10px;">
        Keywords Being Tracked
      </div>
      <div>{kw_badges}</div>
    """
    keywords_card = vml_rounded_container(
        inner_html=keywords_inner_html,
        width_px=EMAIL_WIDTH,
        bg=C["light"],
        border=C["border"],
        radius_px=12,
        pad_px=20
    )

    # Summary table block
    summary_title = f"""
      <div style="font-family:Arial,Helvetica,sans-serif; font-size:16px; color:{C['dark']}; font-weight:700; margin-bottom:10px;">
        Summary by Chamber
      </div>
    """
    summary_table_html = _counts_table_html(counts_by_keyword)
    summary_card = vml_rounded_container(
        inner_html=summary_title + summary_table_html,
        width_px=EMAIL_WIDTH,
        bg=C["light"],
        border=C["border"],
        radius_px=12,
        pad_px=20
    )

    # Combine file sections
    files_section_html = "\n".join(file_sections) if file_sections else vml_rounded_container(
        inner_html="""
          <div style="font-family:Arial,Helvetica,sans-serif; font-size:14px; color:#333;">
            No keyword matches found in new transcripts.
          </div>
        """,
        width_px=EMAIL_WIDTH,
        bg="#FFFFFF",
        border=C["border"],
        radius_px=12,
        pad_px=16
    )

    # Outer wrapper (table-based, Outlook-safe)
    full_html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta http-equiv="x-ua-compatible" content="ie=edge">
  <meta charset="UTF-8">
  <!--[if mso]>
    <style>
      table, td, div, p, a {{ font-family: Arial, sans-serif !important; }}
    </style>
  <![endif]-->
  <title>Hansard Keyword Digest</title>
</head>
<body style="margin:0; padding:0; background:{C['light']};">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background:{C['light']};">
    <tr>
      <td align="center">
        <table role="presentation" width="{EMAIL_WIDTH}" cellpadding="0" cellspacing="0" border="0" style="width:{EMAIL_WIDTH}px;">
          <tr><td height="16" style="line-height:16px;font-size:16px;">&nbsp;</td></tr>
          <tr><td>{header_html}</td></tr>
          <tr><td height="16" style="line-height:16px;font-size:16px;">&nbsp;</td></tr>
          <tr><td>{keywords_card}</td></tr>
          <tr><td height="16" style="line-height:16px;font-size:16px;">&nbsp;</td></tr>
          <tr><td>{summary_card}</td></tr>
          <tr><td height="16" style="line-height:16px;font-size:16px;">&nbsp;</td></tr>
          <tr><td>{files_section_html}</td></tr>
          <tr><td height="24" style="line-height:24px;font-size:24px;">&nbsp;</td></tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>
""".strip()

    return full_html, total_matches, counts_by_keyword

# -----------------------------------------------------------------------------
# Email sending (SMTP, no yagmail)
# -----------------------------------------------------------------------------

def _as_plain_text(html: str) -> str:
    """Very simple HTML to text fallback."""
    # Replace <br> with newline
    text = re.sub(r"(?i)<br\s*/?>", "\n", html)
    # Add newlines before/after block tags
    text = re.sub(r"(?i)</(div|p|h\d|tr|table)>", r"\n", text)
    # Strip tags
    text = re.sub(r"<[^>]+>", "", text)
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()

def load_sent_log():
    """Return set of transcript filenames that have already been emailed."""
    if LOG_FILE.exists():
        return {line.strip() for line in LOG_FILE.read_text(encoding="utf-8").splitlines() if line.strip()}
    return set()

def update_sent_log(files):
    """Append newly emailed filenames to the log."""
    with LOG_FILE.open("a", encoding="utf-8") as f:
        for file in files:
            f.write(f"{Path(file).name}\n")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

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

    body_html, total_hits, _counts = build_digest_html(files, keywords)

    subject = f"Hansard keyword digest — {datetime.now(timezone.utc).strftime('%d %b %Y')}"
    to_list = [addr.strip() for addr in re.split(r"[,\s]+", EMAIL_TO) if addr.strip()]

    # Build MIME message: multipart/mixed -> (multipart/alternative -> plain + html) + attachments
    msg = MIMEMultipart("mixed")
    msg["From"] = EMAIL_USER
    msg["To"] = ", ".join(to_list)
    msg["Subject"] = subject

    alt = MIMEMultipart("alternative")
    msg.attach(alt)

    plain_fallback = _as_plain_text(body_html)
    alt.attach(MIMEText(plain_fallback, "plain", "utf-8"))
    alt.attach(MIMEText(body_html, "html", "utf-8"))

    # Attach transcripts
    for path in files:
        try:
            with open(path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f'attachment; filename="{Path(path).name}"')
            msg.attach(part)
        except Exception as e:
            print(f"Warning: failed to attach {path}: {e}")

    # Send via Gmail SMTP (STARTTLS)
    context = ssl.create_default_context()
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.ehlo()
        server.starttls(context=context)
        server.ehlo()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.sendmail(EMAIL_USER, to_list, msg.as_string())

    update_sent_log(files)
    print(f"✅ Email sent to {EMAIL_TO} with {len(files)} file(s), {total_hits} match(es).")

if __name__ == "__main__":
    main()
