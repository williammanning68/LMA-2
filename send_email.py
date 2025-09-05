#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
send_email.py
================

This script composes and sends the Hansard Monitor digest email based on
transcripts and keywords. It relies on an HTML template that defines
both the visible layout and comment-delimited sub-templates for
dynamically inserting keyword summary rows, per-file sections, and
individual match entries. The Python code here *never* defines any
visual HTML; instead, it extracts these sub-templates from the
template file and fills them with data derived from the transcripts.

Key features:

* Reads keywords from a simple text file (`keywords.txt`), ignoring
  blank lines and comments (lines starting with `#`).
* Scans all `.txt` transcripts under the `transcripts/` folder,
  counting keyword occurrences by chamber (House of Assembly vs
  Legislative Council) and collecting excerpted matches.
* Supports a clear, deterministic filename resolution for the
  template: it looks for `email_template.html` first and falls back
  to `email_template.htm`. If neither exists, it aborts with a
  helpful message.
* Extracts three sub-templates from the template file: `SR_TPL` for
  summary rows, `FS_TPL` for file sections, and `MR_TPL` for match
  rows. These must appear in the template wrapped in comments, as
  described in the companion HTML file. If any are missing, the
  script exits with a diagnostic error.
* Replaces two mount markers (`<!-- MOUNT: SUMMARY_ROWS -->` and
  `<!-- MOUNT: FILE_SECTIONS -->`) with the generated HTML. If
  either marker is missing, the script reports an error.
* Escapes all inserted data for safe HTML. Highlights the first
  keyword matched in each line using a light yellow background.
* Does not run Premailer: the template is already fully inline-styled
  for email clients, and further inlining would risk altering the
  carefully controlled layout.

Environment variables required:

    EMAIL_USER  - sender email address
    EMAIL_PASS  - password or app token for the sender
    EMAIL_TO    - comma-separated list of recipient addresses

Optionally override the House of Assembly prefix via the `HOUSE_PREFIX`
environment variable. This prefix is used to guess the chamber if the
filename does not explicitly mention "House of Assembly" or "Legislative
Council".
"""

import os
import re
import glob
import html
from datetime import datetime
from typing import Dict, List, Tuple

try:
    import yagmail  # type: ignore
except ImportError:
    yagmail = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

KEYWORDS_FILE = "keywords.txt"
TRANSCRIPTS_GLOB = "transcripts/*.txt"

# Template file preference: .html first, then .htm
TEMPLATE_CANDIDATES = ["email_template.html", "email_template.htm"]

# Mount markers inside the template. These comments must exist exactly
SUMMARY_MOUNT = "<!-- MOUNT: SUMMARY_ROWS -->"
FILES_MOUNT = "<!-- MOUNT: FILE_SECTIONS -->"

# Required sub-template identifiers (without the trailing comment lines)
REQUIRED_SUBTEMPLATES = ("SR_TPL", "FS_TPL", "MR_TPL")

# Chamber names. You can override HOUSE_PREFIX via environment variable
HOUSE_HOA = os.getenv("HOUSE_PREFIX", "House of Assembly")
HOUSE_LC = "Legislative Council"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def read_file_text(path: str) -> str:
    """Read a file with UTF-8 encoding, stripping a UTF-8 BOM if present."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    # Remove BOM if present
    return content.lstrip("\ufeff")


def pick_template_file() -> str:
    """Return the first existing template file from the candidates list."""
    for fname in TEMPLATE_CANDIDATES:
        if os.path.exists(fname):
            return fname
    raise FileNotFoundError(
        f"No template file found. Looked for: {', '.join(TEMPLATE_CANDIDATES)}"
    )


def extract_subtemplate(template_html: str, name: str) -> str:
    """
    Extract the inner HTML of a sub-template block. The block is
    declared in the template with a start comment (`<!-- NAME -->`), then a
    comment-wrapped HTML snippet, then an end comment (`<!-- /NAME -->`).

    For example:

        <!-- SR_TPL -->
        <!--
        <tr> ... </tr>
        -->
        <!-- /SR_TPL -->

    This function extracts and returns the `<tr> ... </tr>` snippet.

    Parameters
    ----------
    template_html : str
        The full template HTML.
    name : str
        The identifier of the block (e.g. "SR_TPL").

    Returns
    -------
    str
        The inner HTML snippet. If not found, returns an empty string.
    """
    # Pattern to capture comment with nested comment
    pattern = re.compile(
        rf"<!--\s*{re.escape(name)}\s*-->"  # opening marker
        r"\s*<!--(.*?)-->"                   # inner HTML inside a single comment
        rf"\s*<!--\s*/\s*{re.escape(name)}\s*-->",  # closing marker
        re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(template_html)
    if not match:
        return ""
    inner = match.group(1)
    # Strip leading/trailing whitespace from inner content
    return inner.strip()


def replace_mount_marker(template_html: str, marker: str, html_snippet: str) -> str:
    """Replace the first occurrence of a mount marker with the provided HTML."""
    if marker not in template_html:
        raise ValueError(f"Template missing mount marker: {marker}")
    return template_html.replace(marker, html_snippet)


def highlight_keyword_in_text(text: str, keyword: str) -> str:
    """
    Return a version of the text with the first occurrence of keyword
    highlighted via a yellow background span. Matching is
    case-insensitive, but the replacement preserves the original case.
    """
    # Escape the entire text for HTML safety first
    escaped = html.escape(text)
    # Use case-insensitive replace on escaped text; highlight only the first
    def repl(match: re.Match) -> str:
        return f"<span style=\"background:#FFF3B0\">{match.group(0)}</span>"
    return re.sub(re.escape(html.escape(keyword)), repl, escaped, count=1, flags=re.IGNORECASE)


def normalize_line_breaks(text: str) -> str:
    """Collapse multiple blank lines and unify line breaks to a single newline."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of blank lines
    return re.sub(r"\n{2,}", "\n", text)


def load_keywords(path: str) -> List[str]:
    """
    Load keywords from the given file. Ignores blank lines and lines
    starting with '#'. Returns a list preserving order of appearance.
    """
    if not os.path.exists(path):
        return []
    keywords = []
    for line in read_file_text(path).splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        keywords.append(s)
    return keywords


def detect_chamber(file_path: str, sample_text: str) -> str:
    """
    Determine which chamber a transcript belongs to based on its
    filename or content. If neither heuristics match, return
    'UNKNOWN'.
    """
    lower_path = file_path.lower()
    lower_text = sample_text.lower()
    if HOUSE_HOA.lower() in lower_path or HOUSE_HOA.lower() in lower_text:
        return "HOA"
    if HOUSE_LC.lower() in lower_path or HOUSE_LC.lower() in lower_text:
        return "LC"
    # Try generic terms
    if "assembly" in lower_path or "assembly" in lower_text:
        return "HOA"
    if "council" in lower_path or "council" in lower_text:
        return "LC"
    return "UNKNOWN"


def compile_keyword_patterns(keywords: List[str]) -> Dict[str, re.Pattern[str]]:
    """
    Compile a regex pattern for each keyword that matches the keyword
    case-insensitively and only when not part of a larger alphanumeric
    word. This helps avoid false positives.
    """
    patterns = {}
    for kw in keywords:
        # Word-boundary-ish pattern: match kw surrounded by non-word
        patterns[kw] = re.compile(rf"(?i)(?<![A-Za-z0-9]){re.escape(kw)}(?![A-Za-z0-9])")
    return patterns


def scan_transcripts(keywords: List[str]) -> Tuple[Dict[str, Dict[str, int]], List[Dict], int]:
    """
    Walk through all transcript files, count keyword occurrences by
    chamber, and collect match excerpts.

    Returns a tuple of (summary_counts, file_sections, total_matches):
      - summary_counts: {kw: {'HOA': n, 'LC': n, 'TOTAL': n}}
      - file_sections: list of {filename, matches} where matches is a list
        of dicts with keys idx, title, line_label, excerpt_html
      - total_matches: total number of keyword occurrences across all files
    """
    summary_counts: Dict[str, Dict[str, int]] = {
        kw: {"HOA": 0, "LC": 0, "TOTAL": 0} for kw in keywords
    }
    file_sections: List[Dict] = []
    total_matches = 0

    patterns = compile_keyword_patterns(keywords)

    files = sorted(glob.glob(TRANSCRIPTS_GLOB))
    for path in files:
        text = read_file_text(path)
        if not text.strip():
            continue
        lines = text.splitlines()
        chamber = detect_chamber(path, text)
        match_rows: List[Dict] = []
        idx_counter = 0

        for i, raw in enumerate(lines, start=1):
            line = raw.strip()
            if not line:
                continue
            matched_any = False
            first_kw = None
            for kw, pat in patterns.items():
                hits = list(pat.finditer(line))
                if hits:
                    count = len(hits)
                    summary_counts[kw]["TOTAL"] += count
                    if chamber == "HOA":
                        summary_counts[kw]["HOA"] += count
                    elif chamber == "LC":
                        summary_counts[kw]["LC"] += count
                    if not matched_any:
                        first_kw = kw
                    matched_any = True
            if matched_any and first_kw:
                # Only count this line once for match listing; use first matched keyword
                idx_counter += 1
                total_matches += 1
                title = f"Match: {first_kw}"
                excerpt_html = highlight_keyword_in_text(line, first_kw)
                match_rows.append({
                    "idx": idx_counter,
                    "title": title,
                    "line_label": f"line {i}",
                    "excerpt_html": excerpt_html,
                })
        if match_rows:
            file_sections.append({
                "filename": os.path.basename(path),
                "matches": match_rows,
            })
    return summary_counts, file_sections, total_matches


def render_email(template_html: str, summary_counts: Dict[str, Dict[str, int]], file_sections: List[Dict]) -> str:
    """
    Fill the template with dynamic data using the hidden sub-templates.
    """
    # Extract sub-templates
    sr_tpl = extract_subtemplate(template_html, "SR_TPL")
    fs_tpl = extract_subtemplate(template_html, "FS_TPL")
    mr_tpl = extract_subtemplate(template_html, "MR_TPL")
    if not (sr_tpl and fs_tpl and mr_tpl):
        raise RuntimeError(
            "Template missing one or more required sub-templates. Ensure SR_TPL, FS_TPL, MR_TPL blocks exist."
        )

    # Build summary HTML: sort by total descending
    rows = []
    for kw, counts in sorted(summary_counts.items(), key=lambda kv: (-kv[1]["TOTAL"], kv[0].lower())):
        if counts["TOTAL"] == 0:
            continue
        row = (
            sr_tpl
            .replace("{{KEYWORD}}", html.escape(kw))
            .replace("{{HOA}}", str(counts["HOA"]))
            .replace("{{LC}}", str(counts["LC"]))
            .replace("{{TOTAL}}", str(counts["TOTAL"]))
        )
        rows.append(row)
    summary_html = "".join(rows)

    # Build file sections
    sections_html_parts = []
    for section in sorted(file_sections, key=lambda d: (-len(d["matches"]), d["filename"].lower())):
        match_rows_html = []
        for match in section["matches"]:
            match_row = (
                mr_tpl
                .replace("{{MATCH_INDEX}}", str(match["idx"]))
                .replace("{{MATCH_TITLE}}", html.escape(match["title"]))
                .replace("{{LINE_LABEL}}", html.escape(match["line_label"]))
                .replace("{{EXCERPT_HTML}}", match["excerpt_html"])
            )
            match_rows_html.append(match_row)
        section_html = (
            fs_tpl
            .replace("{{FILENAME}}", html.escape(section["filename"]))
            .replace("{{MATCHES_COUNT}}", str(len(section["matches"])))
            .replace("{{MATCH_ROWS}}", "".join(match_rows_html))
        )
        sections_html_parts.append(section_html)
    files_html = "".join(sections_html_parts)

    # Inject into template
    out = template_html
    out = replace_mount_marker(out, SUMMARY_MOUNT, summary_html)
    out = replace_mount_marker(out, FILES_MOUNT, files_html)
    # Insert run date for placeholder
    out = out.replace("{{RUN_DATE}}", datetime.now().strftime("%d %b %Y"))
    return out


def send_email_message(subject: str, html_body: str) -> None:
    """Send the email via yagmail using environment credentials."""
    if yagmail is None:
        raise ImportError("yagmail is not installed; ensure requirements.txt is installed")
    user = os.environ.get("EMAIL_USER")
    password = os.environ.get("EMAIL_PASS")
    to_raw = os.environ.get("EMAIL_TO")
    if not user or not password or not to_raw:
        raise RuntimeError(
            "EMAIL_USER, EMAIL_PASS, and EMAIL_TO environment variables must be set"
        )
    recipients = [addr.strip() for addr in re.split(r"[;,]", to_raw) if addr.strip()]
    yag = yagmail.SMTP(user=user, password=password)
    yag.send(to=recipients, subject=subject, contents=html_body)
    print(f"✅ Email sent to {', '.join(recipients)}")


def main() -> None:
    """Entry point to build and send the email."""
    try:
        template_file = pick_template_file()
    except FileNotFoundError as e:
        print(str(e))
        return
    template_html = read_file_text(template_file)

    keywords = load_keywords(KEYWORDS_FILE)
    if not keywords:
        print(f"No keywords found in {KEYWORDS_FILE}. Exiting.")
        return

    summary_counts, file_sections, total_matches = scan_transcripts(keywords)

    # Add top-level summary numbers to the template via placeholder replacement
    # These placeholders are optional in the template but if present they will be replaced.
    files_with_matches = len(file_sections)
    run_date_str = datetime.now().strftime("%d %b %Y")
    # Perform replacement pre- and post-render to avoid accidental replacement in sub-templates
    template_html = template_html.replace("{{RUN_DATE}}", run_date_str)
    template_html = template_html.replace("{{TOTAL_FILES}}", str(len(glob.glob(TRANSCRIPTS_GLOB))))
    template_html = template_html.replace("{{FILES_WITH_MATCHES}}", str(files_with_matches))
    template_html = template_html.replace("{{TOTAL_MATCHES}}", str(total_matches))

    try:
        html_out = render_email(template_html, summary_counts, file_sections)
    except Exception as e:
        print("Error rendering email:", str(e))
        return

    subject = f"Hansard Monitor — {run_date_str} — {total_matches} match(es) — {files_with_matches} file(s)"
    try:
        send_email_message(subject, html_out)
    except Exception as e:
        print("Error sending email:", str(e))
        return

    # Append to sent.log
    try:
        with open("sent.log", "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()}\t{subject}\n")
    except Exception:
        # Do not fail if logging fails
        pass


if __name__ == "__main__":
    main()
