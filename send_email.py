import os
import re
import glob
from pathlib import Path
from datetime import datetime
from bisect import bisect_right
import yagmail

# =============================================================================
# Config
# =============================================================================

LOG_FILE = Path("sent.log")
DEFAULT_TITLE = "Hansard Monitor – BETA Version 18.3"

# excerpt/windowing
MAX_SNIPPET_CHARS = 800
WINDOW_PAD_SENTENCES = 1
FIRST_SENT_FOLLOWING = 2
MERGE_IF_GAP_GT = 2

# =============================================================================
# MSO style (kept as a tiny fragment, not a template)
# =============================================================================

MSO_STYLE = """<!--[if mso]><style>
p.MsoNormal,div.MsoNormal,li.MsoNormal{margin:0 0 6pt 0 !important;line-height:normal !important;}
table,td{mso-table-lspace:0pt !important;mso-table-rspace:0pt !important;}
</style><![endif]-->"""

# Path to the external email template file (HTML)
TEMPLATE_PATH = Path(os.environ.get("EMAIL_TEMPLATE_FILE", "email_template.htm"))

# =============================================================================
# Template loader
# =============================================================================

def load_email_template() -> str:
    """
    Load the external HTML email template at runtime.
    The template is expected to contain these placeholders:
      {title}            – Subject/title text (not required in body if you don't use it)
      {mso_style}        – Optional conditional CSS for Outlook (safe to omit in template)
      {date}             – e.g., '03 September 2025'
      {detection_rows}   – <tr>...</tr> rows for the keyword counts table
      {sections}         – HTML for all per-file sections/cards
    """
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(
            f"Email template not found: {TEMPLATE_PATH}. "
            "Set EMAIL_TEMPLATE_FILE env var or place email_template.htm alongside the script."
        )
    return TEMPLATE_PATH.read_text(encoding="utf-8", errors="ignore")

# =============================================================================
# Keyword loading
# =============================================================================

def load_keywords():
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

# =============================================================================
# Transcript segmentation / extraction / html helpers  (UNCHANGED)
# =============================================================================
# ... keep your existing helpers here ...
# (e.g., _build_utterances, _compile_kw_patterns, _collect_hits_in_utterance,
#  _windows_for_hits, _dedup_windows, _merge_windows_far_only, _html_escape,
#  _highlight_keywords_html, _excerpt_from_window_html, extract_matches, etc.)
#
# These appear in your current file as shown by the snippets (preserve them):
# - extract_matches(...) :contentReference[oaicite:2]{index=2}
# - whitespace tighten/minify helpers :contentReference[oaicite:3]{index=3}
# - detection row builder :contentReference[oaicite:4]{index=4}
# - file section builder (cards) :contentReference[oaicite:5]{index=5}

# =============================================================================
# Build the full HTML (now using external template)
# =============================================================================

def build_digest_html(files: list[str], keywords: list[str]):
    run_date = datetime.now().strftime("%d %B %Y")

    # Collect matches + counts
    counts = {kw: {"House of Assembly": 0, "Legislative Council": 0} for kw in keywords}
    sections, total_matches = [], 0

    for f in files:
        text = Path(f).read_text(encoding="utf-8", errors="ignore")
        matches = extract_matches(text, keywords)
        if not matches:
            continue
        total_matches += len(matches)

        chamber = _parse_chamber_from_filename(Path(f).name)
        for kw_set, *_rest in matches:
            for kw in kw_set:
                counts.setdefault(kw, {"House of Assembly": 0, "Legislative Council": 0})
                if chamber in counts[kw]:
                    counts[kw][chamber] += 1

        sections.append(_build_file_section_html(Path(f).name, matches))

    # Detection rows HTML (unchanged logic)
    det_rows = []
    for kw in keywords:
        hoa = counts.get(kw, {}).get("House of Assembly", 0)
        lc  = counts.get(kw, {}).get("Legislative Council", 0)
        det_rows.append(_build_detection_row(kw, hoa, lc, hoa + lc))
    detection_rows_html = "".join(det_rows)

    # Load external template and fill placeholders
    template = load_email_template()
    try:
        html = template.format(
            title=DEFAULT_TITLE,
            mso_style=MSO_STYLE,
            date=run_date,
            detection_rows=detection_rows_html,
            sections="".join(sections),
        )
    except KeyError as e:
        # Provide a friendlier error if the template is missing required placeholders
        missing = str(e).strip("{}")
        raise KeyError(
            f"Email template missing placeholder: {{{missing}}}. "
            "Expected placeholders: {title}, {mso_style}, {date}, {detection_rows}, {sections}"
        )

    # Final whitespace controls (unchanged)
    html = _tighten_outlook_whitespace(html)
    html = _minify_inter_tag_whitespace(html)

    return html, total_matches, counts

# =============================================================================
# Sent-log helpers  (UNCHANGED)
# =============================================================================

def load_sent_log() -> set[str]:
    if LOG_FILE.exists():
        with open(LOG_FILE, encoding="utf-8") as f:
            return {ln.strip() for ln in f if ln.strip()}
    return set()

def update_sent_log(files: list[str]):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        for fp in files:
            f.write(Path(fp).name + "\n")

# =============================================================================
# Main  (UNCHANGED sending path)
# =============================================================================

def main():
    EMAIL_USER = os.environ["EMAIL_USER"]
    EMAIL_PASS = os.environ["EMAIL_PASS"]
    EMAIL_TO   = os.environ["EMAIL_TO"]

    SMTP_HOST      = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT      = int(os.environ.get("SMTP_PORT", "587"))
    SMTP_STARTTLS  = os.environ.get("SMTP_STARTTLS", "1").lower() in ("1", "true", "yes")
    SMTP_SSL       = os.environ.get("SMTP_SSL", "0").lower() in ("1", "true", "yes")

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
    subject = f"{DEFAULT_TITLE} — {datetime.now().strftime('%d %b %Y')}"

    to_list = [addr.strip() for addr in re.split(r"[,\s]+", EMAIL_TO) if addr.strip()]

    yag = yagmail.SMTP(
        user=EMAIL_USER,
        password=EMAIL_PASS,
        host=SMTP_HOST,
        port=SMTP_PORT,
        smtp_starttls=SMTP_STARTTLS,
        smtp_ssl=SMTP_SSL,
    )

    # IMPORTANT: pass ONE HTML string to avoid yagmail inserting extra <br>
    yag.send(
        to=to_list,
        subject=subject,
        contents=body_html,
        attachments=files,  # optional: attach transcripts
    )

    update_sent_log(files)
    print(f"✅ Email sent to {EMAIL_TO} with {len(files)} file(s), {total_hits} match(es).")

if __name__ == "__main__":
    main()
