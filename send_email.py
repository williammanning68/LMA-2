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
...
        return [kw.strip().strip('"') for kw in os.environ["KEYWORDS"].split(",") if kw.strip()]
    return []

# =============================================================================
# Transcript segmentation (utterances)
# =============================================================================

SPEAKER_HEADER_RE = re.compile(
    r"""
    ^
    (?:
        (?P<title>(?i:Mr|Ms|Mrs|Miss|Hon|Dr|Prof|Professor|Premi...he\s+PRESIDENT|The\s+CLERK|Deputy\s+Speaker|Deputy\s+President))
        (?:[\s.]+(?P<name>[A-Z][A-Z'’\-]+(?:\s+[A-Z][A-Z'’\-]+){0,3}))?
      |
        (?P<name_only>[A-Z][A-Z'’\-]+(?:\s+[A-Z][A-Z'’\-]+){0
""",
    re.X | re.M,
)

...

# =============================================================================
# Email HTML (UPDATED to remove MSO classes/blocks and <th> usage)
# =============================================================================

# (1) Removed conditional MSO style block; keep placeholder but empty
MSO_STYLE = ""

# (2) Tidy inline CSS only; no class="MsoNormal" anywhere
EMAIL_TEMPLATE = u"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
{mso_style}
</head>
<body style="margin:0;padding:0;background:#ffffff;">
  <!-- Header -->
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="border-collapse:collapse;">
    <tr><td align="center" style="padding:16px 0;">
      <p style="margin:0 0 6pt 0;text-align:center">
        <span style="font-size:14pt;font-family:'Segoe UI',sans-serif;color:#000;">Hansard Monitor</span>
      </p>
      <p style="margin:0 0 6pt 0;text-align:center">
        <span style="font-size:10pt;font-family:'Segoe UI',sans-serif;color:#6A7682;">{date}</span>
      </p>
    </td></tr>
  </table>

  <!-- Detection Match by Chamber -->
  <table role="presentation" width="100%" cellpadding="0" cellsp...style="border-collapse:collapse;margin:0 auto;max-width:860px;">
    <tr>
      <td style="border-left:3px solid #C5A572;background:#F7F9FA;padding:6px 10px;">
        <div style="font:bold 10pt 'Segoe UI',sans-serif;color:#...ht-rule:exactly;display:block;">Detection Match by Chamber</div>
      </td>
    </tr>
    <tr>
      <td style="border:1px solid #D8DCE0;border-top:none;background:#FFFFFF;padding:0;">
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="border-collapse:collapse;">
          <tr>
            <td align="left" width="28%" style="border-left:1px ...0px;font:10pt 'Segoe UI',sans-serif;color:#24313F;"><b>Keyword</b></td>
            <td align="center" width="28%" style="border-bottom:...0pt 'Segoe UI',sans-serif;color:#24313F;"><b>House of Assembly</b></td>
            <td align="center" width="28%" style="border-bottom:...t 'Segoe UI',sans-serif;color:#24313F;"><b>Legislative Council</b></td>
            <td align="center" width="16%" style="border-right:1... 10px;font:10pt 'Segoe UI',sans-serif;color:#24313F;"><b>Total</b></td>
          </tr>
          {detection_rows}
        </table>
      </td>
    </tr>
  </table>

  <!-- Sections (cards) -->
  <table role="presentation" width="100%" cellpadding="0" cellsp...style="border-collapse:collapse;margin:0 auto;max-width:860px;">
    <tr><td style="padding:6px 8px;">
      {sections}
    </td></tr>
  </table>
</body>
</html>
"""

# =============================================================================
# Builders for HTML sections (UPDATED)
# =============================================================================

def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
         .replace("'", "&#39;")
    )


def _build_detection_row(kw, hoa, lc, tot) -> str:
    kw_html = _html_escape(kw)
    return (
        "<tr>"
        "<td width=\"28%\" style='border-left:1px solid #D8DCE0;border-bottom:1px solid #ECF0F1;padding:8px 10px;'>"
        f"<p style='margin:0;'><span style='font:10pt \"Segoe UI\",sans-serif;color:#000;'><b>{kw_html}</b></span></p></td>"
        "<td width=\"28%\" align='center' style='border-bottom:1px solid #ECF0F1;padding:8px 10px;'>"
        f"<p style='margin:0;'><span style='font:10pt \"Segoe UI\",sans-serif;color:#000;'><b>{hoa}</b></span></p></td>"
        "<td width=\"28%\" align='center' style='border-bottom:1px solid #ECF0F1;padding:8px 10px;'>"
        f"<p style='margin:0;'><span style='font:10pt \"Segoe UI\",sans-serif;color:#000;'><b>{lc}</b></span></p></td>"
        "<td width=\"16%\" align='center' style='border-right:1px solid #D8DCE0;border-bottom:1px solid #ECF0F1;padding:8px 10px;'>"
        f"<p style='margin:0;'><span style='font:10pt \"Segoe UI\",sans-serif;color:#000;'><b>{tot}</b></span></p></td>"
        "</tr>"
    )

...

# =============================================================================
# Rendering
# =============================================================================

def _render_email(title: str, date_text: str, detection_rows_html: str, sections_html: str) -> str:
    html = EMAIL_TEMPLATE.format(
        title=title,
        date=date_text,
        detection_rows=detection_rows_html,
        sections=sections_html,
        mso_style=MSO_STYLE,
    )
    # Remove blank paragraphs/spans that might be produced upstream
    html = re.sub(r"<p[^>]*>\s*(?:&nbsp;|<br\s*/?>)?\s*</p>", "", html, flags=re.I)
    # Minify inter-tag whitespace without touching content
    html = re.sub(r">\s+<", "><", html)
    html = re.sub(r"\s{2,}", " ", html)
    return html

# =============================================================================
# Assembling content (existing logic elided with ...)
# =============================================================================

...

# =============================================================================
# Sending
# =============================================================================

def update_sent_log(files):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with LOG_FILE.open("a", encoding="utf-8") as f:
        for file in files:
            f.write(f"{now}\t{file}\n")


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
...

    # Build detection summary rows (kw, hoa_count, lc_count, tot)
    detection_rows_html = "".join(_build_detection_row(kw, counts[kw]['hoa'], counts[kw]['lc'], counts[kw]['total']) for kw in counts)

    # Build sections/cards markup
    sections_html = build_sections(sections_data)  # existing function (elided above)

    date_text = datetime.now().strftime("%A, %d %B %Y")
    title = os.environ.get("EMAIL_TITLE", DEFAULT_TITLE)

    body_html = _render_email(title, date_text, detection_rows_html, sections_html)

    # Send
    yag = yagmail.SMTP(user=EMAIL_USER, password=EMAIL_PASS, host=SMTP_HOST, port=SMTP_PORT, smtp_starttls=SMTP_STARTTLS, smtp_ssl=SMTP_SSL)

    to_list = [addr.strip() for addr in re.split(r"[,;]", EMAIL_TO) if addr.strip()]
    subject = f"{title} – {date_text}"

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
