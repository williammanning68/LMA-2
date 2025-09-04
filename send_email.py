#!/usr/bin/env python3
"""
send_email.py

Two sending paths:

1) Outlook + OFT TEMPLATE  (preferred on Windows)
   - Looks for an .oft file (default: ./email_template.oft or ./templates/email_template.oft)
   - Sends the template EXACTLY as-is.
   - Optionally replaces a single token (default: "[DATE]") in the HTML body.
   - Attaches your transcripts.

2) SMTP + yagmail (fallback)
   - Builds a simple HTML digest of keyword hits from transcripts and sends via SMTP.
   - Requires EMAIL_USER/EMAIL_PASS if used.

Environment variables:
  # Required for BOTH paths
  EMAIL_TO="alice@example.com,bob@example.com"

  # Optional: control where transcripts live (default: ./transcripts)
  TRANSCRIPTS_DIR="transcripts"

  # ---- OFT path (Windows / Outlook) ----
  # Optional: explicit path to template (absolute or relative to this script)
  OFT_TEMPLATE="path\\to\\email_template.oft"
  # Optional: open compose window instead of auto-send (0/1, true/false, yes/no)
  OFT_PREVIEW="1"
  # Optional: token to replace in template body; if not present, nothing changes
  OFT_DATE_TOKEN="[DATE]"

  # ---- SMTP fallback (yagmail) ----
  EMAIL_USER="you@example.com"
  EMAIL_PASS="app-password-or-password"
  SMTP_HOST="smtp.gmail.com"
  SMTP_PORT="587"
  SMTP_STARTTLS="1"
  SMTP_SSL="0"

  # Keywords file (optional; used by SMTP fallback to build digest):
  # - If KEYWORDS is set, use comma-separated keywords from env
  # - Otherwise read keywords.txt (one per line) if present
  KEYWORDS="budget, health, education"
"""

import os
import re
import glob
import json
from pathlib import Path
from datetime import datetime
from html import escape as html_escape

# Optional Outlook automation (Windows only). If unavailable, we fall back to SMTP.
try:
    import win32com.client as win32  # pip install pywin32
    HAVE_OUTLOOK = True
except Exception:
    HAVE_OUTLOOK = False


# =============================================================================
# Utility / Config
# =============================================================================

DEFAULT_TITLE = "Hansard Monitor – BETA Version 18.3"
SENT_LOG_PATH = "sent_log.json"


def load_sent_log():
    p = Path(SENT_LOG_PATH)
    if p.exists():
        try:
            return set(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            return set()
    return set()


def update_sent_log(files):
    sent = load_sent_log()
    for f in files:
        sent.add(Path(f).name)
    Path(SENT_LOG_PATH).write_text(json.dumps(sorted(sent), indent=2), encoding="utf-8")


def resolve_transcripts_dir():
    base = Path(__file__).resolve().parent
    d = os.getenv("TRANSCRIPTS_DIR", "transcripts")
    p = (Path(d) if os.path.isabs(d) else (base / d)).resolve()
    return p


def resolve_oft_path():
    """
    Locate the .oft template:
      1) OFT_TEMPLATE env var (absolute or relative)
      2) ./email_template.oft (next to this script)
      3) ./templates/email_template.oft
    Return Path or None if not found.
    """
    base = Path(__file__).resolve().parent

    env_path = os.getenv("OFT_TEMPLATE")
    if env_path:
        p = (Path(env_path) if os.path.isabs(env_path) else (base / env_path)).resolve()
        if p.exists():
            return p

    p1 = (base / "email_template.oft").resolve()
    if p1.exists():
        return p1

    p2 = (base / "templates" / "email_template.oft").resolve()
    if p2.exists():
        return p2

    return None


def parse_recipients(env_value):
    if not env_value:
        return []
    return [addr.strip() for addr in re.split(r"[,\s]+", env_value) if addr.strip()]


# =============================================================================
# Outlook (OFT) sending
# =============================================================================

def _send_with_outlook_from_oft(
    oft_path,
    to_list,
    attachments=None,
    date_token="[DATE]",
    date_value=None,
    subject_override=None,
    preview=False,
):
    """
    Opens an .oft template in Outlook, optionally replaces a token (default: [DATE]),
    sets recipients and attachments, then Display() or Send().

    The template's subject/body are kept EXACTLY as-is unless you pass subject_override.
    """
    if not HAVE_OUTLOOK:
        raise RuntimeError("Outlook automation (pywin32) not available on this machine.")

    outlook = win32.Dispatch("Outlook.Application")
    mail = outlook.CreateItemFromTemplate(str(Path(oft_path).resolve()))

    # Optional token replacement — no-op if token not present.
    if date_token:
        html = mail.HTMLBody
        html = html.replace(date_token, date_value or datetime.now().strftime("%d %B %Y"))
        mail.HTMLBody = html

    if subject_override:
        mail.Subject = subject_override  # otherwise keep the template's subject verbatim

    # Recipients
    mail.To = "; ".join([x for x in to_list if x])

    # Attachments (e.g., your transcripts)
    for fp in (attachments or []):
        mail.Attachments.Add(str(Path(fp).resolve()))

    if preview:
        mail.Display()  # open compose window for review
    else:
        mail.Send()     # send immediately


# =============================================================================
# SMTP fallback helpers (yagmail) — only imported if needed
# =============================================================================

def load_keywords():
    # Priority: KEYWORDS env (comma-separated), then keywords.txt (one per line)
    env_kw = os.getenv("KEYWORDS")
    if env_kw:
        kws = [k.strip() for k in env_kw.split(",") if k.strip()]
        return sorted(set(kws), key=str.lower)

    p = Path("keywords.txt")
    if p.exists():
        kws = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return sorted(set(kws), key=str.lower)

    return []


def _file_hits(path, keywords):
    """
    Return list of tuples (line_no, line_text, matched_keywords_set) for lines that match.
    Case-insensitive whole-line search for any keyword occurrence.
    """
    hits = []
    try:
        txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return hits

    lines = txt.splitlines()
    low_lines = [ln.lower() for ln in lines]
    low_keywords = [k.lower() for k in keywords]

    for i, (ln, low_ln) in enumerate(zip(lines, low_lines), start=1):
        matched = {k for k in low_keywords if k in low_ln}
        if matched:
            hits.append((i, ln, matched))
    return hits


def build_digest_html(files, keywords):
    """
    Build a minimal HTML digest summarizing keyword hits across files.
    Returns (html, total_hits, counts_dict).
    """
    total_hits = 0
    counts = {}
    rows = []

    for fp in files:
        fhits = _file_hits(fp, keywords) if keywords else []
        counts[Path(fp).name] = len(fhits)
        total_hits += len(fhits)

        # Summarize per file
        file_section = []
        file_section.append(f"<h3>{html_escape(Path(fp).name)}</h3>")
        if not fhits:
            file_section.append("<p><em>No keyword hits.</em></p>")
        else:
            file_section.append("<ul>")
            for (ln_no, ln_text, matched) in fhits[:50]:  # cap to keep things readable
                m_str = ", ".join(sorted(matched))
                file_section.append(
                    f"<li><strong>Line {ln_no}</strong> "
                    f"(matches: {html_escape(m_str)}): "
                    f"{html_escape(ln_text)}</li>"
                )
            if len(fhits) > 50:
                file_section.append(f"<li>…and {len(fhits)-50} more lines</li>")
            file_section.append("</ul>")

        rows.append("\n".join(file_section))

    today_long = datetime.now().strftime("%d %B %Y")
    today_short = datetime.now().strftime("%d %b %Y")
    title = f"{DEFAULT_TITLE} — {today_short}"

    html = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>{html_escape(title)}</title>
  </head>
  <body style="font-family: Arial, Helvetica, sans-serif;">
    <h2>{html_escape(DEFAULT_TITLE)}</h2>
    <p><strong>Program Run:</strong> {html_escape(today_long)}</p>
    <p>Total files: {len(files)} | Total keyword hits: {total_hits}</p>
    {'<hr>'.join(rows)}
  </body>
</html>
"""
    return html, total_hits, counts


# =============================================================================
# Main
# =============================================================================

def main():
    # Recipients (required)
    email_to = os.environ.get("EMAIL_TO", "")
    to_list = parse_recipients(email_to)
    if not to_list:
        raise SystemExit("EMAIL_TO is required (comma-separated list).")

    # Collect transcripts (attachments)
    transcripts_dir = resolve_transcripts_dir()
    all_files = sorted(str(p) for p in transcripts_dir.glob("*.txt"))
    if not all_files:
        raise SystemExit(f"No transcripts found in {transcripts_dir}")

    # Skip files already sent
    sent = load_sent_log()
    files = [f for f in all_files if Path(f).name not in sent]
    if not files:
        print("No new transcripts to email.")
        return

    # Try Outlook/OFT first
    oft_path = resolve_oft_path()
    if oft_path and HAVE_OUTLOOK:
        oft_preview = os.getenv("OFT_PREVIEW", "0").lower() in ("1", "true", "yes")
        oft_date_token = os.getenv("OFT_DATE_TOKEN", "[DATE]")

        _send_with_outlook_from_oft(
            oft_path=oft_path,
            to_list=to_list,
            attachments=files,
            date_token=oft_date_token,
            date_value=datetime.now().strftime("%d %B %Y"),
            subject_override=None,  # keep template's subject EXACTLY as-is
            preview=oft_preview,
        )

        update_sent_log(files)
        print(f"✅ Outlook/OFT email sent to {', '.join(to_list)} with {len(files)} file(s).")
        return

    # Fallback: SMTP/yagmail
    # Lazily import yagmail so Windows+OFT users don't need it installed.
    try:
        import yagmail  # pip install yagmail
    except Exception as e:
        raise SystemExit(
            "Outlook template not available and yagmail is not installed. "
            "Install pywin32 for Outlook or 'pip install yagmail' for SMTP."
        ) from e

    email_user = os.environ.get("EMAIL_USER")
    email_pass = os.environ.get("EMAIL_PASS")
    if not (email_user and email_pass):
        raise SystemExit("SMTP fallback requires EMAIL_USER and EMAIL_PASS.")

    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_starttls = os.environ.get("SMTP_STARTTLS", "1").lower() in ("1", "true", "yes")
    smtp_ssl = os.environ.get("SMTP_SSL", "0").lower() in ("1", "true", "yes")

    # Keywords for digest (optional)
    keywords = load_keywords()

    body_html, total_hits, _counts = build_digest_html(files, keywords)
    subject = f"{DEFAULT_TITLE} — {datetime.now().strftime('%d %b %Y')}"

    yag = yagmail.SMTP(
        user=email_user,
        password=email_pass,
        host=smtp_host,
        port=smtp_port,
        smtp_starttls=smtp_starttls,
        smtp_ssl=smtp_ssl,
    )

    # IMPORTANT: pass ONE HTML string to avoid extra <br> insertion
    yag.send(
        to=to_list,
        subject=subject,
        contents=body_html,
        attachments=files,
    )

    update_sent_log(files)
    print(
        f"✅ SMTP email sent to {', '.join(to_list)} with {len(files)} file(s), {total_hits} total hit(s)."
    )


if __name__ == "__main__":
    main()
