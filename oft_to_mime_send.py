#!/usr/bin/env python3
"""
oft_to_mime_send.py
One-file sender for GitHub-hosted runners.
- Reads an Outlook .OFT template directly (no Outlook/COM).
- Extracts the exact HTML + any inline attachments (CIDs) from the OFT.
- Builds a proper MIME (multipart/related) with the HTML body base64-encoded (prevents relay rewrap).
- Sends via SMTP using env vars.

Env:
  OFT_PATH                default: email_template.oft
  SMTP_HOST               required
  SMTP_PORT               default: 587
  SMTP_SSL                default: "0"  (use SMTPS/465 if "1")
  SMTP_STARTTLS           default: "1"  (use STARTTLS if "1" and not using SSL)
  SMTP_USER               optional (required if server requires auth)
  SMTP_PASS               optional
  FROM_ADDR               default: SMTP_USER
  TO                      required, comma-separated
  SUBJECT                 default: "Email"
"""
import os, sys, mimetypes, smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email import encoders
from pathlib import Path

def _bool(name: str, default=False) -> bool:
    val = os.environ.get(name, "1" if default else "0").lower()
    return val in ("1", "true", "yes", "on")

def extract_from_oft(oft_path: str):
    try:
        import extract_msg  # pip install extract_msg olefile
    except Exception as e:
        print("ERROR: missing dependency extract_msg (pip install extract_msg olefile)", file=sys.stderr)
        raise
    msg = extract_msg.Message(oft_path)
    html = msg.htmlBody or ""
    if not html:
        # Fallback: not ideal, but keeps something rendering if HTML missing
        html = (msg.body or "").replace("\n", "<br>")
    # harvest inline attachments (with Content-ID), and other files
    inlines = []  # (bytes, mimetype, filename, cid)
    attaches = [] # (bytes, mimetype, filename)
    for att in msg.attachments:
        data = att.data
        name = att.longFilename or att.shortFilename or "file.bin"
        cid = getattr(att, "cid", None) or getattr(att, "contentId", None)
        mt, _ = mimetypes.guess_type(name)
        if not mt and name.lower().endswith(".jpg"):
            mt = "image/jpeg"
        mt = mt or "application/octet-stream"
        if cid:
            inlines.append((data, mt, name, cid.strip("<>")))
        else:
            attaches.append((data, mt, name))
    return html, inlines, attaches

def build_mime(from_addr: str, to_list, subject: str, html: str, inlines, attaches):
    # multipart/related -> multipart/alternative -> text/plain + text/html(base64)
    root = MIMEMultipart("related")
    alt = MIMEMultipart("alternative")
    root.attach(alt)
    alt.attach(MIMEText("This message contains HTML content.", "plain", "utf-8"))
    html_part = MIMEText(html, "html", "utf-8")
    encoders.encode_base64(html_part)  # ensure base64 CTE
    alt.attach(html_part)

    # inline CIDs
    for data, mt, name, cid in inlines:
        maintype, subtype = mt.split("/", 1)
        if maintype == "image":
            part = MIMEImage(data, _subtype=subtype)
        else:
            part = MIMEBase(maintype, subtype)
            part.set_payload(data)
            encoders.encode_base64(part)
        part.add_header("Content-ID", f"<{cid}>")
        part.add_header("Content-Disposition", f'inline; filename="{name}"')
        root.attach(part)

    # regular attachments (if any)
    for data, mt, name in attaches:
        maintype, subtype = mt.split("/", 1)
        part = MIMEBase(maintype, subtype)
        part.set_payload(data)
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{name}"')
        root.attach(part)

    root["From"] = from_addr
    root["To"] = ", ".join(to_list)
    root["Subject"] = subject
    return root

def send_smtp(msg, host, port, user, password, use_ssl=False, use_starttls=True):
    if use_ssl:
        server = smtplib.SMTP_SSL(host, port)
    else:
        server = smtplib.SMTP(host, port)
    try:
        server.ehlo()
        if use_starttls and not use_ssl:
            server.starttls()
            server.ehlo()
        if user:
            server.login(user, password or "")
        server.sendmail(msg["From"], [a.strip() for a in msg["To"].split(",")], msg.as_bytes())
    finally:
        server.quit()

def main():
    oft_path = os.environ.get("OFT_PATH", "email_template.oft")
    smtp_host = os.environ.get("SMTP_HOST")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    use_ssl = _bool("SMTP_SSL", False)
    use_starttls = _bool("SMTP_STARTTLS", True)
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_PASS", "")
    to_list = [a.strip() for a in (os.environ.get("TO") or os.environ.get("EMAIL_TO") or "").split(",") if a.strip()]
    if not smtp_host or not to_list:
        print("SMTP_HOST and TO (or EMAIL_TO) must be set.", file=sys.stderr)
        sys.exit(2)
    from_addr = os.environ.get("FROM_ADDR") or smtp_user or "no-reply@example.com"
    subject = os.environ.get("SUBJECT", "Email")

    if not Path(oft_path).exists():
        print(f"File not found: {oft_path}", file=sys.stderr)
        sys.exit(1)

    html, inlines, attaches = extract_from_oft(oft_path)
    msg = build_mime(from_addr, to_list, subject, html, inlines, attaches)
    send_smtp(msg, smtp_host, smtp_port, smtp_user, smtp_pass, use_ssl=use_ssl, use_starttls=use_starttls)
    print(f"✅ Sent to {len(to_list)} recipient(s) via SMTP.")

if __name__ == "__main__":
    main()
