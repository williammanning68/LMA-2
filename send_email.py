import os, re, glob, smtplib
from bisect import bisect_right
from pathlib import Path
from datetime import datetime, UTC
from zoneinfo import ZoneInfo
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ---- existing imports you already had ----
# (keep all your extraction helpers exactly as-is)
# ... [KEEP ALL YOUR EXISTING FUNCTIONS UNCHANGED up to build_digest_html()] ...

TEMPLATE_CANDIDATES = [
    "Hansard Monitor - Email Format - Version 2 - 03092025.txt",
    "email_template.html",
]

def _load_verbatim_template() -> str:
    """
    Load the exact Outlook/Word HTML you provided so the email uses it verbatim.
    We only make minimal string insertions (date, summary rows, match blocks).
    """
    for p in TEMPLATE_CANDIDATES:
        if Path(p).exists():
            return Path(p).read_text(encoding="windows-1252", errors="ignore")
    raise FileNotFoundError(
        "Template not found. Put your provided file next to this script "
        "(e.g., 'Hansard Monitor - Email Format - Version 2 - 03092025.txt')."
    )

def _html_escape(s: str) -> str:
    return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def _highlight_keywords_html(text_html: str, keywords: list[str]) -> str:
    # identical logic, visual-only span to match Word highlight
    out = text_html
    for kw in sorted(keywords, key=len, reverse=True):
        if " " in kw:
            pat = re.compile(re.escape(_html_escape(kw)), re.IGNORECASE)
        else:
            pat = re.compile(rf"\b{re.escape(_html_escape(kw))}\b", re.IGNORECASE)
        out = pat.sub(lambda m: f"<b><span style=\"background:silver;mso-highlight:silver\">{m.group(0)}</span></b>", out)
    return out

def _summary_row_html(kw: str, hoa: int, lc: int, tot: int) -> str:
    # Row markup mirrors the template’s table widths/borders/colors exactly.
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

def _inject_date(tmpl: str, hobart_dt: datetime) -> str:
    # Replace the literal [DATE] token in the hero: Program Run: [DATE]
    # (The token exists in your template hero block.)
    date_str = hobart_dt.strftime("%d %B %Y %I:%M %p %Z")
    return tmpl.replace("Program Run: [DATE]", f"Program Run: {date_str}")

def _inject_summary_rows(tmpl: str, rows_html: str) -> str:
    """
    Find the summary inner table (the one that has Keyword/House of Assembly/Legislative Council/Total)
    and insert our rows right AFTER that header <tr>.
    """
    header_regex = re.compile(
        r"(<tr[^>]*>\s*"
        r"<td[^>]*?>.*?>\s*.*?Keyword.*?</td>\s*"
        r"<td[^>]*?>.*?>\s*.*?House of Assembly.*?</td>\s*"
        r"<td[^>]*?>.*?>\s*.*?Legislative Council.*?</td>\s*"
        r"<td[^>]*?>.*?>\s*.*?Total.*?</td>\s*"
        r"</tr>)",
        re.IGNORECASE | re.DOTALL,
    )
    return header_regex.sub(r"\1" + rows_html, tmpl, count=1)

def _file_block_html(filename: str, match_count: int, match_cards_html: str) -> str:
    # Reuse your white box with gold underline style from the template section.
    return (
        "<table class='MsoNormalTable' border='1' cellspacing='0' cellpadding='0' width='100%' "
        "style='width:100.0%;mso-cellspacing:0cm;background:white;border:solid #D8DCE0 1.0pt;"
        "mso-border-alt:solid #D8DCE0 .75pt;mso-yfti-tbllook:1184;mso-padding-alt:0cm 0cm 0cm 0cm'>"
        "<tr><td style='border:none;border-bottom:solid #C5A572 2.25pt;padding:12.0pt 13.5pt'>"
        f"<p class='MsoNormal' style='margin:0;line-height:normal'><b><span style='font-family:\"Segoe UI\",sans-serif;color:black'>{_html_escape(filename)}</span></b>"
        f"<span style='font-family:\"Segoe UI\",sans-serif;color:#8795A1'> — {match_count} match(es)</span></p>"
        "</td></tr>"
        f"{match_cards_html}"
        "</table>"
    )

def _match_card_html(speaker: str, lines_str: str, excerpt_html: str) -> str:
    return (
        "<tr><td style='border:none;padding:9.0pt 12.0pt'>"
        "<table class='MsoNormalTable' border='0' cellspacing='0' cellpadding='0' width='100%' "
        "style='width:100.0%;background:white;border-collapse:collapse;border:solid #D8DCE0 1.0pt;mso-border-alt:solid #D8DCE0 .75pt;'>"
        "<tr>"
        "<td style='border-left:solid #C5A572 3.0pt;border-top:none;border-bottom:none;border-right:none;padding:7.5pt 9.0pt'>"
        f"<p class='MsoNormal' style='margin:0 0 6.0pt 0;line-height:normal'><b><span style='font-family:\"Segoe UI\",sans-serif;color:#475560'>{_html_escape(speaker)}</span></b>"
        f"<span style='font-family:\"Segoe UI\",sans-serif;color:#8795A1'> — {lines_str}</span></p>"
        f"<p class='MsoNormal' style='margin:0;line-height:22px'><span style='font-family:\"Segoe UI\",sans-serif;color:#475560'>{excerpt_html}</span></p>"
        "</td></tr></table>"
        "</td></tr>"
    )

def build_digest_html(files, keywords):
    # ----- UNCHANGED: your parsing/matching logic -----
    chambers = ["House of Assembly", "Legislative Council"]
    counts = {ch: {kw: 0 for kw in keywords} for ch in chambers}
    totals = {kw: 0 for kw in keywords}
    total_matches = 0

    # Build per-file HTML blocks
    file_blocks = []
    for f in sorted(files, key=lambda x: Path(x).name):
        text = Path(f).read_text(encoding="utf-8", errors="ignore")
        matches = extract_matches(text, keywords)
        match_cards = []

        for (kw_set, excerpt_html, speaker, line_list, win_start, _win_end) in matches:
            for kw in kw_set:
                if "house_of_assembly" in Path(f).name.lower():
                    counts["House of Assembly"][kw] += 1
                elif "legislative_council" in Path(f).name.lower():
                    counts["Legislative Council"][kw] += 1
                totals[kw] += 1

            first_line = min(line_list) if line_list else win_start
            lines_str = ("line " if len(line_list) <= 1 else "lines ") + (
                ", ".join(str(n) for n in sorted(set(line_list))) if line_list else str(first_line)
            )
            speaker_display = speaker if speaker and not _looks_suspicious(speaker) else "Unknown"
            match_cards.append(_match_card_html(speaker_display, lines_str, excerpt_html))

        total_matches += len(matches)
        file_blocks.append(_file_block_html(Path(f).name, len(matches), "".join(match_cards)))

    # Build summary rows (exact widths/borders)
    summary_rows_html = "".join(
        _summary_row_html(kw, counts["House of Assembly"].get(kw, 0), counts["Legislative Council"].get(kw, 0), totals.get(kw, 0))
        for kw in keywords
    )

    # Load template verbatim and inject date, rows, and file blocks
    tmpl = _load_verbatim_template()
    hobart_now = datetime.now(ZoneInfo("Australia/Hobart"))
    html = _inject_date(tmpl, hobart_now)
    html = _inject_summary_rows(html, summary_rows_html)

    # Append file blocks after the grey summary section (right after its closing container table)
    # Anchor: the “Detection Match by Chamber” block exists; we’ll append after its closing </table></td></tr></table>
    anchor = re.search(r"(Detection Match by Chamber.*?</table>\s*</td>\s*</tr>\s*</table>)", html, re.IGNORECASE | re.DOTALL)
    if anchor:
        insert_at = anchor.end()
        html = html[:insert_at] + "".join(file_blocks) + html[insert_at:]
    else:
        # Fallback: just append to the end if the anchor can’t be found (rare template changes)
        html += "".join(file_blocks)

    return html, total_matches, counts

# --- Main send: use raw MIME so nothing reflows your HTML ---
def main():
    EMAIL_USER = os.environ["EMAIL_USER"]
    EMAIL_PASS = os.environ["EMAIL_PASS"]
    EMAIL_TO = os.environ["EMAIL_TO"]
    SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")           # switch to 'smtp.office365.com' for Outlook
    SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
    SMTP_STARTTLS = os.environ.get("SMTP_STARTTLS", "1").lower() in ("1","true","yes")

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

    # Build MIME (text/html; charset=windows-1252 because that’s what your template declares)
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Hansard keyword digest — {datetime.now().strftime('%d %b %Y')}"
    msg["From"] = EMAIL_USER
    msg["To"] = EMAIL_TO

    part_html = MIMEText(body_html.encode("windows-1252", errors="ignore"), "html", "windows-1252")
    msg.attach(part_html)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        if SMTP_STARTTLS:
            s.starttls()
        s.login(EMAIL_USER, EMAIL_PASS)
        s.sendmail(EMAIL_USER, [a.strip() for a in re.split(r"[,\s]+", EMAIL_TO) if a.strip()], msg.as_string())

    update_sent_log(files)
    print(f"✅ Email sent to {EMAIL_TO} with {len(files)} file(s), {total_hits} match(es)}")

if __name__ == "__main__":
    main()
