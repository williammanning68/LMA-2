#!/usr/bin/env python3
"""
Scan the Hansard "Quick Search" by current (Hobart) year and download any
transcripts not yet in transcripts/ as .txt via the viewer's "As Text" option.

Environment (optional):
  WAIT_BEFORE_DOWNLOAD_SECONDS   default "15"
  MAX_PAGES                      default "5"
"""

import os
import re
from pathlib import Path
from time import sleep
from datetime import datetime
from zoneinfo import ZoneInfo

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

ROOT = Path(__file__).parent.resolve()
OUT_DIR = ROOT / "transcripts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WAIT_BEFORE_DOWNLOAD = int(os.environ.get("WAIT_BEFORE_DOWNLOAD_SECONDS", "15"))
MAX_PAGES = int(os.environ.get("MAX_PAGES", "5"))

HOBART_TZ = ZoneInfo("Australia/Hobart")

def sanitise_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_") + ".txt"

def download_current_year_new():
    year = datetime.now(HOBART_TZ).year
    url = "https://www.parliament.tas.gov.au/hansard"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(accept_downloads=True)
        context.set_default_timeout(60000)
        page = context.new_page()

        print(f"Opening Hansard home… ({url})")
        page.goto(url, wait_until="domcontentloaded")

        # Quick Search form:
        # form#queryForm with input#value[name="IW_FIELD_ADVANCE_PHRASE"]
        page.wait_for_selector('#queryForm', timeout=30000)
        page.wait_for_selector('#value', timeout=30000)
        page.fill('#value', str(year))
        page.click('#queryForm button[type="submit"]')

        total_downloaded = 0
        page_num = 1
        while page_num <= MAX_PAGES:
            print(f"Scanning results page {page_num}…")
            try:
                page.wait_for_selector('a[href*="/doc/"]', timeout=30000)
            except PWTimeout:
                print("No results on this page.")
                break

            # Title links (avoid green DOCX link)
            result_links = page.locator(
                "td.resultColumnB a[href*='/doc/']:not(:has-text('Download Document'))"
            )
            count = result_links.count()
            if count == 0:
                print("No result links found.")
                break

            for i in range(count):
                link = result_links.nth(i)
                title = link.inner_text().strip()

                out_path = OUT_DIR / sanitise_filename(title)
                if out_path.exists():
                    continue

                print(f"→ Opening: {title}")
                try:
                    with page.expect_download(timeout=5000) as dl:
                        link.click()
                    download = dl.value
                    download.save_as(str(out_path))
                    print(f"   ✅ Saved: {out_path.name}")
                    total_downloaded += 1
                    continue
                except PWTimeout:
                    pass

                # Viewer toolbar overlay
                try:
                    page.wait_for_selector('#viewer_toolbar', timeout=60000)
                except PWTimeout:
                    print("   ❌ Viewer toolbar not found; skipping.")
                    continue

                # Safety delay
                print(f"   Waiting {WAIT_BEFORE_DOWNLOAD}s before download…")
                sleep(WAIT_BEFORE_DOWNLOAD)

                # Open download menu
                page.click('#viewer_toolbar .btn.btn-download, div[onclick*="downloadMenu"]')

                # Choose "As Text"
                page.wait_for_selector('#viewer_toolbar_download li:has-text("As Text")', timeout=60000)
                with page.expect_download(timeout=60000) as dl:
                    page.click('#viewer_toolbar_download li:has-text("As Text")')
                download = dl.value
                download.save_as(str(out_path))
                print(f"   ✅ Saved: {out_path.name}")
                total_downloaded += 1

                # Close viewer
                try:
                    page.click('#viewer_toolbar .btn.btn-close')
                    sleep(0.5)
                except PWTimeout:
                    pass

            # Next page?
            next_btn = page.locator('#isys_var_nextbatch, a.page:has-text("Next")')
            if next_btn.count() > 0 and next_btn.first.is_visible():
                next_btn.first.click()
                page_num += 1
            else:
                break

        browser.close()
        print(f"Done. New downloads this run: {total_downloaded}")

if __name__ == "__main__":
    download_current_year_new()
