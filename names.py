import argparse
import csv
import re
from pathlib import Path
from typing import Dict
from urllib.parse import urljoin

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

BASE_URL = "https://www.myscheme.gov.in"
DEFAULT_CATEGORY_URL = (
    "https://www.myscheme.gov.in/search/category/Science,%20IT%20&%20Communications"
)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def slug_to_name(url: str) -> str:
    slug = url.rstrip("/").split("/")[-1]
    return slug.replace("-", " ").strip().title()


def extract_from_page(page) -> Dict[str, str]:
    data: Dict[str, str] = {}
    links = page.locator("a[href*='/schemes/']")
    count = links.count()

    for i in range(count):
        anchor = links.nth(i)
        href = anchor.get_attribute("href")
        if not href:
            continue
        full_url = urljoin(BASE_URL, href)
        if "/schemes/" not in full_url:
            continue

        text = normalize_space(anchor.inner_text())
        if not text:
            text = normalize_space(anchor.text_content() or "")
        if not text:
            text = slug_to_name(full_url)

        if full_url not in data:
            data[full_url] = text

    return data


def write_csv(rows: Dict[str, str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not output_path.exists()) or output_path.stat().st_size == 0
    with output_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["scheme_name", "scheme_url"])
        for url, name in sorted(rows.items(), key=lambda item: item[1].lower()):
            writer.writerow([name, url])


def fallback_from_urls_file(input_path: Path, output_path: Path) -> int:
    if not input_path.exists():
        return 0

    rows: Dict[str, str] = {}
    for line in input_path.read_text(encoding="utf-8").splitlines():
        url = line.strip()
        if not url:
            continue
        rows[url] = slug_to_name(url)

    if not rows:
        return 0

    write_csv(rows, output_path)
    return len(rows)


def scrape_category(category_url: str, max_pages: int, timeout_ms: int) -> Dict[str, str]:
    all_rows: Dict[str, str] = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        print(f"Loading initial URL: {category_url}")
        try:
            page.goto(category_url, wait_until="networkidle", timeout=timeout_ms)
        except PlaywrightTimeoutError:
            print("Initial load timed out, but trying to proceed...")

        for page_no in range(1, max_pages + 1):
            print(f"--- Scraping Page {page_no} ---")
            
            # Wait for schemes to appear on the current page
            page.wait_for_selector("a[href*='/schemes/']", timeout=10000)
            
            # 1. Extract data
            current_data = extract_from_page(page)
            all_rows.update(current_data)
            print(f"Collected {len(current_data)} links (Total: {len(all_rows)})")

            # 2. Locate the "Next Arrow" SVG
            # Based on your inspect image, the next button is the last SVG in that ul
            next_arrow = page.locator("ul.list-none.flex svg").last
            
            # Check if we've reached the end
            # Usually, the 'last' page hides the arrow or adds a 'disabled' class to the parent <li>
            parent_li = page.locator("ul.list-none.flex li").last
            if "disabled" in (parent_li.get_attribute("class") or ""):
                print("Reached the end of pagination.")
                break

            if next_arrow.is_visible():
                print("Clicking the next arrow...")
                next_arrow.click()
                
                # 3. Wait for the cards to refresh
                # We wait for the first scheme's text to change or just a small timeout
                page.wait_for_timeout(2000) 
                page.wait_for_load_state("networkidle")
            else:
                print("Next arrow not visible. Stopping.")
                break

        browser.close()
    return all_rows

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape myScheme category pages and save scheme name + URL to CSV."
    )
    parser.add_argument("--category-url", default=DEFAULT_CATEGORY_URL)
    parser.add_argument("--max-pages", type=int, default=84)
    parser.add_argument("--output", default="scheme_names_urls.csv")
    parser.add_argument("--timeout-ms", type=int, default=60000)
    parser.add_argument(
        "--fallback-urls-file",
        default="all_urls.txt",
        help="If online scrape returns no rows, fallback by converting URLs in this file.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    rows = scrape_category(
        category_url=args.category_url,
        max_pages=args.max_pages,
        timeout_ms=args.timeout_ms,
    )

    if rows:
        write_csv(rows, output_path)
        print(f"Saved {len(rows)} rows to {output_path}")
        return

    fallback_count = fallback_from_urls_file(Path(args.fallback_urls_file), output_path)
    if fallback_count > 0:
        print(
            "Online scrape returned 0 rows. "
            f"Used fallback file and saved {fallback_count} rows to {output_path}"
        )
        return

    print("No rows found from scrape and fallback file.")


if __name__ == "__main__":
    main()
