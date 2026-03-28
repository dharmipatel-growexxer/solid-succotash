"""
Script to extract scheme data from rem_schemes.csv and save JSON files
to their respective section folders (my_schemes_{section_no}).
"""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from data_scraper import (
    normalize_space,
    extract_scheme_text,
    wait_for_content,
    expand_faqs,
    should_retry,
    sections_are_effectively_empty,
    build_sections,
    classify_location,
    safe_filename,
    unique_output_path,
)


DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "schemes_urls" / "rem_schemes.csv"
JSON_DIR = DATA_DIR / "schemes_data_json"


def get_section_name_mapping() -> Dict[str, str]:
    """
    Build a mapping from section_no to section_name based on CSV filenames.
    E.g., '4' -> 'education_learning', '10' -> 'social_welfare_empowerment'
    """
    mapping = {}
    csv_dir = DATA_DIR / "schemes_urls"
    if csv_dir.exists():
        for csv_file in csv_dir.glob("*.csv"):
            # Parse filename like "4_education_learning.csv"
            stem = csv_file.stem
            if "_" in stem:
                parts = stem.split("_", 1)
                if len(parts) == 2 and parts[0].isdigit():
                    mapping[parts[0]] = parts[1]
    return mapping


def iter_rem_csv_rows(csv_path: Path) -> List[Tuple[str, str, str]]:
    """
    Read rem_schemes.csv and yield (scheme_name, scheme_url, section_no) tuples.
    Handles malformed CSV with unquoted commas in scheme names.
    """
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = {normalize_space(x).lower(): x for x in (reader.fieldnames or [])}

        name_col = fields.get("scheme_name") or fields.get("name")
        url_col = fields.get("scheme_url") or fields.get("url") or fields.get("link")
        section_col = fields.get("section_no") or fields.get("section")

        if not name_col or not url_col:
            raise ValueError(
                "CSV must have columns: scheme_name/scheme_url or name/link."
            )
        if not section_col:
            raise ValueError(
                "CSV must have a section_no or section column."
            )

        for row in reader:
            # Handle malformed CSV: if section looks like a URL, columns got shifted
            raw_section = row.get(section_col, "")
            raw_url = row.get(url_col, "")
            
            # If section contains "myscheme", the columns got shifted due to comma in name
            if "myscheme" in raw_section.lower():
                # The actual URL is in section_no, section is in None key
                url = normalize_space(raw_section)
                extra = row.get(None, [])
                section = normalize_space(extra[0]) if extra else ""
            else:
                url = normalize_space(raw_url)
                section = normalize_space(raw_section)
            
            name = normalize_space(row.get(name_col, ""))
            
            if name and url and section:
                rows.append((name, url, section))

    return rows


def get_section_output_dir(section_no: str, section_name: str = "") -> Path:
    """
    Return the output directory for a given section.
    Uses {section_no}_{section_name} format if section_name provided,
    otherwise falls back to {section_no}_{section_name} from mapping.
    Creates the directory if it doesn't exist.
    """
    if not section_name:
        mapping = get_section_name_mapping()
        section_name = mapping.get(section_no, f"section_{section_no}")
    output_dir = JSON_DIR / f"{section_no}_{section_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def scrape_and_save_by_section(
    csv_path: Path,
    headless: bool,
    timeout_ms: int,
    limit: int = 0,
) -> None:
    """
    Scrape schemes from rem_schemes.csv and save JSON files to their
    respective section folders ({section_no}_{section_name}).
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    rows = iter_rem_csv_rows(csv_path)
    if limit > 0:
        rows = rows[:limit]

    # Group rows by section for summary
    section_counts: Dict[str, int] = {}
    for _, _, section in rows:
        section_counts[section] = section_counts.get(section, 0) + 1

    section_mapping = get_section_name_mapping()
    
    print(f"Found {len(rows)} schemes across {len(section_counts)} sections:")
    for section, count in sorted(section_counts.items(), key=lambda x: int(x[0])):
        section_name = section_mapping.get(section, f"section_{section}")
        print(f"  - {section}_{section_name}: {count} scheme(s)")
    print()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            viewport={"width": 1365, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            locale="en-US",
        )
        page = context.new_page()

        success_count = 0
        fail_count = 0

        section_mapping = get_section_name_mapping()

        for idx, (scheme_name, scheme_url, section_no) in enumerate(rows, start=1):
            section_name = section_mapping.get(section_no, f"section_{section_no}")
            print(f"[{idx}/{len(rows)}] Scraping: {scheme_name}")
            print(f"           Section: {section_no}_{section_name}/")

            html = ""
            page_title = ""
            state_name = ""
            faqs: List[Tuple[str, str]] = []
            full_text = ""
            sections: Dict[str, Any] = {}

            for attempt in range(1, 4):
                try:
                    page.goto(scheme_url, wait_until="domcontentloaded", timeout=timeout_ms)
                    wait_for_content(page, timeout_ms)
                    page.wait_for_timeout(1200)
                    expand_faqs(page)
                    page.wait_for_timeout(600)
                    html = page.content()
                except PlaywrightTimeoutError:
                    print(f"  -> Timeout on {scheme_url} (attempt {attempt})")
                    continue
                except Exception as exc:
                    print(f"  -> Failed on {scheme_url} (attempt {attempt}): {exc}")
                    continue

                page_title, state_name, faqs, full_text, sections = extract_scheme_text(html)
                if not should_retry(page_title, sections):
                    break

            if not page_title and sections_are_effectively_empty(sections):
                print(f"  -> Empty content for {scheme_url}, skipped.")
                fail_count += 1
                continue

            location_name, location_type = classify_location(state_name)

            # Fallback if DOM section extraction returned nothing useful.
            if sections_are_effectively_empty(sections):
                sections = build_sections(full_text)

            # Get the output directory for this section
            output_dir = get_section_output_dir(section_no, section_name)

            stem = safe_filename(scheme_name)
            out_file = unique_output_path(output_dir, stem, suffix=".json")

            payload = {
                "scheme_name": scheme_name,
                "scheme_url": scheme_url,
                "section_no": section_no,
                "section_name": section_name,
                "location_name": location_name,
                "location_type": location_type,
                "page_title": page_title,
                "scraped_at": timestamp,
                "sections": sections,
                "faqs": [
                    {
                        "question": q,
                        "answer": a if a else "Answer not captured.",
                    }
                    for q, a in faqs
                ],
            }

            with out_file.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            print(f"  -> Saved: {out_file}")
            success_count += 1
            page.wait_for_timeout(400)

        context.close()
        browser.close()

    print()
    print("=" * 50)
    print(f"Scraping completed!")
    print(f"  Success: {success_count}")
    print(f"  Failed:  {fail_count}")
    print(f"  Total:   {len(rows)}")
    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract scheme data from rem_schemes.csv and save to respective section folders."
    )
    parser.add_argument(
        "--csv",
        default=str(CSV_PATH),
        help="Input CSV file path (default: data/schemes_urls/rem_schemes.csv)",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=90000,
        help="Page load timeout in milliseconds (default: 90000)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of schemes to process (0 means all)",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run browser in headed mode for debugging",
    )
    args = parser.parse_args()

    scrape_and_save_by_section(
        csv_path=Path(args.csv),
        headless=not args.headed,
        timeout_ms=args.timeout_ms,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
