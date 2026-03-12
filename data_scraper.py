import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from bs4 import BeautifulSoup
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


NOISE_LINES = {
    "something went wrong. please try again later.",
    "ok",
    "are you sure you want to sign out?",
    "cancel",
    "sign out",
    "you're being redirected to an external website. by proceeding, your details will be shared with the relevant department.",
    "your mobile number will be shared with jan samarth and you will be redirected to external website.",
    "you have already submitted an application for this scheme. you may apply again only after 30 days i.e. after",
    "you need to sign in before applying for schemes",
    "it seems you have already initiated your application earlier.",
    "to know more please visit",
}

DROP_LINES = {
    "back",
    "feedback",
    "apply now",
    "check eligibility",
    "sign in",
    "share",
    "scheme guidelines",
    "news and updates",
    "no new news and updates available",
    "connect on social media",
    "quick links",
    "useful links",
    "get in touch",
}

STOP_AT_LINES = {
    "was this helpful?",
    "powered by",
    "about us",
    "contact us",
    "screen reader",
    "accessibility statement",
    "disclaimer",
    "terms & conditions",
    "dashboard",
}

SECTION_MAP = {
    "details": "details",
    "benefits": "benefits",
    "eligibility": "eligibility",
    "application process": "application_process",
    "documents required": "documents_required",
    "sources and references": "sources_and_references",
    "frequently asked questions": "faqs",
}

STATE_NAMES = {
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh", "goa",
    "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka", "kerala",
    "madhya pradesh", "maharashtra", "manipur", "meghalaya", "mizoram", "nagaland",
    "odisha", "punjab", "rajasthan", "sikkim", "tamil nadu", "telangana", "tripura",
    "uttar pradesh", "uttarakhand", "west bengal", "andaman and nicobar islands",
    "chandigarh", "dadra and nagar haveli and daman and diu", "delhi", "jammu and kashmir",
    "ladakh", "lakshadweep", "puducherry",
}

UNION_TERRITORIES = {
    "andaman and nicobar islands",
    "chandigarh",
    "dadra and nagar haveli and daman and diu",
    "delhi",
    "jammu and kashmir",
    "ladakh",
    "lakshadweep",
    "puducherry",
}

DATA_DIR = Path("data")
CSV_DIR = DATA_DIR / "schemes_urls"
JSON_DIR = DATA_DIR / "schemes_data_json"






def is_noise_line(text: str) -> bool:
    return normalize_space(text).casefold() in NOISE_LINES


def clean_line(text: str) -> str:
    return normalize_space((text or "").replace("\ufeff", ""))


def get_clean_lines(text: str) -> List[str]:
    raw_lines = [clean_line(line) for line in text.splitlines()]
    lines: List[str] = []
    for line in raw_lines:
        if not line:
            continue
        low = line.casefold()
        if is_noise_line(line):
            continue
        if low in DROP_LINES:
            continue
        if low in STOP_AT_LINES:
            break
        lines.append(line)
    return lines


def normalize_lines(text: str) -> str:
    lines = get_clean_lines(text)

    deduped: List[str] = []
    seen = set()
    for line in lines:
        key = line.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(line)
    return "\n".join(deduped)


def safe_filename(name: str) -> str:
    cleaned = re.sub(r'[\\/*?:"<>|]+', "", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().strip(".")
    return cleaned[:140] if cleaned else "scheme"


def unique_output_path(base_dir: Path, stem: str, suffix: str = ".json") -> Path:
    candidate = base_dir / f"{stem}{suffix}"
    idx = 2
    while candidate.exists():
        candidate = base_dir / f"{stem}_{idx}{suffix}"
        idx += 1
    return candidate


def pick_content_root(soup: BeautifulSoup):
    candidates = []
    for selector in ["main", "article", "#__next"]:
        node = soup.select_one(selector)
        if node:
            candidates.append(node)
    if not candidates:
        candidates = [soup.body or soup]
    return max(candidates, key=lambda n: len(n.get_text(" ", strip=True)))


def clean_root_for_text(root) -> None:
    for bad in root.select("script, style, noscript, header, nav, footer, form, svg"):
        bad.decompose()


def dedupe_lines(lines: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for line in lines:
        cleaned = clean_line(line)
        if not cleaned:
            continue
        low = cleaned.casefold()
        if low in seen:
            continue
        seen.add(low)
        out.append(cleaned)
    return out


def extract_lines_from_node(node) -> List[str]:
    if not node:
        return []
    lines = get_clean_lines(node.get_text("\n", strip=True))
    return dedupe_lines(lines)


def best_section_node(root, section_id: str) -> Optional[object]:
    # myScheme renders desktop and mobile blocks with same IDs. Pick richer one.
    candidates = root.select(f"#{section_id}")
    if not candidates:
        return None
    return max(candidates, key=lambda n: len(normalize_space(n.get_text(" ", strip=True))))


def parse_application_process(lines: List[str]) -> Dict[str, List[str]]:
    buckets: Dict[str, List[str]] = {
        "online": [],
        "offline": [],
        "unspecified": [],
    }
    current = "unspecified"
    for line in lines:
        low = line.casefold()
        if low in {"online", "offline"}:
            current = low
            continue
        if low.replace(" ", "") == "applicationprocess":
            continue
        buckets[current].append(line)

    return {k: dedupe_lines(v) for k, v in buckets.items() if v}


def extract_sections_from_dom(root) -> Dict[str, Any]:
    id_map = {
        "details": "details",
        "benefits": "benefits",
        "eligibility": "eligibility",
        "application-process": "application_process",
        "documents-required": "documents_required",
        "sources": "sources_and_references",
    }

    sections: Dict[str, Any] = {
        "details": [],
        "benefits": [],
        "eligibility": [],
        "application_process": {"unspecified": []},
        "documents_required": [],
        "sources_and_references": [],
    }

    for html_id, key in id_map.items():
        node = best_section_node(root, html_id)
        lines = extract_lines_from_node(node)
        # Remove section label if present as first line.
        if lines and lines[0].casefold().replace(" ", "") in {
            "details",
            "benefits",
            "eligibility",
            "applicationprocess",
            "documentsrequired",
            "sourcesandreferences",
        }:
            lines = lines[1:]
        if key == "application_process":
            sections[key] = parse_application_process(lines)
        else:
            sections[key] = lines

    return sections


def extract_faqs(root) -> List[Tuple[str, str]]:
    faq_lines: List[str] = []
    faq_headers = root.find_all(
        lambda tag: tag.name in {"h1", "h2", "h3", "h4", "h5", "h6"}
        and re.search(r"\b(faq|frequently asked questions)\b", tag.get_text(" ", strip=True), re.I)
    )

    for header in faq_headers:
        container = header.find_parent(["section", "article", "div"]) or header.parent
        if not container:
            continue

        for node in container.find_all(["dt", "dd", "h3", "h4", "button", "p", "li", "strong"]):
            text = clean_line(node.get_text(" ", strip=True))
            if not text:
                continue
            if is_noise_line(text):
                continue
            if text.casefold() in DROP_LINES:
                continue
            if re.search(r"\b(faq|frequently asked questions)\b", text, re.I):
                continue
            if len(text) < 3:
                continue
            faq_lines.append(text)

    # Fallback: if explicit FAQ block was not detected, collect question-like lines.
    if not faq_lines:
        for line in root.get_text("\n", strip=True).splitlines():
            text = clean_line(line)
            if is_noise_line(text):
                continue
            if text.casefold() in DROP_LINES:
                continue
            if text.endswith("?") and len(text) > 5:
                faq_lines.append(text)

    # Convert flat FAQ text lines into Q/A pairs.
    qas: List[Tuple[str, str]] = []
    current_q = ""
    current_a: List[str] = []
    for line in faq_lines:
        if line.endswith("?"):
            if current_q:
                qas.append((current_q, " ".join(current_a).strip()))
            current_q = line
            current_a = []
        elif current_q:
            current_a.append(line)

    if current_q:
        qas.append((current_q, " ".join(current_a).strip()))

    deduped_qas: List[Tuple[str, str]] = []
    seen = set()
    for q, a in qas:
        qn = normalize_space(q)
        an = normalize_space(a)
        if not qn:
            continue
        key = (qn.casefold(), an.casefold())
        if key in seen:
            continue
        seen.add(key)
        deduped_qas.append((qn, an))
    return deduped_qas


def extract_state_name(root, page_title: str) -> str:
    # State label usually appears as a short h3 right above scheme title.
    for node in root.select("h3, h4"):
        text = clean_line(node.get_text(" ", strip=True))
        if not text:
            continue
        low = text.casefold()
        if low == page_title.casefold():
            continue
        if low in DROP_LINES or low in STOP_AT_LINES or is_noise_line(text):
            continue
        if low in STATE_NAMES:
            return text
    return ""

def classify_location(name: str) -> Tuple[str, str]:
    if not name:
        return "", ""
    low = name.casefold()
    if low in UNION_TERRITORIES:
        return name, "union_territory"
    if low in STATE_NAMES:
        return name, "state"
    return name, ""


def extract_scheme_text(
    html: str,
) -> Tuple[str, str, List[Tuple[str, str]], str, Dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    title = normalize_space(soup.title.get_text()) if soup.title else ""

    root = pick_content_root(soup)
    clean_root_for_text(root)

    state_name = extract_state_name(root, title)
    full_text = normalize_lines(root.get_text("\n", strip=True))
    faqs = extract_faqs(root)
    sections = extract_sections_from_dom(root)
    return title, state_name, faqs, full_text, sections


def wait_for_content(page, timeout_ms: int) -> None:
    # myScheme is client-rendered; wait for section containers to appear.
    try:
        page.wait_for_selector(
            "#details, #benefits, #eligibility, #application-process, #documents-required",
            timeout=timeout_ms,
        )
    except PlaywrightTimeoutError:
        pass


def build_sections(full_text: str) -> Dict[str, Any]:
    sections: Dict[str, Any] = {
        "details": [],
        "benefits": [],
        "eligibility": [],
        "application_process": [],
        "documents_required": [],
        "sources_and_references": [],
    }
    lines = get_clean_lines(full_text)
    current = "details"
    for line in lines:
        key = SECTION_MAP.get(line.casefold())
        if key:
            if key == "faqs":
                current = "details"
                continue
            current = key
            continue
        # Keep only informative lines.
        if len(line) <= 2:
            continue
        sections[current].append(line)
    sections = rebalance_sections_if_needed(sections)
    sections["application_process"] = parse_application_process(sections["application_process"])
    return sections


def sections_are_effectively_empty(sections: Dict[str, Any]) -> bool:
    total = 0
    for key, value in sections.items():
        if key == "application_process" and isinstance(value, dict):
            total += sum(len(x) for x in value.values())
        elif isinstance(value, list):
            total += len(value)
    return total == 0


def rebalance_sections_if_needed(sections: Dict[str, Any]) -> Dict[str, Any]:
    # On myscheme pages, tab labels appear first and actual content follows.
    # In that case, naive section switching can dump all content into one bucket.
    non_source_count = sum(
        len(sections[k])
        for k in sections
        if k != "sources_and_references"
    )
    if non_source_count > 0:
        return sections

    lines = sections.get("sources_and_references", [])
    if not lines:
        return sections

    rebuilt: Dict[str, Any] = {
        "details": [],
        "benefits": [],
        "eligibility": [],
        "application_process": [],
        "documents_required": [],
        "sources_and_references": [],
    }

    # Remove lines that are already represented as FAQ entries.
    faq_questions = {x.casefold() for x in lines if x.endswith("?")}
    faq_mode = False

    current = "details"
    for line in lines:
        low = line.casefold()

        if low in DROP_LINES or low in STOP_AT_LINES or is_noise_line(line):
            continue

        # Ignore FAQ payload from section buckets; keep FAQ only under `faqs`.
        if low.endswith("?") or low in faq_questions:
            faq_mode = True
            continue
        if faq_mode:
            # First non-question after FAQ question is likely FAQ answer.
            # Skip these in section buckets.
            if line and not line.endswith("?"):
                continue

        # Strong section cues
        if "step 1" in low or low == "offline" or low == "online" or "where to apply" in low:
            current = "application_process"
        elif "applicant should" in low or "eligible" in low or "must be" in low:
            current = "eligibility"
        elif (
            "aadhaar" in low
            or "certificate" in low
            or "marksheet" in low
            or "passport-size" in low
            or "passport size" in low
            or "affidavit" in low
            or "identity card" in low
            or "school leaving" in low
            or "documents required" in low
        ):
            current = "documents_required"
        elif (
            "benefit" in low
            or "stipend" in low
            or "assistance" in low
            or "₹" in line
            or "rs." in low
        ):
            current = "benefits"
        elif "guidelines" in low or "http://" in low or "https://" in low or "source" in low:
            current = "sources_and_references"

        if len(line) <= 2:
            continue
        rebuilt[current].append(line)

    return rebuilt


def expand_faqs(page) -> None:
    # Expand common FAQ accordion triggers before extracting HTML.
    selectors = [
        "main button[aria-expanded='false']",
        "main [role='button'][aria-expanded='false']",
        "main div.cursor-pointer",
        "main button",
        "main [role='button']",
    ]

    for _ in range(3):
        clicked = 0
        for selector in selectors:
            loc = page.locator(selector)
            count = min(loc.count(), 400)
            for i in range(count):
                item = loc.nth(i)
                try:
                    text = normalize_space(item.inner_text())
                except Exception:
                    continue
                if not text:
                    continue
                # Avoid touching non-FAQ UI controls.
                if is_noise_line(text):
                    continue
                if not text.endswith("?") and "faq" not in text.casefold():
                    continue
                try:
                    item.click(timeout=700, force=True)
                    clicked += 1
                except Exception:
                    continue
        if clicked == 0:
            break
        page.wait_for_timeout(300)


def should_retry(page_title: str, sections: Dict[str, Any]) -> bool:
    if not page_title:
        return True
    if sections_are_effectively_empty(sections):
        return True
    return False


def iter_csv_rows(csv_path: Path) -> Iterable[Tuple[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = {normalize_space(x).lower(): x for x in (reader.fieldnames or [])}

        name_col = fields.get("scheme_name") or fields.get("name")
        url_col = fields.get("scheme_url") or fields.get("url") or fields.get("link")
        if not name_col or not url_col:
            raise ValueError(
                "CSV must have columns: scheme_name/scheme_url or name/link."
            )

        for row in reader:
            name = normalize_space(row.get(name_col, ""))
            url = normalize_space(row.get(url_col, ""))
            if name and url:
                yield name, url


def scrape_and_save(
    csv_path: Path,
    output_dir: Path,
    headless: bool,
    timeout_ms: int,
    limit: int = 0,
) -> None:
    if not csv_path.exists() and not csv_path.is_absolute():
        csv_path = CSV_DIR / csv_path
    output_dir = output_dir / csv_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    rows = list(iter_csv_rows(csv_path))
    if limit > 0:
        rows = rows[:limit]

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

        for idx, (scheme_name, scheme_url) in enumerate(rows, start=1):
            print(f"[{idx}/{len(rows)}] Scraping: {scheme_name}")
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
                continue

            location_name, location_type = classify_location(state_name)
            # Fallback only if DOM section extraction returned nothing useful.
            if sections_are_effectively_empty(sections):
                sections = build_sections(full_text)
            stem = safe_filename(scheme_name)
            out_file = unique_output_path(output_dir, stem, suffix=".json")

            payload = {
                "scheme_name": scheme_name,
                "scheme_url": scheme_url,
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
            page.wait_for_timeout(400)

        context.close()
        browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read schemes CSV and save scheme details + FAQs into JSON files."
    )
    # parser.add_argument("--csv", default="my_schemes_11.csv", help="Input CSV file path")
    # parser.add_argument("--output-dir", default="data", help="Folder for txt files")
    parser.add_argument(
        "--csv",
        default=str(CSV_DIR / "my_schemes_5.csv"),
        help="Input CSV file path",
    )
    parser.add_argument(
        "--output-dir",
        default=str(JSON_DIR),
        help="Folder for json output",
    )
    parser.add_argument("--timeout-ms", type=int, default=90000)
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows")
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run browser in headed mode for debugging",
    )
    args = parser.parse_args()

    scrape_and_save(
        csv_path=Path(args.csv),
        output_dir=Path(args.output_dir),
        headless=not args.headed,
        timeout_ms=args.timeout_ms,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
