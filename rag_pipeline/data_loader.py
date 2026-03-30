"""
Data loader for government scheme JSON files.
Loads, validates, and normalizes scheme data.
"""
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from config import (
    CATEGORY_MAP,
    CATEGORY_NAME_TO_ID,
    CENTRAL_GOVT_LABEL,
    DATA_DIR,
    DEFAULT_LANGUAGE,
    LOCATION_TYPE_CENTRAL,
)


@dataclass
class SchemeData:
    """Normalized scheme data structure."""
    
    scheme_id: str
    scheme_name: str
    scheme_url: str
    location_type: str
    location_name: str
    category_id: int
    category_name: str
    page_title: str
    scraped_at: str
    language: str
    sections: Dict[str, Any]
    faqs: List[Dict[str, str]]
    source_file: str
    
    @classmethod
    def from_json(cls, data: Dict[str, Any], source_file: Path) -> "SchemeData":
        """Create SchemeData from raw JSON dict."""
        
        # Generate unique ID
        scheme_id = generate_scheme_id(
            data.get("scheme_name", ""),
            data.get("scheme_url", "")
        )
        
        # Normalize location
        location_name = data.get("location_name", "").strip()
        location_type = data.get("location_type", "").strip()
        
        if not location_name or not location_type:
            location_name = CENTRAL_GOVT_LABEL
            location_type = LOCATION_TYPE_CENTRAL
        
        # Extract category from folder name
        folder_name = source_file.parent.name
        category_id, category_name = parse_category_from_folder(folder_name)
        
        return cls(
            scheme_id=scheme_id,
            scheme_name=data.get("scheme_name", "").strip(),
            scheme_url=data.get("scheme_url", "").strip(),
            location_type=location_type,
            location_name=location_name,
            category_id=category_id,
            category_name=category_name,
            page_title=data.get("page_title", "").strip(),
            scraped_at=data.get("scraped_at", ""),
            language=DEFAULT_LANGUAGE,
            sections=data.get("sections", {}),
            faqs=data.get("faqs", []),
            source_file=str(source_file),
        )
    
    def to_metadata_base(self) -> Dict[str, Any]:
        """Return base metadata dict (without chunk-specific fields)."""
        return {
            "scheme_id": self.scheme_id,
            "scheme_name": self.scheme_name,
            "scheme_url": self.scheme_url,
            "location_type": self.location_type,
            "location_name": self.location_name,
            "category_id": self.category_id,
            "category_name": self.category_name,
            "language": self.language,
            "scraped_at": self.scraped_at,
        }


def generate_scheme_id(scheme_name: str, scheme_url: str) -> str:
    """Generate a unique scheme ID from name and URL."""
    combined = f"{scheme_name}|{scheme_url}"
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def parse_category_from_folder(folder_name: str) -> tuple[int, str]:
    """
    Parse category ID and name from folder name.
    E.g., "4_education_learning" -> (4, "education_learning")
    """
    if "_" in folder_name:
        parts = folder_name.split("_", 1)
        if parts[0].isdigit():
            category_id = int(parts[0])
            category_name = parts[1]
            return category_id, category_name
    
    # Fallback: try to match by name
    for cat_id, cat_name in CATEGORY_MAP.items():
        if cat_name in folder_name:
            return cat_id, cat_name
    
    return 0, "unknown"


def load_scheme_json(file_path: Path) -> Optional[Dict[str, Any]]:
    """Load and parse a single JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"Error loading {file_path}: {e}")
        return None


def iter_scheme_files(data_dir: Path = DATA_DIR) -> Generator[Path, None, None]:
    """Iterate over all JSON files in the data directory."""
    for json_file in data_dir.rglob("*.json"):
        yield json_file


def load_all_schemes(
    data_dir: Path = DATA_DIR,
    verbose: bool = True
) -> List[SchemeData]:
    """
    Load all scheme JSON files and return normalized SchemeData objects.
    
    Args:
        data_dir: Directory containing scheme JSON files
        verbose: Print progress information
    
    Returns:
        List of SchemeData objects
    """
    schemes = []
    errors = []
    
    json_files = list(iter_scheme_files(data_dir))
    total = len(json_files)
    
    if verbose:
        print(f"Loading {total} scheme files from {data_dir}...")
    
    for idx, json_file in enumerate(json_files):
        if verbose and (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{total} files...")
        
        data = load_scheme_json(json_file)
        if data is None:
            errors.append(str(json_file))
            continue
        
        try:
            scheme = SchemeData.from_json(data, json_file)
            schemes.append(scheme)
        except Exception as e:
            errors.append(f"{json_file}: {e}")
    
    if verbose:
        print(f"\nLoaded {len(schemes)} schemes successfully.")
        if errors:
            print(f"Errors: {len(errors)} files failed to load.")
            for err in errors[:5]:
                print(f"  - {err}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more")
    
    return schemes


def get_scheme_stats(schemes: List[SchemeData]) -> Dict[str, Any]:
    """Generate statistics about loaded schemes."""
    from collections import Counter
    
    location_types = Counter(s.location_type for s in schemes)
    locations = Counter(s.location_name for s in schemes)
    categories = Counter(s.category_name for s in schemes)
    
    return {
        "total_schemes": len(schemes),
        "location_types": dict(location_types),
        "top_locations": dict(locations.most_common(10)),
        "categories": dict(categories),
    }


if __name__ == "__main__":
    # Test the data loader
    schemes = load_all_schemes(verbose=True)
    
    if schemes:
        stats = get_scheme_stats(schemes)
        print("\n=== Statistics ===")
        print(f"Total schemes: {stats['total_schemes']}")
        print(f"\nLocation types: {stats['location_types']}")
        print(f"\nTop 10 locations: {stats['top_locations']}")
        print(f"\nCategories: {stats['categories']}")
        
        # Print sample scheme
        print("\n=== Sample Scheme ===")
        sample = schemes[0]
        print(f"Name: {sample.scheme_name}")
        print(f"ID: {sample.scheme_id}")
        print(f"Location: {sample.location_name} ({sample.location_type})")
        print(f"Category: {sample.category_name} (ID: {sample.category_id})")
        print(f"Sections: {list(sample.sections.keys())}")
        print(f"FAQs: {len(sample.faqs)}")
