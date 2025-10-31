import json
import sqlite3
from pathlib import Path
from typing import Dict

import pytest

from photo_sorter import PhotoSorter, SorterConfig


class StubOllamaClient:
    def __init__(self, responses: Dict[str, Dict]):
        self._responses = responses

    def analyze_photo(self, image_path: Path, *, metadata=None, neighbor=None) -> Dict:
        return json.loads(json.dumps(self._responses[image_path.name]))


@pytest.fixture()
def sample_analysis() -> Dict[str, Dict]:
    return {
        "contact.jpg": {
            "category": "business",
            "confidence": 0.92,
            "business_insights": {
                "type": "contacts",
                "summary": "Two people exchanging business cards near a booth.",
                "slide_description": None,
                "contacts": [
                    {
                        "name": "Jordan Smith",
                        "role": "Partnerships Lead",
                        "company": "TechNova",
                        "email": "jordan@technova.example",
                        "phone": "+1-555-0100",
                        "website": "https://technova.example",
                    }
                ],
                "startup_profiles": [],
            },
            "sightseeing_details": {
                "location_guess": None,
                "visual_elements": [],
                "vibe": None,
            },
            "notable_text": ["TechNova"],
            "safety_checks": {
                "contains_sensitive_info": False,
                "notes": None,
            },
            "_raw_response": '{"category":"business"}',
        },
        "skyline.jpg": {
            "category": "sightseeing",
            "confidence": 0.87,
            "business_insights": {
                "type": None,
                "summary": "",
                "slide_description": None,
                "contacts": [],
                "startup_profiles": [],
            },
            "sightseeing_details": {
                "location_guess": "San Francisco skyline",
                "visual_elements": ["skyscrapers", "evening sky"],
                "vibe": "Relaxed",
            },
            "notable_text": [],
            "safety_checks": {
                "contains_sensitive_info": False,
                "notes": None,
            },
            "_raw_response": '{"category":"sightseeing"}',
        },
    }


def test_photo_sorter_moves_and_logs(tmp_path: Path, sample_analysis: Dict[str, Dict]) -> None:
    source_dir = tmp_path / "img"
    dest_dir = tmp_path / "sorted"
    db_path = tmp_path / "photo.sqlite"
    source_dir.mkdir()

    contact_photo = source_dir / "contact.jpg"
    skyline_photo = source_dir / "skyline.jpg"
    contact_photo.write_bytes(b"\xFF\xD8\xFF\xE0")
    skyline_photo.write_bytes(b"\xFF\xD8\xFF\xE0")

    config = SorterConfig(source_dir=source_dir, output_dir=dest_dir, database=db_path)
    sorter = PhotoSorter(config, dry_run=False)
    sorter._ollama = StubOllamaClient(sample_analysis)  # type: ignore[assignment]

    try:
        sorter.run()
    finally:
        sorter.close()

    business_target = dest_dir / "business" / "contacts" / "contact.jpg"
    sightseeing_target = dest_dir / "sightseeing" / "skyline.jpg"

    assert business_target.exists(), "Business photo should be moved into contacts folder."
    assert sightseeing_target.exists(), "Sightseeing photo should be moved into sightseeing folder."

    with sqlite3.connect(db_path) as conn:
        rows = {
            row[1]: (row[2], row[3], row[4])
            for row in conn.execute(
                "SELECT file_name, category, business_type, destination_path FROM photo_analysis"
            )
        }

    assert rows["contact.jpg"][0] == "business"
    assert rows["contact.jpg"][1] == "contacts"
    assert Path(rows["contact.jpg"][2]) == business_target
    assert rows["skyline.jpg"][0] == "sightseeing"
    assert rows["skyline.jpg"][1] is None
    assert Path(rows["skyline.jpg"][2]) == sightseeing_target
