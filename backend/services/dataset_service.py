"""Dataset service for loading and querying How2Sign annotations."""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional

from backend.config import get_settings
from backend.schemas import DatasetSample

logger = logging.getLogger(__name__)


class DatasetService:
    """Service to load, index, and query How2Sign dataset samples and context pairs."""

    def __init__(self, search_directories: Optional[List[str]] = None):
        settings = get_settings()
        self.search_directories = search_directories or [
            r"c:\Users\manoj\Downloads\context_aware_howsign_30samples_with_csvs\ASL\data\annotations",
            r"c:\Users\manoj\Downloads",
            str(Path(settings.DATA_DIR).resolve() / "annotations"),
            str(Path(settings.DATA_DIR).resolve()),
            str(Path(__file__).resolve().parents[2] / "data" / "annotations"),
            str(Path(__file__).resolve().parents[2] / "data"),
        ]
        self._cache: Dict[str, List[DatasetSample]] = {}
        self._id_index: Dict[str, DatasetSample] = {}

    def _find_csv_for_split(self, split: str) -> Optional[Path]:
        """Search configured directories for a CSV file matching the split."""
        split_lower = split.lower()
        candidate_patterns = [
            f"how2sign_realigned_{split_lower}.csv",
            f"{split_lower}.csv",
        ]
        if split_lower == "val":
            candidate_patterns.extend(["how2sign_realigned_dev.csv", "dev.csv"])

        for directory in self.search_directories:
            dir_path = Path(directory)
            if not dir_path.is_dir():
                continue

            # Exact name matches first
            for pattern in candidate_patterns:
                target = dir_path / pattern
                if target.is_file():
                    return target

            # Glob pattern match
            try:
                for file_path in dir_path.glob("*.csv"):
                    name = file_path.name.lower()
                    if split_lower in name or (split_lower == "val" and "dev" in name):
                        return file_path
            except Exception as e:
                logger.debug("Error globbing directory %s: %s", dir_path, e)

        return None

    def _parse_csv_file(self, file_path: Path, split: str) -> List[DatasetSample]:
        """Parse a How2Sign annotation CSV/TSV file and build context pairs."""
        raw_rows = []
        try:
            # Detect encoding and delimiter
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                first_line = f.readline()
                delimiter = "\t" if "\t" in first_line else ","
                f.seek(0)
                reader = csv.DictReader(f, delimiter=delimiter)

                for row_idx, row in enumerate(reader):
                    # Clean and normalize keys
                    clean_row = {
                        k.strip().upper() if k else f"COL_{idx}": v.strip() if v else ""
                        for idx, (k, v) in enumerate(row.items())
                    }
                    clean_row["__ROW_IDX__"] = str(row_idx)
                    raw_rows.append(clean_row)
        except Exception as e:
            logger.error("Failed to read CSV file %s: %s", file_path, e)
            return []

        # Helper to extract value from multiple possible column keys
        def get_col_val(r: dict, prefixes: List[str], default: str = "") -> str:
            for p in prefixes:
                for k, v in r.items():
                    if k.startswith(p):
                        return v
            return default

        # Parse into temporary records
        records_by_video: Dict[str, List[dict]] = {}
        ordered_records: List[dict] = []

        for r in raw_rows:
            sentence_name = get_col_val(r, ["SENTENCE_NAME", "SAMPLE_ID", "ID"])
            video_id = get_col_val(r, ["VIDEO_ID", "VIDEO_NAME"])
            sentence_id = get_col_val(r, ["SENTENCE_ID", "SENTENCE_NUM"])
            sentence_text = get_col_val(r, ["SENTENCE", "TEXT", "TRANSCRIPT"])
            start_str = get_col_val(r, ["START_REALIGNED", "START_REALI", "START"])
            end_str = get_col_val(r, ["END_REALIGNED", "END_REALI", "END"])

            # Fallbacks if columns are missing
            if not sentence_name:
                sentence_name = f"sample_{r.get('__ROW_IDX__', '0')}"
            if not video_id:
                # Often sentence_name starts with video_id (e.g., '1a2b3c-5-rgb_front')
                video_id = sentence_name.split("-")[0] if "-" in sentence_name else "default_video"
            if not sentence_id:
                sentence_id = r.get("__ROW_IDX__", "0")

            try:
                start_val = float(start_str) if start_str else 0.0
            except ValueError:
                start_val = 0.0

            try:
                end_val = float(end_str) if end_str else 0.0
            except ValueError:
                end_val = 0.0

            rec = {
                "id": sentence_name,
                "video_id": video_id,
                "sentence_id": sentence_id,
                "sentence": sentence_text,
                "start": start_val,
                "end": end_val,
            }
            ordered_records.append(rec)
            records_by_video.setdefault(video_id, []).append(rec)

        # Build context pairs within each video sequence
        # Map record id to prior record in the same video
        context_map: Dict[str, Optional[dict]] = {}
        for vid, v_recs in records_by_video.items():
            # Ensure ordered by start time
            v_recs_sorted = sorted(v_recs, key=lambda x: x["start"])
            for i, item in enumerate(v_recs_sorted):
                if i > 0:
                    context_map[item["id"]] = v_recs_sorted[i - 1]
                else:
                    context_map[item["id"]] = None

        # Build final DatasetSample objects
        samples = []
        for rec in ordered_records:
            prior = context_map.get(rec["id"])
            sample = DatasetSample(
                id=rec["id"],
                video_id=rec["video_id"],
                sentence_id=rec["sentence_id"],
                sentence=rec["sentence"],
                start=rec["start"],
                end=rec["end"],
                prev_sentence=prior["sentence"] if prior else None,
                prev_sentence_id=prior["id"] if prior else None,
                split=split,
            )
            samples.append(sample)
            self._id_index[sample.id] = sample

        logger.info("Loaded %d samples for split '%s' from %s", len(samples), split, file_path)
        return samples

    def _ensure_split_loaded(self, split: str):
        """Ensure the specified split is parsed and cached."""
        split_key = split.lower()
        if split_key in self._cache:
            return

        csv_path = self._find_csv_for_split(split_key)
        if csv_path and csv_path.exists():
            self._cache[split_key] = self._parse_csv_file(csv_path, split_key)
        else:
            logger.warning("No CSV file found for split '%s'", split_key)
            self._cache[split_key] = []

    def get_samples(self, split: str = "train", limit: int = 20, offset: int = 0) -> List[DatasetSample]:
        """Return a paginated slice of samples for the specified split."""
        split_key = split.lower()
        self._ensure_split_loaded(split_key)
        all_samples = self._cache.get(split_key, [])
        return all_samples[offset : offset + limit]

    def get_total(self, split: str = "train") -> int:
        """Return total sample count in the specified split."""
        split_key = split.lower()
        self._ensure_split_loaded(split_key)
        return len(self._cache.get(split_key, []))

    def get_sample(self, sample_id: str) -> Optional[DatasetSample]:
        """Find a sample by ID across all loaded splits."""
        if sample_id in self._id_index:
            return self._id_index[sample_id]

        # If not indexed yet, load default splits
        for split in ["train", "val", "test"]:
            self._ensure_split_loaded(split)
            if sample_id in self._id_index:
                return self._id_index[sample_id]

        return None


# Global singleton instance
_dataset_service: Optional[DatasetService] = None


def get_dataset_service() -> DatasetService:
    """Return singleton instance of DatasetService."""
    global _dataset_service
    if _dataset_service is None:
        _dataset_service = DatasetService()
    return _dataset_service
