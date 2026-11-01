from pathlib import Path

import pandas as pd

from .clean import load_data, deduplicate_traffic, quarantine_outliers


def generate_data_quality_report(
    raw_dir: Path = Path("data/raw"),
    report_path: Path = Path("reports/data_quality.md"),
) -> Path:
    traffic, _, _ = load_data(raw_dir)
    raw_rows = len(traffic)

    duplicate_candidates = traffic.duplicated(["road_id", "timestamp"]).sum()
    deduplicated = deduplicate_traffic(traffic)
    quarantined = quarantine_outliers(deduplicated)

    negative_volume = (deduplicated["traffic_volume"] < 0).sum()
    invalid_speed = (
        (deduplicated["avg_speed"] < 0) | (deduplicated["avg_speed"] > 200)
    ).sum()
    invalid_occupancy = (
        (deduplicated["occupancy"] < 0) | (deduplicated["occupancy"] > 100)
    ).sum()
    newly_quarantined_volume = (
        quarantined["traffic_volume"].isna() & deduplicated["traffic_volume"].notna()
    ).sum()
    newly_quarantined_speed = (
        quarantined["avg_speed"].isna() & deduplicated["avg_speed"].notna()
    ).sum()
    newly_quarantined_occupancy = (
        quarantined["occupancy"].isna() & deduplicated["occupancy"].notna()
    ).sum()

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = f"""# FlowCast data quality report

## Source traffic data

- Raw traffic rows: {raw_rows:,}
- Duplicate primary keys before cleaning: {int(duplicate_candidates):,}
- Rows after deduplication: {len(deduplicated):,}

## Quarantined values

- Negative traffic-volume values: {int(negative_volume):,}
- Invalid average-speed values outside 0–200 km/h: {int(invalid_speed):,}
- Invalid occupancy values outside 0–100%: {int(invalid_occupancy):,}
- Additional traffic-volume values quarantined: {int(newly_quarantined_volume):,}
- Additional average-speed values quarantined: {int(newly_quarantined_speed):,}
- Additional occupancy values quarantined: {int(newly_quarantined_occupancy):,}

## Final datasets

- Cleaned dataset: `data/interim/traffic_cleaned.parquet`
- Feature dataset: `data/processed/northline_corridor_processed.parquet`
"""
    report_path.write_text(report, encoding="utf-8")
    return report_path


if __name__ == "__main__":
    path = generate_data_quality_report()
    print(f"Generated {path}")
