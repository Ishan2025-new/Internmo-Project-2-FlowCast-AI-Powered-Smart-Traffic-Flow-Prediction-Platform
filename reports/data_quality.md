# FlowCast data quality report

## Source traffic data

- Raw traffic rows: 178,468
- Duplicate primary keys before cleaning: 1,767
- Rows after deduplication: 176,701

## Quarantined values

- Negative traffic-volume values: 239
- Invalid average-speed values outside 0–200 km/h: 236
- Invalid occupancy values outside 0–100%: 229
- Additional traffic-volume values quarantined: 268
- Additional average-speed values quarantined: 236
- Additional occupancy values quarantined: 229

## Final datasets

- Cleaned dataset: `data/interim/traffic_cleaned.parquet`
- Feature dataset: `data/processed/northline_corridor_processed.parquet`
