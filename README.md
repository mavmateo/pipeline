# ETL pipeline implemented with the medallion architecture

data_pipeline/
├── configs/settings.yaml     # All config in one place
├── src/
│   ├── config.py             # Pydantic-validated config loader
│   ├── logger.py             # Centralised rotating file logger
│   ├── validators.py         # Schema + business rule checks
│   ├── cleaners.py           # Pure cleaning functions
│   ├── transformers.py       # Business-logic enrichment
│   └── pipeline.py           # Orchestrator
├── tests/
│   ├── test_cleaners.py
│   └── test_validators.py
└── main.py                   # CLI entrypoint



## Selecting a medallion level

The pipeline runs medallion layers cumulatively:

- `bronze`: normalize and deduplicate column names, apply column aliases, normalize
  Unicode/control characters, trim strings, standardize source null tokens, and add
  ingestion metadata.
- `silver`: run bronze, then parse JSON columns, normalize currency/phone/category
  values, drop unusable columns, deduplicate rows while ignoring metadata, normalize
  emails, resolve configured business-key duplicates, remove PII, cast types, normalize
  datetime timezones, apply numeric range rules, add missingness flags, quarantine rows
  missing required fields, clip outliers, and impute missing values.
- `gold`: run bronze and silver, then add analytics-ready fields including tenure,
  age buckets, high-value flags, lifetime value, RFM metrics, customer segments,
  period aggregates, dimension keys, data quality scores, and cohort features.

Examples:

```bash
python main.py --input data/raw/customers.csv --medallion-level bronze
python main.py --input data/raw/customers.csv --medallion-level silver
python main.py --input data/raw/customers.csv --medallion-level gold
```

When `--output` is omitted, files are written under the configured medallion path:
`data/bronze/`, `data/silver/`, or `data/gold/`. The default level is configured in
`configs/settings.yaml` and currently remains `gold` to preserve the original full
pipeline behavior.