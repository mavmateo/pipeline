# pipeline

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



