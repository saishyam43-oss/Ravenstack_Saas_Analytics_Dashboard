# Source Code Overview

This folder contains all Python scripts used to generate, clean, transform, and analyze data for the Ravenstack SaaS Analytics project.

Scripts are designed to be run **sequentially**, reflecting the full analytics pipeline from raw data simulation to analysis-ready outputs.

---

## Execution Order

1. **00_create_messy_files.py**  
   Creates intentionally corrupted datasets to simulate real-world data quality issues.

2. **01_etl1.py**  
   Performs data cleaning, validation, and correction.  
   Outputs cleaned tables and data quality metrics to `data/etl1/`.

3. **02_etl2.py**  
   Applies feature engineering and prepares analysis-ready datasets.  
   Outputs feature-engineered tables and EDA inputs to `data/etl2/`.

4. **03_eda1.py**  
   Generates account-level, subscription-level, and cohort-level exploratory analysis outputs.

5. **04_eda2.py**  
   Produces advanced EDA outputs used for product usage, retention, and strategic opportunity analysis.

---

## Notes
- Scripts are modular and idempotent where possible.
- CSV outputs are persisted intentionally to support transparency and reproducibility.
- No SQL databases are used; all transformations are file-based.
