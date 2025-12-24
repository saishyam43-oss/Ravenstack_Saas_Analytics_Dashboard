# Data Directory Overview

This folder contains all datasets used in the Ravenstack SaaS Analytics project, organized by data lifecycle stage to ensure transparency, traceability, and reproducibility.

---

## Folder Structure

### raw/
Source-level data before any transformations.

- **original/**  
  Original datasets as received. These files are immutable and never modified.

- **messy/**  
  Intentionally corrupted versions of the original data, used to simulate real-world data quality issues for ETL validation.

- **merged/**  
  Unified raw datasets aligned to a consistent schema before cleaning.

These files represent the starting point of the pipeline.

---

### etl1/ – Data Cleaning & Validation Outputs
Outputs from **ETL1**, focused on data quality correction and validation.

Includes:
- Cleaned tables (`*_clean.csv`)
- Data quality summaries and validation reports
- Dirty row flags and correction logs
- Column-level and table-level quality metrics

ETL1 ensures the data is **accurate, consistent, and analysis-ready** before feature engineering.

---

### etl2/ – Feature Engineering & Analysis-Ready Data
Outputs from **ETL2**, focused on feature engineering and exploratory analysis.

Key components:
- **Feature-engineered tables** (`*_fe.csv`)
- **EDA outputs** grouped by analytical theme:
  - `accounts/`
  - `subscriptions/`
  - `cohorts/`
  - `features/`
  - `strategy/`
- **Diagnostics/**  
  Deep validation checks used to verify churn logic, date consistency, and edge cases

These datasets directly feed the dashboards and business analysis.

---

## Data Usage Notes
- No SQL databases were used in this project.
- Initial data inspection and cleaning summaries were validated using **Excel**.
- All transformations and feature engineering were implemented using **Python**.
- CSV outputs are intentionally retained to allow easy inspection without running code.

---

## Dashboard Lineage
The final dashboards are built using:
- **Excel** → Data Cleaning Summary
- **Tableau** → Growth, Retention, Product Stickiness, and Strategic Opportunity dashboards

All dashboard metrics can be traced back to files within `etl1/` and `etl2/`.

---

## Purpose
This structure is designed to:
- Make data lineage explicit
- Separate cleaning from feature engineering
- Support reproducibility and auditability
- Reflect real-world analytics workflows
