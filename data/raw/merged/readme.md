\# RavenStack Merged Datasets (Raw + Messy)



This folder contains the merged versions of the RavenStack SaaS datasets, where the original "clean" tables have been appended with the corresponding `\*\_MESSY\_ONLY.csv` rows to simulate more realistic, production-like data quality issues.



The goal of these merged files is to provide a single source per table that you can use for:

\- Data cleaning and preprocessing practice.

\- Building and testing data quality rules.

\- Designing robust analytics and BI models that tolerate imperfect data.



---



\## Files in This Folder



Each merged file combines:

\- The original base table (clean RavenStack dataset).

\- The corresponding `\*\_MESSY\_ONLY.csv` file with injected anomalies.



Expected files:



\- `ravenstack\_accounts\_MERGED.csv`

\- `ravenstack\_subscriptions\_MERGED.csv`

\- `ravenstack\_feature\_usage\_MERGED.csv`

\- `ravenstack\_support\_tickets\_MERGED.csv`

\- `ravenstack\_churn\_events\_MERGED.csv`



All merged tables retain the original schema. \*\*Note:\*\* These files do not include a `messiness\_description` column—the messy rows are blended directly with clean data, requiring you to identify and diagnose issues through data profiling and validation logic.



---



\## Injected Data Quality Issues by Table



Below is a reference for the types of issues present in approximately 10% of rows in each merged dataset. These are intentionally injected anomalies designed to test your data cleaning and validation skills.



\### accounts\_MERGED



\- Missing values in `industry` or `country` fields.

\- Inconsistent `churn\_flag` values that conflict with subscription activity.



\### subscriptions\_MERGED



\- Missing `end\_date` for subscriptions that may have actually ended.

\- Null values in `plan\_tier`, `mrr\_amount`, or `arr\_amount`.

\- Inverted `churn\_flag` values creating lifecycle contradictions.



\### feature\_usage\_MERGED



\- Negative values in `usage\_count` or `usage\_duration\_secs` (should always be non-negative).

\- Null values in `feature\_name` despite having usage records.



\### support\_tickets\_MERGED



\- Missing `priority` classification.

\- Null `satisfaction\_score` values beyond expected missing feedback.

\- Negative `resolution\_time\_hours` (impossible in real scenarios).



\### churn\_events\_MERGED



\- Missing `reason\_code` for churn records.

\- Null `feedback\_text` where customer feedback should exist.

\- Negative `refund\_amount\_usd` (refunds should be zero or positive).



---



\## How to Use These Merged Files



Typical workflows you can practice with the merged datasets:



\- \*\*Data profiling:\*\*

&nbsp; - Calculate null percentages for each column.

&nbsp; - Identify negative values in numeric fields that should be non-negative.

&nbsp; - Check for logical inconsistencies (e.g., churn\_flag conflicts, missing mandatory fields).



\- \*\*Cleaning \& transformation:\*\*

&nbsp; - Decide on strategies: impute nulls, drop invalid rows, or flag for manual review.

&nbsp; - Correct negative values or treat them as missing data.

&nbsp; - Reconcile inconsistent flags by cross-referencing with related tables.



\- \*\*Quality metrics \& monitoring:\*\*

&nbsp; - Track % of rows with nulls, negatives, or logical conflicts.

&nbsp; - Build validation rules that alert when key business constraints are violated.

&nbsp; - Create data quality dashboards showing issue distribution by segment.



\- \*\*BI / Analytics:\*\*

&nbsp; - Build SQL views or Power BI models that explicitly handle or filter dirty data.

&nbsp; - Compare metrics on "assumed clean" vs "validated clean" subsets to measure data quality impact.

&nbsp; - Document data quality assumptions and limitations in your analysis.



---



\## Best Practices for Working with Dirty Data



1\. \*\*Profile first:\*\* Before cleaning, understand the extent and patterns of issues.

2\. \*\*Document decisions:\*\* Keep a log of how you handled nulls, negatives, and inconsistencies.

3\. \*\*Validate relationships:\*\* Check foreign key integrity and cross-table consistency.

4\. \*\*Test transformations:\*\* Verify that your cleaning logic doesn't accidentally remove valid edge cases.

5\. \*\*Version your data:\*\* Keep both raw merged and cleaned versions for reproducibility.



---



\## Upstream Source



These merged datasets are derived from the original RavenStack synthetic SaaS dataset by River @ Rivalytics, with an additional messiness layer added for educational and portfolio purposes.



Please ensure that when you publish analyses or projects using this data, you still credit the original dataset author:



\*\*RavenStack Synthetic SaaS Dataset – River @ Rivalytics\*\*



