\## Additional Messiness Layer (Data Quality Scenarios)





For data cleaning, validation, and anomaly-detection practice, a “messy” layer has been added by sampling at least 10% of rows from each table and injecting controlled inconsistencies. These rows are saved separately in files named `\*\_MESSY\_ONLY.csv`, each including a `messiness\_description` column that documents the issues applied to that row.



\- accounts\_MESSY\_ONLY.csv  

&nbsp; - Possible issues (per row):  

&nbsp;   - `null\_industry` or `null\_country` – missing segmentation or geography values.  

&nbsp;   - `inconsistent\_churn\_flag` – churn flag flipped to create logical conflicts with related activity.



\- subscriptions\_MESSY\_ONLY.csv  

&nbsp; - Possible issues (per row):  

&nbsp;   - `null\_end\_date`, `null\_plan\_tier`, `null\_mrr\_amount`, or `null\_arr\_amount` – incomplete subscription lifecycle or billing fields.  

&nbsp;   - `flipped\_churn\_flag` – churn status inverted to create contradictions with dates and revenue.



\- feature\_usage\_MESSY\_ONLY.csv  

&nbsp; - Possible issues (per row):  

&nbsp;   - `negative\_usage\_count` or `negative\_usage\_duration\_secs` – invalid usage volumes or durations.  

&nbsp;   - `null\_feature\_name` – usage records with missing feature identifiers.



\- support\_tickets\_MESSY\_ONLY.csv  

&nbsp; - Possible issues (per row):  

&nbsp;   - `null\_priority` – tickets without a severity classification.  

&nbsp;   - `null\_satisfaction\_score` – missing CSAT values beyond the original null patterns.  

&nbsp;   - `negative\_resolution\_time` – tickets whose resolution\_time\_hours is negative.



\- churn\_events\_MESSY\_ONLY.csv  

&nbsp; - Possible issues (per row):  

&nbsp;   - `null\_reason\_code` – churn records without a categorical reason.  

&nbsp;   - `null\_feedback\_text` – missing free-text feedback.  

&nbsp;   - `negative\_refund` – refund\_amount\_usd set to a negative value.



Recommended usage:



\- Append each `\*\_MESSY\_ONLY.csv` file into its corresponding base table when you want to simulate dirty production data.  

\- Use `messiness\_description` to filter, profile, and validate cleaning logic in SQL, Python, or BI tools.  

\- Build tests that reconcile business rules (e.g., negative durations, inconsistent churn flags, missing keys for segmentation) against these injected anomalies.





---

