<h1 align="center"> Ravenstack SaaS Analytics: Growth, Retention & Strategic Focus </h1>

> *This project evaluates whether Ravenstack’s SaaS growth is structurally durable and forces a strategic choice between volume-led acquisition and Enterprise-led retention.*

![Status: Diagnosis](https://img.shields.io/badge/Status-Diagnosis_Complete-success) ![Focus: Retention](https://img.shields.io/badge/Focus-Retention_%26_Growth-blue) ![Outcome: Strategy_Pivot](https://img.shields.io/badge/Outcome-Strategy_Pivot-orange)

---

<p align="center">
  <img src="logo.png" width="200"/>
</p>

---

## 🧭 Decision Summary

**Primary Decision:** Pivot Ravenstack from volume-led acquisition to Enterprise-led, retention-first growth.

**Why:** Acquisition efficiency masks a structural retention failure, ~66% of accounts churn before delivering durable value.

**Core Insight:** Churn is front-loaded and driven by delayed time-to-value and unfocused product usage, not long-term dissatisfaction.

**Strategic Trade-off:** Accept slower top-line growth by deprioritizing SMB volume to protect long-term revenue durability.

**Primary Action:** Re-center Product, Growth, and Customer Success around Enterprise onboarding, activation, and expansion.

**Owner:** Product Analytics / Growth Strategy

---

## ⚡ Executive Snapshot

**The Problem**  
At a surface level, **Ravenstack appears healthy**, with strong MRR growth and high acquisition efficiency (Quick Ratio ≈ 3.0). However, this growth masks a **structural imbalance**: approximately **66% of acquired accounts eventually churn**, preventing acquisition-led growth from compounding into durable long-term value.

**The Diagnosis**  
Retention analysis shows that **subscription disengagement begins early**, often before users experience meaningful value. Product usage patterns explain why, users explore broadly across features but fail to anchor on a clear value driver, resulting in **flat engagement and delayed time-to-value**. As a result, revenue durability increasingly depends on a small subset of Enterprise customers.

**The Solution**  
Ravenstack must shift from **volume-led acquisition** to **retention-first, value-led growth**. This requires accelerating time-to-value, reducing product complexity during onboarding, and deliberately prioritizing **Enterprise customers**, where revenue durability and operational efficiency compound over time.

---

## 🏢 Client Background & Project Context

**Client:** Ravenstack (fictional SaaS company)  
**Business Model:** Subscription-based B2B SaaS

As the business enters a scaling phase, leadership requires deeper visibility into:
- Whether revenue growth is sustainable
- Where and why customers churn
- Whether product usage translates into long-term value
- Which customer segments truly justify continued investment

### Key Stakeholders
- **Product Leadership** – responsible for onboarding experience and feature adoption  
- **Growth & Marketing** – focused on acquisition efficiency and funnel performance  
- **Customer Success** – accountable for churn reduction and expansion  
- **Revenue & Finance** – concerned with MRR stability, LTV, and concentration risk  

### Context
Ravenstack is a fictional **B2B subscription-based SaaS company** operating across multiple plan tiers (SMB, Mid-Market, Enterprise).  
As the business enters a **scaling phase**, leadership faces growing **churn risk** and requires cross-functional visibility across **Product, Growth, Customer Success, and Revenue teams**.

This analysis evaluates whether Ravenstack’s growth is **structurally sustainable** and identifies **levers to stabilize long-term revenue**.

---

## 🎯 Business Problem & Objective

### Business Problem
Despite strong acquisition efficiency, Ravenstack faces:
- High lifetime account attrition
- Weak retention durability
- Revenue concentration risk

### Objectives
- Assess whether growth efficiency translates into long-term value
- Identify **when** customers disengage
- Diagnose **why** customers fail to retain or expand
- Define a **strategic path forward** to improve revenue durability

---

## ⭐ North Star Metrics & Analytical Focus

The analysis is anchored around a focused set of North Star metrics that collectively describe growth health, retention strength, and customer value:

- **Total Monthly Recurring Revenue (MRR)**  
  Measures business scale and revenue concentration risk.

- **Quick Ratio**  
  Evaluates growth efficiency by comparing revenue gains to churn and contraction.

- **Churn Rate (Lifetime & Early-Stage)**  
  Identifies whether acquisition efforts translate into durable customers.

- **Net Revenue Retention (NRR)**  
  Assesses expansion, contraction, and long-term revenue stability.

- **Time-to-Value**  
  Measures how quickly users realize meaningful product value.

- **ARPU & LTV (Proxy)**  
  Used to compare customer segment value and prioritize investment.

These metrics guide every dashboard and insight in the project.

---

## 🧭 Analytical Approach

The analysis follows a **top-down diagnostic framework**, moving from growth outcomes to root causes:

1. **Growth Efficiency Analysis**  
   Evaluates whether revenue gains outpace churn and contraction.

2. **Retention & Cohort Analysis**  
   Identifies when customers churn and how retention patterns evolve over time.

3. **Product Usage & Time-to-Value Analysis**  
   Assesses whether users meaningfully engage with the product before churning.

4. **Segment-Level Value & Efficiency Analysis**  
   Compares customer segments by lifetime value relative to operational cost.

**Note:** Net Revenue Retention (NRR) and cohort-based retention metrics were computed using Python  
(see [`src/03_eda1.py`](src/03_eda1.py) and [`src/04_eda2.py`](src/04_eda2.py) for cohort and retention logic).

This structured approach ensures insights are **diagnostic and causal, not merely descriptive**.

---

## 📊 Executive Summary (North Star View)

<p align="center">
  <img src="dashboards/01_the_north_star.png"/>
</p>

### Key Insights
- **Efficient but peaking acquisition:** Quick Ratio remains strong (>3.0) but has normalized as scale increased, suggesting diminishing marginal returns from acquisition alone.
- **High lifetime account attrition (≈66%):** A majority of acquired accounts eventually exit, preventing acquisition-led growth from compounding into durable value.
- **Revenue concentration risk:** A disproportionate share of revenue is driven by a small subset of Enterprise customers, increasing downside exposure.
- **Asymmetric value creation:** Enterprise accounts generate significantly higher revenue per interaction, indicating that not all customers contribute equally to growth quality.

### Executive Takeaway
Growth is **efficient but fragile**. Without retention improvements and customer mix rebalancing, continued acquisition will amplify churn faster than durable value.

---

## 🪣 Retention Audit: The “Leaky Bucket” (Churn Is Front-Loaded)

> *Following the North Star assessment of fragile growth, this section examines **when** customers disengage during the subscription lifecycle.*

---

<p align="center">
  <img src="dashboards/02_the_leaky_bucket.png"/>
</p>

### Business Question
Are customers exiting gradually over time, or is disengagement concentrated at specific lifecycle stages?

### Key Insights
- **Subscription exits are front-loaded:** Disengagement is most pronounced early in the subscription lifecycle, before long-term value is realized.
- **Early exits limit compounding:** Although exit likelihood declines over time, early losses prevent customers from ever contributing durable revenue.
- **Cohort quality is deteriorating:** Newer cohorts degrade ~2× faster than older cohorts, indicating that recent growth came at the cost of retention quality.
- **Revenue decay persists:** Net Revenue Retention declines steadily, showing limited expansion even among retained customers.

### Why This Matters
Early subscription exits signal **expectation mismatch or onboarding gaps**, while poor NRR ensures that even retained customers fail to compound value.

### Action Plan
- **Fix early onboarding clarity (Days 0–7):** Reduce expectation mismatch by clarifying value propositions and guiding first successful actions.
- **Close the early value gap (Days 8–30):** Ensure users experience tangible value before trial or contract expiration.
- **Improve cohort quality:** Reassess acquisition channels and ICP alignment contributing low-retention cohorts.
- **Retention target:** Reduce early subscription exits (<30 days) to below 10% and stabilize post-onboarding retention.

---

## ⏱️ Product Stickiness: The “Time-to-Value” Crisis (Value Arrives Too Late)

> *After identifying **when** users disengage, this section examines **why** retention fails at the product level.*

---

<p align="center">
  <img src="dashboards/03_the_stickiness.png"/>
</p>

### Key Insights
- **Severe time-to-value gap:** Average time-to-first-value (~76 days) far exceeds early disengagement timelines.
- **Flat engagement curve:** Usage intensity does not deepen over time, indicating a lack of compounding product stickiness.
- **Broad exploration, higher attrition:** Accounts with wider feature usage exhibit higher lifetime attrition, suggesting exploration without value anchoring.
- **No dominant value driver:** No single feature acts as a clear “hero” that consistently drives retention.

### Interpretation
Broad exploration reflects **cognitive overload**, not product stickiness. Users fail to internalize a clear “aha” moment before disengaging.

### Strategic Implications
- **Reduce cognitive load:** Narrow early workflows to focus users on a small set of high-impact actions.
- **Create a guided value path:** Replace feature discovery with opinionated onboarding that leads to a clear “aha” moment.
- **Accelerate activation:** Compress time-to-value from weeks to days to prevent disengagement before value realization.

---

## 🐋 Strategic Opportunity: The “Whale Hunt” (Pivoting from Volume to Value)

> *With root causes identified, this section evaluates **where** Ravenstack should focus to maximize durable growth.*

---

<p align="center">
  <img src="dashboards/04_the_opportunity.png"/>
</p>

### Key Insights
- **Enterprise drives disproportionate value:** ~22% of accounts generate ~47% of total revenue.
- **Superior revenue durability:** Enterprise customers show a ~30% Net Revenue Retention advantage over SMB.
- **Operational efficiency advantage:** Revenue per support interaction is orders-of-magnitude higher for Enterprise customers.
- **SMB drag on growth quality:** SMB volume contributes disproportionately to churn and operational overhead.

### Strategic Decision
**Pivot from volume-led growth to value-led expansion.**

### Recommended Actions
- **Rebalance growth strategy:** Prioritize Enterprise acquisition, retention, and expansion over pure volume growth.
- **Segment operating model:** Shift SMB and Mid-Market toward low-touch or self-serve experiences.
- **Align teams around durability:** Focus Product, CS, and GTM efforts on customers where revenue compounds.
- **NRR objective:** Stabilize and grow Net Revenue Retention through expansion-led Enterprise growth.

---

## 🔗 Cross-Dashboard Narrative: From Growth to Strategy

Taken together, the four dashboards reveal a consistent story.

Ravenstack’s **acquisition engine is efficient**, as reflected by a strong Quick Ratio and growing MRR. However, this efficiency masks a deeper issue: **customers disengage or exit before they realize meaningful value**, causing long-term revenue leakage.

Retention analysis shows that churn is **front-loaded**, concentrated in the first few weeks of the customer lifecycle. Product usage data explains why: users are overwhelmed by feature breadth and fail to reach meaningful value quickly. As a result, engagement remains shallow and time-to-value exceeds time-to-disengagement by a wide margin.

Segment-level analysis resolves the strategic tension. While SMB customers drive volume, **Enterprise customers deliver durable revenue with far greater efficiency**. Treating all segments equally has diluted focus and increased operational cost.

The implication is clear: Ravenstack’s challenge is not growth, but **growth quality**. Solving retention and prioritizing high-value segments offers a far higher return than accelerating acquisition.

---

## 💡 Key Business Insights (Consolidated)

- Growth is **efficient but fragile**, with churn offsetting acquisition gains over time.
- Retention failures **begin early and compound over time**, driven by onboarding gaps and poor expansion rather than long-term dissatisfaction.
- Product usage does not naturally deepen over time, confirming a **time-to-value gap**.
- Feature breadth without guidance increases churn instead of retention.
- Enterprise customers generate outsized value relative to operational effort.
- SMB growth adds volume but introduces disproportionate cost and revenue risk.

---

## 🧭 The Decision This Analysis Forces

Ravenstack cannot simultaneously optimize for high-volume SMB acquisition and durable revenue growth.

This analysis forces a clear strategic choice.

---

### 1️⃣ Stop Optimizing for Volume-Led Growth

SMB acquisition drives top-line growth but contributes disproportionately to early churn, operational load, and weak Net Revenue Retention.

**Decision:**  
Deprioritize SMB volume as a primary growth lever.

**Explicit Sacrifice:**  
Slower logo growth and lower short-term MRR acceleration.

---

### 2️⃣ Commit to Enterprise-Led Durability

Enterprise accounts demonstrate superior retention, expansion behavior, and revenue efficiency per operational touch.

**Decision:**  
Re-center growth strategy around Enterprise acquisition, onboarding, and expansion.

**Primary Metric Owner:**  
Net Revenue Retention (NRR)

---

### 3️⃣ Treat Time-to-Value as a First-Class Growth Constraint

Users disengage before meaningful value is realized, making acquisition efficiency irrelevant beyond the first few weeks.

**Decision:**  
Shift Product and CS priorities from feature breadth to accelerated, opinionated time-to-value.

**Primary Metric Owner:**  
Time-to-First-Value

---

## 🚫 What This Strategy Requires Us to Stop Doing

- Treating all customer segments as equally valuable  
- Optimizing acquisition efficiency without retention durability  
- Measuring growth primarily through logo count or gross MRR  
- Shipping onboarding experiences that prioritize exploration over value anchoring  
- Allowing SMB churn to subsidize the appearance of growth

---

## 🏁 Final Conclusion

Ravenstack’s growth engine is not broken, but it is **misaligned**.

Acquisition efficiency without retention durability creates **illusory growth**. By re-centering strategy around **time-to-value, onboarding clarity, and Enterprise prioritization**, Ravenstack can convert efficient growth into **sustainable, compounding revenue**.

---

## 🧹 **Data Quality & Cleaning Summary**

Before analysis, a dedicated data validation and cleaning pipeline (ETL1) was executed to ensure **accuracy, consistency, and auditability** across all datasets and establish a **reliable analytical foundation**.

A summary dashboard highlighting:
- Data quality issues identified  
- Corrections and imputations applied  
- Validation coverage across core tables  

is available here:

📊 **[View Data Cleaning Summary Dashboard](./dashboards/00_data_cleaning_summary.png)**

Detailed cleaning logic, validation outputs, and correction logs are documented in the `data/etl1/` and `excel/` folders.

---

## 🔮 **What I’d Do Next With More Data**

With access to additional data, this analysis could be extended to:

- **Session-level product logs**  
  → Identify precise drop-off moments within onboarding and build early churn predictors.

- **Contract terms and billing data**  
  → Replace LTV proxies with true lifetime value and renewal risk modeling.

- **Customer feedback and support sentiment**  
  → Quantify qualitative friction points contributing to early churn.

These additions would enable **predictive retention modeling and targeted intervention strategies**.

---

## ⚠️ **Assumptions & Limitations**

- LTV is estimated using revenue proxies due to limited contract duration data.
- Usage intensity is aggregated and does not reflect session-level behavior.
- Retention analysis is based on observed churn events rather than predictive labels.
- The company and data are fictional but structured to reflect real-world SaaS behavior.

These limitations are acknowledged and do not invalidate the directional insights.

---

## 📂 **Repository Structure**

The repository is organized to reflect a real-world analytics workflow, separating raw data, transformation logic, and business outputs.

```markdown
📦 Project Root
├── data/            # Raw, cleaned, and feature-engineered datasets (CSV)
│   ├── raw/         # Original and intentionally messy source data
│   ├── etl1/        # Cleaning, validation, and data quality outputs
│   └── etl2/        # Feature engineering and analysis-ready datasets
├── src/             # Python scripts for ETL and EDA
├── dashboards/      # Final dashboard images (Excel + Tableau)
├── excel/           # Excel files related to data cleaning
└── README.md        # Project documentation and narrative
```

This structure ensures transparency, reproducibility, and easy navigation for both technical and non-technical reviewers.

---

## 🧰 **Technical Stack**

- **Data Validation & Summaries:** Excel  
- **Data Processing & Feature Engineering:** Python  
- **Visualization:** Tableau  
- **Storage Format:** CSV-based, file-driven analytics pipeline

---

## 👥 **Stakeholder Lens**

This analysis is designed to support **Product, Growth, Customer Success, and Revenue leadership** in making informed decisions around retention, onboarding, and customer segment prioritization.

---

## ⭐ **Call-to-Action**

```markdown
# 📢 Call to Action

If you would like to:
- Explore the dataset in detail  
- Request a walkthrough of the dashboard  
- Discuss how to build similar BI solutions  
- Collaborate on analytics or portfolio projects  

Feel free to reach out or open an issue in the repository.

🚀 **Happy analyzing!**
```
