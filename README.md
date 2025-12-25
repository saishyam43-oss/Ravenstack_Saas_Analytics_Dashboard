# Ravenstack SaaS Analytics: Growth, Retention & Strategic Focus

> *Diagnosing growth quality, churn risk, and product value delivery in a subscription SaaS business*

---

<p align="center">
  <img src="logo.png" width="500"/>
</p>

---

## The Growth Paradox: High Revenue, Higher Risk  
**Analyzing the disconnect between Enterprise growth and severe customer attrition risk**

---

## ⚡ Executive TL;DR

**The Problem**  
At a surface level, **Ravenstack appears healthy**, with strong MRR growth and high acquisition efficiency (Quick Ratio ≈ 3.0). However, deeper analysis reveals a **structural imbalance**. Approximately **66% of acquired accounts eventually churn**, preventing acquisition-led growth from compounding into durable long-term value.

**The Diagnosis**  
Retention analysis shows that **subscription disengagement begins early**, often before users experience meaningful value, while revenue durability depends heavily on a small segment of Enterprise customers.

Product usage patterns further explain this behavior. Users explore broadly across features but fail to anchor on a clear value driver, resulting in **flat engagement and delayed time-to-value**.

Together, these signals indicate that Ravenstack’s growth is **efficient but fragile**, requiring a shift from volume-led acquisition toward **retention-first growth and deliberate Enterprise prioritization**.

---

## 🏢 Client Background & Project Context

**Client:** Ravenstack (fictional SaaS company)  
**Business Model:** Subscription-based B2B SaaS

As the business enters a scaling phase, leadership requires deeper visibility into whether:
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
- Acquisition efficiency remains strong (Quick Ratio > 3.0) but has normalized with scale
- **66% lifetime account attrition** structurally limits compounding growth
- Revenue is increasingly concentrated among a small set of Enterprise customers
- Enterprise accounts generate significantly higher revenue per interaction

### Executive Takeaway
Growth is **efficient but fragile**. Without retention improvements and customer mix rebalancing, continued acquisition will amplify churn faster than durable value.

---

## 🪣 Retention Audit: The “Leaky Bucket” (Churn Is Front-Loaded)

*Following the North Star assessment of fragile growth, this section examines **when** customers disengage during the subscription lifecycle.*

<p align="center">
  <img src="dashboards/02_the_leaky_bucket.png"/>
</p>

### Business Question
Are customers exiting gradually over time, or is disengagement concentrated at specific lifecycle stages?

### Key Insights
- Subscription exits are **front-loaded**, with noticeable disengagement in the early lifecycle
- Exit likelihood declines over time, but early losses prevent long-term value realization
- Newer cohorts degrade ~2× faster than older cohorts
- Net Revenue Retention declines steadily, indicating limited expansion

### Why This Matters
Early subscription exits signal **expectation mismatch or onboarding gaps**, while poor NRR ensures that even retained customers fail to compound value.

### Action Plan
- Fix early onboarding clarity and activation (Days 0–7)
- Close the value gap before trial or contract expiration
- Re-evaluate acquisition channels contributing low-retention cohorts
- Target reduction of early subscription exits (<30 days) to <10%

---

## ⏱️ Product Stickiness: The “Time-to-Value” Crisis (Value Arrives Too Late)

*After identifying **when** users disengage, this section examines **why** retention fails at the product level.*

<p align="center">
  <img src="dashboards/03_the_stickiness.png"/>
</p>

### Key Insights
- Average time-to-first-value (~76 days) far exceeds time-to-disengagement
- Engagement intensity remains flat over time, indicating no compounding usage
- Accounts with broader feature exploration show **higher lifetime attrition**
- No single feature acts as a dominant value anchor

### Interpretation
Broad exploration reflects **cognitive overload**, not product stickiness. Users fail to internalize a clear “aha” moment before disengaging.

### Strategic Implications
- Reduce cognitive load by narrowing early workflows
- Guide users toward a small set of core value paths
- Accelerate time-to-value from weeks to days

---

## 🐋 Strategic Opportunity: The “Whale Hunt” (Pivoting from Volume to Value)

*With root causes identified, this section evaluates **where** Ravenstack should focus to maximize durable growth.*

<p align="center">
  <img src="dashboards/04_the_opportunity.png"/>
</p>

### Key Insights
- Enterprise accounts represent ~22% of volume but ~47% of revenue
- Enterprise customers show **~30% Net Revenue Retention advantage** vs SMB
- Revenue efficiency per support interaction is orders-of-magnitude higher for Enterprise
- SMB volume contributes disproportionately to churn and operational cost

### Strategic Decision
**Pivot from volume-led growth to value-led expansion.**

### Recommended Actions
- Prioritize Enterprise acquisition, retention, and expansion
- Automate or self-serve SMB where possible
- Align Product, CS, and GTM around Enterprise durability
- Target NRR stabilization above 100% through expansion-led growth

---

## 🔗 Cross-Dashboard Narrative: From Growth to Strategy

Taken together, the four dashboards reveal a consistent story.

Ravenstack’s **acquisition engine is efficient**, as reflected by a strong Quick Ratio and growing MRR. However, this efficiency masks a deeper issue: **customers churn faster than they realize value**, causing long-term revenue leakage.

Retention analysis shows that churn is **front-loaded**, concentrated in the first few weeks of the customer lifecycle. Product usage data explains why: users are overwhelmed by feature breadth and fail to reach meaningful value quickly. As a result, engagement remains shallow and time-to-value exceeds time-to-churn by a wide margin.

Segment-level analysis resolves the strategic tension. While SMB customers drive volume, **Enterprise customers deliver durable revenue with far greater efficiency**. Treating all segments equally has diluted focus and increased operational cost.

The implication is clear: Ravenstack’s challenge is not growth, but **growth quality**. Solving retention and prioritizing high-value segments offers a far higher return than accelerating acquisition.

---

## 💡 Key Business Insights (Consolidated)

- Growth is **efficient but fragile**, with churn offsetting acquisition gains over time.
- Retention failures are **early-stage**, pointing to onboarding and expectation mismatch rather than long-term dissatisfaction.
- Product usage does not naturally deepen over time, confirming a **time-to-value gap**.
- Feature breadth without guidance increases churn instead of retention.
- Enterprise customers generate outsized value relative to operational effort.
- SMB growth adds volume but introduces disproportionate cost and revenue risk.

---

## 8️⃣ Final Conclusion

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
