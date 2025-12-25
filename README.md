# Ravenstack SaaS Growth & Retention Analytics

**Diagnosing growth quality, churn risk, and product value delivery in a subscription SaaS**

<p align="center">
  <img src="./logo.png" alt="Ravenstack Logo" width="500"/>
</p>

---

## 🏢 Client Background & Analytics Context

**Ravenstack** is a fictional B2B SaaS company operating on a subscription-based revenue model with multiple plan tiers (SMB, Mid-Market, Enterprise). The company has successfully scaled customer acquisition and revenue but is experiencing **increasing uncertainty around growth quality and customer retention**.

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

This analysis was commissioned to support **cross-functional decision-making**, not just reporting.

---

## 🎯 Business Problem & Objective

### Business Problem
Despite strong headline revenue growth, Ravenstack faces several unresolved risks:
- High customer churn offsets acquisition gains
- Retention appears to be weakening for newer cohorts
- Product usage does not clearly correlate with customer longevity
- A small segment of customers contributes a disproportionate share of revenue

These signals raise a critical question:  
**Is Ravenstack scaling efficiently, or accumulating hidden risk beneath top-line growth?**

### Objective
The objective of this analysis is to:
- Evaluate the **quality of growth**, not just its magnitude
- Identify **where in the customer lifecycle value breaks down**
- Assess whether **product adoption drives retention**
- Quantify **customer segment value and operational efficiency**
- Surface **clear, actionable strategic recommendations**

---

## ⭐ North Star Metrics & Analytical Focus

The analysis is anchored around a focused set of **North Star metrics** that collectively describe growth health, retention strength, and customer value:

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

This structured approach ensures insights are **causal, not descriptive**.

---

## 📊 Executive Summary (North Star View)

At a surface level, Ravenstack appears healthy, with strong MRR growth and a Quick Ratio consistently above 3.0, indicating efficient acquisition.

However, deeper analysis reveals a critical imbalance. **Lifetime churn reaches 73%**, meaning the majority of acquired customers eventually exit the platform, significantly eroding long-term value. A large share of churn occurs within the **first 7–30 days**, pointing to onboarding and expectation mismatch rather than pricing or competition.

Product analysis shows that users often churn **before discovering meaningful value**, with average time-to-value far exceeding the time it takes customers to decide to leave. Meanwhile, revenue is increasingly concentrated among a small set of high-value Enterprise customers, creating both opportunity and risk.

Together, these signals indicate that Ravenstack’s growth is **efficient but fragile**, requiring an immediate shift in focus from acquisition to **retention, onboarding, and strategic customer prioritization**.

<p align="center">
  <img src="./dashboards/01_the_north_star.png" alt="North Star Growth Dashboard" width="900"/>
</p>

---

## 🪣 Retention Audit: The “Leaky Bucket” Diagnosis

**Business Question**  
Are customers leaving gradually over time, or is churn concentrated at specific moments in the lifecycle?

### Key Findings
- **73% lifetime churn** indicates that most acquired customers eventually exit the platform.
- A significant share of churn occurs **within the first 7–30 days**, pointing to early disengagement rather than long-term dissatisfaction.
- **Newer cohorts degrade nearly 2× faster** than older cohorts, suggesting that recent growth has come at the cost of retention quality.
- Net Revenue Retention (NRR) shows a **steady downward trajectory**, with limited evidence of meaningful upsell or expansion.

### Why This Matters
High acquisition efficiency is negated if customers churn before delivering long-term value. Early churn signals an **expectation mismatch or onboarding failure**, not pricing or competitive pressure.

Without intervention, continued acquisition will compound churn rather than compound growth.

<p align="center">
  <img src="./dashboards/02_the_leaky_bucket.png" alt="Retention Audit – Leaky Bucket" width="900"/>
</p>

---

## ⏱️ Product Stickiness: The “Time-to-Value” Crisis

**Business Question**  
Do customers experience meaningful value from the product before deciding to churn?

### Key Findings
- Average **time-to-churn is ~7 days**, while average **time-to-first meaningful adoption is ~76 days**.
- Users are making churn decisions **long before discovering product value**.
- Usage intensity remains **flat over time**, indicating shallow engagement rather than progressive adoption.
- Customers who explore more features (“Broad Explorers”) paradoxically show **higher churn**, suggesting confusion rather than value realization.
- No single feature emerges as a clear “hero” driver of retention.

### Why This Matters
The product is failing to guide users to value quickly. Feature breadth without direction increases cognitive load, delaying value discovery and accelerating churn.

This is not a feature quantity problem, but a **value delivery and onboarding design problem**.

<p align="center">
  <img src="./dashboards/03_the_stickiness.png" alt="Product Stickiness – Time-to-Value Crisis" width="900"/>
</p>

---

## 🐋 Strategic Opportunity: The “Whale Hunt” Initiative

**Business Question**  
Are all customers equally valuable, or should Ravenstack focus on a narrower, higher-impact segment?

### Key Findings
- The **top 10% of customers drive ~47% of total revenue**, indicating heavy revenue concentration.
- **Enterprise customers generate ~3,000× more revenue per support ticket** compared to SMBs.
- Enterprise cohorts show **NRR stabilization post-onboarding**, while SMB cohorts continue to contract.
- SMB customers represent the majority of volume but deliver **low lifetime value and high operational cost**.

### Why This Matters
Not all growth is equal. Treating all segments identically dilutes focus and increases operational drag.

The data supports a clear strategic pivot:
- **Prioritize Enterprise retention and expansion**
- **Aggressively migrate SMB customers to self-serve**
- Align Product, CS, and Growth efforts around high-LTV accounts

This “Whale Hunt” strategy offers the most realistic path to stabilizing NRR and reducing churn-driven revenue loss.

<p align="center">
  <img src="./dashboards/04_the_opportunity.png" alt="Strategic Opportunity – Whale Hunt" width="900"/>
</p>

---
