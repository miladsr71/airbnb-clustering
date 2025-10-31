# Chicago Airbnb Market Clustering

## Project Overview

This project applies **unsupervised learning (K-Means)** to uncover structure in the Chicago Airbnb market.  
Two main analyses were conducted:

1. **Listing Profiling (K=3)** — Segment properties into **Budget**, **Mid-range**, and **Premium** tiers.
2. **Host Profiling (K=2)** — Classify hosts as **Casual** or **Professional** operators.

---

## Methodology

- **Outlier Management:** Removed extreme high-priced listings using the **Interquartile Range (IQR)** method to prevent distortion in clustering results.
- **Feature Scaling:** Applied **StandardScaler** to normalize features such as price and reviews so each contributes equally to distance calculations.
- **Model:** Implemented **K-Means Clustering** with fixed values of K (3 for listings, 2 for hosts).

---

## Findings

- **Budget & Mid-range clusters** (Blue/Red): Spread across established residential areas.
- **Premium cluster** (Gold): Concentrated in tourist and downtown zones, such as _The Loop_ and _Near North Side_.
- **Host segmentation:** Based on features like _listing count_, _recent reviews_, and _availability_, distinguishing **Professional** from **Casual** hosts.

---

## Tools Used

- Python
- pandas
- scikit-learn
- seaborn
- matplotlib

---

## Author

**Milad Zahmatkesh**  
M.S. Computer Science, Southern Illinois University Edwardsville
