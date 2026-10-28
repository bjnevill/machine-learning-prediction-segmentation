# Machine Learning: Wine Quality Prediction & Retail Customer Segmentation

**Individual academic project · Ngee Ann Polytechnic, Diploma in Data Science · 2026**

Two end-to-end machine learning studies: supervised regression on white-wine quality and unsupervised segmentation of retail customers.

## Part 1 — Wine quality prediction
- Explored 4,898 samples and analysed feature distributions, correlations and outliers.
- Compared **Linear Regression**, **K-Nearest Neighbours** and **Decision Tree Regressor**.
- Used training-only preprocessing, IQR capping, transformations, standardisation and GridSearchCV.
- Testing results reported in the assignment: KNN **RMSE 0.714, R² 0.341**; tuned Decision Tree **RMSE 0.725, R² 0.322**.
- Selected the tuned tree for a balance of interpretability and generalisation, rather than simply choosing the top test metric.

## Part 2 — Retail customer segmentation
- Cleaned online retail transactions, excluding invalid records, cancellations and returns.
- Aggregated transactions per customer; engineered **Recency, Frequency, Monetary (RFM)** and average basket value.
- Compared **K-means (4 clusters; silhouette 0.409)** against **hierarchical clustering (2 clusters; silhouette 0.448)**.
- Preferred four K-means groups for more actionable business segments: loyal high-value, at-risk low-value, high-basket occasional and other customers.

## Skills
Python · pandas · scikit-learn · EDA · Feature Engineering · Regression · Clustering · Model Evaluation · Business Interpretation

## Source files
Open [`analysis.ipynb`](./analysis.ipynb) for the original analysis code (with student identifiers and stored outputs removed), and [`requirements.txt`](./requirements.txt) for Python dependencies. The notebook requires its original datasets to run. Raw coursework datasets are not redistributed here.

*Educational analysis; model metrics are from the submitted academic report.*
