
# 🏥 Medical Insurance Charges Predictor

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-ff4b4b?logo=streamlit)](https://financial-risk-and-healthcare-premium-pricing-jmidxcxdyhctm65f.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Random%20Forest-orange?logo=scikit-learn)](https://scikit-learn.org/)

> **🔗 Try the live app:** [financial-risk-and-healthcare-premium-pricing-jmidxcxdyhctm65f.streamlit.app](https://financial-risk-and-healthcare-premium-pricing-jmidxcxdyhctm65f.streamlit.app)

---

## 1. What Problem Does This Solve?

The healthcare industry faces a significant challenge in actuarial risk assessment. Traditional methods of pricing insurance premiums are often rigid and fail to capture the complex, non-linear interactions between a person's lifestyle and their health risks.

This project solves the **Pricing Accuracy** problem. By leveraging machine learning, we automate the estimation of medical charges, ensuring:

- **Insurers** can set premiums that accurately reflect risk, maintaining profitability.
- **Customers** receive fair, data-driven pricing based on their specific health profile.

---

## 2. Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Scikit-Learn | Model training & preprocessing |
| Pandas & NumPy | Data manipulation |
| Streamlit | Web app deployment |
| Joblib | Model serialization |

---

## 3. What Data Was Used?

The model was trained on a medical insurance claimants dataset. Key features (cost drivers) include:

- **Demographics:** Age, Sex, Region
- **Health Metrics:** BMI — a critical indicator of potential chronic conditions
- **Lifestyle & Family:** Smoking status (strongest correlation to cost) and number of dependants

---

## 4. Model & Approach

The core engine is a **Random Forest Regressor**.

**Why Random Forest?** It excels at handling both categorical and numerical data while capturing complex non-linear relationships — for example, how BMI interacts differently with cost depending on smoking status.

**Hyperparameter Tuning:** `RandomizedSearchCV` was used to optimise `n_estimators`, `max_depth`, and `min_samples_split`, ensuring the model generalises well to unseen data.

---

## 5. Results

| Metric | Baseline Model | Tuned Model |
|---|---|---|
| R² Score | 0.82 | **0.88** |
| MAE | — | **$1,906** |
| RMSE | — | **$3,347** |

- **R² of 0.88** means the model explains 88% of the variance in medical charges.
- **RMSE of $3,347** is the primary metric — in insurance pricing, large errors are disproportionately costly, making RMSE the most relevant measure of model quality.
- Minimal gap between training and test performance confirms the model is robust and not overfitted.

---

## 6. Project Structure

```
├── raw_data/          # Source dataset
├── notebooks/         # EDA and model development
├── src/
│   ├── features/      # Data ingestion & preprocessing pipeline
│   └── inference/     # Prediction module
├── models/            # Saved model artifacts (.pkl)
├── streamlit app/     # Web application
├── predictions/       # Output predictions
└── requirements.txt
```

---

## 7. How to Run Locally

```bash
# Clone the repo
git clone https://github.com/MphadiJ/Financial-Risk-and-healthcare-Premium-Pricing.git
cd Financial-Risk-and-healthcare-Premium-Pricing

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run "streamlit app/app.py"
```

---

## 8. Why Should You Care?

This project demonstrates end-to-end ML product thinking:

- **Data Engineering** — structured pipeline separating ingestion, feature engineering, and inference
- **Statistical Rigour** — model tuned and validated with industry-standard metrics (R², MAE, RMSE)
- **Business Integration** — Streamlit UI bridges the gap between a black-box model and real end-users such as insurance agents or customers
- **Modular Architecture** — professional software engineering patterns for maintainability and scalability
