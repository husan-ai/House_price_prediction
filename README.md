# Toshkent Uy Narxlarini Bashorat Qilish

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completed-green)
![Methodology](https://img.shields.io/badge/Methodology-CRISP--DM-purple)

## Project Overview

This project predicts apartment prices in **Tashkent, Uzbekistan** using a **Linear Regression** model built with a Scikit-learn Pipeline. The project follows the **CRISP-DM** methodology and uses real estate listings data collected in February 2021.

---

## Dataset

- **Source:** [Tashkent Housing Data (Feb 2021)](https://raw.githubusercontent.com/anvarnarz/praktikum_datasets/main/housing_data_08-02-2021.csv)
- **Samples:** 7,565 listings × 7 columns
- **Target:** `price` (USD)

| Feature | Description |
|---------|-------------|
| `location` | Full address of the listing |
| `district` | District in Tashkent |
| `rooms` | Number of rooms (1–10) |
| `size` | Apartment area (m²) |
| `level` | Floor of the apartment (1–19) |
| `max_levels` | Total floors in the building (1–25) |
| `price` | Sale price in USD *(target)* |

---

## Data Preprocessing

- Converted `price` and `size` columns from `object` to `int64`
- Replaced non-numeric values with **median** values
- Dropped `location` column — too many unique values (1,600+), would cause high-dimensional encoding
- **Categorical encoding:** `district` → `OneHotEncoder` (handle_unknown='ignore')
- **Numerical scaling:** `rooms`, `size`, `level`, `max_levels` → `StandardScaler`
- Train/Test split: **80% / 20%** (`random_state=42`)

---

## Exploratory Data Analysis

- Histogram plots for all numeric features
- **Top 10 locations** by listing count (bar chart)
- **Top 10 districts** by listing count (Seaborn barplot)
- Most listings concentrated in: Chilanzar, Yakkasaray, Mirzo-Ulugbek districts

---

## Model: Linear Regression (Pipeline)

Used a **Scikit-learn Pipeline** combining preprocessing and model in one step:

```python
Pipeline([
    ("prep", ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])),
    ("lr", LinearRegression())
])
```

---

## Results

| Metric | Value |
|--------|-------|
| **RMSE** | 224,608 USD |

**Sample Predictions vs Actual:**

| Actual Price | Predicted Price | Difference % |
|-------------|----------------|--------------|
| $41,000 | $44,703 | -9% |
| $91,000 | $155,373 | -71% |
| $95,000 | $153,729 | -62% |

### Analysis

- RMSE of ~$224K is high relative to typical apartment prices
- Some predictions go **negative** (e.g., -$26,484) — model weakness
- Linear Regression struggles with non-linear price distributions
- Root cause: `district` alone is insufficient to capture location value; dropping `location` lost key spatial information

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| `Python` | Core language |
| `Pandas / NumPy` | Data manipulation |
| `Matplotlib / Seaborn` | Visualization |
| `Scikit-learn` | Pipeline, preprocessing, model |

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/husan-ai/tashkent-house-price-prediction.git
cd tashkent-house-price-prediction

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook Toshkentdagi_uylarni_bashorat_qilish_ML.ipynb
```

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
```

---

## Future Improvements

- [ ] Try non-linear models: **Random Forest**, **XGBoost**, **Gradient Boosting**
- [ ] Better encode `location` using **target encoding** or clustering
- [ ] Add price-per-m² as a derived feature
- [ ] Handle outliers in `price` and `size`
- [ ] Cross-validation for more reliable evaluation
- [ ] Deploy as a price estimator web app using **Streamlit**

---

## Author

**Husan**  
ML | DL | NLP  
GitHub: [@husan-ai](https://github.com/husan-ai)
