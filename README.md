#  House Price Predictor
 
Predicts residential house sale prices using the Ames Iowa Housing dataset. Compares Linear Regression, Random Forest, and XGBoost — Linear Regression achieved the best results after log-transforming the target variable.
 
---
 
## Results
 
| Model | R² Score |
|-------|----------|
| **Linear Regression** | **0.9194** ✅ |
| XGBoost | 0.8940 |
| Random Forest | 0.8723 |
 
**Key finding:** Linear Regression outperformed tree-based models after log-transforming SalePrice, confirming strong linear relationships once target skew is corrected. OverallQual is the single strongest predictor, accounting for ~54% of Random Forest feature importance.
 
---
 
## Dataset
 
[Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) — 1,460 samples, 79 features. Download `train.csv` and place in the project root.
 
---
 
## Project Structure
 
```
├── train.py       # Data pipeline + model training → saves pkl files
├── predict.py     # Loads saved model → returns price prediction
├── app.py         # FastAPI REST API — exposes predict.py via HTTP
├── main.py        # Full evaluation, model comparison, visualizations
├── requirements.txt
└── .gitignore
```
 
---
 
## Setup
 
```bash
pip install -r requirements.txt
python train.py       # generates pkl files
python predict.py     # runs a sample prediction
python main.py        # full evaluation and plots
```
 
---
 
## API Usage
 
```bash
uvicorn app:app --reload
```
 
Then open `http://localhost:8000/docs` for the interactive API UI.
 
**POST** `/predict`
 
```json
{
  "features": {
    "OverallQual": 7,
    "GrLivArea": 1500,
    "TotalBsmtSF": 800,
    "GarageCars": 2,
    "YearBuilt": 2000
  }
}
```
 
**Response:**
```json
{
  "predicted_price": 183360.01,
  "log_prediction": 12.1192
}
```
 
Missing features are automatically filled with training data medians.
 
---
 
## Tech Stack
 
Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, FastAPI, Uvicorn
 
---

## Running With Docker

```bash
docker pull aturkieh/house-price-predictor
docker run -p 8000:8000 aturkieh/house-price-predictor
```

Then visit http://localhost:8000/docs
