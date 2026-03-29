import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#load 
df = pd.read_csv('train.csv')
#fill columns that have NaN but mean is not applicable with 'None'
none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 
             'FireplaceQu', 'GarageType', 'GarageFinish',
             'GarageQual', 'GarageCond', 'BsmtQual', 
             'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
             'BsmtFinType2', 'MasVnrType']
for col in none_cols:
    df[col] = df[col].fillna('None')

# numerical — fill with median
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)

# categorical — fill with most common value
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

#columns to drop
cols_drop = ['Id', 'Utilities', 'Street', 'LandSlope', 'BsmtFinType2', 'BsmtFinSF2', 'Exterior2nd', 'MoSold' , 'MiscFeature', 'MiscVal', 'PoolQC', 'PoolArea', '3SsnPorch', 'LowQualFinSF']
df = df.drop(cols_drop, axis=1)


#ordinal encoding
quality_order = ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex']

ordinal_cols = {
    'ExterQual': quality_order,
    'ExterCond': quality_order,
    'BsmtQual': quality_order,
    'BsmtCond': quality_order,
    'HeatingQC': quality_order,
    'KitchenQual': quality_order,
    'FireplaceQu': quality_order,
    'GarageQual': quality_order,
    'GarageCond': quality_order,
    'Fence': ['None', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv'],
    'BsmtExposure': ['None', 'No', 'Mn', 'Av', 'Gd'],
    'BsmtFinType1': ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
    'GarageFinish': ['None', 'Unf', 'RFn', 'Fin'],
    'PavedDrive': ['N', 'P', 'Y'],
    'LotShape': ['IR3', 'IR2', 'IR1', 'Reg'],
    'LandContour': ['Low', 'HLS', 'Bnk', 'Lvl'],
}


for col, order in ordinal_cols.items():
    encoder = OrdinalEncoder(categories=[order])
    df[col] = encoder.fit_transform(df[[col]]).ravel()

#nominal encoding
nominal_cols = ['MSZoning', 'Alley', 'LotConfig', 'Neighborhood',
                'Condition1', 'BldgType', 'HouseStyle', 'RoofStyle',
                'RoofMatl', 'Exterior1st', 'MasVnrType', 'Foundation',
                'Heating', 'CentralAir', 'Electrical', 'Functional',
                'GarageType', 'SaleCondition', 'MSSubClass', 'Condition2', 'SaleType']
df = pd.get_dummies(df, columns=nominal_cols)



df['SalePrice'] = pd.to_numeric(df['SalePrice'], errors='coerce')

remaining = df.select_dtypes(include='object').columns.tolist()
remaining = [col for col in remaining if col != 'SalePrice']
if remaining:
    df = pd.get_dummies(df, columns=remaining)
    print(f"Auto-encoded: {remaining}")


#defining x and y
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']
y = np.log1p(y)


#splitting 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#scaling x
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#fitting model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("--------Linear Regression Performance--------:")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R^2 Score: {r2}")



y_test_actual = np.expm1(y_test)
y_pred_actual = np.expm1(y_pred)

plt.figure(figsize=(8, 6))
plt.scatter(y_test_actual, y_pred_actual, alpha=0.5)
plt.plot([y_test_actual.min(), y_test_actual.max()], [y_test_actual.min(), y_test_actual.max()], 'r--', linewidth = 2)
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs Predicted Sale Price')
plt.show()



#residual plot

residuals = y_test_actual - y_pred_actual
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_actual, residuals, alpha=0.5)
plt.axhline(0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Predicted Sale Price')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Sale Price')
plt.show()


#random forest 
rforest_model = RandomForestRegressor(n_estimators=300,
    max_depth=None,   
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt', 
    random_state=42)
rforest_model.fit(X_train, y_train)
y_pred_rf = rforest_model.predict(X_test)
print("--------Random Forest Performance--------:")
print(r2_score(y_test, y_pred_rf))

#importance of top 20 features in rforest model
importances = pd.Series(
    rforest_model.feature_importances_, 
    index=X.columns
).sort_values(ascending=False)
print(importances.head(20))

#xgb boost
xgb_model = XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
print("--------XGBoost Performance--------:")
print(f"XGBoost R²: {r2_score(y_test, y_pred_xgb):.4f}")

print("\n--------Model Comparison--------:")
results = {
    'Linear Regression': r2_score(y_test, y_pred),
    'Random Forest': r2_score(y_test, y_pred_rf),
    'XGBoost': r2_score(y_test, y_pred_xgb)
}
for model, score in results.items():
    print(f"{model}: {score:.4f}")


importances.head(15).plot(kind='barh', figsize=(10, 6))
plt.title('Top 15 Most Important Features')
plt.xlabel('Feature Importance Score')
plt.tight_layout()
plt.show()
