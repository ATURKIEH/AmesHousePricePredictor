import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
 

def load_and_clean(filepath):
    #load 
    df = pd.read_csv(filepath)
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
    return df

def drop_columns(df):
    cols_drop = ['Id', 'Utilities', 'Street', 'LandSlope', 'BsmtFinType2', 'BsmtFinSF2', 'Exterior2nd', 'MoSold' , 'MiscFeature', 'MiscVal', 'PoolQC', 'PoolArea', '3SsnPorch', 'LowQualFinSF']
    df = df.drop(cols_drop, axis=1)
    return df

def encode(df):
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
    return df


def train(): 
    df = load_and_clean('train.csv')
    df = drop_columns(df)
    df = encode(df)
 
    # define X and y
    X = df.drop('SalePrice', axis=1)
    y = np.log1p(df['SalePrice'])
 
    #split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    feature_medians = X_train.median()
    pickle.dump(feature_medians, open('feature_medians.pkl', 'wb'))
 
    #scale — fit only on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
 
    #train
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
 
    # save model, scaler, and feature columns
    pickle.dump(model,            open('model.pkl', 'wb'))
    pickle.dump(scaler,           open('scaler.pkl', 'wb'))
    pickle.dump(X.columns.tolist(), open('feature_columns.pkl', 'wb'))
    

    print("Training complete. Model saved to model.pkl")
    print(f"Features saved: {len(X.columns)} columns")
 
 
if __name__ == '__main__':
    train()