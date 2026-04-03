import numpy as np
import pandas as pd
import pickle

def load_artifacts():
    model           = pickle.load(open('model.pkl', 'rb'))
    scaler          = pickle.load(open('scaler.pkl', 'rb'))
    feature_columns = pickle.load(open('feature_columns.pkl', 'rb'))
    feature_medians = pickle.load(open('feature_medians.pkl', 'rb'))
    return model, scaler, feature_columns, feature_medians


def predict_price(input_data: dict)->dict:
    model, scaler, feature_columns, feature_medians = load_artifacts()
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feature_columns)
    input_df = input_df.fillna(feature_medians)
 
    input_scaled = scaler.transform(input_df)
 
    log_prediction   = model.predict(input_scaled)[0]
    predicted_price  = np.expm1(log_prediction)
 
    return {
        'predicted_price': round(predicted_price, 2),
        'log_prediction':  round(log_prediction, 4)
    }

