"""
Inference and SHAP explanation module.
"""
import pandas as pd
import numpy as np
import shap
from typing import Dict, Any, Tuple

from src.features import add_features

def predict_price(model, input_dict: Dict[str, Any]) -> float:
    """
    Predicts the price for a single house.
    
    Args:
        model: Fitted pipeline model.
        input_dict (Dict[str, Any]): Dictionary of feature values.
        
    Returns:
        float: Predicted actual price.
    """
    df = pd.DataFrame([input_dict])
    df_eng = add_features(df)
    log_pred = model.predict(df_eng)[0]
    return float(np.expm1(log_pred))

def predict_with_shap(model, input_dict: Dict[str, Any]) -> Tuple[float, np.ndarray, list]:
    """
    Predicts the price and calculates SHAP values.
    
    Args:
        model: Fitted pipeline model.
        input_dict (Dict[str, Any]): Dictionary of feature values.
        
    Returns:
        Tuple[float, np.ndarray, list]: Predicted price, SHAP values, feature names.
    """
    df = pd.DataFrame([input_dict])
    df_eng = add_features(df)
    
    # Predict
    log_pred = model.predict(df_eng)[0]
    pred_price = float(np.expm1(log_pred))
    
    # Extract preprocessor and regressor
    preprocessor = model.named_steps['preprocessor']
    regressor = model.named_steps['regressor']
    
    # Transform input
    X_transformed = preprocessor.transform(df_eng)
    
    # Get feature names from preprocessor
    num_features = preprocessor.transformers_[0][2]
    cat_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out().tolist()
    feature_names = num_features + cat_features
    
    # Calculate SHAP values based on model type
    model_name = regressor.__class__.__name__
    
    try:
        if model_name in ['RandomForestRegressor', 'XGBRegressor', 'LGBMRegressor']:
            explainer = shap.TreeExplainer(regressor)
            shap_values = explainer.shap_values(X_transformed)
        elif model_name in ['LinearRegression', 'Ridge', 'Lasso']:
            # Assuming linear models
            # We use an independent masker for LinearExplainer or just KernelExplainer
            explainer = shap.LinearExplainer(regressor, shap.maskers.Independent(X_transformed, max_samples=100))
            shap_values = explainer.shap_values(X_transformed)
        else:
            # Fallback
            shap_values = np.zeros((1, len(feature_names)))
            
        # Get base value for single prediction
        if isinstance(shap_values, list):
            shap_val_array = shap_values[0][0]
        else:
            shap_val_array = shap_values[0]
            
        return pred_price, shap_val_array, feature_names
    except Exception as e:
        # Fallback if SHAP fails
        print(f"SHAP error: {e}")
        return pred_price, np.zeros(len(feature_names)), feature_names