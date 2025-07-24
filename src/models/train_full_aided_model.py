import pandas as pd
import statsmodels.api as sm
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

def train_ols_model(interaction):
    model_name = "All Markets - Aided - Manager - Interaction Effects" if interaction else "All Markets - Aided - Manager"

    # Read selected features
    selected_path = (
        'data/processed/selected_cols_manager_interaction_poly.txt'
        if interaction else
        'data/processed/selected_cols_manager.txt'
    )
    with open(selected_path, 'r') as f:
        cols = [line.strip() for line in f if line.strip()]

    # Load data
    final = pd.read_csv('data/processed/manager_metrics.csv')

    # Optional: manually expand interactions if necessary (not needed if already included in cols)
    if interaction:
        base_X = final.drop(columns=['market', 'manager', 'aided_recognition', 'unaided_recognition'])
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(base_X)
        feature_names = poly.get_feature_names_out(base_X.columns)
        poly_df = pd.DataFrame(X_poly, columns=feature_names)
        X = poly_df[cols]  # select only the chosen features
    else:
        X = final[cols]
        
    y = final['aided_recognition']

    # Train/test split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Add constant term
    x_train = sm.add_constant(x_train)
    x_test = sm.add_constant(x_test)

    # Fit OLS model
    model = sm.OLS(y_train, x_train)
    results = model.fit()

    # Predict and evaluate
    y_pred = results.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    # Save model summary
    os.makedirs('outputs/model_summaries', exist_ok=True)
    with open(f'outputs/model_summaries/{model_name}.txt', 'w') as f:
        f.write(results.summary().as_text())
        f.write(f"\n\nTest R^2: {r2:.4f}")
        f.write(f"\nTest RMSE: {mse:.4f}")

    # Save model object
    os.makedirs('models', exist_ok=True)
    with open(f'models/{model_name}.pkl', 'wb') as f:
        pickle.dump(results, f)

# Example usage:
train_ols_model(interaction=True)
train_ols_model(interaction=False)