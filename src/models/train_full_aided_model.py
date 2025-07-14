import pandas as pd
import statsmodels.api as sm
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

model_name = "All Markets - Aided - Manager"

# Load and prepare data
final = pd.read_csv('data/processed/manager_metrics.csv')

x = final[['unit_count', 'asset_count', 'count_within_1', 'count_within_5', 'count_within_10', 'count_within_20', 
           'building_age', 'years_since_reno', 'hs_diploma_perc', 'bachelors_perc', 'masters_perc', 'number_stories',
           'manager_brand', 'miles_from_city_center', 'rating', 'property_quality']]

# x = final.drop(columns=['aided_recognition', 'unaided_recognition', 'market', 'manager'])
y = final['aided_recognition']

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Add constant
x_train = sm.add_constant(x_train)
x_test = sm.add_constant(x_test)

# Fit model on train
model = sm.OLS(y_train, x_train)
results = model.fit()

# Predict and evaluate on test
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
