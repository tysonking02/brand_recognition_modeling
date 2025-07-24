import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

def clean_df(df):
    return df.loc[:, ~df.columns.str.startswith('Unnamed')]

distances_from_city_center = clean_df(pd.read_csv('data/processed/distances_from_city_center.csv'))
hellodata_features = clean_df(pd.read_csv('data/processed/hellodata_features.csv'))
brand_recognition = clean_df(pd.read_csv('data/processed/brand_recognition.csv'))
google_reviews = clean_df(pd.read_csv('data/processed/google_reviews.csv', usecols=['property_id', 'rating']))
costar_export = clean_df(pd.read_csv('data/raw/costar_export.csv'))
asset_preds = clean_df(pd.read_csv('data/processed/asset_preds.csv'))

full_data = pd.merge(asset_preds, distances_from_city_center, on='property_id', how='inner')

full_data = pd.merge(full_data, hellodata_features, left_on='property_id', right_on='costar_id', how='inner').drop(columns=['costar_id'])

full_data = pd.merge(full_data, brand_recognition, on=['market', 'manager'], how='right')
full_data.dropna(subset=['property_id'], inplace=True)

full_data = pd.merge(full_data, google_reviews, on='property_id', how='left')

full_data = pd.merge(full_data, costar_export.drop(columns=['unit_count', 'latitude', 'longitude', 'zip_code']), on='property_id', how='left')

# One-hot encode selected categorical columns
dummify_cols = ['area_type', 'style', 'building_class']
full_data = pd.get_dummies(full_data, columns=dummify_cols)

# Convert all dummy cols to int
for col in full_data.columns:
    if any(prefix in col for prefix in ['style_', 'building_class_', 'area_type_', 'manager_brand']):
        full_data[col] = full_data[col].astype(int)

full_data['building_age'] = (datetime.now().year - full_data['year_built']).round()

full_data['year_renovated'] = full_data['year_renovated'].fillna(full_data['year_built'])
full_data['years_since_reno'] = (datetime.now().year - full_data['year_renovated']).round()

full_data['years_since_acquisition'] = (datetime.now().year - full_data['year_acquired']).round()
full_data['years_since_acquisition'] = full_data['years_since_acquisition'].fillna(full_data['years_since_reno'])

bad_cols = full_data.columns[
    full_data.isnull().any() 
]

# Print them with counts
for col in bad_cols:
    full_data[col] = full_data[col].fillna(full_data[col].mean())

# Drop irrelevant columns
full_data = full_data.drop(columns=[
    'property_id', 'branded', 'property', 'owner', 'latitude', 'longitude',
    'submarket', 'zip_code', 'closest_city_center', 'total_responses',
    'year_built', 'year_renovated', 'year_acquired', 'is_affordable', 'is_lease_up',
    'manager', 'market', 'property_name', 'property_address', 'city', 'state', 
    'market_name', 'submarket_name'
])

X = full_data.drop(columns=['aided_recognition', 'unaided_recognition', 'impact'])

y = full_data['impact']

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add constant term
x_train = sm.add_constant(x_train)
x_test = sm.add_constant(x_test)

# Fit OLS model
model = sm.OLS(y_train, x_train)
results = model.fit()

with open(f'outputs/model_summaries/All Markets - Aided - Asset Impact.txt', 'w') as f:
    f.write(results.summary().as_text())