import pandas as pd
import numpy as np
from datetime import datetime

def clean_df(df):
    return df.loc[:, ~df.columns.str.startswith('Unnamed')]

def assemble_df(branded_sites):
    distances_from_city_center = clean_df(pd.read_csv('data/processed/distances_from_city_center.csv'))
    hellodata_features = clean_df(pd.read_csv('data/processed/hellodata_features.csv'))
    brand_recognition = clean_df(pd.read_csv('data/processed/brand_recognition.csv'))
    google_reviews = clean_df(pd.read_csv('data/processed/google_reviews.csv', usecols=['property_id', 'rating']))
    costar_export = clean_df(pd.read_csv('data/raw/costar_export.csv'))

    print(f'Branded Sites Length: {len(branded_sites)}')

    full_data = pd.merge(branded_sites, distances_from_city_center, on='property_id', how='inner')
    print(f'Length Post Location Merge: {len(full_data)}')

    full_data = pd.merge(full_data, hellodata_features, left_on='property_id', right_on='costar_id', how='inner').drop(columns=['costar_id'])
    print(f'Length Post HelloData Merge: {len(full_data)}')

    full_data = pd.merge(full_data, brand_recognition, on=['market', 'manager'], how='right')
    full_data.dropna(subset=['property_id'], inplace=True)
    print(f'Length Post Survey Data Merge: {len(full_data)}')

    full_data = pd.merge(full_data, google_reviews, on='property_id', how='left')

    full_data = pd.merge(full_data, costar_export.drop(columns=['unit_count', 'latitude', 'longitude', 'zip_code']), on='property_id', how='left')

    # Filter out lease-up and affordable properties
    full_data = full_data[(full_data['is_affordable'] == 0) & (full_data['is_lease_up'] == 0)]
    full_data = full_data.drop(columns=['is_affordable', 'is_lease_up'])

    # One-hot encode selected categorical columns
    dummify_cols = ['area_type', 'style', 'building_class']
    full_data = pd.get_dummies(full_data, columns=dummify_cols)

    # Convert all dummy cols to int
    for col in full_data.columns:
        if any(prefix in col for prefix in ['style_', 'building_class_', 'area_type_']):
            full_data[col] = full_data[col].astype(int)

    full_data['building_age'] = (datetime.now().year - full_data['year_built']).round()

    full_data['year_renovated'] = full_data['year_renovated'].fillna(full_data['year_built'])
    full_data['years_since_reno'] = (datetime.now().year - full_data['year_renovated']).round()

    full_data['years_since_acquisition'] = (datetime.now().year - full_data['year_acquired']).round()

    full_data['years_since_first_acquisition'] = (
        full_data['years_since_acquisition'] -
        full_data.groupby(['manager', 'market'])['years_since_acquisition'].transform('min')
    )

    # Drop irrelevant columns
    full_data = full_data.drop(columns=[
        'property_id', 'branded', 'property', 'owner', 'latitude', 'longitude',
        'submarket', 'zip_code', 'closest_city_center', 'total_responses',
        'year_built', 'year_renovated', 'year_acquired'
    ])

    # Separate columns to aggregate differently
    dummy_cols = [col for col in full_data.columns if any(prefix in col for prefix in ['style_', 'building_class_', 'area_type_', 'unit_count'])]
    numeric_cols = [col for col in full_data.select_dtypes(include='number').columns if col not in dummy_cols]

    # Group and aggregate
    means = full_data.groupby(['manager', 'market'])[numeric_cols + ['manager_brand']].mean()
    counts = full_data.groupby(['manager', 'market'])[dummy_cols].sum()

    manager_metrics = pd.concat([means, counts], axis=1).reset_index()

    return full_data, manager_metrics


def main():
    branded_sites = clean_df(pd.read_csv('data/processed/branded_sites.csv'))
    full_data, manager_metrics = assemble_df(branded_sites)
    full_data.to_csv('data/processed/asset_metrics.csv', index=False)
    manager_metrics.to_csv('data/processed/manager_metrics.csv', index=False)

if __name__ == "__main__":
    main()