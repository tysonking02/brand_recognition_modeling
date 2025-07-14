import pandas as pd
import numpy as np

def clean_df(df):
    return df.loc[:, ~df.columns.str.startswith('Unnamed')]

def assemble_df(costar_export):

    distances_from_city_center = clean_df(pd.read_csv('data/processed/distances_from_city_center.csv'))
    hellodata_features = clean_df(pd.read_csv('data/processed/hellodata_features.csv'))
    brand_recognition = clean_df(pd.read_csv('data/processed/brand_recognition.csv'))
    google_reviews = clean_df(pd.read_csv('data/processed/google_reviews.csv', usecols=['property_id', 'rating']))

    print(f'Branded Sites Length: {len(costar_export)}')

    full_data = pd.merge(costar_export, distances_from_city_center, on='property_id', how='inner')
    print(f'Length Post Location Merge: {len(full_data)}')

    full_data = pd.merge(full_data, hellodata_features, left_on='property_id', right_on='costar_id', how='inner').drop(columns=['costar_id'])
    print(f'Length Post HelloData Merge: {len(full_data)}')


    full_data = pd.merge(full_data, brand_recognition, on=['market', 'manager'], how='right')
    full_data.dropna(subset=['property_id'], inplace=True)
    print(f'Length Post Survey Data Merge: {len(full_data)}')

    full_data = pd.merge(full_data, google_reviews, on='property_id', how='left')

    full_data = full_data[(full_data['is_affordable'] == 0) & (full_data['is_lease_up'] == 0)].drop(columns=['is_affordable', 'is_lease_up'])

    dummify_cols = ['area_type', 'style', 'building_class']

    full_data = pd.get_dummies(full_data, columns=dummify_cols)

    bool_cols = [col for col in full_data.columns if any(key in col for key in ['style', 'building_class', 'area_type', 'manager_brand'])]

    for col in bool_cols:
        full_data[col] = full_data[col].astype(int)

    full_data = full_data.drop(columns=['property_id', 'branded', 'property', 'owner', 'latitude', 'longitude', 'submarket', 'zip_code', 'closest_city_center', 'total_responses'])

    numeric_cols = full_data.select_dtypes(include='number').columns

    manager_metrics = full_data.groupby(['manager', 'market'])[numeric_cols].mean().reset_index()

    return full_data, manager_metrics

def main():
    costar_export = clean_df(pd.read_csv('data/processed/branded_sites.csv'))
    full_data, manager_metrics = assemble_df(costar_export)
    full_data.to_csv('data/processed/asset_metrics.csv', index=False)
    manager_metrics.to_csv('data/processed/manager_metrics.csv', index=False)

if __name__ == "__main__":
    main()