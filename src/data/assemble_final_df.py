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

    full_data = full_data[(full_data['IsAffordable'] == 0) & (full_data['IsLeaseUp'] == 0)].drop(columns=['IsAffordable', 'IsLeaseUp'])

    full_data['MedianHHI'] = full_data['MedianHHI'].replace('-', np.nan)
    full_data['MedianHHI'] = pd.to_numeric(full_data['MedianHHI'], errors='coerce')


    dummify_cols = ['area_type', 'market']

    full_data = pd.get_dummies(full_data, columns=dummify_cols)

    bool_cols = ['manager_brand', 'area_type_Rural', 'area_type_Suburban', 'area_type_Urban']

    for col in bool_cols:
        full_data[col] = full_data[col].astype(int)

    full_data = full_data.drop(columns=['property_id', 'branded', 'property', 'owner', 'latitude', 'longitude', 'submarket', 'zip_code', 'closest_city_center', 'total_responses'])

    cols_to_fill = [col for col in full_data.columns if col not in ['unaided_recognition', 'market', 'manager']]

    error_cols = []

    for col in cols_to_fill:
        full_data[col] = full_data[col].fillna(full_data[col].mean())

    # Create distance bands as indicator columns
    full_data['within_1'] = (full_data['miles_from_city_center'] <= 1).astype(int)
    full_data['within_5'] = (full_data['miles_from_city_center'] <= 5).astype(int)
    full_data['within_10'] = (full_data['miles_from_city_center'] <= 10).astype(int)
    full_data['within_20'] = (full_data['miles_from_city_center'] <= 20).astype(int)

    # Group by manager/market and count how many properties fall in each band
    grouped_counts = (
        full_data.groupby(['manager', 'market'])[['within_1', 'within_5', 'within_10', 'within_20']]
        .sum()
        .rename(columns={
            'within_1': 'count_within_1',
            'within_5': 'count_within_5',
            'within_10': 'count_within_10',
            'within_20': 'count_within_20',
        })
    )

    # Compute log(count + 1) to model diminishing returns
    for col in grouped_counts.columns:
        grouped_counts[f'log_{col}'] = np.log1p(grouped_counts[col])

    # Compute unit/asset counts
    unit_asset_counts = (
        full_data.groupby(['manager', 'market'])
        .agg(
            unit_count=('unit_count', 'sum'),
            asset_count=('unit_count', 'count')
        )
    )

    # Get remaining numeric features (excluding engineered columns)
    excluded = [
        'unit_count', 'distance_to_city_center',
        'within_1', 'within_5', 'within_10', 'within_20'
    ]
    numeric_cols = full_data.select_dtypes(include='number').columns.difference(excluded)

    # Get means of other numeric features
    other_metrics = full_data.groupby(['manager', 'market'])[numeric_cols].mean()

    # Combine all together
    manager_metrics = (
        pd.concat([unit_asset_counts, grouped_counts, other_metrics], axis=1)
        .reset_index()
    )

    return full_data, manager_metrics

def main():
    costar_export = clean_df(pd.read_csv('data/processed/branded_sites.csv'))
    full_data, manager_metrics = assemble_df(costar_export)
    full_data.to_csv('data/processed/asset_metrics.csv', index=False)
    manager_metrics.to_csv('data/processed/manager_metrics.csv', index=False)