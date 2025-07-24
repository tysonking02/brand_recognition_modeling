import pandas as pd
import statsmodels.api as sm
import pickle
from datetime import datetime
import shutup
from tqdm import tqdm

shutup.please()

def clean_df(df):
    return df.loc[:, ~df.columns.str.startswith('Unnamed')]

distances_from_city_center = clean_df(pd.read_csv('data/processed/distances_from_city_center.csv'))
hellodata_features = clean_df(pd.read_csv('data/processed/hellodata_features.csv'))
brand_recognition = clean_df(pd.read_csv('data/processed/brand_recognition.csv'))
google_reviews = clean_df(pd.read_csv('data/processed/google_reviews.csv', usecols=['property_id', 'rating']))
costar_export = clean_df(pd.read_csv('data/raw/costar_export.csv'))
branded_sites = clean_df(pd.read_csv('data/processed/branded_sites.csv'))

branded_sites = pd.merge(branded_sites, brand_recognition, on=['market', 'manager'], how='right').drop(columns=['aided_recognition','unaided_recognition','total_responses'])

def assemble_df(branded_sites):
    full_data = pd.merge(branded_sites, distances_from_city_center, on='property_id', how='inner')

    full_data = pd.merge(full_data, hellodata_features, left_on='property_id', right_on='costar_id', how='inner').drop(columns=['costar_id'])

    full_data = pd.merge(full_data, brand_recognition, on=['market', 'manager'], how='right')
    full_data.dropna(subset=['property_id'], inplace=True)

    full_data = pd.merge(full_data, google_reviews, on='property_id', how='left')

    full_data = pd.merge(full_data, costar_export.drop(columns=['unit_count', 'latitude', 'longitude', 'zip_code']), on='property_id', how='left')

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
        'branded', 'property', 'owner', 'latitude', 'longitude',
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

model_name = 'models/All Markets - Aided - Manager.pkl'

with open(model_name, 'rb') as f:
    model = pickle.load(f)

cols = model.model.exog_names

asset_preds = pd.DataFrame()
num_iters = len(branded_sites['property_id'].unique())
progress = tqdm(total=num_iters, desc='Processing...')

for manager in branded_sites['manager'].unique():
    for market in branded_sites['market'].unique():

        filtered = branded_sites[(branded_sites['manager'] == manager) & (branded_sites['market'] == market)].copy()

        full_data, overall_train = assemble_df(filtered)

        if overall_train.empty:
            continue

        actual_rec = float(
            brand_recognition.loc[
                (brand_recognition['manager'] == manager) &
                (brand_recognition['market'] == market),
                'aided_recognition'
            ]
        )

        overall_train = sm.add_constant(overall_train, has_constant='add')
        overall_train = overall_train.reindex(columns=cols, fill_value=0)
        overall_pred = model.predict(overall_train).mean()

        for _, row in filtered.iterrows():
            without_cur = filtered[filtered['property_id'] != row['property_id']]
            full_data, train = assemble_df(without_cur)
            train = sm.add_constant(train, has_constant='add')
            train = train.reindex(columns=cols, fill_value=0)

            pred = model.predict(train).mean()
            impact = overall_pred - pred

            if pd.isna(impact):
                progress.update(1)
                continue

            row_data = row.to_dict()
            row_data['impact'] = impact
            asset_preds = pd.concat([asset_preds, pd.DataFrame([row_data])], axis=0, ignore_index=True)

            progress.update(1)

progress.close()

asset_preds.to_csv('data/processed/asset_preds.csv')





