import pandas as pd


cols = [
    'costar_id', 
    'is_affordable', 'is_lease_up', 'property_quality',
    'total_pop', 'land_area', 'pop_density', 'unemployed_perc', 'hs_diploma_perc', 
    'bachelors_perc', 'masters_perc'
]


property_details = pd.read_csv('data/raw/property_details_full.csv')
property_details = property_details[cols]

property_details.to_csv('data/processed/hellodata_features.csv')