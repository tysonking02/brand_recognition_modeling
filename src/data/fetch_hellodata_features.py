import pandas as pd

hellodata_costar_ref = pd.read_csv('data/raw/hellodata_costar_ref.csv')

hellodata_cols = [
    'IsAffordable', 'IsLeaseUp', 'NumberStories', 'PropertyQuality',
    'BuildingAge', 'MedianHHI', 'MedianAge', 'PersonsPerHousehold',
    'PercentRenters', 'Population', 'LandArea', 'PopDensity', 'UnemployedPerc',
    'UndergradPerc', 'HighSchoolDiplomaPerc', 'BachelorsPerc', 'GraduatePerc', 'MastersPerc',
    'FamilyHouseholdsPerc', 'has_bldg_gated_community_access',
]


property_details = pd.read_csv('data/raw/property_details.csv')
property_details = property_details[['HelloDataID'] + hellodata_cols]

property_details = pd.merge(property_details, hellodata_costar_ref, left_on='HelloDataID', right_on='property_id', how='inner').drop(columns=['HelloDataID', 'property_id']).dropna(subset='costar_id')

cols = ['costar_id'] + [col for col in property_details.columns if col != 'costar_id']
property_details = property_details[cols]

property_details.to_csv('data/processed/hellodata_features.csv')