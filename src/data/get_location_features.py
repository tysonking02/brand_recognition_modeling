import pandas as pd
import numpy as np

costar_export = pd.read_csv('data/processed/branded_sites.csv')

# region Find City Centers and Save to File

print(f'Finding city centers for {len(costar_export['market'].unique())} markets\n')

bin_side_length = 0.01

costar_export['lat'] = round((costar_export['latitude'] / bin_side_length).round() * bin_side_length, 3)
costar_export['lon'] = round((costar_export['longitude'] / bin_side_length).round() * bin_side_length, 3)

tile_density = (
    costar_export.groupby(['market', 'lat', 'lon'])
    .agg(
        total_units=('unit_count', 'sum'),
        total_properties=('property_id', 'count')
    )
    .reset_index()
)

city_centers = (
    tile_density.sort_values(['market', 'total_units'], ascending=[True, False])
    .groupby('market')
    .head(5)
    .copy()
)

city_centers['rank'] = (
    city_centers.groupby('market')
    .cumcount() + 1
)

def find_submarket(lat, lon):
    lat_min = lat - bin_side_length / 2
    lat_max = lat + bin_side_length / 2
    lon_min = lon - bin_side_length / 2
    lon_max = lon + bin_side_length / 2

    properties_in_bin = costar_export[
        (costar_export['lat'] >= lat_min) & (costar_export['lat'] < lat_max) &
        (costar_export['lon'] >= lon_min) & (costar_export['lon'] < lon_max)
    ].dropna(subset='submarket')

    if not properties_in_bin.empty:
        return properties_in_bin['submarket'].mode().iloc[0]
    else:
        return None

city_centers_list = []

for i, row in city_centers.iterrows():
    submarket = find_submarket(row['lat'], row['lon'])

    city_centers_list.append({
        'market': row['market'],
        'rank': row['rank'],
        'total_units': row['total_units'],
        'total_properties': row['total_properties'],
        'lat': row['lat'],
        'lon': row['lon'],
        'submarket': submarket
    })

city_centers_df = pd.DataFrame(city_centers_list)

city_centers_df.to_csv('data/processed/city_centers.csv')

# endregion

# region Find Distance to Closest City Center from Costar Data

def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8  # Earth radius in miles
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

def find_closest_city_center(lat, lon, market):
    centers = city_centers_df[city_centers_df['market'] == market]

    if centers.empty:
        return None, None

    distances = haversine(lat, lon, centers['lat'], centers['lon'])
    closest_idx = distances.idxmin()
    closest_row = centers.loc[closest_idx]
    closest_distance = distances.loc[closest_idx]

    return closest_row, closest_distance

def get_closest_info(row):
    center, distance = find_closest_city_center(row['latitude'], row['longitude'], row['market'])
    return pd.Series([center['submarket'] if center is not None else None, distance])

print(f'Finding closest city center for {len(costar_export)} properties\n')

costar_export[['closest_city_center', 'miles_from_city_center']] = costar_export.apply(get_closest_info, axis=1)

costar_export[['property_id', 'closest_city_center', 'miles_from_city_center']].to_csv('data/processed/distances_from_city_center.csv')

# endregion

