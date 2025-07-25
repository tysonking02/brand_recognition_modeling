import streamlit as st
import pandas as pd
import leafmap.foliumap as leafmap
from folium.plugins import HeatMap
import re
from shapely.geometry import Polygon
import geopandas as gpd
import unicodedata
import matplotlib.colors as mcolors
import pickle
import numpy as np
import statsmodels.api as sm
from datetime import datetime
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

def clean_df(df):
    return df.loc[:, ~df.columns.str.startswith('Unnamed')]

distances_from_city_center = clean_df(pd.read_csv('data/processed/distances_from_city_center.csv'))
hellodata_features = clean_df(pd.read_csv('data/processed/hellodata_features.csv'))
brand_recognition = clean_df(pd.read_csv('data/processed/brand_recognition.csv'))
google_reviews = clean_df(pd.read_csv('data/processed/google_reviews.csv', usecols=['property_id', 'rating']))
costar_export = clean_df(pd.read_csv('data/raw/costar_export.csv'))


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

def clean_text(val):
    if isinstance(val, str):
        return unicodedata.normalize("NFKD", val).encode("ascii", "ignore").decode("ascii")
    return val

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df['manager'] = df['manager'].fillna('').astype(str).str.strip()
    return df

with open('models/All Markets - Aided - Manager.pkl', 'rb') as f:
    model = pickle.load(f)

df = load_data('data/processed/branded_sites.csv')

df['property'] = df['property'].apply(clean_text)
df['manager'] = df['manager'].apply(clean_text)
df['owner'] = df['owner'].apply(clean_text)

manager_logo_map = {
    'AMLI': 'https://media.licdn.com/dms/image/v2/C560BAQGnxhMQWfLpjA/company-logo_200_200/company-logo_200_200/0/1630613538782/amli_residential_logo?e=2147483647&v=beta&t=V-mBBUgDZ6KajQ6XKjkbSpIqh382Wb2hN6_8BkMnUM0',
    'AvalonBay': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTSjZB_WdcnA6UHxoTZh9e7ewgMARfPHRdsGg&s',
    'Bell': 'https://media.glassdoor.com/sqll/2590407/bell-management-squarelogo-1644392295469.png',
    'Camden': 'https://www.camdenliving.com/images/camden-logo-black.png',
    'Cortland': 'https://www.multifamilyexecutive.com/wp-content/uploads/sites/3/2018/cortland-logo-stacked-rgb.png?w=1024',
    'FPA': 'https://images1.apartments.com/i2/6yzvf1jIv9iWltohdLCwGaBOliNpGgXLLH7jE3BUH5Y/110/image.png',
    'Fairfield': 'https://images1.apartments.com/i2/AejWy_Wu0361EimbUJCaiGPRRQ5TwLcgkECUYjSupZo/110/image.jpg',
    'GID': 'https://media.licdn.com/dms/image/v2/D560BAQFec0alUyaRAQ/company-logo_200_200/B56ZZfFBgeGcAM-/0/1745351876017/windsorm_logo?e=2147483647&v=beta&t=Pv905gG85FrKuZrGBKgY4P135mWvkgnXiUQYgZ6w7fs',
    'Greystar': 'https://mma.prnewswire.com/media/1761396/greystar_Logo.jpg?p=facebook',
    'MAA': 'https://cdn.cookielaw.org/logos/7d7d223a-000d-4093-b8e4-356348d54408/018fdf5b-7162-738e-a17d-542945beefb7/9f203eab-91c5-4baf-8b5b-c4592cd027e3/MAA_logo_with_R.png',
    'Mill Creek': 'https://mma.prnewswire.com/media/224987/mill_creek_logo.jpg?p=facebook'
}

bin_side_length = 0.01
num_city_centers = 5

# Calculate city centers with dynamic bin size
df['lat'] = round((df['latitude'] / bin_side_length).round() * bin_side_length, 3)
df['lon'] = round((df['longitude'] / bin_side_length).round() * bin_side_length, 3)

tile_density = (
    df.groupby(['market', 'lat', 'lon'])
    .agg(
        total_units=('unit_count', 'sum'),
        total_assets=('property_id', 'count')
    )
    .reset_index()
)

tile_density = tile_density[tile_density['total_assets'] >= 3]

top_tiles = (
    tile_density.sort_values(['market', 'total_units'], ascending=[True, False])
    .groupby('market')
    .head(num_city_centers)
    .copy()
)

filtered = df[df['manager'].isin([
    'Fairfield', 'Bell', 'Mill Creek', 'GID', 'FPA', 'AMLI', 
    'Greystar', 'Camden', 'Cortland', 'AvalonBay', 'MAA'
])]

# Read in Survey Data

market_map = {
    'Atlanta': 'Atlanta, GA',
    'Austin': 'Austin, TX',
    'Charlotte': 'Charlotte, NC',
    'Columbus': 'Columbus, OH',
    'DC': 'Washington, DC',
    'Dallas': 'Dallas-Fort Worth, TX',
    'Denver': 'Denver, CO',
    'Houston': 'Houston, TX',
    'Nashville': 'Nashville, TN',
    'Orlando': 'Orlando, FL',
    'Phoenix': 'Phoenix, AZ',
    'Raleigh': 'Raleigh, NC',
    'South Florida': 'Miami, FL',
    'Tampa': 'Tampa, FL',
    'Tucson': 'Tucson, AZ'
}

manager_map = {
    'amli': 'AMLI',
    'avalon': 'AvalonBay',
    'camden': 'Camden',
    'cortland': 'Cortland',
    'greystar': 'Greystar',
    'maa': 'MAA',
    'pb_bell': 'Bell',
    'windsor': 'GID'
}

recognition_df = pd.read_csv('data/processed/brand_recognition.csv')

# only markets with 50+ properties
counts = filtered['market'].value_counts()
eligible_markets = counts[(counts >= 50) & (counts.index.isin(recognition_df['market']))].index.tolist()
markets = ['All'] + sorted(eligible_markets)

atl_i = markets.index("Atlanta, GA")

st.sidebar.subheader('Brand Recognition Contribution')

market = st.sidebar.selectbox("Market", markets, index=atl_i)

filtered = filtered[filtered['market'] == market] if market != 'All' else filtered
center_lat = filtered['latitude'].mean()
center_lon = filtered['longitude'].mean()
zoom = 10

hotspots = top_tiles[top_tiles['market'] == market]

# Rename columns
hotspots = hotspots.rename(columns={
    'lat': 'latitude',
    'lon': 'longitude',
    'total_units': '# Units',
    'total_assets': '# Assets'
})

hotspots['Center Rank'] = (
    hotspots.groupby('market')['# Units']
    .rank(method='first', ascending=False)
    .astype(int)
)

# Create popup text
hotspots[''] = (
    'City Center #' + hotspots['Center Rank'].astype(str) + '<br>' +
    'Lat: ' + hotspots['latitude'].astype(str) + '<br>' +
    'Lon: ' + hotspots['longitude'].astype(str) + '<br>' +
    '# Units: ' + hotspots['# Units'].astype(str) + '<br>' +
    '# Assets: ' + hotspots['# Assets'].astype(str)
)

valid_pairs = recognition_df[['manager', 'market']].drop_duplicates()

filtered = (
    filtered
    .merge(valid_pairs, on=['manager', 'market'], how='inner')
)

managers = sorted(filtered['manager'].unique())

default_manager = ['Cortland'] if 'Cortland' in managers else [managers[0]]

manager_select = st.sidebar.multiselect("Management", managers, default=default_manager)

if len(manager_select) == 0:
    st.warning("Please select at least one management company to proceed.")
    st.stop()

default_colors = {
    'AMLI': '#a7292e',
    'AvalonBay': '#4a2e89',
    'Bell': '#31999b',
    'Camden': '#84be30',
    'Cortland': '#284692',
    'FPA': '#f59128',
    'Fairfield': '#a7632d',
    'GID': '#222707',
    'Greystar': '#eac584',
    'MAA': '#e6752e',
    'Mill Creek': '#7aaeb6'
}

manager_color_map = {}

with st.sidebar.expander(label='Heatmap Settings', expanded=True):
    heatmap_radius = st.slider("Radius", min_value=5, max_value=50, value=20, step=1)
    heatmap_blur = st.slider("Blur", min_value=1, max_value=30, value=10, step=1)

    st.markdown("Customize brand colors:")
    for mgr in manager_select:
        default = default_colors.get(mgr, '#888888')
        manager_color_map[mgr] = st.color_picker(f"{mgr} Color", value=default)

filtered = filtered[filtered['manager'].isin(manager_select)]
filtered['impact'] = 0

expected_cols = model.model.exog_names

important_feats = pd.DataFrame()

for manager in filtered['manager'].unique():

    if market == "All":
        actual_rec = recognition_df[recognition_df['manager'] == manager]['aided_recognition'].mean()
        zoom = 4
    else:
        actual_rec = float(recognition_df.loc[(recognition_df['manager'] == manager) & (recognition_df['market'] == market), 'aided_recognition'])

    cur_manager_filtered = filtered[filtered['manager'] == manager].copy()

    full_data, overall_train = assemble_df(cur_manager_filtered)

    important_feats = pd.concat([important_feats, full_data])

    overall_train.drop(columns=['manager', 'market'], inplace=True)
    overall_train = sm.add_constant(overall_train, has_constant='add')
    overall_train = overall_train.reindex(columns=expected_cols, fill_value=0)

    overall_pred = model.predict(overall_train).mean()

    marginal_impacts = []
    property_ids = []

    progress_bar = st.progress(0)
    total = len(cur_manager_filtered)
    status = st.empty()

    for i, (_, asset) in enumerate(cur_manager_filtered.iterrows()):   
        status.text(f"Processing {manager}...")     
        property_ids.append(asset['property_id'])

        without_asset_df = cur_manager_filtered[cur_manager_filtered['property_id'] != asset['property_id']]
        _, train_data = assemble_df(without_asset_df)
        train_data.drop(columns=['manager', 'market'], inplace=True)
        train_data = sm.add_constant(train_data, has_constant='add')
        train_data = train_data.reindex(columns=expected_cols, fill_value=0)

        pred_without = model.predict(train_data).mean()
        impact = overall_pred - pred_without
        marginal_impacts.append(impact)

        progress_bar.progress((i + 1) / total)
    
    progress_bar.empty()
    status.empty()

    marginal_impacts = np.array(marginal_impacts)

    min_impact = marginal_impacts.min()
    shifted = marginal_impacts - min_impact 

    if shifted.sum() > 0:
        scaled = (shifted / shifted.sum()) * overall_pred
    else:
        scaled = np.full_like(shifted, overall_pred / len(shifted))

    for prop_id, impact in zip(property_ids, scaled):
        adj_impact = (actual_rec / overall_pred) * impact
        filtered.loc[filtered['property_id'] == prop_id, 'impact'] = round(adj_impact, 4)

filtered = pd.merge(filtered, important_feats.drop(columns=['manager', 'market', 'lat', 'lon', 'impact', 'unit_count', 'manager_brand']), on=['property_id'], how='left')


data = []

for mgr in manager_select:
    filtered_df = filtered[filtered['manager'] == mgr]
    total_assets = len(filtered_df)
    branded_assets = len(filtered_df[filtered_df['branded'] == True])
    total_units = filtered_df['unit_count'].sum()
    branded_units = filtered_df.loc[filtered_df['branded'] == True, 'unit_count'].sum()

    if market != 'All':
        match = recognition_df[
            (recognition_df['manager'] == mgr) & (recognition_df['market'] == market)
        ]

        if not match.empty:
            recognition_row = match.iloc[0]
            aided = f"{recognition_row['aided_recognition']:.2%}"
            unaided = f"{recognition_row['unaided_recognition']:.2%}"
        else:
            aided = 'N/A'
            unaided = 'N/A'

        market_label = market
    else:
        manager_rows = recognition_df[recognition_df['manager'] == mgr]

        if not manager_rows.empty:
            total_count = manager_rows['total_responses'].sum()
            aided_val = (manager_rows['aided_recognition'] * manager_rows['total_responses']).sum() / total_count
            unaided_val = (manager_rows['unaided_recognition'] * manager_rows['total_responses']).sum() / total_count
            aided = f"{aided_val:.2%}"
            unaided = f"{unaided_val:.2%}"
        else:
            aided = 'N/A'
            unaided = 'N/A'

        market_label = 'National'

    logo_url = manager_logo_map.get(mgr, "")
    logo_html = f'<div style="text-align:center;"><img src="{logo_url}" width="100"/></div>' if logo_url else ''

    data.append({
        'Logo': logo_html,
        'Manager': mgr,
        'Market': market_label,
        'Aided Recognition': aided,
        'Unaided Recognition': unaided,
        'Total Assets': f"{total_assets:,}",
        'Branded Assets': f"{branded_assets:,}",
        'Total Units': f"{round(total_units):,}",
        'Branded Units': f"{round(branded_units):,}"
    })

df_display = pd.DataFrame(data)

html_table = df_display.to_html(escape=False, index=False)

html_table = html_table.replace(
    "<thead>",
    "<thead><style>th { text-align: center !important; }</style>",
)

st.markdown(html_table, unsafe_allow_html=True)

filtered['value'] = filtered['impact'] * 100

def create_gradient(base_hex):
    base_rgb = mcolors.to_rgb(base_hex)
    gradient = {
        0.2: mcolors.to_hex([min(1, c + 1) for c in base_rgb]),
        0.5: mcolors.to_hex([min(1, c + 0.5) for c in base_rgb]),
        0.8: base_hex,
        1.0: mcolors.to_hex([max(0, c - 0.5) for c in base_rgb])
    }
    return gradient

# Create map
m = leafmap.Map()
m.add_basemap("CartoDB.Positron")

for manager in manager_select:
    manager_df = filtered[filtered['manager'] == manager]
    base_color = manager_color_map.get(manager, '#888888')
    gradient = create_gradient(base_color)

    m.add_heatmap(
        data=manager_df,
        latitude="latitude",
        longitude="longitude",
        value="value",
        name=f"{manager} Heatmap",
        radius=heatmap_radius,
        blur=heatmap_blur,
        gradient=gradient
    )

def find_submarket(lat, lon):
    lat_min = lat - bin_side_length / 2
    lat_max = lat + bin_side_length / 2
    lon_min = lon - bin_side_length / 2
    lon_max = lon + bin_side_length / 2

    properties_in_bin = df[
        (df['lat'] >= lat_min) & (df['lat'] < lat_max) &
        (df['lon'] >= lon_min) & (df['lon'] < lon_max)
    ].dropna(subset='submarket')

    if not properties_in_bin.empty:
        return properties_in_bin['submarket'].mode().iloc[0]
    else:
        return None

if market != 'All':
    # Create square polygons with dynamic bin size
    half_bin_size = bin_side_length / 2
    geometries = []
    popups = []

    for _, row in hotspots.iterrows():
        lat, lon = row['latitude'], row['longitude']
        rank = row['Center Rank']

        city_center_name = find_submarket(lat, lon)

        square = Polygon([
            (lon - half_bin_size, lat - half_bin_size),
            (lon + half_bin_size, lat - half_bin_size),
            (lon + half_bin_size, lat + half_bin_size),
            (lon - half_bin_size, lat + half_bin_size),
            (lon - half_bin_size, lat - half_bin_size)
        ])
        geometries.append(square)

        popups.append(
            f"<b>{city_center_name}</b><br><br>"
            f"Rank: {rank}<br>"
            f"Market: {row['market']}<br>"
            f"Lat: {lat}<br>"
            f"Lon: {lon}<br>"
            f"# Units: {row['# Units']}<br>"
            f"# Assets: {row['# Assets']}"
        )

    gdf = gpd.GeoDataFrame({
        '': popups,
        'geometry': geometries
    })

    m.add_gdf(gdf, layer_name='City Centers', info_mode='on_click', stroke_color='black', fill_opacity=0.5, show=True)

def format_popup(row):
    return (
        f"<br><br><b>Property:</b> {row['property']}<br>"
        f"<b>Manager:</b> {row['manager']}<br>"
        f"<b>Impact:</b> {row['impact']:.1%}<br>"
        f"<b>Miles from City Center:</b> {row['miles_from_city_center']:.2f}<br>"
        f"<b>Manager Branded:</b> {row['manager_brand']}<br>"
        f"<b>Stories:</b> {int(row['number_of_stories']) if not pd.isnull(row['number_of_stories']) else 'N/A'}<br>"
        f"<b>Unit Count:</b> {row['unit_count']:.0f}<br>"
        f"<b>Building Age:</b> {row['building_age']:.0f}<br>"
        f"<b>Years Since Reno:</b> {int(row['years_since_reno']) if not pd.isnull(row['years_since_reno']) else 'N/A'}<br>"
        f"<b>Years Since Acquisition:</b> {int(row['years_since_acquisition']) if not pd.isnull(row['years_since_acquisition']) else 'N/A'}<br>"
        f"<b>HelloData Quality Score:</b> {row['property_quality']:.3f}<br>"
    )

# Apply to filtered
filtered['Details'] = filtered.apply(format_popup, axis=1)

m.add_circle_markers_from_xy(
    data=filtered,
    x='longitude',
    y='latitude',
    popup='Details',
    layer_name='Properties',
    radius=4,
    fill_opacity=0.3,
    stroke=False,
    show=True
)

filtered['impact'] = (filtered['impact'] * 100).round(2)

styled_df = filtered[['property', 'manager', 'impact']].sort_values('impact', ascending=False).rename(columns={'property': 'Property', 'manager': 'Manager', 'impact': 'Brand Awareness Contribution'})

gb = GridOptionsBuilder.from_dataframe(styled_df)
gb.configure_selection(
    selection_mode="single",    
    use_checkbox=True 
)

gb.configure_default_column(filter=True)
grid_options = gb.build()

st.markdown("<br>", unsafe_allow_html=True)

grid_response = AgGrid(
    styled_df,
    gridOptions=grid_options,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    allow_unsafe_jscode=True,
    height=300,
    theme="light" 
)

st.markdown("<br>", unsafe_allow_html=True)

selected_rows = grid_response["selected_rows"]
if selected_rows is not None:
    selected_asset = selected_rows["Property"]
    row = filtered[filtered['property'] == selected_asset[0]]

    m.set_center(lat=row['latitude'].values[0], lon=row['longitude'].values[0], zoom=15)

    m.add_marker(location=(row['latitude'].values[0], row['longitude'].values[0]), 
                 tooltip=row['Details'].values[0])
else:
    m.set_center(lat=center_lat, lon=center_lon, zoom=zoom)

m.to_streamlit(height=700)