import requests
import pandas as pd
from tqdm import tqdm

with open('google_api_key.txt', 'r') as f:
    api_key = f.read().strip()

def fetch_places_id(building_name, lat, lon):
    
    search_url = "https://places.googleapis.com/v1/places:searchText"

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.id"
    }

    body = {
        "textQuery": building_name,
        "locationBias": {
            "circle": {
                "center": {"latitude": lat, "longitude": lon},
                "radius": 5000
            }
        }
    }

    search_resp = requests.post(search_url, headers=headers, json=body)
    search_resp.raise_for_status()
    search_data = search_resp.json()

    places = search_data.get("places", [])

    if not places:
        print(f"No results found for '{building_name}' near {lat},{lon}")
        return
    
    place_id = places[0]["id"]

    return place_id

def fetch_review_data(place_id):

    details_url = f"https://places.googleapis.com/v1/places/{place_id}"
    
    params = {
        "key": api_key,
        "fields": "rating,userRatingCount"
    }

    details_resp = requests.get(details_url, params=params)
    details_resp.raise_for_status()
    details_data = details_resp.json()

    return {
        "rating": details_data.get("rating"),
        "count_reviews": details_data.get("userRatingCount")
    }

asset_details = pd.read_csv('data/processed/branded_sites.csv', usecols=['property_id', 'property', 'manager', 'market', 'latitude', 'longitude'])
brand_recognition = pd.read_csv('data/processed/brand_recognition.csv')

asset_details = asset_details[asset_details['manager'].isin(brand_recognition['manager'].unique()) & (asset_details['market'].isin(brand_recognition['market'].unique()))]

google_reviews_list = []

for i, row in tqdm(asset_details.iterrows(), total=len(asset_details), desc="Fetching Google reviews"):
    building_name = row['property']
    lat = row['latitude']
    lon = row['longitude']

    try:
        place_id = fetch_places_id(building_name, lat, lon)
        review_data = fetch_review_data(place_id)

        google_reviews_list.append({
            'property_id': row['property_id'],
            'property': building_name,
            'google_places_id': place_id,
            'rating': review_data['rating'],
            'count_reviews': review_data['count_reviews']
        })

    except Exception as e:
        print(f"Failed for {building_name}: {e}")

google_reviews_df = pd.DataFrame(google_reviews_list)

google_reviews_df.to_csv('data/processed/google_reviews.csv')