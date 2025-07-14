import pandas as pd
import requests
import time
import json
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import shutup

shutup.please()

api_key = "f3898958-b76a-4a47-816f-0294f0c5103d"

BASE_URL = "https://api.hellodata.ai"

HEADERS = {
    "x-api-key": api_key
}

costar = pd.read_csv('data/raw/costar_export.csv')
costar = costar[(costar['unit_count'] >= 100)]


def make_request(url, headers, params=None, max_retries=5, backoff_factor=1, timeout=10):
    """Helper function to make a GET request with exponential backoff for rate limits, timeouts, and 503 errors."""
    attempt = 0
    while attempt < max_retries:
        try:
            response = requests.get(url, headers=headers, params=params, timeout=timeout)
            if response.status_code == 200:
                return response
            elif response.status_code in [429, 503]:
                retry_after = response.headers.get('Retry-After')
                if retry_after:
                    sleep_time = int(retry_after)
                else:
                    sleep_time = backoff_factor * (2 ** attempt)
                print(f"\nReceived {response.status_code} on {url}. Sleeping for {sleep_time} seconds (attempt {attempt+1}/{max_retries})")
                time.sleep(sleep_time)
                attempt += 1
            else:
                response.raise_for_status()
        except requests.exceptions.Timeout:
            sleep_time = backoff_factor * (2 ** attempt)
            print(f"\nTimeout on {url}. Sleeping for {sleep_time} seconds (attempt {attempt+1}/{max_retries})")
            time.sleep(sleep_time)
            attempt += 1
        except requests.exceptions.RequestException as e:
            print(f"\nRequest failed on {url}: {e}")
            raise
    raise ValueError(f"Max retries exceeded for URL: {url}")

def fetch_property_data(property, address, lat, lon):
    """Function to fetch property data using property name and zip code."""
    querystring = {}
    querystring['q'] = f'{property} {address}'
    querystring['lat'] = lat
    querystring['lon'] = lon

    url = f"{BASE_URL}/property/search"
    response = make_request(url, headers=HEADERS, params=querystring)
    try:
        data = response.json()
        return data if data and len(data) > 0 else None
    except ValueError as e:
        raise ValueError(f"Error parsing JSON response from property search: {e}")

def fetch_property_details(property_id):
    """Function to fetch details for a specific property."""
    url = f"{BASE_URL}/property/{property_id}"
    response = make_request(url, headers=HEADERS)
    try:
        data = response.json()
        return data
    except ValueError as e:
        raise ValueError(f"Error parsing JSON response from property details: {e}")
    
def aggregate_details(costar_id, property_name, property_data, property_details):
    
    if len(property_data) == 0 or len(property_details) == 0:
        return pd.DataFrame()
    
    hellodata_id = property_data[0].get('id')
    city = property_data[0].get('city')
    state = property_data[0].get('state')
    num_units = property_data[0].get('number_units')
    street_address = property_data[0].get('street_address')

    number_stories = property_details.get('number_stories')

    is_lease_up = property_details.get('is_lease_up')
    is_single_family = property_details.get('is_single_family')
    is_condo = property_details.get('is_condo')
    is_apartment = property_details.get('is_apartment')
    is_senior = property_details.get('is_senior')
    is_student = property_details.get('is_student')
    is_affordable = property_details.get('is_affordable')

    cats_monthly_rent = property_details.get('cats_monthly_rent', 0) or 0
    cats_one_time_fee = property_details.get('cats_one_time_fee', 0) or 0
    cats_deposit = property_details.get('cats_deposit', 0) or 0
    dogs_monthly_rent = property_details.get('dogs_monthly_rent', 0) or 0
    dogs_one_time_fee = property_details.get('dogs_one_time_fee', 0) or 0
    dogs_deposit = property_details.get('dogs_deposit', 0) or 0
    admin_fee = property_details.get('admin_fee', 0) or 0
    amenity_fee = property_details.get('amenity_fee', 0) or 0
    application_fee = property_details.get('application_fee', 0) or 0
    storage_fee = property_details.get('storage_fee', 0) or 0

    building_quality = property_details.get('building_quality', {})
    property_quality = building_quality.get('property_overall_quality')
    bedroom_quality = building_quality.get('avg_quality_score_bedroom')
    kitchen_quality = building_quality.get('avg_quality_score_kitchen')
    bathroom_quality = building_quality.get('avg_quality_score_bathroom')
    dining_room_quality = building_quality.get('avg_quality_score_dining_room')
    common_areas_quality = building_quality.get('avg_quality_score_common_areas')
    fitness_center_quality = building_quality.get('avg_quality_score_fitness_center')
    laundry_room_quality = building_quality.get('avg_quality_score_laundry_room')
    living_room_quality = building_quality.get('avg_quality_score_living_room')
    main_entrance_quality = building_quality.get('avg_quality_score_main_entrance')
    stairs_hallway_quality = building_quality.get('avg_quality_score_stairs_hallway')
    swimming_pool_quality = building_quality.get('avg_quality_score_swimming_pool')
    walk_in_closet_quality = building_quality.get('avg_quality_score_walk_in_closet')

    demographics = property_details.get('demographics', {})

    total_pop = demographics.get('total_pop')
    land_area = demographics.get('land_area')
    pop_density = demographics.get('pop_density')
    unemployed_perc = demographics.get('unemployed_pop_perc')
    undergrad_perc = demographics.get('in_undergrad_college_perc')
    hs_diploma_perc = demographics.get('high_school_diploma_perc')
    bachelors_perc = demographics.get('bachelors_degree_perc')
    graduate_perc = demographics.get('graduate_professional_degree_perc')
    masters_perc = demographics.get('masters_degree_perc')

    details = pd.DataFrame([{
        'costar_id': costar_id,
        'hellodata_id': hellodata_id,
        'property': property_name,
        'city': city,
        'state': state,
        'street_address': street_address,
        'num_units': num_units,
        'number_stories': number_stories,
        'is_lease_up': is_lease_up,
        'is_single_family': is_single_family,
        'is_condo': is_condo,
        'is_apartment': is_apartment,
        'is_senior': is_senior,
        'is_student': is_student,
        'is_affordable': is_affordable,
        'cats_monthly_rent': cats_monthly_rent,
        'cats_one_time_fee': cats_one_time_fee,
        'cats_deposit': cats_deposit,
        'dogs_monthly_rent': dogs_monthly_rent,
        'dogs_one_time_fee': dogs_one_time_fee,
        'dogs_deposit': dogs_deposit,
        'admin_fee': admin_fee,
        'amenity_fee': amenity_fee,
        'application_fee': application_fee,
        'storage_fee': storage_fee,
        'property_quality': property_quality,
        'bedroom_quality': bedroom_quality,
        'kitchen_quality': kitchen_quality,
        'bathroom_quality': bathroom_quality,
        'dining_room_quality': dining_room_quality,
        'common_areas_quality': common_areas_quality,
        'fitness_center_quality': fitness_center_quality,
        'laundry_room_quality': laundry_room_quality,
        'living_room_quality': living_room_quality,
        'main_entrance_quality': main_entrance_quality,
        'stairs_hallway_quality': stairs_hallway_quality,
        'swimming_pool_quality': swimming_pool_quality,
        'walk_in_closet_quality': walk_in_closet_quality,
        'total_pop': total_pop,
        'land_area': land_area,
        'pop_density': pop_density,
        'unemployed_perc': unemployed_perc,
        'undergrad_perc': undergrad_perc,
        'hs_diploma_perc': hs_diploma_perc,
        'bachelors_perc': bachelors_perc,
        'graduate_perc': graduate_perc,
        'masters_perc': masters_perc
    }])

    return details

def process_property(args):
    _, row = args 
    try:
        costar_id = row['PropertyID']
        property_name = row['PropertyName']
        address = row['PropertyAddress']
        lat = row['Latitude']
        lon = row['Longitude']

        property_data = fetch_property_data(property_name, address, lat, lon)
        property_id = property_data[0].get('id') if property_data else None

        if not property_id:
            return None

        property_details = fetch_property_details(property_id)

        return aggregate_details(costar_id, property_name, property_data, property_details)

    except Exception as e:
        print(e)
        return None

def main():
    args = list(costar.iterrows())

    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_property, args), total=len(args)))

    property_details = [r for r in results if r is not None]
    property_details_df = pd.concat(property_details, ignore_index=True)

    property_details_df.to_csv('data/raw/property_details_full.csv')

if __name__ == "__main__":
    main()