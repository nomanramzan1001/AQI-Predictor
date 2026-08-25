"""
Feature pipeline: It fetches current AQI from Open-Meteo and writes it to Hopsworks.

"""

import json
import requests
import hopsworks
from datetime import datetime, timezone
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")
FEATURE_GROUP_NAME = os.getenv("FEATURE_GROUP_NAME")
FEATURE_GROUP_VERSION = os.getenv("FEATURE_GROUP_VERSION")

LATITUDE = 31.5204
LONGITUDE = 74.3587
CITY_NAME = "LAHORE"


# --------------------------------------------------------------------------------------
# Fetch Air Quality Data (Core Pollutants) and Weather Data (Meteorological Factors)
# -------------------------------------------------------------------------------------
def fetch_raw_data() -> dict:
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "current": [
            "pm10", "pm2_5", "carbon_monoxide",
            "nitrogen_dioxide", "ozone", "sulphur_dioxide",
            "us_aqi",
        ],
    }
    response = requests.get(url, params = params, timeout = 60)
    response.raise_for_status()
    return response.json()  


# ---------------------------------------------------------------------------
# Build feature row
# ---------------------------------------------------------------------------

def compute_features(data : dict) -> pd.DataFrame:

    now = datetime.now(timezone.utc)

    current = data.get('current' , {})

    aqi = int(current.get('us_aqi' , 0))

    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        host="eu-west.cloud.hopsworks.ai",
        project=HOPSWORKS_PROJECT_NAME
    )

    # Access the feature store
    fs = project.get_feature_store()

    # Example: Read an existing feature group
    fg = fs.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=1
    )

    # Read data from the feature group
    df = fg.read()
    
    last_row = df.iloc[-1]

    pre_aqi = last_row["us_aqi"]

    row = {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "city": CITY_NAME,
        "hour": now.hour,
        "day": now.day,
        "month": now.month,
        "day_of_week": now.weekday(),
        "pm2_5": float(current.get("pm2_5", 0) or 0),
        "pm10": float(current.get("pm10", 0) or 0),
        "carbon_monoxide": float(current.get("carbon_monoxide", 0) or 0),
        "nitrogen_dioxide": float(current.get("nitrogen_dioxide", 0) or 0),
        "ozone": float(current.get("ozone", 0) or 0),
        "sulphur_dioxide": float(current.get("sulphur_dioxide", 0) or 0),
        "us_aqi": aqi,
        "aqi_change_rate": int(aqi-pre_aqi)
    }

    df = pd.DataFrame([row])

    return df

# ---------------------------------------------------------------------------
# Fetch IDs (project_id, feature_store_id , feature_group_id)
# ---------------------------------------------------------------------------

headers = {
    "Authorization": f"ApiKey {HOPSWORKS_API_KEY}",
    "Accept": "application/json",
    }

def get_hopsworks_project_id() -> int:
    
    url = f"https://eu-west.cloud.hopsworks.ai/hopsworks-api/api/project/getProjectInfo/{HOPSWORKS_PROJECT_NAME}"
    response = requests.get(
        url =url,
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()

    project_id = response.json()["projectId"]

    return project_id


def get_feature_store_id(project_id : int) -> int:

    response = requests.get(
        f"https://eu-west.cloud.hopsworks.ai/hopsworks-api/api/project/{project_id}/featurestores",
        headers=headers,
        timeout=30,
        )

    response.raise_for_status()

    stores_list = response.json()

    # The default feature store has the same name as the (project_name in lower-cased)_featurestore
    project_feature_store_name = f"{HOPSWORKS_PROJECT_NAME.lower()}_featurestore"

    for store in stores_list:

        if store.get("featurestoreName", "").lower() == project_feature_store_name:

            feature_store_id = store["featurestoreId"]

            print(f"feature_store_id = {feature_store_id}")
            return feature_store_id

    # Fallback: return the first one
    feature_store_id = stores_list[0]["featurestoreId"]
    print(f"feature_store_id (fallback_case) = {feature_store_id}")
    return feature_store_id


def get_feature_group_id(project_id : int , feature_store_id : int) -> int:

    resp = requests.get(
        f"https://eu-west.cloud.hopsworks.ai/hopsworks-api/api/project/{project_id}/featurestores/{feature_store_id}/featuregroups/{FEATURE_GROUP_NAME}",
        headers=headers,
        params={"version": FEATURE_GROUP_VERSION},
        timeout=30,
    )

    resp.raise_for_status()

    response = resp.json()

    fg_id = response[0]["id"]

    print(f"feature_group_id = {fg_id}")

    return fg_id

# ---------------------------------------------------------------------------
# Insert features into the Hopsworks
# ---------------------------------------------------------------------------

def insert_features(df: pd.DataFrame):
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        host="eu-west.cloud.hopsworks.ai",
        project=HOPSWORKS_PROJECT_NAME
    )

    fs = project.get_feature_store()

    fg = fs.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION
    )

    print(f"Feature group type: {type(fg).__name__}")

    print("Inserting row...")
    fg.insert(df)
    print("Insert complete.")


if __name__ == "__main__":
    print("--- Step 1: Fetch AQI data ---")
    raw_data = fetch_raw_data()
    print("Fetched.")

    print("--- Step 2: Compute features ---")
    df = compute_features(raw_data)
    print(df)

    print("--- Step 3: Resolve Hopsworks IDs ---")
    project_id = get_hopsworks_project_id()
    fs_id = get_feature_store_id(project_id)
    fg_id = get_feature_group_id(project_id, fs_id)

    print("--- Step 4: Insert ---")
    insert_features(df)

    print("Done!")


