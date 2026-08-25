import requests
import pandas as pd
from datetime import datetime, timedelta
import hopsworks
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

def fetch_historical_aqi(start_date , end_date) -> dict:
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": [
            "pm10", "pm2_5", "carbon_monoxide","nitrogen_dioxide", 
            "ozone", "sulphur_dioxide","us_aqi", 
                ],
        "start_date": start_date,
        "end_date": end_date,
    }
    response = requests.get(url, params=params)

    response.raise_for_status()

    return response.json()

def process_historical_date(data) -> pd.DataFrame:

    hourly = data.get("hourly", {})

    times_list = hourly.get("time" , [])

    rows = []

    prev_aqi = None

    for i, time_str in enumerate(times_list):

        date_time = datetime.fromisoformat(time_str)

        aqi = hourly["us_aqi"][i] or 0
        aqi_change_rate = int(aqi - prev_aqi) if prev_aqi is not None else 0
        prev_aqi = aqi


        rows.append({
            "timestamp": date_time.strftime("%Y-%m-%d %H:%M:%S"),
            "city": CITY_NAME,
            "hour": date_time.hour,
            "day": date_time.day,
            "month": date_time.month,
            "day_of_week": date_time.weekday(),
            "pm2_5": hourly["pm2_5"][i] or 0,
            "pm10": hourly["pm10"][i] or 0,
            "carbon_monoxide": hourly["carbon_monoxide"][i] or 0,
            "nitrogen_dioxide": hourly["nitrogen_dioxide"][i] or 0,
            "ozone": hourly["ozone"][i] or 0,
            "sulphur_dioxide": hourly["sulphur_dioxide"][i] or 0,
            "us_aqi": aqi,
            "aqi_change_rate": aqi_change_rate,
        })

    df = pd.DataFrame(rows)

    return df

def store_data_in_hopsworks(df):

    project = hopsworks.login(
        project=HOPSWORKS_PROJECT_NAME,
        host="eu-west.cloud.hopsworks.ai",
        api_key_value=HOPSWORKS_API_KEY
    )

    # Access the feature store
    fs = project.get_feature_store()

    # Read an existing feature group and create if not already exist.
    fg = fs.get_or_create_feature_group(
        name = FEATURE_GROUP_NAME, #The name of the group to create ot retrieve
        version = FEATURE_GROUP_VERSION,
        description = "AQI Features for Pattoki fetched from Open-Meteo",
        primary_key = ["timestamp", "city"],
        #time_travel_format = "HUDI"
    )

    fg.insert(df, write_options={"kafka_timeout" : 60})

    print(f"Stored {len(df)} records successfully.") 

if __name__=="__main__":

    end_date = datetime.now().strftime("%Y-%m-%d")

    start_date = (datetime.now()-timedelta(days=365)).strftime("%Y-%m-%d")

    print(f"Fetching historical data from {start_date} to {end_date}")
    raw_data = fetch_historical_aqi(start_date , end_date)

    print("Processing raw data...")
    df = process_historical_date(raw_data)
    print(f"Shape of the dataframe : {df.shape}")
    print(df.head())

    print("Storing data in Hopsworks...")
    store_data_in_hopsworks(df)
    print("Backfill complete!")

    

