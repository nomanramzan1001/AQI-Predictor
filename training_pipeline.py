import json
import os
import time
from datetime import datetime, timezone
import shutil
import hopsworks
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")
FEATURE_GROUP_NAME = os.getenv("FEATURE_GROUP_NAME")
FEATURE_GROUP_VERSION = os.getenv("FEATURE_GROUP_VERSION")


MODELS = {
    "random_forest": ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42)),
    "ridge": ("Ridge Regression", Ridge()),
    "gradient_boosting": ("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, random_state=42)),
    "extra_trees": ("Extra Trees", ExtraTreesRegressor(n_estimators=100, random_state=42)),
    "knn": ("K-Nearest Neighbors", KNeighborsRegressor(n_neighbors=7)),
    "xgboost" : ("XGBRegressor" , XGBRegressor(n_estimators=500,learning_rate=0.05,max_depth=6,random_state=42)),
    "catboost" : ("CatBoost" , CatBoostRegressor(iterations=500,learning_rate=0.05,depth=6,loss_function="RMSE",random_seed=42))
}

def get_features():
    project = hopsworks.login(
        host="eu-west.cloud.hopsworks.ai",
        project=HOPSWORKS_PROJECT_NAME,
        api_key_value=HOPSWORKS_API_KEY,
    )

    fs = project.get_feature_store()

    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)

    print("Reading data...")
    df = fg.read()

    return df,project

def evaluate_model(estimator , X_train, X_test, y_train , y_test):

    estimator.fit(X_train, y_train)

    y_pred = estimator.predict(X_test)

    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    return {
        "mse" : round(mse , 2), 
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "r2": round(r2, 4),
        "r2_pct": round(r2 * 100, 1),
        "estimator": estimator,
        "predictions": y_pred,
    }

FEATURES = [
    "hour",
    "day",
    "month",
    "day_of_week",
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "ozone",
    "sulphur_dioxide",
    "aqi_change_rate",
]

TARGET = "us_aqi"

def train_ml_models(df):

    X = df[FEATURES]
    y = df[TARGET]

    # train_test_split
    X_train, X_test, y_train, y_test = train_test_split( X , y , test_size = 0.20, random_state = 42)

    results = {}

    for key, (display_name, estimator) in MODELS.items():

        print(f"Training {display_name}...")

        output = evaluate_model(estimator, X_train, X_test, y_train, y_test)
        results[key] = {
            "display_name": display_name,
            "rmse": output["rmse"],
            "mae": output["mae"],
            "r2": output["r2"],
            "r2_pct": output["r2_pct"],
            "estimator": output["estimator"],
            "predictions": output["predictions"],
        }
        print(f"  → RMSE: {output['rmse']}, MAE: {output['mae']}, R²: {output['r2']} ({output['r2_pct']}%)")

        best_key = max(results, key=lambda k: results[k]["r2"])
        best_model = results[best_key]

        print(f"Best model: {best_model['display_name']} with (R² = {best_model['r2']})")

        metrics_report = {
        "best_model": best_key,
        "best_model_display_name": best_model["display_name"],
        "models": {
            key: {
                "display_name": v["display_name"],
                "rmse": v["rmse"],
                "mae": v["mae"],
                "r2": v["r2"],
                "r2_pct": v["r2_pct"],
            }
            for key, v in results.items()
        }
    }

    best_estimator = results[best_key]["estimator"]

    return best_estimator, best_key, metrics_report, X_train, X_test

MODEL_REGISTRY_NAME = "aqi_predictor"

def save_model(model, model_name, metrics_report, project, max_retries=3):
    """
    Stage the model artifacts locally and upload them to the Hopsworks Model Registry.
    """
    # Clean and prepare a single staging directory
    registry_dir = "model/registry"
    if os.path.isdir(registry_dir):
        shutil.rmtree(registry_dir)
    os.makedirs(registry_dir, exist_ok=True)
 
    staged_model_path = os.path.join(registry_dir, "best_model.pkl")
    joblib.dump(model, staged_model_path)
    
    staged_metrics_path = os.path.join(registry_dir, "metrics.json")
    with open(staged_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=4)
        
    print(f"Artifacts staged locally in '{registry_dir}'")

    best_metrics = metrics_report["models"][metrics_report["best_model"]]
    mr = project.get_model_registry()
     
    hw_model = mr.sklearn.create_model(
        name=MODEL_REGISTRY_NAME, 
        metrics={
            "rmse": best_metrics["rmse"],
            "r2": best_metrics["r2"],
            "mae": best_metrics["mae"],
        },
        description=(
            f"AQI prediction for Lahore — best model: "
            f"{metrics_report['best_model_display_name']} ({metrics_report['best_model']})"
        ),
    )

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Uploading '{model_name}' to Hopsworks Model Registry (attempt {attempt}/{max_retries})...")
            hw_model.save(
                registry_dir,
                upload_configuration={"max_chunk_retries": 3},
            )
            print("Model successfully saved to Hopsworks!")
            return
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"  Registry upload failed ({exc}); retrying in {wait}s...")
                time.sleep(wait)
                
    raise last_exc


if __name__ == "__main__":
    df, project = get_features()
    print(f"Loaded {len(df)} records")
    print(df.head())

    print("\nTraining models...")
    model, model_name, metrics_report, X_train, X_test = train_ml_models(df)

    # print("\nComputing SHAP and LIME explanations...")
    # try:
    #     metrics_report["explainability"] = build_explainability_report(
    #         model, X_train, X_test, x_instance=df.dropna().iloc[-1]
    #     )
    #     print("  → SHAP and LIME summaries saved in metrics.json")
    # except Exception as exc:
    #     print(f"  → Explainability skipped: {exc}")

    clean = df.dropna()
    metrics_report["trained_at"] = datetime.now(timezone.utc).isoformat()
    metrics_report["data_through"] = pd.Timestamp(clean["timestamp"].max()).isoformat()
    metrics_report["total_samples"] = len(clean)

    print("\nSaving best model and metrics...")
    try:
        save_model(model, model_name, metrics_report, project)
    except Exception as exc:
        # Hopsworks Model Registry upload failed.
        # Model and metrics are already saved locally — pipeline still succeeds.
        print(f"WARNING: Hopsworks Model Registry upload failed: {exc}")
        print("Model saved locally at model/best_model.pkl — registry upload skipped.")
    print("Training pipeline complete!")

