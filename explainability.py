"""SHAP and LIME explanations for AQI regression models."""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import shap

FEATURES = ["hour" , "day" , "month" , "day_of_week" , "pm2_5" , "pm10",
    "carbon_monoxide" , "nitrogen_dioxide" , "ozone" , "sulphur_dioxide" , "aqi_change_rate"]

FEATURE_LABELS = ["Hour" , "Day" , "Month" , "Day of week" , "PM2.5" , "PM10",
    "CO" , "NO₂" , "Ozone" , "SO₂" , "AQI Δ rate"]

TARGET = "us_aqi"


def sample_rows(X: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(X) <= max_rows:
        return X
    return X.sample(max_rows, random_state=42)


def shap_explainer(model, X_background: pd.DataFrame):

    if isinstance(
        model,
        (RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, XGBRegressor, CatBoostRegressor),
    ):
        return shap.TreeExplainer(model), "tree"
    if isinstance(model, Ridge):
        return shap.LinearExplainer(model = model, data = X_background), "linear"
    background = shap.sample(X_background, min(80, len(X_background)), random_state=42)
    return shap.KernelExplainer(model.predict, background), "kernel"


def compute_shap_summary(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    max_background: int = 150,
    max_explain: int = 120,
    ) -> dict:
    """Mean absolute SHAP values on a held-out sample (global importance)."""

    X_bg = sample_rows(X_train, max_background)
    X_ex = sample_rows(X_test, max_explain)
    explainer, explainer_type = shap_explainer(model, X_bg)
    shap_values = explainer.shap_values(X_ex)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim > 2:
        shap_values = shap_values.reshape(len(X_ex), -1)

    mean_abs = np.abs(shap_values).mean(axis=0)
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = float(np.asarray(base).ravel()[0])
    else:
        base = float(base)

    return {
        "method": "shap",
        "explainer": explainer_type,
        "feature_names": FEATURES,
        "feature_labels": FEATURE_LABELS,
        "mean_abs_shap": {
            name: round(float(v), 4) for name, v in zip(FEATURES, mean_abs)
        },
        "base_value": round(base, 2),
        "samples_explained": len(X_ex),
    }


def compute_lime_explanation(
    model,
    X_train: pd.DataFrame,
    x_instance: pd.Series | pd.DataFrame,
    *,
    num_features: int | None = None,
) -> dict:
    """Local LIME explanation for one prediction."""
    from lime.lime_tabular import LimeTabularExplainer

    if isinstance(x_instance, pd.Series):
        row = x_instance
    else:
        row = x_instance.iloc[0]

    n_feat = num_features or len(FEATURES)
    explainer = LimeTabularExplainer(
        X_train.values.astype(float),
        feature_names=FEATURES,
        mode="regression",
        random_state=42,
    )
    exp = explainer.explain_instance(
        row[FEATURES].values.astype(float),
        model.predict,
        num_features=n_feat,
    )
    pred = float(model.predict(pd.DataFrame([row[FEATURES]]))[0])
    def _lime_label(feat: str) -> str:
        for i, name in enumerate(FEATURES):
            if name == feat or name in feat:
                return FEATURE_LABELS[i]
        return feat

    weights = [
        {"feature": feat, "label": _lime_label(feat), "weight": round(w, 4)}
        for feat, w in exp.as_list()
    ]

    return {
        "method": "lime",
        "prediction": round(pred, 1),
        "intercept": round(float(exp.intercept[0]), 2),
        "weights": weights,
        "instance": {k: round(float(row[k]), 4) for k in FEATURES},
    }


def build_explainability_report(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    x_instance: pd.Series
) -> dict:
    """SHAP global summary + LIME local explanation for dashboard and metrics.json."""
    report = {"shap": compute_shap_summary(model, X_train, X_test)}
    instance = x_instance if x_instance is not None else X_test.iloc[len(X_test) // 2]
    report["lime"] = compute_lime_explanation(model, X_train, instance)
    return report