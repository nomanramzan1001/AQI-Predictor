# Deploy on Streamlit Community Cloud

## Why the app fails on Python 3.14

Your logs show:

```text
Using Python 3.14.5 environment at /home/adminuser/venv
...
TypeError: Metaclasses with custom tp_new are not supported.
```

This project **cannot run on Python 3.14** because:

1. **Hopsworks** (`hopsworks==4.7.5`) supports Python `>=3.9,<3.14` only.
2. **protobuf 4.x** (required by Hopsworks) breaks on Python 3.14 before Streamlit even starts.

**Streamlit Cloud does not read `.python-version` for an app that is already deployed.**  
You must pick Python **3.12** (or 3.11 / 3.13) in the deploy UI.

---

## Fix (required): redeploy with Python 3.12

You cannot switch Python on a live app. Redeploy:

1. Open [share.streamlit.io](https://share.streamlit.io) → your app **noman-aqi-predictor**.
2. Note your **subdomain**, **repo**, **branch**, **main file** (`appstreamlit_app.py`), and **Secrets**.
3. **Delete** the app (Manage app → Delete).
4. **Create app** again from the same GitHub repo.
5. Set **Main file path:** `app/streamlit_app.py`
6. Click **Advanced settings**.
7. **Python version:** choose **3.12** (not 3.14).
8. Paste **Secrets** (same as local `.env`):

   ```
   HOPSWORKS_API_KEY = "your-key"
   HOPSWORKS_PROJECT_NAME = "your-project"
   FEATURE_GROUP_NAME = "your-group-name"
   FEATURE_GROUP_VERSION = "your-group-version"
   ```


9. **Save** → **Deploy**.

After reboot, logs should show **Python 3.12.x**, not 3.14.5.

---

## Checklist before deploy

| Item | Status |
|------|--------|
| GitHub repo pushed with latest `requirements.txt` | |
| Hopsworks secrets set in Streamlit | |
| At least one successful **Train Pipeline** on GitHub Actions | |
| Model `aqi_predictor` exists in Hopsworks Model Registry | |

---

## Optional: local test on Python 3.12

```powershell
py -3.12 -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```