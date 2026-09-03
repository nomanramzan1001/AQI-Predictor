# AQI-Predictor
Live Link : https://noman-aqi-predictor.streamlit.app/
Air quality monitoring and 7-day AQI forecasting for Lahore, Pakistan.

**Documentation:** open [`report.html`](report.html) for project flow, 7-model training, Overview UI, GitHub Actions CI/CD, artifacts, libraries, setup, and troubleshooting.

## Quick start (local)

```powershell
venv/Scripts/Activate.ps1
pip install -r requirements.txt
pip install -r requirements-explain.txt
streamlit run app/streamlit_app.py
```

If `pip install -r requirements.txt` fails on Windows with a **twofish** / C++ build error, Hopsworks may already be installed — run `pip install -r requirements-explain.txt` only, then continue.

## GitHub Actions (CI/CD)

Workflows in [`.github/workflows/`](.github/workflows/):

| Workflow | Schedule (UTC) | Script |
|----------|------------------|--------|
| [`feature_pipeline.yml`](.github/workflows/feature_pipeline.yml) | Every hour (`0 * * * *`) | `feature_pipeline.py` |
| [`train_pipeline.yml`](.github/workflows/train_pipeline.yml) | Daily at 06:00 (`0 6 * * *`) | `training_pipeline.py` |

Both support manual **Run workflow** from the Actions tab.

### Required repository secrets

**Settings → Secrets and variables → Actions:**

| Secret | Required |
|--------|----------|
| `HOPSWORKS_API_KEY` | Yes |
| `HOPSWORKS_PROJECT_NAME` | Yes |
| `FEATURE_GROUP_NAME` | Yes |
| `FEATURE_GROUP_VERSION` | Yes |

### First-time setup

1. Push this repo to GitHub and add the secrets above.
2. Run `python backfill.py` locally once (or manually) to seed historical data.
3. Let Actions run hourly (features) and daily (training), or trigger workflows manually.
4. Run the Streamlit app locally to view the dashboard.



## Deploy dashboard (Streamlit Community Cloud)

**Full guide:** [DEPLOY_STREAMLIT.md](DEPLOY_STREAMLIT.md)

**Important:** Use **Python 3.12** in the deploy **Advanced settings** dropdown. Python **3.14** breaks this app (`protobuf` + Hopsworks). Rebooting is not enough — if logs show `Python 3.14.5`, delete the app and redeploy with **3.12** selected.


1. Connect the GitHub repo at [share.streamlit.io](https://share.streamlit.io).
2. **Main file:** `app/streamlit_app.py`
3. **Advanced settings → Python version:** **3.12** (required)

## Screenshots
<img width="1816" height="901" alt="Image" src="https://github.com/user-attachments/assets/e956a29e-a86e-4e33-b167-0421c6e6d5e3" />

<img width="1792" height="913" alt="Image" src="https://github.com/user-attachments/assets/2d26544b-566a-4aae-af17-b7f375954a4a" />

<img width="1778" height="904" alt="Image" src="https://github.com/user-attachments/assets/8fbb5f8b-49d7-4c04-af49-b945fe45cbaa" />

<img width="1799" height="915" alt="Image" src="https://github.com/user-attachments/assets/b56549dd-98e8-42f4-961e-37af37d23f05" />

<img width="1783" height="898" alt="Image" src="https://github.com/user-attachments/assets/98cd9a65-f0d6-4dc3-8fea-d8af82010157" />

<img width="1791" height="913" alt="Image" src="https://github.com/user-attachments/assets/1d07310e-f676-49a4-aba9-ce24822c998a" />

<img width="1777" height="905" alt="Image" src="https://github.com/user-attachments/assets/d390e59f-7d8c-442e-ba95-e6cdf90aa097" />

<img width="1812" height="882" alt="Image" src="https://github.com/user-attachments/assets/a5dab28a-5d4c-4719-8109-ba6f45f30a01" />

<img width="1822" height="906" alt="Image" src="https://github.com/user-attachments/assets/37692e35-defc-4ba5-862e-31f735d175ba" />










4. **Secrets:** `HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT_NAME`, `FEATURE_GROUP_NAME`, `FEATURE_GROUP_VERSION`
5. Train pipeline must have run at least once so Hopsworks has `aqi_predictor`