# Employee Attrition Early-Warning System

A full-stack ML app: a Logistic Regression model (trained in the included
notebook) predicting employee attrition risk, served through a FastAPI
backend and displayed in a black-themed Streamlit frontend — built to
showcase the project to recruiters (dataset, EDA, model, and the business
cost reasoning behind the decision threshold).

## Project Structure

```
employee-attrition-predictor/
├── backend/                FastAPI app — serves live predictions
│   ├── main.py
│   ├── schemas.py
│   ├── model_artifacts/    trained pipeline.pkl + supporting JSON
│   └── requirements.txt
├── frontend/                Streamlit app — 4 pages
│   ├── app.py               Overview (landing page)
│   ├── pages/
│   │   ├── 1_Predict.py
│   │   ├── 2_Dataset_and_EDA.py
│   │   └── 3_Model_and_Business_Metrics.py
│   ├── utils/                api_client.py, styling.py
│   ├── assets/                precomputed chart JSON (no live recompute)
│   ├── .streamlit/config.toml  black theme
│   └── requirements.txt
├── scripts/
│   └── export_artifacts.py   re-run this any time the notebook logic changes —
│                               it regenerates everything in backend/model_artifacts
│                               and frontend/assets
├── notebooks/                 your original notebook, untouched
└── data/                      the CSV dataset
```

## Prerequisites

- Python 3.10+ (developed and tested on 3.12)
- Two terminal windows/tabs (backend and frontend run as separate processes)

## 1. Set up the backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Check it's working: open http://127.0.0.1:8000/docs — you should see the
interactive Swagger UI with `/health` and `/predict`.

## 2. Set up the frontend (in a second terminal)

```bash
cd frontend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

This opens the app at http://localhost:8501. The Overview page will show a
green "Prediction API is live" banner if the backend (step 1) is running.

## 3. Try it

- **Predict** page — fill the form, submit, see a live probability + risk
  tier + which factors pushed it up/down.
- **Dataset & EDA** and **Model & Business Metrics** pages work even
  without the backend running (they read precomputed data).

## Regenerating artifacts

If you change anything in the notebook (features, model, cost assumptions),
re-run this from the project root to refresh everything both apps use:

```bash
pip install joblib scikit-learn pandas numpy plotly --break-system-packages
python scripts/export_artifacts.py
```

This overwrites `backend/model_artifacts/*` and `frontend/assets/*`. Restart
both servers afterward.

## Deployment (when you're ready)

- **Backend** → Render / Railway (free tier). Start command:
  `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Frontend** → Streamlit Community Cloud. Set `BACKEND_URL` in
  `frontend/.streamlit/secrets.toml` (or as a Streamlit Cloud secret) to
  your deployed backend's URL.
