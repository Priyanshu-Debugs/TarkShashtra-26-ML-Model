# Early Academic Risk Detection & Student Intervention Platform (Monorepo)

This repo separates ML inference from the core API so you can deploy the ML service independently and reduce rate-limit pressure on the main app.

## Structure
- services/ml_api: FastAPI service for model inference
- services/core_api: FastAPI gateway + intervention endpoint (calls ml_api)
- training: training pipeline that produces model artifacts
- datasets: dataset files used for training
- Early_Academic_Risk_Detection_Student_Intervention_Platform.ipynb: exploration notebook

## Folder structure
```
.
├── datasets/
│   └── TS-PS12.csv
├── services/
│   ├── ml_api/
│   │   ├── app/
│   │   │   ├── main.py
│   │   │   └── model.py
│   │   ├── models/
│   │   │   ├── model.pkl
│   │   │   ├── model_regression.pkl
│   │   │   ├── label_encoder.pkl
│   │   │   └── model_metadata.json
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   └── core_api/
│       ├── app/
│       │   └── main.py
│       ├── requirements.txt
│       └── Dockerfile
├── training/
│   ├── train.py
│   └── requirements.txt
├── Early_Academic_Risk_Detection_Student_Intervention_Platform.ipynb
├── FASTAPI_DEPLOYMENT_GUIDE.md
├── RENDER_DEPLOYMENT_GUIDE.md
├── render.yaml
└── README.md
```

Optional notebook exports in repo root: best_model_classification.pkl, best_model_regression.pkl, label_encoder.pkl.

## Project innovation (closing the loop)
This platform goes beyond risk prediction by closing the loop from early detection to measurable impact:
- Explainable risk, not just a label/score: the ML API returns a `reasons` list derived from feature percentiles so faculty can see why a student is at risk. Evidence: [services/ml_api/app/model.py](services/ml_api/app/model.py), [services/ml_api/app/main.py](services/ml_api/app/main.py)
- Actionable interventions: the Core API adds rule-based recommendations via `/intervention` so staff get next steps, not just a risk status. Evidence: [services/core_api/app/main.py](services/core_api/app/main.py)
- Measurable pre/post impact: intervention and performance logging enable before/after comparison with deltas. Evidence: [services/core_api/app/main.py](services/core_api/app/main.py)
- Deployable at scale: ML inference is separated from the core gateway for independent scaling and rate limiting. Evidence: [render.yaml](render.yaml)

## Prerequisites
- Python 3.10+ installed
- pip available
- (Optional) Git and Docker

## Step-by-step setup (Windows cmd)

0) Open a terminal in this folder
- Use Windows cmd and set your working directory to this repo root.
- You should see these folders: datasets, services, training.

1) Create and activate a virtual environment (recommended)
- python -m venv .venv
- .venv\Scripts\activate
- Confirm: where python

2) Install notebook requirements (optional, for the .ipynb)
- pip install -r requirements.txt

3) Train and export the model
- pip install -r training/requirements.txt
- python training/train.py

This writes model artifacts to:
- services/ml_api/models/model.pkl
- services/ml_api/models/model_regression.pkl
- services/ml_api/models/label_encoder.pkl
- services/ml_api/models/model_metadata.json

If these files are missing after training:
- Confirm datasets/TS-PS12.csv exists
- Re-run: python training/train.py

4) Run the ML API (in a new terminal)
- Open a new cmd window in the same repo root
- Activate the same venv: .venv\Scripts\activate
- pip install -r services/ml_api/requirements.txt
- uvicorn app.main:app --reload --port 8001 --app-dir services/ml_api

Verify ML API is running:
- Health: http://localhost:8001/health (should show "model_loaded": true)
- Docs: http://localhost:8001/docs
- Note: http://localhost:8001/ shows {"detail":"Not Found"} because no root route is defined.

5) Run the Core API (in a new terminal)
- Open another cmd window in the repo root
- Activate the same venv: .venv\Scripts\activate
- pip install -r services/core_api/requirements.txt
- set ML_API_URL=http://localhost:8001/predict
- uvicorn app.main:app --reload --port 8000 --app-dir services/core_api

Verify Core API is running:
- Health: http://localhost:8000/health
- Docs: http://localhost:8000/docs

6) Test the Core API (predict)
- Make sure both services are running:
	- ML API: http://localhost:8001/health should return {"status":"ok"}
	- Core API: http://localhost:8000/health should return {"status":"ok"}
- Send a prediction request (Windows cmd):
	- curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"assignment\":65,\"attendance\":72,\"lms\":55,\"marks\":68}"
- Expected response shape:
	- {"risk_label":"Medium","risk_label_id":1,"probabilities":{...},"risk_score_predicted":...,"risk_score_calculated":...,"reasons":[...],"suggestions":[...]}
- If you get 500, confirm the ML API is running and the model files exist in services/ml_api/models

7) Try the intervention endpoint
- This endpoint calls the ML API and adds recommended actions.
- Run in Windows cmd:
	- curl -X POST http://localhost:8000/intervention -H "Content-Type: application/json" -d "{\"assignment\":65,\"attendance\":72,\"lms\":55,\"marks\":68}"
- Expected response shape:
	- {"risk_label":"Medium","recommendations":["..."],"suggestions":["..."],"model":{...}}
- If you see a timeout, ensure ML_API_URL is set correctly and ML API is running

8) Log an intervention record
- Use this when a mentor takes an action (counselling, remedial class, extension).
- curl -X POST http://localhost:8000/interventions -H "Content-Type: application/json" -d "{\"student_id\":\"S-001\",\"action_type\":\"counselling\",\"mentor\":\"Dr. Shah\",\"notes\":\"Attendance plan agreed\",\"class_id\":\"CS-A\",\"subject\":\"Math\"}"

9) Track performance before/after intervention
- curl -X POST http://localhost:8000/performance -H "Content-Type: application/json" -d "{\"student_id\":\"S-001\",\"class_id\":\"CS-A\",\"subject\":\"Math\",\"before\":{\"assignment\":50,\"attendance\":55,\"lms\":40,\"marks\":48,\"risk_score\":45},\"after\":{\"assignment\":68,\"attendance\":72,\"lms\":60,\"marks\":70,\"risk_score\":30}}"

10) Generate high-risk alerts from a batch
- curl -X POST http://localhost:8000/alerts/high-risk -H "Content-Type: application/json" -d "{\"items\":[{\"student_id\":\"S-001\",\"class_id\":\"CS-A\",\"subject\":\"Math\",\"assignment\":40,\"attendance\":45,\"lms\":30,\"marks\":42},{\"student_id\":\"S-002\",\"class_id\":\"CS-A\",\"subject\":\"Math\",\"assignment\":82,\"attendance\":88,\"lms\":75,\"marks\":85}]}"

11) View at-risk dashboard list (after predict_batch)
- Call /predict_batch first to store recent predictions.
- curl "http://localhost:8000/dashboard/at_risk?class_id=CS-A&subject=Math&severity=High"

## Notebook usage
- Open Early_Academic_Risk_Detection_Student_Intervention_Platform.ipynb
- Run cells top-to-bottom to reproduce dataset analysis, model comparison, and export
- The export cell writes both classification and regression models plus label encoder

If you want the notebook to populate the ML API models folder:
- Copy best_model_classification.pkl to services/ml_api/models/model.pkl
- Copy best_model_regression.pkl to services/ml_api/models/model_regression.pkl
- Copy label_encoder.pkl to services/ml_api/models/label_encoder.pkl

## Environment variables

ML API (.env.example in services/ml_api):
- MODEL_PATH=models/model.pkl
- REGRESSION_MODEL_PATH=models/model_regression.pkl
- LABEL_ENCODER_PATH=models/label_encoder.pkl
- MODEL_METADATA_PATH=models/model_metadata.json
- RATE_LIMIT=60/minute
- PREDICT_RATE_LIMIT=30/minute

Core API (.env.example in services/core_api):
- ML_API_URL=http://localhost:8001/predict
- ML_API_BATCH_URL=http://localhost:8001/predict_batch
- ML_API_TIMEOUT=10
- RATE_LIMIT=120/minute
- PREDICT_RATE_LIMIT=60/minute
- INTERVENTION_RATE_LIMIT=30/minute
- ALERT_SCORE_THRESHOLD=70

## Deployment to Render

This repo includes render.yaml with two services (ml_api and core_api). Deploy them as separate services so the ML service can scale or rate-limit independently.

Recommended steps:
1) Push this repo to GitHub
2) In Render, create a new Blueprint from render.yaml
3) After ML API deploys, update ML_API_URL in core_api to the ML API URL
4) Redeploy core_api

## Docker (optional)

ML API:
- cd services\ml_api
- docker build -t tark-ml-api .
- docker run -p 8001:8000 tark-ml-api

Core API:
- cd services\core_api
- docker build -t tark-core-api .
- docker run -p 8000:8000 -e ML_API_URL=http://host.docker.internal:8001/predict tark-core-api

## Rate-limit mitigation
- ML API and Core API both include IP-based rate limiting.
- You can tune limits via env vars in each service's .env.example.
- If you integrate a third-party LLM later, keep that call in core_api and limit it separately.

## Troubleshooting
- If ML API returns 500, verify model files exist in services/ml_api/models
- If core_api cannot reach ML API, confirm ML_API_URL and that ML API is running
- If core_api returns Internal Server Error on /predict, call ML API /predict directly to verify it responds and check the ML API terminal logs
- If you see import errors, re-check the requirements for that service
- If you see {"detail":"Not Found"} at http://localhost:8001/, use /health or /docs
- If prediction fails, confirm all 4 features are sent in the JSON body
