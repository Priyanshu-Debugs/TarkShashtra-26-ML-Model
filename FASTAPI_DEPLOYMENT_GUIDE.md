# TS-12 FastAPI Deployment Guide for Vercel Team Integration

## 1) Final goal

Your frontend or backend team will host the main website on Vercel. The ML API runs as a persistent Python service, and the Core API acts as the gateway that Vercel calls over HTTPS.

Recommended architecture:

- Website and Node layer: Vercel
- Core API (FastAPI gateway): Render (recommended) or Railway
- ML inference API (FastAPI + model files): Render (recommended) or Railway
- Data storage: managed by your full-stack team

Why this split:

- FastAPI with heavy ML libraries is not ideal on Vercel serverless for stable low-latency inference
- Render or Railway provides a long-running Python web process better suited for model serving
- Core API can apply business logic and guardrails without exposing ML service directly

## 2) API endpoints (current)

Core API (recommended for frontend use):

- GET /health
- POST /predict
- POST /predict_batch
- POST /intervention
- POST /interventions
- GET /interventions
- POST /performance
- GET /performance
- POST /alerts/high-risk
- GET /dashboard/at_risk

ML API (internal or direct inference use):

- GET /health
- POST /predict
- POST /predict_batch

Notes:

- Vercel should call the Core API for end-to-end workflows (interventions, alerts, dashboard).
- The Core API calls the ML API internally for inference.

## 3) Files you must have before deploy

Ensure these exist for the ML API:

- services/ml_api/app/main.py
- services/ml_api/app/model.py
- services/ml_api/requirements.txt
- services/ml_api/models/model.pkl
- services/ml_api/models/model_regression.pkl
- services/ml_api/models/label_encoder.pkl
- services/ml_api/models/model_metadata.json

Generate artifacts locally:

```powershell
pip install -r training/requirements.txt
python training/train.py
```

## 4) Local validation before deployment

Start ML API:

```powershell
.venv\Scripts\activate
pip install -r services/ml_api/requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload --app-dir services/ml_api
```

Start Core API:

```powershell
.venv\Scripts\activate
pip install -r services/core_api/requirements.txt
set ML_API_URL=http://localhost:8001/predict
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --app-dir services/core_api
```

Test locally:

- http://127.0.0.1:8001/health
- http://127.0.0.1:8000/health
- http://127.0.0.1:8000/docs

Confirm:

- API starts without model loading errors
- /predict works with a sample student payload

## 4.1) Local frontend integration (Vercel app on localhost)

If your Vercel app is still in local development (for example Next.js on localhost), set the base URL in your frontend:

```bash
# .env.local (frontend)
FASTAPI_BASE_URL=http://127.0.0.1:8000
# or, if you call from the browser
NEXT_PUBLIC_FASTAPI_BASE_URL=http://127.0.0.1:8000
```

If the frontend runs on a different laptop, use a LAN URL:

- http://YOUR_LOCAL_IP:8000

If LAN access fails, allow inbound firewall for 8000:

```powershell
New-NetFirewallRule -DisplayName "FastAPI 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
```

## 5) Deploy FastAPI to Render (recommended)

Use the existing render.yaml in the repo root to deploy two services.

### Free-tier setup (no card required)

Render lets you deploy on the free tier without adding a payment method. To stay on free:

- Choose the **Free** instance type for both services when creating them.
- Do **not** enable paid add-ons (databases, autoscaling, or private networking).
- If prompted for billing, skip and continue with free services only.

Free tier notes:

- Services can spin down when idle, so the first request may be slower (cold start).
- Keep `/health` checks for quick wake-ups before demos.

### Step 1: Push code to GitHub

Push your latest main branch with model artifacts included:

- services/ml_api/models/*.pkl
- services/ml_api/models/model_metadata.json

### Step 2: Create a Render Blueprint

1) Login to Render
2) New + -> Blueprint
3) Connect your GitHub repository
4) Render detects render.yaml and creates two services
5) For each service, pick **Free** as the instance type

### Step 3: Set environment variables

ML API:

- MODEL_PATH=models/model.pkl
- REGRESSION_MODEL_PATH=models/model_regression.pkl
- LABEL_ENCODER_PATH=models/label_encoder.pkl
- MODEL_METADATA_PATH=models/model_metadata.json

Core API:

- ML_API_URL=https://<your-ml-api>.onrender.com/predict
- ML_API_BATCH_URL=https://<your-ml-api>.onrender.com/predict_batch
- ML_API_TIMEOUT=10
- ALERT_SCORE_THRESHOLD=70

### Step 4: Deploy and verify

After deploy, verify:

- https://<your-ml-api>.onrender.com/health
- https://<your-core-api>.onrender.com/health
- https://<your-core-api>.onrender.com/docs

## 6) What to share with your Vercel team

Share only these items:

- Core API base URL, for example: https://<your-core-api>.onrender.com
- Endpoint list and request schema
- Expected response fields for each endpoint

Required frontend environment variable on Vercel:

- FASTAPI_BASE_URL=https://<your-core-api>.onrender.com

## 7) Vercel integration pattern (recommended)

Use server-side calls from Next.js API routes or server actions instead of direct browser calls when possible.

Example server-side fetch pattern:

```ts
const apiBase = process.env.FASTAPI_BASE_URL;

const response = await fetch(`${apiBase}/predict`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    student_id: "STU001",
    attendance: 62,
    marks: 48,
    assignment: 55,
    lms: 44,
  }),
});

const result = await response.json();
```

Benefits:

- Keeps API base URL and operational logic centralized
- Easier auth/rate-limit wrapping later
- Better control over retries and error handling

## 8) CORS rules (only if browser calls the API directly)

If the browser calls the Core API directly, add CORS middleware in services/core_api/app/main.py and allow only your Vercel domains.

Example snippet:

```python
from fastapi.middleware.cors import CORSMiddleware

allowed = ["https://your-vercel-app.vercel.app"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 9) Production checklist

Before handing off to the website team:

- API URL is HTTPS and publicly reachable
- /health returns healthy
- /docs is visible
- /predict returns expected fields
- Model files are present in deployment
- Response time is acceptable for expected traffic

## 10) Updating models later (without breaking the team)

When you retrain:

1. Replace files in services/ml_api/models
2. Commit and push
3. Trigger redeploy on Render
4. Re-test /health and /predict
5. Inform frontend team only if response schema changed

Best practice:

- Keep response schema stable
- Add version info later if needed

## 11) Quick local sharing (same Wi-Fi only)

If a teammate is on the same LAN:

1. Run:
   - uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --app-dir services/core_api
2. Share:
   - http://YOUR_LOCAL_IP:8000/docs
3. Allow firewall inbound on port 8000 if needed

This is only for temporary local testing, not production.

## 12) If you need a public temporary demo link

Use one of these:

- ngrok
- cloudflared tunnel

This is useful for hackathon demos before full deployment.
