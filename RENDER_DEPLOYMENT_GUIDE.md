# Render Deployment Guide

This guide explains how to deploy the monorepo to Render using the included render.yaml.

## Prerequisites

- A Render account
- This repository pushed to GitHub
- Model artifacts present (see Step 4)

## Step 1: Verify render.yaml

The file render.yaml exists in the repo root and defines two services:
- tark-ml-api (ML inference service)
- tark-core-api (Core API gateway)

## Step 2: Push code to GitHub

Make sure your latest changes are pushed:
- git status
- git add .
- git commit -m "<your message>"
- git push -u origin main

## Step 3: Create a Render Blueprint

1) Log in to Render.
2) Click New + and choose Blueprint.
3) Connect your GitHub repository and select this repo.
4) Render will detect render.yaml and show two services.
5) Click Apply to create the services.

## Step 4: Upload model artifacts (important)

The ML API requires these files inside services/ml_api/models:
- model.pkl
- model_regression.pkl
- label_encoder.pkl
- model_metadata.json

You can create them by running locally:
- pip install -r training/requirements.txt
- python training/train.py

Then commit and push the artifacts if you want Render to load them at build time.

If you prefer not to commit model files, upload them using one of these approaches:
- Add a startup script that pulls the model from cloud storage
- Store the model in a private bucket and download it on startup

## Step 5: Set environment variables in Render

For tark-ml-api:
- MODEL_PATH=models/model.pkl
- REGRESSION_MODEL_PATH=models/model_regression.pkl
- LABEL_ENCODER_PATH=models/label_encoder.pkl
- MODEL_METADATA_PATH=models/model_metadata.json
- RATE_LIMIT=60/minute
- PREDICT_RATE_LIMIT=30/minute

For tark-core-api:
- ML_API_URL=https://<your-ml-api>.onrender.com/predict
- ML_API_BATCH_URL=https://<your-ml-api>.onrender.com/predict_batch
- ML_API_TIMEOUT=10
- RATE_LIMIT=120/minute
- PREDICT_RATE_LIMIT=60/minute
- INTERVENTION_RATE_LIMIT=30/minute
- ALERT_SCORE_THRESHOLD=70

## Step 6: Deploy and verify

After Render finishes deploying:
- ML API health: https://<your-ml-api>.onrender.com/health
- Core API health: https://<your-core-api>.onrender.com/health
- Core API docs: https://<your-core-api>.onrender.com/docs

## Step 7: Test endpoints

Predict:
- curl -X POST https://<your-core-api>.onrender.com/predict -H "Content-Type: application/json" -d "{\"assignment\":65,\"attendance\":72,\"lms\":55,\"marks\":68}"

Intervention:
- curl -X POST https://<your-core-api>.onrender.com/intervention -H "Content-Type: application/json" -d "{\"assignment\":65,\"attendance\":72,\"lms\":55,\"marks\":68}"

Batch predict:
- curl -X POST https://<your-core-api>.onrender.com/predict_batch -H "Content-Type: application/json" -d "{\"items\":[{\"student_id\":\"S-001\",\"class_id\":\"CS-A\",\"subject\":\"Math\",\"assignment\":40,\"attendance\":45,\"lms\":30,\"marks\":42},{\"student_id\":\"S-002\",\"class_id\":\"CS-A\",\"subject\":\"Math\",\"assignment\":82,\"attendance\":88,\"lms\":75,\"marks\":85}]}"

## Troubleshooting

- If you see {"detail":"Not Found"} at /, use /health or /docs.
- If ML API returns 500, confirm the model files exist on the server.
- If core_api cannot reach ML API, confirm ML_API_URL is correct.
- If deployments fail, check the Render logs for build errors and missing packages.
