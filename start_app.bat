@echo off
set MLFLOW_TRACKING_URI=file:C:/Users/User/AppData/Roaming/zenml/local_stores/<your-store-id>/mlruns
set /p RUN_ID=Enter your MLflow Run ID: 
start cmd /k mlflow models serve --model-uri "runs:/%RUN_ID%/model" --port 5005 --env-manager=local
timeout /t 5
streamlit run streamlit_app.py