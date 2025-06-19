# 🏦 Loan Amount Prediction System

A machine learning system to predict the **Loan Sanction Amount (USD)** for applicants based on personal, financial, and property-related features. This project uses **ZenML** to structure the pipeline, **MLflow** for model tracking and serving, and **Streamlit** to provide a user-friendly web interface for predictions.

---

## Problem Statement

Banks and financial institutions need reliable tools to estimate how much loan can be sanctioned to a customer based on their profile and credit data. This project predicts loan amounts using a trained ML model while demonstrating MLOps best practices for development, deployment, and interaction.

---

## Tech Stack

- [ZenML](https://zenml.io/) – MLOps pipeline orchestration
- [MLflow](https://mlflow.org/) – Model tracking and deployment
- [Scikit-learn](https://scikit-learn.org/) – Model building
- [Streamlit](https://streamlit.io/) – Interactive frontend
- Python 3.10+, Pandas, NumPy, etc.

---

## Project Features

### Training & Deployment Pipeline
Built using ZenML, the pipeline includes:
- `ingest_data`: Load and validate the dataset
- `clean_data`: Clean missing values, handle outliers, transform & scale features
- `train_model`: Train Random Forest Regressor
- `evaluation`: Evaluate R² and RMSE
- `deployment_trigger`: Compare model accuracy with minimum threshold
- `mlflow_model_deployer_step`: Deploy model using MLflow if accuracy ≥ threshold

>  MLflow model serving is started manually due to daemon limitations on Windows.

### Streamlit UI
- Predicts loan amounts based on live user input
- Validates input feature alignment with trained model
- Compatible with manually served MLflow REST API

---

## Project Structure

Loan-Amount-Prediction-System/
├── data/
│   └── train.csv                       # Training dataset
│
├── pipelines/
│   └── deployment_pipeline.py         # Full training and deployment pipeline (ZenML)
│
├── steps/                              # ZenML pipeline steps
│   ├── ingest_data.py
│   ├── clean_data.py
│   ├── model_train.py
│   ├── evaluation.py
│   └── config.py
│
├── streamlit_app.py                   # Streamlit UI for live predictions
├── run_deployment.py                 # CLI entry point to run ZenML pipeline
├── start_app.bat                     # One-click startup for MLflow server and Streamlit (Windows)
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Excluded files for clean version control
└── README.md                         # Project documentation


---

## How to Use

### 1. Clone and Setup

```bash
git clone <this-repo-url>
cd Loan-Amount-Prediction-System
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Run the Pipeline

```bash
python run_deployment.py --config deploy
```

If you're on Windows, MLflow cannot auto-deploy in background. You must serve manually:

```bash
mlflow models serve --model-uri "runs:/<your_run_id>/model" --port 5005 --env-manager=local
```

### Launch the Streamlit App

```bash
streamlit run streamlit_app.py
```
The app connects to http://127.0.0.1:5005/invocations for model predictions.






