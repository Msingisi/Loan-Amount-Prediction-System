from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.model_train import train_model
from steps.config import ModelNameConfig

@pipeline(enable_cache=False)
def train_pipeline(data_path: str, model_name: str = "RandomForest"):
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model_config = ModelNameConfig(model_name="RandomForest")
    model = train_model(X_train, X_test, y_train, y_test, config=model_config)
    r2, rmse = evaluate_model(model, X_test, y_test)