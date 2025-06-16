import pandas as pd
import numpy as np

from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.steps.mlflow_deployer import mlflow_model_deployer_step
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer

from pydantic import BaseModel

from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.model_train import train_model
from steps.config import ModelNameConfig

# Enable MLflow integration in Dockerized step execution
docker_settings = DockerSettings(required_integrations=[MLFLOW])

# ---------------------
# Deployment Trigger Step
# ---------------------

class DeploymentTriggerConfig(BaseModel):
    min_accuracy: float = 0.60

@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig
) -> bool:
    """Decide whether to deploy model based on accuracy threshold."""
    return accuracy >= config.min_accuracy

# ---------------------
# Deployment Pipeline
# ---------------------

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    data_path: str,
    min_accuracy: float = 0.60,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    """Full training and deployment pipeline."""
    df = ingest_df(data_path=data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model_config = ModelNameConfig(model_name="RandomForest")
    model = train_model(X_train, X_test, y_train, y_test, config=model_config)
    r2, rmse = evaluate_model(model, X_test, y_test)
    should_deploy = deployment_trigger(accuracy=r2, config=DeploymentTriggerConfig(min_accuracy=min_accuracy))
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=should_deploy,
        workers=workers,
        timeout=timeout,
    )