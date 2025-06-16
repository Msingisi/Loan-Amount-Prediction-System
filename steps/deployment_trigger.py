from zenml.steps import step
from pydantic import BaseModel

class DeploymentTriggerConfig(BaseModel):
    min_accuracy: float = 0.70

@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig
) -> bool:
    """Return True if accuracy exceeds the threshold to deploy the model."""
    return accuracy >= config.min_accuracy