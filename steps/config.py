from pydantic import BaseModel

class ModelNameConfig(BaseModel):
    """Model Configurations"""
    model_name: str = "RandomForest" 