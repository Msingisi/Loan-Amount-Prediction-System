from pipelines.training_pipeline import train_pipeline
# from pipelines.deployment_pipeline import continous_deployment_pipeline
from zenml.client import Client

if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path="data/train/train.csv")