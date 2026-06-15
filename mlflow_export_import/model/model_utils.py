
import mlflow

from mlflow_export_import.client.client_utils import create_mlflow_client

def _extract_model_id(source):

    idx = source.find("models")
    if idx == 0:
        return source.split('models:/')[1]
    else:
        return source.split("models/")[1].split("/")[0]

def _get_logged_model_artifact_path(model_id, mlflow_client=None):
    mlflow_client = mlflow_client or create_mlflow_client()
    return mlflow_client.get_logged_model(model_id).artifact_location
