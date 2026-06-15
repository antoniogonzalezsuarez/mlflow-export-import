from mlflow_export_import.client.client_registry import (
    get_mlflow_client,
    get_http_client,
    get_dbx_client,
    reset_clients,
    sync_pool_with_threads,
    HttpSessionRegistry,
)


def create_http_client(mlflow_client, model_name=None):
    """
    Create MLflow HTTP client from MlflowClient.
    If model_name is a Unity Catalog (UC) model, the returned client is UC-enabled.

    :param mlflow_client: Source MlflowClient for host credentials.
    :type mlflow_client: mlflow.tracking.MlflowClient
    :param model_name: Registered model name for UC routing.
    :type model_name: str, optional
    :return: Shared HttpClient or MlflowHttpClient instance.
    :rtype: mlflow_export_import.client.http_client.HttpClient
    """
    return get_http_client(mlflow_client, model_name)


def create_dbx_client(mlflow_client):
    """
    Create Databricks HTTP client from MlflowClient.
    Returns None if not using Databricks backend.

    :param mlflow_client: Source MlflowClient for host credentials.
    :type mlflow_client: mlflow.tracking.MlflowClient
    :return: Shared DatabricksHttpClient or None.
    :rtype: mlflow_export_import.client.http_client.DatabricksHttpClient or None
    """
    return get_dbx_client(mlflow_client)


def create_mlflow_client():
    """
    Create MLflowClient. If MLFLOW_TRACKING_URI is UC, then set MlflowClient.tracking_uri to the non-UC variant.

    :return: Shared MlflowClient instance.
    :rtype: mlflow.tracking.MlflowClient
    """
    return get_mlflow_client()
