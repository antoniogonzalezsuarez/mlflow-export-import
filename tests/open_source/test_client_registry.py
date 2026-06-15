"""
Tests for shared MLflow and HTTP client registries.
"""

import os
from unittest import mock

import pytest
import requests

from mlflow_export_import.client.client_registry import (
    get_mlflow_client,
    get_http_client,
    get_dbx_client,
    reset_clients,
    sync_pool_with_threads,
    HttpSessionRegistry,
)
from mlflow_export_import.client.http_client import HttpClient, MlflowHttpClient


@pytest.fixture(autouse=True)
def clear_client_registry():
    reset_clients()
    yield
    reset_clients()


def test_get_mlflow_client_returns_same_instance_for_same_uri():
    with mock.patch("mlflow_export_import.client.client_registry.mlflow.MlflowClient") as mock_client_ctor:
        mock_client_ctor.return_value = mock.Mock(name="client_a")
        client_one = get_mlflow_client("http://localhost:5000", None)
        client_two = get_mlflow_client("http://localhost:5000", None)
        assert client_one is client_two
        assert mock_client_ctor.call_count == 1


def test_get_mlflow_client_returns_different_instances_for_different_uris():
    with mock.patch("mlflow_export_import.client.client_registry.mlflow.MlflowClient") as mock_client_ctor:
        mock_client_ctor.side_effect = [mock.Mock(name="client_a"), mock.Mock(name="client_b")]
        client_one = get_mlflow_client("http://localhost:5000", None)
        client_two = get_mlflow_client("http://localhost:5001", None)
        assert client_one is not client_two
        assert mock_client_ctor.call_count == 2


def test_get_mlflow_client_normalizes_uc_tracking_uri():
    with mock.patch("mlflow_export_import.client.client_registry.mlflow.MlflowClient") as mock_client_ctor:
        mock_client_ctor.return_value = mock.Mock(name="client_uc")
        get_mlflow_client("databricks-uc://profile", "databricks-uc://profile")
        mock_client_ctor.assert_called_once_with("databricks://profile", "databricks-uc://profile")


def test_http_session_registry_reuses_session_for_same_host():
    session_one = HttpSessionRegistry.get_session("http://localhost:5000")
    session_two = HttpSessionRegistry.get_session("http://localhost:5000")
    assert session_one is session_two


def test_http_client_uses_shared_session():
    host = "http://localhost:5999"
    session = HttpSessionRegistry.get_session(host)
    client = HttpClient("api/2.0/mlflow", host, token="token", session=session)
    assert client._session is session


def test_mlflow_http_client_uses_shared_session_by_default():
    host = "http://localhost:5998"
    with mock.patch.object(HttpSessionRegistry, "get_session", wraps=HttpSessionRegistry.get_session) as mock_get_session:
        client = MlflowHttpClient(host, token="token")
        mock_get_session.assert_called_once_with(host)
        assert client._session is HttpSessionRegistry.get_session(host)


def test_get_http_client_caches_by_host_and_api():
    mock_mlflow_client = mock.Mock()
    mock_creds = mock.Mock(host="http://localhost:5000", token="token", username=None, password=None)
    mock_mlflow_client._tracking_client.store.get_host_creds.return_value = mock_creds
    with mock.patch("mlflow_export_import.common.model_utils.is_unity_catalog_model", return_value=False):
        client_one = get_http_client(mock_mlflow_client)
        client_two = get_http_client(mock_mlflow_client)
        assert client_one is client_two


def test_get_dbx_client_returns_none_for_non_databricks_backend():
    mock_mlflow_client = mock.Mock()
    mock_mlflow_client._tracking_client.store.get_host_creds.side_effect = AttributeError("no creds")
    assert get_dbx_client(mock_mlflow_client) is None


def test_get_dbx_client_caches_instance():
    mock_mlflow_client = mock.Mock()
    mock_creds = mock.Mock(host="http://localhost:5000", token="token", username=None, password=None)
    mock_mlflow_client._tracking_client.store.get_host_creds.return_value = mock_creds
    client_one = get_dbx_client(mock_mlflow_client)
    client_two = get_dbx_client(mock_mlflow_client)
    assert client_one is client_two


def test_reset_clients_clears_mlflow_client_cache():
    with mock.patch("mlflow_export_import.client.client_registry.mlflow.MlflowClient") as mock_client_ctor:
        mock_client_ctor.return_value = mock.Mock(name="client_a")
        get_mlflow_client("http://localhost:5000", None)
        reset_clients()
        get_mlflow_client("http://localhost:5000", None)
        assert mock_client_ctor.call_count == 2


def test_sync_pool_with_threads_sets_environment_defaults():
    env_keys = [
        "MLFLOW_HTTP_POOL_MAXSIZE",
        "MLFLOW_HTTP_POOL_CONNECTIONS",
        "MLFLOW_EXPORT_IMPORT_HTTP_POOL_MAXSIZE",
    ]
    saved = {key: os.environ.pop(key, None) for key in env_keys}
    try:
        sync_pool_with_threads(24)
        assert os.environ["MLFLOW_HTTP_POOL_MAXSIZE"] == "24"
        assert os.environ["MLFLOW_HTTP_POOL_CONNECTIONS"] == "24"
        assert os.environ["MLFLOW_EXPORT_IMPORT_HTTP_POOL_MAXSIZE"] == "24"
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_http_session_uses_configured_pool_size():
    saved = os.environ.pop("MLFLOW_EXPORT_IMPORT_HTTP_POOL_MAXSIZE", None)
    try:
        os.environ["MLFLOW_EXPORT_IMPORT_HTTP_POOL_MAXSIZE"] = "20"
        reset_clients()
        session = HttpSessionRegistry.get_session("http://localhost:5997")
        https_adapter = session.get_adapter("https://")
        assert https_adapter._pool_maxsize == 20
        assert https_adapter._pool_connections == 20
    finally:
        reset_clients()
        if saved is None:
            os.environ.pop("MLFLOW_EXPORT_IMPORT_HTTP_POOL_MAXSIZE", None)
        else:
            os.environ["MLFLOW_EXPORT_IMPORT_HTTP_POOL_MAXSIZE"] = saved


def test_export_experiments_passes_shared_client_to_workers():
    from mlflow_export_import.bulk import export_experiments

    shared_client = mock.Mock(name="shared_client")
    with mock.patch("mlflow_export_import.bulk.export_experiments.create_mlflow_client", return_value=shared_client) as mock_create:
        with mock.patch("mlflow_export_import.bulk.export_experiments.bulk_utils.get_experiment_ids", return_value=["1"]):
            with mock.patch("mlflow_export_import.bulk.export_experiments._export_experiment") as mock_export_exp:
                mock_export_exp.return_value = export_experiments.Result("exp", 1, 0)
                with mock.patch("mlflow_export_import.bulk.export_experiments.io_utils.write_export_file"):
                    export_experiments.export_experiments(
                        experiments=["1"],
                        output_dir="/tmp/test_export",
                        use_threads=True,
                        mlflow_client=shared_client,
                    )
                    mock_create.assert_not_called()
                    mock_export_exp.assert_called_once()
                    assert mock_export_exp.call_args[0][0] is shared_client
