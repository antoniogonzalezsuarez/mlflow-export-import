"""
Thread-safe registries for shared MLflow and HTTP clients with connection pooling.
"""

import os
import threading

import mlflow
import requests
from requests.adapters import HTTPAdapter

_DEFAULT_POOL_SIZE = 10
_mlflow_clients = {}
_http_clients = {}
_dbx_clients = {}
_sessions = {}
_registry_lock = threading.RLock()


def _resolve_pool_size():
    """
    Resolve HTTP connection pool size from environment variables.

    :return: Pool size as integer.
    :rtype: int
    """
    export_import_size = os.environ.get("MLFLOW_EXPORT_IMPORT_HTTP_POOL_MAXSIZE")
    if export_import_size:
        return int(export_import_size)
    mlflow_size = os.environ.get("MLFLOW_HTTP_POOL_MAXSIZE")
    if mlflow_size:
        return int(mlflow_size)
    return _DEFAULT_POOL_SIZE


def _normalize_tracking_uri(tracking_uri):
    """
    Normalize tracking URI for UC compatibility.

    :param tracking_uri: MLflow tracking URI.
    :type tracking_uri: str
    :return: Normalized tracking URI.
    :rtype: str
    """
    if tracking_uri:
        return tracking_uri.replace("databricks-uc", "databricks")
    return tracking_uri


def _mlflow_client_key(tracking_uri, registry_uri):
    """
    Build cache key for MlflowClient instances.

    :param tracking_uri: MLflow tracking URI.
    :type tracking_uri: str
    :param registry_uri: MLflow registry URI.
    :type registry_uri: str or None
    :return: Cache key tuple.
    :rtype: tuple
    """
    return (_normalize_tracking_uri(tracking_uri), registry_uri)


def _http_client_key(api_name, host, token, username, password):
    """
    Build cache key for HttpClient instances.

    :param api_name: API path prefix.
    :type api_name: str
    :param host: Host URL.
    :type host: str
    :param token: Auth token.
    :type token: str or None
    :param username: Basic auth username.
    :type username: str or None
    :param password: Basic auth password.
    :type password: str or None
    :return: Cache key tuple.
    :rtype: tuple
    """
    return (api_name, host, token, username, password)


class HttpSessionRegistry:
    """
    Registry of shared requests.Session objects keyed by host.
    """

    @classmethod
    def get_session(cls, host):
        """
        Return a shared Session for the given host with connection pooling.

        :param host: Host URL used as session cache key.
        :type host: str
        :return: Shared requests Session.
        :rtype: requests.Session
        """
        with _registry_lock:
            if host not in _sessions:
                _sessions[host] = cls._create_session()
            return _sessions[host]

    @classmethod
    def _create_session(cls):
        """
        Create a requests Session with a sized HTTPAdapter.

        :return: Configured requests Session.
        :rtype: requests.Session
        """
        pool_size = _resolve_pool_size()
        adapter = HTTPAdapter(
            pool_connections=pool_size,
            pool_maxsize=pool_size,
            pool_block=True,
        )
        session = requests.Session()
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session


def sync_pool_with_threads(max_workers):
    """
    Ensure MLflow SDK and custom HTTP pool sizes are at least max_workers.

    :param max_workers: Number of parallel worker threads.
    :type max_workers: int
    """
    pool_size = max(max_workers, _resolve_pool_size())
    os.environ.setdefault("MLFLOW_HTTP_POOL_MAXSIZE", str(pool_size))
    os.environ.setdefault("MLFLOW_HTTP_POOL_CONNECTIONS", str(pool_size))
    os.environ.setdefault("MLFLOW_EXPORT_IMPORT_HTTP_POOL_MAXSIZE", str(pool_size))


def get_mlflow_client(tracking_uri=None, registry_uri=None):
    """
    Return a cached MlflowClient for the given URIs.

    :param tracking_uri: MLflow tracking URI, defaults to mlflow.get_tracking_uri().
    :type tracking_uri: str, optional
    :param registry_uri: MLflow registry URI, defaults to mlflow.get_registry_uri().
    :type registry_uri: str, optional
    :return: Shared MlflowClient instance.
    :rtype: mlflow.tracking.MlflowClient
    """
    tracking_uri = tracking_uri if tracking_uri is not None else mlflow.get_tracking_uri()
    registry_uri = registry_uri if registry_uri is not None else mlflow.get_registry_uri()
    key = _mlflow_client_key(tracking_uri, registry_uri)
    with _registry_lock:
        if key not in _mlflow_clients:
            normalized_tracking_uri = _normalize_tracking_uri(tracking_uri)
            if registry_uri:
                _mlflow_clients[key] = mlflow.MlflowClient(normalized_tracking_uri, registry_uri)
            else:
                _mlflow_clients[key] = mlflow.MlflowClient()
        return _mlflow_clients[key]


def get_http_client(mlflow_client, model_name=None):
    """
    Return a cached HttpClient derived from an MlflowClient.

    :param mlflow_client: Source MlflowClient for host credentials.
    :type mlflow_client: mlflow.tracking.MlflowClient
    :param model_name: Registered model name for UC routing.
    :type model_name: str, optional
    :return: Shared HttpClient or MlflowHttpClient instance.
    :rtype: mlflow_export_import.client.http_client.HttpClient
    """
    from mlflow_export_import.common import model_utils
    from mlflow_export_import.client.http_client import HttpClient, MlflowHttpClient

    creds = mlflow_client._tracking_client.store.get_host_creds()
    if model_name and model_utils.is_unity_catalog_model(model_name):
        api_name = "api/2.0/mlflow/unity-catalog"
    else:
        api_name = "api/2.0/mlflow"
    key = _http_client_key(api_name, creds.host, creds.token, creds.username, creds.password)
    with _registry_lock:
        if key not in _http_clients:
            session = HttpSessionRegistry.get_session(creds.host)
            if model_name and model_utils.is_unity_catalog_model(model_name):
                _http_clients[key] = HttpClient(
                    api_name,
                    creds.host,
                    creds.token,
                    creds.username,
                    creds.password,
                    session=session,
                )
            else:
                _http_clients[key] = MlflowHttpClient(
                    creds.host,
                    creds.token,
                    creds.username,
                    creds.password,
                    session=session,
                )
        return _http_clients[key]


def get_dbx_client(mlflow_client):
    """
    Return a cached DatabricksHttpClient derived from an MlflowClient.

    :param mlflow_client: Source MlflowClient for host credentials.
    :type mlflow_client: mlflow.tracking.MlflowClient
    :return: Shared DatabricksHttpClient or None for non-Databricks backends.
    :rtype: mlflow_export_import.client.http_client.DatabricksHttpClient or None
    """
    from mlflow_export_import.client.http_client import DatabricksHttpClient

    try:
        creds = mlflow_client._tracking_client.store.get_host_creds()
    except AttributeError:
        return None
    key = _http_client_key("api/2.0", creds.host, creds.token, None, None)
    with _registry_lock:
        if key not in _dbx_clients:
            session = HttpSessionRegistry.get_session(creds.host)
            _dbx_clients[key] = DatabricksHttpClient(creds.host, creds.token, session=session)
        return _dbx_clients[key]


def reset_clients():
    """
    Clear all cached clients and HTTP sessions. Intended for tests.
    """
    with _registry_lock:
        for session in _sessions.values():
            session.close()
        _mlflow_clients.clear()
        _http_clients.clear()
        _dbx_clients.clear()
        _sessions.clear()
