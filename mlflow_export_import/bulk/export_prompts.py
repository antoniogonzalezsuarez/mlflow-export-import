"""
Exports multiple MLflow prompts to a directory.

Note: This implementation uses standard MLflow APIs that are available across different
MLflow deployments. Version discovery is done by iteratively checking version numbers 
1-10 to ensure compatibility with various MLflow configurations.
"""

import os
import sys
import click
import mlflow
from concurrent.futures import ThreadPoolExecutor

from mlflow_export_import.common import utils, io_utils
from mlflow_export_import.common.click_options import opt_output_dir
from mlflow_export_import.common.version_utils import has_prompt_support, log_version_info
from mlflow_export_import.client.client_utils import create_mlflow_client
from mlflow_export_import.client.client_registry import sync_pool_with_threads
from mlflow_export_import.prompt.export_prompt import export_prompt, _get_prompt_safe

_logger = utils.getLogger(__name__)


def export_prompts(
        output_dir,
        prompt_names=None,
        use_threads=False,
        mlflow_client=None
    ):
    """
    Export multiple prompts to a directory.
    
    :param output_dir: Output directory.
    :param prompt_names: List of prompt names to export. If None, exports all prompts.
    :param use_threads: Use multithreading for export.
    :param mlflow_client: MLflow client.
    :return: Summary of export results.
    """
    
    if not has_prompt_support():
        _logger.warning(f"Prompt registry not supported in MLflow {mlflow.__version__} (requires 2.21.0+)")
        return {"unsupported": True, "mlflow_version": mlflow.__version__}
    
    mlflow_client = mlflow_client or create_mlflow_client()
    log_version_info()
    
    try:
        if prompt_names:
            prompts_to_export = _get_specified_prompts(prompt_names, mlflow_client)
        else:
            prompts_to_export = _get_all_prompt_versions(mlflow_client)
        
        _logger.info(f"Found {len(prompts_to_export)} prompts to export")
        
        os.makedirs(output_dir, exist_ok=True)
        
        if use_threads:
            max_workers = utils.get_threads(use_threads=True)
            sync_pool_with_threads(max_workers)
            results = _export_prompts_threaded(prompts_to_export, output_dir, mlflow_client)
        else:
            results = _export_prompts_sequential(prompts_to_export, output_dir, mlflow_client)
        
        successful = [r for r in results if r is not None]
        failed = len(results) - len(successful)
        
        summary = {
            "total_prompts": len(prompts_to_export),
            "successful_exports": len(successful),
            "failed_exports": failed
        }
        
        io_utils.write_export_file(output_dir, "prompts_summary.json", __file__, summary)
        
        _logger.info(f"Prompt export completed: {summary}")
        return summary
        
    except Exception as e:
        _logger.error(f"Bulk prompt export failed: {str(e)}")
        return {"error": str(e)}


def _search_prompts_with_pagination(search_func):
    """Helper to handle pagination for any search_prompts function."""
    all_prompts = []
    page_token = None
    
    while True:
        response = search_func(max_results=1000, page_token=page_token)
        prompts = list(response)
        all_prompts.extend(prompts)
        
        page_token = response.token if hasattr(response, 'token') else None
        if not page_token:
            break
    
    return all_prompts


def _get_all_prompts(mlflow_client):
    """
    Get all available prompts from the registry with pagination support.
    Tries multiple APIs for compatibility across MLflow versions (2.21+ and 3.0+).

    :param mlflow_client: Shared MLflow client.
    :type mlflow_client: mlflow.tracking.MlflowClient
    :return: List of prompts.
    :rtype: list
    """
    try:
        import mlflow.genai
        if hasattr(mlflow.genai, 'search_prompts'):
            return _search_prompts_with_pagination(mlflow.genai.search_prompts)
    except (ImportError, AttributeError, Exception):
        pass
    
    try:
        if hasattr(mlflow_client, 'search_prompts'):
            return _search_prompts_with_pagination(mlflow_client.search_prompts)
    except (ImportError, AttributeError, Exception):
        pass
    
    try:
        if hasattr(mlflow, 'search_prompts'):
            return _search_prompts_with_pagination(mlflow.search_prompts)
    except (ImportError, AttributeError, Exception):
        pass
    
    raise Exception(f"No compatible prompt search API found in MLflow {mlflow.__version__}. Ensure prompt registry is supported.")


def _get_all_prompt_versions(mlflow_client):
    """Get all prompt versions from all prompts."""
    all_prompts = _get_all_prompts(mlflow_client)
    prompt_versions = []
    
    for prompt in all_prompts:
        versions = _get_prompt_versions(prompt.name, mlflow_client)
        prompt_versions.extend(versions)
    
    return prompt_versions


def _get_prompt_versions(prompt_name, mlflow_client):
    """Get all versions of a specific prompt using best available API."""
    try:
        if hasattr(mlflow_client, 'search_prompt_versions'):
            _logger.debug(f"Using search_prompt_versions API for '{prompt_name}'")
            
            all_versions = []
            page_token = None
            
            while True:
                response = mlflow_client.search_prompt_versions(
                    prompt_name, 
                    max_results=1000,
                    page_token=page_token
                )
                
                versions = list(response.prompt_versions) if hasattr(response, 'prompt_versions') else list(response)
                all_versions.extend(versions)
                
                page_token = response.token if hasattr(response, 'token') else None
                if not page_token:
                    break
            
            if all_versions:
                all_versions = sorted(all_versions, key=lambda v: int(v.version))
                _logger.info(f"Found {len(all_versions)} version(s) for prompt '{prompt_name}' via search API")
                return all_versions
    except Exception as e:
        _logger.debug(f"search_prompt_versions not available or failed: {e}")
    
    _logger.debug(f"Using iterative version discovery for '{prompt_name}' (fallback method)")
    versions = []
    
    version_num = 1
    consecutive_missing = 0
    max_consecutive_missing = 3
    
    while True:
        try:
            prompt_version = _get_prompt_safe(prompt_name, str(version_num), mlflow_client)
            if prompt_version:
                versions.append(prompt_version)
                consecutive_missing = 0
            else:
                consecutive_missing += 1
        except Exception:
            consecutive_missing += 1
        
        if consecutive_missing >= max_consecutive_missing:
            break
        
        version_num += 1
        
        if version_num > 1000 and version_num % 100 == 0:
            _logger.warning(f"Still searching for versions of '{prompt_name}' at version {version_num}...")
    
    if not versions:
        _logger.warning(f"No versions found for prompt '{prompt_name}'")
    else:
        _logger.info(f"Found {len(versions)} version(s) for prompt '{prompt_name}' via iteration")
    
    return versions


def _get_specified_prompts(prompt_names, mlflow_client):
    """Get specified prompts with their latest versions."""
    prompts = []
    for prompt_name in prompt_names:
        try:
            prompt_versions = _get_prompt_versions(prompt_name, mlflow_client)
            if prompt_versions:
                latest = max(prompt_versions, key=lambda x: int(x.version))
                prompts.append(latest)
            else:
                _logger.warning(f"Prompt '{prompt_name}' not found")
        except Exception as e:
            _logger.error(f"Error getting prompt '{prompt_name}': {e}")
    
    return prompts


def _export_prompts_sequential(prompts, output_dir, mlflow_client):
    """Export prompts sequentially."""
    results = []
    for prompt in prompts:
        prompt_dir = os.path.join(output_dir, f"{prompt.name}_v{prompt.version}")
        result = export_prompt(prompt.name, prompt.version, prompt_dir, mlflow_client=mlflow_client)
        results.append(result)
    return results


def _export_prompts_threaded(prompts, output_dir, mlflow_client):
    """Export prompts using multithreading."""
    def export_single(prompt):
        prompt_dir = os.path.join(output_dir, f"{prompt.name}_v{prompt.version}")
        return export_prompt(prompt.name, prompt.version, prompt_dir, mlflow_client=mlflow_client)
    
    max_workers = utils.get_threads(use_threads=True)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(export_single, prompts))
    
    return results


@click.command()
@opt_output_dir
@click.option("--prompts",
    help="Prompt names: 'all' for all prompts, comma-delimited list (e.g., 'prompt1,prompt2'), \
or file path ending with '.txt' containing prompt names (one per line).",
    type=str,
    required=True
)
@click.option("--use-threads",
    help="Use multithreading for export.",
    is_flag=True,
    default=False
)
def main(output_dir, prompts, use_threads):
    _logger.info("Options:")
    for k, v in locals().items():
        _logger.info(f"  {k}: {v}")
    
    if prompts.endswith(".txt"):
        with open(prompts, "r", encoding="utf-8") as f:
            prompt_names_list = f.read().splitlines()
    elif prompts.lower() == "all":
        prompt_names_list = None
    else:
        prompt_names_list = [name.strip() for name in prompts.split(",")]
    
    result = export_prompts(
        output_dir=output_dir,
        prompt_names=prompt_names_list,
        use_threads=use_threads
    )
    
    if result is None:
        _logger.error("Prompt export failed with unknown error")
        sys.exit(1)
    elif "unsupported" in result:
        _logger.error(f"Prompt registry not supported in MLflow {result.get('mlflow_version')} (requires 2.21.0+)")
        sys.exit(1)
    elif "error" in result:
        _logger.error(f"Prompt export failed: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
