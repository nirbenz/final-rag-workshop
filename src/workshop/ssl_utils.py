# SSL utilities for corporate proxy environments
# Handles SSL verification bypass for LiteLLM proxies with self-signed certificates

import os
import ssl
from typing import TYPE_CHECKING
import warnings

from loguru import logger

if TYPE_CHECKING:
    pass

_ssl_bypass_applied = False


def should_disable_ssl_verification() -> bool:
    """
    Check if SSL verification should be disabled.

    Returns:
        True if SSL_VERIFY environment variable is set to "false"
    """
    ssl_verify_status = os.getenv("SSL_VERIFY", "true").lower()
    return ssl_verify_status == "false" or ssl_verify_status == "0"


def apply_ssl_bypass() -> None:
    """
    Apply SSL verification bypass for environments with self-signed certificates.

    This patches:
    1. Global SSL context to disable hostname and certificate verification
    2. OpenAI client to use unverified httpx clients (sync and async)
    3. Environment variables for requests/curl

    Should be called once at application startup before any network requests.
    Only applies patches if SSL_VERIFY=false or SSL_VERIFY=0 environment variable is set.
    """
    global _ssl_bypass_applied

    if _ssl_bypass_applied:
        return

    if not should_disable_ssl_verification():
        return

    logger.warning(
        f"SSL verification is DISABLED (SSL_VERIFY={os.getenv('SSL_VERIFY')}). Use only in trusted environments."
    )

    # Suppress SSL warnings
    warnings.filterwarnings("ignore", message="Unverified HTTPS request")

    # Set environment variables for various HTTP libraries
    os.environ["PYTHONHTTPSVERIFY"] = "0"
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""

    # Patch SSL context creation to disable verification
    _original_create_default_context = ssl.create_default_context

    def _create_unverified_context(purpose=ssl.Purpose.SERVER_AUTH, *args, **kwargs):
        context = _original_create_default_context(purpose, *args, **kwargs)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context

    ssl.create_default_context = _create_unverified_context
    ssl._create_default_https_context = _create_unverified_context  # type: ignore[attr-defined]

    # Patch OpenAI clients to use unverified httpx clients
    _patch_openai_clients()

    _ssl_bypass_applied = True
    logger.info("SSL bypass patches applied successfully")


def _patch_openai_clients() -> None:
    """
    Patch OpenAI sync and async clients to disable SSL verification.

    This is needed because pydantic-ai uses OpenAI clients internally,
    and they need to be configured to skip SSL verification for proxies
    with self-signed certificates.
    """
    import httpx
    import openai

    # Patch sync client
    _original_init = openai.OpenAI.__init__

    def _patched_init(self, *args, **kwargs):
        if "http_client" not in kwargs:
            kwargs["http_client"] = httpx.Client(verify=False)
        return _original_init(self, *args, **kwargs)

    openai.OpenAI.__init__ = _patched_init

    # Patch async client (used by Pydantic-AI)
    _original_async_init = openai.AsyncOpenAI.__init__

    def _patched_async_init(self, *args, **kwargs):
        if "http_client" not in kwargs:
            kwargs["http_client"] = httpx.AsyncClient(verify=False)
        # Ensure api_key is set from environment if not provided
        if "api_key" not in kwargs or kwargs.get("api_key") == "litellm-placeholder":
            import os
            env_key = os.getenv("OPENAI_API_KEY")
            if env_key:
                kwargs["api_key"] = env_key
                logger.debug(f"Patched AsyncOpenAI with API key from environment")
        return _original_async_init(self, *args, **kwargs)

    openai.AsyncOpenAI.__init__ = _patched_async_init

    # Patch the AsyncAPIClient._build_request to intercept body before serialization
    from openai._base_client import AsyncAPIClient
    _original_build_request = AsyncAPIClient._build_request

    def _patched_build_request(self, *args, **kwargs):
        # Get the options object which contains json_data
        options = kwargs.get("options") if "options" in kwargs else (args[0] if args else None)

        if options and hasattr(options, "json_data") and isinstance(options.json_data, dict):
            model = options.json_data.get("model", "")
            if ("cohere" in model.lower() or model.startswith("litellm:")) and "encoding_format" in options.json_data:
                original_format = options.json_data["encoding_format"]
                # Remove encoding_format entirely - Cohere/Bedrock doesn't support it at all
                del options.json_data["encoding_format"]
                logger.info(f"[BUILD_REQUEST PATCH] Removed encoding_format='{original_format}' for model: {model}")

        return _original_build_request(self, *args, **kwargs)

    AsyncAPIClient._build_request = _patched_build_request
    logger.info("Patched AsyncAPIClient._build_request to remove encoding_format for Cohere")
