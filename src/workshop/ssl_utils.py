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
        return _original_async_init(self, *args, **kwargs)

    openai.AsyncOpenAI.__init__ = _patched_async_init

    # Patch embeddings.create to set encoding_format=None for Cohere/LiteLLM models
    from openai.resources import AsyncEmbeddings, Embeddings

    _original_async_embeddings_create = AsyncEmbeddings.create
    _original_sync_embeddings_create = Embeddings.create

    async def _patched_async_embeddings_create(self, *args, **kwargs):
        model = kwargs.get("model", "")
        # For Cohere/LiteLLM models, explicitly set encoding_format=None to prevent SDK default
        if ("cohere" in model.lower() or model.startswith("litellm:")) and "encoding_format" not in kwargs:
            kwargs["encoding_format"] = None
            logger.info(f"Set encoding_format=None for async embeddings with model: {model}")

        return await _original_async_embeddings_create(self, *args, **kwargs)

    def _patched_sync_embeddings_create(self, *args, **kwargs):
        model = kwargs.get("model", "")
        # For Cohere/LiteLLM models, explicitly set encoding_format=None to prevent SDK default
        if ("cohere" in model.lower() or model.startswith("litellm:")) and "encoding_format" not in kwargs:
            kwargs["encoding_format"] = None
            logger.info(f"Set encoding_format=None for sync embeddings with model: {model}")

        return _original_sync_embeddings_create(self, *args, **kwargs)

    AsyncEmbeddings.create = _patched_async_embeddings_create
    Embeddings.create = _patched_sync_embeddings_create
    logger.info("Patched embeddings.create (sync and async) to set encoding_format=None for Cohere/LiteLLM models")
