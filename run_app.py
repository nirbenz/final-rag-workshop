#!/usr/bin/env python3
"""
Startup script that disables SSL verification before loading the main app.
This is needed for LiteLLM proxies with self-signed certificates.
"""
import os
import ssl
import sys
import warnings

# Suppress SSL warnings
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# Disable SSL verification globally
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

# Patch SSL context creation to disable verification
_original_create_default_context = ssl.create_default_context

def _create_unverified_context(purpose=ssl.Purpose.SERVER_AUTH, *args, **kwargs):
    context = _original_create_default_context(purpose, *args, **kwargs)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context

ssl.create_default_context = _create_unverified_context
ssl._create_default_https_context = _create_unverified_context

# Now run the main app module
if __name__ == "__main__":
    import runpy
    sys.argv[0] = 'python -m nicegui_app.main'
    runpy.run_module('nicegui_app.main', run_name='__main__')
