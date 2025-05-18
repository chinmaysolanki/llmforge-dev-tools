# This file serves as the entry point for Vercel to run the FastAPI app.

# Adjust the import path if your main FastAPI app file is named differently or located elsewhere.
# Assuming advanced_code_assistant.py is in the root directory (one level up from 'api')
from ..advanced_code_assistant import app

# Vercel will look for an 'app' instance that is a WSGI/ASGI compatible application.
# FastAPI 'app' is ASGI compatible. 