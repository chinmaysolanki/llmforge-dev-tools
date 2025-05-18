# LLMForge – where AI models forge your dev tools.

A powerful AI-powered coding and portfolio assistant. This assistant uses large language models via OpenRouter to generate code, analyze Python projects, and create static or React-based web portfolios from prompts.

## Features

- Code generation from natural language prompts via OpenRouter LLMs.
- Python project analysis (technical metrics and LLM-generated summaries).
- Static web portfolio generation (HTML, CSS, JS) from descriptions.
- React.js portfolio source code generation.
- Web interface for all functionalities.
- Designed for Vercel deployment.

## Requirements

- Python 3.8+
- Internet connection for OpenRouter API access.

## Installation & Local Development

1. Clone this repository:
```bash
git clone <your-repo-url> # Replace <your-repo-url> with your actual repository URL
cd llmforge 
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional, for local .env use) Create a `.env` file in the root directory:
```
OPENROUTER_API_KEY="your_openrouter_api_key_here"
# OPENROUTER_BASE_URL="https://openrouter.ai/api/v1" # Default is usually fine
```
   *Note: The `OPENROUTER_API_KEY` is currently hardcoded in `advanced_code_assistant.py` but using an environment variable is best practice.*

## Usage

### Starting the API Server Locally

To run the server locally (after installation and setting up `.env` if not using the hardcoded key):
```bash
python3 advanced_code_assistant.py
```

The server will start on `http://127.0.0.1:8000` by default (if the `uvicorn.run` block is uncommented in `advanced_code_assistant.py`).

### Accessing the Web Interface

Open your browser and navigate to `http://127.0.0.1:8000` (or your Vercel deployment URL).

### API Endpoints (served by FastAPI)

- **`/`**: Serves the main HTML interface.
- **`/api/openrouter-models` (GET)**: Lists available OpenRouter models.
- **`/api/generate-code` (POST)**: Generates code snippets.
  - Payload: `{"prompt": "user_prompt", "openrouter_model_name": "model_id"}`
- **`/api/upload-and-analyze-project` (POST)**: Uploads a Python project ZIP, analyzes it, and provides a summary.
  - Query Params: `openrouter_model_name`
  - Body: `multipart/form-data` with `file` (the .zip file).
- **`/api/generate-portfolio` (POST)**: Generates a static portfolio ZIP.
  - Payload: `{"prompt": "portfolio_description", "openrouter_model_name": "model_id"}`
- **`/api/generate-react-portfolio` (POST)**: Generates React portfolio source code ZIP.
  - Payload: `{"prompt": "react_portfolio_description", "openrouter_model_name": "model_id"}`

## Deployment to Vercel

This project is configured for easy deployment to Vercel:

1. Push your code to a Git repository (GitHub, GitLab, Bitbucket).
2. Import the repository into Vercel.
3. Vercel should automatically detect the Python environment using `vercel.json` and `requirements.txt`.
4. **Important**: If you choose *not* to rely on the hardcoded API key, set your `OPENROUTER_API_KEY` as an environment variable in the Vercel project settings.
5. Deploy. Vercel will provide a public URL.

## License

MIT License (Please update if different)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 