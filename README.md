# central-prompt

Unified, provider-agnostic prompt management for MLflow and Langfuse via one class-based API.

- Create and fetch prompts from either provider through a single interface.
- Use a unified handle exposing `compile(**vars)` across providers.
- Strong input validation and clear, categorized errors.
- Optional extras install provider SDKs only when needed.

## Features
- Class-first API: `CentralPrompt(provider, ...)` with `.set_prompt(...)` and `.get_prompt(...)`.
- Providers supported: `"mlflow"`, `"langfuse"`.
- Template support:
  - Text templates (string with `{{ var }}` placeholders)
  - Chat templates (list of `{role, content}` dicts with placeholders)
- Robust error handling with custom exception types
- Minimal base dependencies; provider SDKs are optional extras

## Requirements
- Python 3.8+
- For MLflow usage: `mlflow` with `mlflow.genai` support
- For Langfuse usage: `langfuse` and `python-dotenv`

## Installation (Private)
You have multiple private install approaches. Pick the one that fits your workflow.

### 1) Editable install for local development
```bash
python -m pip install --upgrade pip setuptools
python -m pip install -e .[all]
```

### 2) Build a wheel and install locally
```bash
python -m pip install --upgrade build
python -m build  # creates dist/*.whl and sdist
python -m pip install "dist/central_prompt-0.1.0-py3-none-any.whl"
# Or with extras (auto-install deps):
python -m pip install "dist/central_prompt-0.1.0-py3-none-any.whl[all]"
```

### 3) Private Git repository
Push this directory to a private repo and install directly from Git.
```bash
python -m pip install "git+ssh://git@github.com/<org>/<repo>.git#egg=central-prompt[all]"
# or HTTPS with a token
python -m pip install "git+https://<token>@github.com/<org>/<repo>.git#egg=central-prompt[all]"
```

### 4) Private package index (e.g., Nexus/Artifactory/GitHub Packages)
Upload the built wheel to your private index and install using your index URL.
```bash
python -m pip install --index-url https://<your-private-index>/simple central-prompt[all]
```

## Quick Start
```python
from central_prompt import CentralPrompt

# MLflow: environment-based tracking
# Ensure MLFLOW_TRACKING_URI (and optionally MLFLOW_EXPERIMENT_NAME) are set in .env/ENV.
cp_ml = CentralPrompt("mlflow", experiment="traces-quickstart")
cp_ml.set_prompt(
    name="sentiment-text-prompt",
    template="Classify the sentiment as Positive or Negative for: {{ text }}",
    tags={"purpose": "sentiment"},
)
handle_ml = cp_ml.get_prompt(name="sentiment-text-prompt", version=1)
print(handle_ml.compile(text="I love this product!"))

# Langfuse: chat prompt with labels
cp_lf = CentralPrompt("langfuse")
cp_lf.set_prompt(
    name="support-chat-prompt",
    template=[
        {"role": "system", "content": "You are a helpful assistant specialized in {{ domain }}."},
        {"role": "user", "content": "Issue: {{ issue_description }}"},
    ],
    labels=["production"],
)
handle_lf = cp_lf.get_prompt(name="support-chat-prompt", label="production")
print(handle_lf.compile(domain="billing", issue_description="Can't see my invoice"))
```

## API

### CentralPrompt(provider, *, experiment: Optional[str] = None, load_env: bool = True)
- **provider**: `Literal["mlflow","langfuse"]` (case-normalized)
- **experiment** (MLflow): Optional experiment name. Tracking URI is taken from `.env`/ENV via `MLFLOW_TRACKING_URI`.
- **load_env**: If `True`, attempts to load `.env` for provider configuration.

#### set_prompt(name, template, *, prompt_type=None, labels=None, tags=None, response_format=None, commit_message=None) -> dict
- Creates/registers a prompt with the selected provider.
- Langfuse:
  - `prompt_type`: `"text"` or `"chat"` (auto-inferred if omitted)
  - `labels`: optional `List[str]`
- MLflow:
  - `tags`: `Dict[str, str]`
  - `response_format`: optional Pydantic model class for structured output
  - `commit_message`: optional
  - Returns `{"provider": ..., "name": ..., "version": ...}`

#### get_prompt(name=None, *, version=None, path=None, label=None) -> PromptHandle
- MLflow: provide `name+version` or `path` like `prompts:/<name>/<version>`.
- Langfuse: provide `name` and optionally either `version` or `label` (mutually exclusive).

### PromptHandle
- `compile(**vars)`: Inserts variables into the prompt template.
  - MLflow: forwards to `format(**vars)`
  - Langfuse: forwards to `compile(**vars)`
- `repr(handle)`: includes provider, name, version, and underlying type.

## Template Format
- **Text**: a single string with `{{ variable }}` placeholders.
- **Chat**: a list of dicts with `role` and `content` keys.

## Environment configuration
- MLflow: `.env` or environment variables
  - `MLFLOW_TRACKING_URI`
  - `MLFLOW_EXPERIMENT_NAME` (optional if passed in code)
- Langfuse: `.env` or environment variables
  - `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`

## Exceptions
- `PromptProviderError` — invalid/unsupported provider or bad arguments
- `PromptCreationError` — provider-specific failure during creation
- `PromptFetchError` — provider-specific failure during fetch/load
- `PromptCompilationError` — error during `compile(**vars)`

## Testing
- A comprehensive offline test suite is included using in-memory stubs for both providers.
```bash
python test.py
# or
python -m unittest -v test.py
```

## Versioning
- Current version: `0.1.0`
- Update `project.version` in `pyproject.toml` when cutting a new release.

## License
- Private/Internal use only. Do not publish to public indexes.
