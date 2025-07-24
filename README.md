# langchain-heroku

This package contains the LangChain integration with Heroku

## Project Structure

```
langchain-heroku/
  README.md
  LICENSE.txt
  pyproject.toml
  poetry.lock
  Makefile
  CODE_OF_CONDUCT.md
  CODEOWNERS
  CONTRIBUTING.md
  SECURITY.md
  docs/
    ...
  langchain_heroku/
    __init__.py
    chat_models.py
    py.typed
    ...
  scripts/
    check_imports.py
    lint_imports.sh
  tests/
    test_chat_models.py
    test_chat_models_integration.py
    test_compile.py
    ...
  .gitignore
```

## Installation

```bash
pip install -U langchain-heroku
```

And you should configure credentials by setting the following environment variables:

* `INFERENCE_URL` - Your Heroku Inference API URL (e.g., `https://us.inference.heroku.com`)
* `INFERENCE_KEY` - Your Heroku Inference API key
* `INFERENCE_MODEL_ID` - The model ID to use (e.g., `claude-3-5-sonnet-latest`, `claude-3-5-haiku-latest`)

### Setting up Heroku AI

To use this integration, you need to set up Heroku's Managed Inference and Agents add-on:

1. **Install the Heroku CLI and AI Plugin**:
   ```bash
   # Install Heroku CLI if you haven't already
   # Then install the AI plugin
   heroku plugins:install @heroku/plugin-ai
   ```

2. **Create a Heroku App** (if you don't have one):
   ```bash
   heroku create <your-new-app-name>
   ```

3. **Provision an AI Model Resource**:
   ```bash
   # List available models
   heroku ai:models:list
   
   # Create and attach a model to your app
   heroku ai:models:create -a $APP_NAME $MODEL_ID
   
   # Example for Claude 3.5 Sonnet
   heroku ai:models:create -a my-app claude-3-5-sonnet-latest
   ```

4. **Get Your Config Variables**: After attaching a model resource, your app will have three new config variables. You can view them with:
   ```bash
   heroku config -a $APP_NAME
   ```

5. **Export Environment Variables**: You can export these as environment variables with:
   ```bash
   eval $(heroku config -a $APP_NAME --shell | grep '^INFERENCE_' | tee /dev/tty | sed 's/^/export /')
   ```

**Available Models**:
- `claude-3-5-sonnet-latest` - Claude 3.5 Sonnet (recommended for best intelligence)
- `claude-3-5-haiku-latest` - Claude 3.5 Haiku (cost-effective and fast)
- `claude-3-opus-latest` - Claude 3 Opus (most capable)
- `claude-3-sonnet-latest` - Claude 3 Sonnet
- `claude-3-haiku-latest` - Claude 3 Haiku

**Pricing**: Models are billed per token used. See the [Heroku AI pricing page](https://devcenter.heroku.com/articles/heroku-ai-pricing) for current rates.

## 🧪 Running Tests

This project uses [Poetry](https://python-poetry.org/) for dependency management and [pytest](https://docs.pytest.org/) for testing.

### 1. Install dependencies (including test dependencies)

```bash
poetry install --with test
```

### 2. Run the test suite

```bash
poetry run pytest
```

Or to run a specific test file:

```bash
poetry run pytest tests/unit_tests/test_chat_models.py
```

### 3. (Optional) Run with verbose output

```bash
poetry run pytest -v
```

### Notes

- Make sure you are using a compatible Python version (see `pyproject.toml`).
- If you add new test dependencies, add them to the `[tool.poetry.group.test.dependencies]` section in `pyproject.toml`.

## Chat Models

`ChatHeroku` class exposes chat models from Heroku using the Inference API.

```python
from langchain_heroku import ChatHeroku
from langchain_core.messages import HumanMessage

chat = ChatHeroku()
result = chat.invoke([HumanMessage(content="Sing a ballad of LangChain.")])
print(result.content)
```

### Advanced Usage

```python
from langchain_core.messages import HumanMessage, SystemMessage

# With system message
messages = [
    SystemMessage(content="You are a helpful assistant that speaks in a friendly tone."),
    HumanMessage(content="What's the weather like?")
]

chat = ChatHeroku(temperature=0.7, max_tokens=256)
result = chat.invoke(messages)
print(result.content)
```

### Streaming

```python
chat = ChatHeroku(streaming=True)
for chunk in chat.stream([HumanMessage(content="Tell me a story.")]):
    print(chunk.content, end="")
```
