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

* TODO: fill this out

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

`ChatHeroku` class exposes chat models from Heroku.

```python
from langchain_heroku import ChatHeroku

llm = ChatHeroku()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`HerokuEmbeddings` class exposes embeddings from Heroku.

```python
from langchain_heroku import HerokuEmbeddings

embeddings = HerokuEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`HerokuLLM` class exposes LLMs from Heroku.

```python
from langchain_heroku import HerokuLLM

llm = HerokuLLM()
llm.invoke("The meaning of life is")
```
