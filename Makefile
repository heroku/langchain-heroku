.PHONY: all format lint test tests integration_tests help clean install build

# Default target executed when no arguments are given to make.
all: help

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/
integration_test integration_tests: TEST_FILE = tests/integration_tests/


# unit tests are run with the --disable-socket flag to prevent network calls
test tests:
	poetry run pytest --disable-socket --allow-unix-socket $(TEST_FILE)

test_watch:
	poetry run ptw --snapshot-update --now . -- -vv $(TEST_FILE)

# integration tests are run without the --disable-socket flag to allow network calls
integration_test integration_tests:
	poetry run pytest $(TEST_FILE)

# LangGraph 101 integration tests with ChatHeroku
langgraph_test:
	poetry run pytest tests/integration_tests/test_langgraph_101_integration.py -v -s

embeddings_test:
	poetry run pytest tests/integration_tests/test_embeddings_integration.py -v -s

######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --relative=libs/partners/heroku --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langchain_heroku
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests:
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff check $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && poetry run mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

lint_fix lint_fix_diff lint_fix_package lint_fix_tests:
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff check --fix $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff format $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && poetry run mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff format $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff check --select I --fix $(PYTHON_FILES)

spell_check:
	poetry run codespell --toml pyproject.toml

spell_fix:
	poetry run codespell --toml pyproject.toml -w

check_imports: $(shell find langchain_heroku -name '*.py')
	poetry run python ./scripts/check_imports.py $^

######################
# DEVELOPMENT TASKS
######################

install:
	poetry install

build:
	poetry build

clean:
	rm -rf .mypy_cache .mypy_cache_test
	rm -rf .pytest_cache
	rm -rf __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

######################
# HELP
######################

help:
	@echo 'Available targets:'
	@echo '=================='
	@echo 'install                    - install dependencies'
	@echo 'build                      - build the package'
	@echo 'clean                      - remove cache files and compiled Python files'
	@echo 'check_imports              - check imports'
	@echo 'format                     - run code formatters'
	@echo 'format_diff                - format only changed files'
	@echo 'lint                       - run linters'
	@echo 'lint_fix                   - run linters and fix issues automatically'
	@echo 'lint_diff                  - lint only changed files'
	@echo 'lint_fix_diff              - lint and fix only changed files'
	@echo 'lint_package               - lint only the main package'
	@echo 'lint_fix_package           - lint and fix only the main package'
	@echo 'lint_tests                 - lint only test files'
	@echo 'lint_fix_tests             - lint and fix only test files'
	@echo 'spell_check                - check spelling'
	@echo 'spell_fix                  - fix spelling issues'
	@echo 'test                       - run unit tests'
	@echo 'tests                      - run unit tests'
	@echo 'test_watch                 - run tests in watch mode'
	@echo 'langgraph_test             - test LangGraph 101 integration with ChatHeroku'
	@echo 'integration_tests          - run integration tests'
	@echo 'test TEST_FILE=<test_file> - run all tests in file'
