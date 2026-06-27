PYTHON?=uv run python

TESTS_ARGS=-m pytest --tb=short -s
COV_OPTIONS=--cov=src --cov-report term --cov-report html --cov-report xml:coverage.xml
DEFAULT_TEST_FILES=tests/

DIRS=src tests


.PHONY: help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(firstword $(MAKEFILE_LIST)) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-17s\033[0m %s\n", $$1, $$2}'

vendor: ## Install all necessary packages
	uv sync

fmt: vendor ## Format code
	@echo === FMT ===
	# Fix linting violations
	$(PYTHON) -m ruff check --fix $(DIRS) || true
	# Full code formatting
	$(PYTHON) -m ruff format $(DIRS)

lint-ruff:
	@echo === LINT RUFF ===
	$(PYTHON) -m ruff check --output-format=pylint $(DIRS)

lint-mypy:
	@echo === LINT MYPY ===
	$(PYTHON) -m mypy \
		--skip-cache-mtime-checks \
		--exclude .*\\.ipynb \
		$(DIRS)

lint: vendor lint-ruff lint-mypy ## Lint code

test: vendor ## Run test. You can run `make test tests/path/file.py`
	@TEST_FILES="$(filter-out $@,$(MAKECMDGOALS))"; \
	if [ -z "$$TEST_FILES" ]; then TEST_FILES=$(DEFAULT_TEST_FILES); fi; \
	$(PYTHON) $(TESTS_ARGS) $$TEST_FILES

test-cov: vendor ## Run all test with coverage. You can run `make test-cov tests/path/file.py`
	@TEST_FILES="$(filter-out $@,$(MAKECMDGOALS))"; \
	if [ -z "$$TEST_FILES" ]; then TEST_FILES=$(DEFAULT_TEST_FILES); fi; \
	$(PYTHON) $(TESTS_ARGS) $(COV_OPTIONS) $$TEST_FILES

clean: ## Clean generated files
	@find . -name '*.py[cod]' -exec rm -f {} +
	@find . -name '__pycache__' -exec rm -rf {} +
	@find . -name '*$py.class' -exec rm -rf {} +

# Prevent make from trying to build .py files as targets
.PHONY: $(filter %.py,$(MAKECMDGOALS))
$(filter %.py,$(MAKECMDGOALS)):
	@:
