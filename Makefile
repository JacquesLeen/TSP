
# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Variables
PYTHON := python3
PIP := pip
PYTEST := pytest --verbose
PYLINT := pylint
BLACK := black
SCRIPT := fetch_cities_lat_long.py
TEST_FILE := test_fetch_cities_lat_long.py
ALL_FILES := *.py

install:
	@echo "$(BLUE)Installing dependencies...$(NC)"
	@$(PIP) install --upgrade $(PIP) &&\
		$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Dependencies installed$(NC)"
	

lint:
	@echo "$(BLUE)Running pylint...$(NC)"
	@$(PYLINT) $(SCRIPT) || (EXIT_CODE=$$?; \
		if [ $$EXIT_CODE -eq 0 ]; then \
			echo "$(GREEN)✓ No issues found$(NC)"; \
		elif [ $$EXIT_CODE -eq 4 ]; then \
			echo "$(YELLOW)⚠ Warnings found (exit code 4 - ignored)$(NC)"; \
			exit 0; \
		elif [ $$EXIT_CODE -eq 8 ]; then \
			echo "$(YELLOW)⚠ Refactor suggestions found (exit code 8 - ignored)$(NC)"; \
			exit 0; \
		elif [ $$EXIT_CODE -eq 12 ]; then \
			echo "$(YELLOW)⚠ Warnings + Refactor (exit code 12 - ignored)$(NC)"; \
			exit 0; \
		elif [ $$EXIT_CODE -eq 16 ]; then \
			echo "$(YELLOW)⚠ Convention violations found (exit code 16 - ignored)$(NC)"; \
			exit 0; \
		elif [ $$EXIT_CODE -eq 20 ]; then \
			echo "$(YELLOW)⚠ Convention + Warnings (exit code 20 - ignored)$(NC)"; \
			exit 0; \
		elif [ $$EXIT_CODE -eq 24 ]; then \
			echo "$(YELLOW)⚠ Convention + Refactor (exit code 24 - ignored)$(NC)"; \
			exit 0; \
		elif [ $$EXIT_CODE -eq 28 ]; then \
			echo "$(YELLOW)⚠ Convention + Refactor + Warnings (exit code 28 - ignored)$(NC)"; \
			exit 0; \
		elif [ $$EXIT_CODE -eq 1 ] || [ $$EXIT_CODE -eq 2 ]; then \
			echo "$(RED)✗ Fatal errors or syntax errors found (exit code $$EXIT_CODE)$(NC)"; \
			exit $$EXIT_CODE; \
		else \
			echo "$(RED)✗ Pylint found issues (exit code $$EXIT_CODE)$(NC)"; \
			exit $$EXIT_CODE; \
		fi)
	@echo "$(GREEN)✓ Linting passed$(NC)"

test:
	@echo "$(BLUE)Running tests...$(NC)"
	$(PYTEST) $(TEST_FILE)
	@echo "$(GREEN)✓ Tests passed$(NC)"

format:
	@echo "$(BLUE)Formatting code with black...$(NC)"
	$(BLACK) $(ALL_FILES)
	@echo "$(GREEN)✓ Code formatted$(NC)"

all: install lint test format