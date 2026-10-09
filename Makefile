# Raccourcis de développement. `make help` liste les cibles.
PYTHON ?= python

.PHONY: help install test lint format check demo clean

help: ## affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-10s %s\n", $$1, $$2}'

install: ## installe le paquet en mode développement avec les outils de test
	$(PYTHON) -m pip install -e ".[dev]"

test: ## lance la suite de tests (aucun accès réseau)
	$(PYTHON) -m pytest

lint: ## vérifie le code (ruff check + ruff format --check)
	ruff check .
	ruff format --check .

format: ## reformate le code avec ruff
	ruff format .
	ruff check --fix .

check: lint test ## lint puis tests (ce que fait l'intégration continue)

demo: ## backtest de toutes les stratégies sur des données synthétiques
	$(PYTHON) -m tradingbot demo

clean: ## supprime les caches Python, pytest et ruff
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
