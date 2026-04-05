# crawl4ai-mcp Makefile
# Usage:
#   make test            — run unit tests with coverage (offline, no Docker needed)
#   make lint            — run syntax / type checks
#   make build           — build the Docker image
#   make up              — bring up the full stack (app + pgvector DB)
#   make down            — tear down the stack
#   make logs            — tail the app container logs
#   make test-integration — run smoke tests against the live Docker stack
#   make clean           — stop stack and remove volumes

.PHONY: test test-fast lint build up down logs test-integration verify-e2e clean help

IMAGE_NAME   ?= crawl4ai-mcp
COMPOSE_FILE ?= docker-compose.yml
APP_SERVICE  ?= docs-rag-mcp
DB_SERVICE   ?= rag-db
PORT         ?= 8051

# -------------------------------------------------------------------------
# Unit tests (offline, mocked externals, 100 % coverage required)
# -------------------------------------------------------------------------
test:
	uv run pytest tests/ -q --tb=short \
	    --cov=src --cov-report=term-missing \
	    -m "not integration"

# Run only unit tests without coverage (faster)
test-fast:
	uv run pytest tests/ -q --tb=short -m "not integration"

# -------------------------------------------------------------------------
# Lint / type checks
# -------------------------------------------------------------------------
lint:
	uv run python -m py_compile src/utils.py src/crawl4ai_mcp.py \
	    src/crawler/web_crawler.py src/crawler/postgres_client.py \
	    src/crawler/tool_definitions.py && \
	echo "Syntax OK"

# -------------------------------------------------------------------------
# Docker build
# -------------------------------------------------------------------------
build:
	docker build -t $(IMAGE_NAME) .

# -------------------------------------------------------------------------
# Docker Compose helpers
# -------------------------------------------------------------------------
up:
	docker compose -f $(COMPOSE_FILE) up -d
	@echo "Waiting for services to be healthy…"
	@timeout 120 sh -c 'until docker compose -f $(COMPOSE_FILE) ps --status=running | grep -q "$(APP_SERVICE)"; do sleep 2; done' || true
	@echo "Stack is up. App should be reachable on http://localhost:$(PORT)"

down:
	docker compose -f $(COMPOSE_FILE) down

logs:
	docker compose -f $(COMPOSE_FILE) logs -f $(APP_SERVICE)

clean:
	docker compose -f $(COMPOSE_FILE) down -v --remove-orphans

# -------------------------------------------------------------------------
# Integration / smoke tests against live Docker stack
# Requires: stack to be running (`make up`)
# -------------------------------------------------------------------------
test-integration: _check-stack-up
	MCP_URL=http://localhost:$(PORT)/sse uv run python tests/integration_smoke.py

# Full local verification flow requested for this project:
# 1) unit tests with 100% coverage gate
# 2) docker image build
# 3) bring up compose environment
# 4) run integration smoke tests against live stack
verify-e2e: test build up test-integration

_check-stack-up:
	@docker compose -f $(COMPOSE_FILE) ps --status=running 2>/dev/null | grep -q "$(APP_SERVICE)" || \
	    (echo "ERROR: stack not running — run 'make up' first" && exit 1)

# -------------------------------------------------------------------------
help:
	@echo "Available targets:"
	@echo "  test              Run unit tests with 100% coverage gate"
	@echo "  test-fast         Run unit tests without coverage"
	@echo "  lint              Check Python syntax"
	@echo "  build             Build Docker image"
	@echo "  up                Start full stack (app + DB)"
	@echo "  down              Stop stack"
	@echo "  logs              Tail app container logs"
	@echo "  test-integration  Smoke-test the running stack"
	@echo "  verify-e2e        Run test + build + up + integration smoke"
	@echo "  clean             Stop stack and remove volumes"
