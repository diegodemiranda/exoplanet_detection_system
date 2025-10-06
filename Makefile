.PHONY: help cert-dev up down logs ps grafana build-frontend

help:
	@echo "Available targets:"
	@echo "  cert-dev       - Generate a self-signed TLS certificate for Nginx (localhost)"
	@echo "  up             - Start all services with Docker Compose (builds frontend first if present)"
	@echo "  down           - Stop and remove services"
	@echo "  logs           - Stream API logs"
	@echo "  ps             - List running services"
	@echo "  grafana        - Show Grafana URL"
	@echo "  build-frontend - Build the frontend locally (if present)"

cert-dev:
	@mkdir -p nginx/ssl
	@echo "Creating self-signed certificate under nginx/ssl/..."
	@openssl req -x509 -newkey rsa:2048 -sha256 -days 365 -nodes \
		-keyout nginx/ssl/key.pem -out nginx/ssl/cert.pem \
		-subj "/CN=localhost" \
		-addext "subjectAltName=DNS:localhost,IP:127.0.0.1"
	@echo "Certificate created: nginx/ssl/cert.pem"
	@echo "Private key:        nginx/ssl/key.pem"

# Build the frontend locally (if the frontend source exists).
# This is useful for local development so the static assets are available
# to the nginx/frontend service without relying on the docker build stage.
build-frontend:
	@echo "Checking for frontend source in src/frontend..."
	@if [ -d "src/frontend" ]; then \
		echo "Building frontend (src/frontend)..."; \
		cd src/frontend && npm ci --no-audit --fund=false && npm run build; \
		echo "Frontend build finished."; \
	else \
		echo "No src/frontend directory found — skipping frontend build."; \
	fi

up:
	@$(MAKE) build-frontend
	docker compose up --build -d

down:
	docker compose down -v

logs:
	docker compose logs -f exoplanet-api

ps:
	docker compose ps
