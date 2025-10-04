PHONY: help cert-dev up down logs ps grafana

help:
	@echo "Targets disponíveis:"
	@echo "  cert-dev       - Gera certificado autoassinado para Nginx (localhost)"
	@echo "  up             - Sobe todos os serviços com Docker Compose"
	@echo "  down           - Para e remove os serviços"
	@echo "  logs           - Mostra logs da API"
	@echo "  ps             - Lista serviços em execução"
	@echo "  grafana        - Mostra URL do Grafana"

cert-dev:
	@mkdir -p nginx/ssl
	@echo "Gerando certificado autoassinado em nginx/ssl/..."
	@openssl req -x509 -newkey rsa:2048 -sha256 -days 365 -nodes \
		-keyout nginx/ssl/key.pem -out nginx/ssl/cert.pem \
		-subj "/CN=localhost" \
		-addext "subjectAltName=DNS:localhost,IP:127.0.0.1"
	@echo "Certificado gerado: nginx/ssl/cert.pem"
	@echo "Chave privada:      nginx/ssl/key.pem"

up:
	docker compose up --build -d

down:
	docker compose down -v

logs:
	docker compose logs -f exoplanet-api

ps:
	docker compose ps


