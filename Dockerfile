# Jarvis (bot do Telegram) em contêiner. O Ollama roda em outro serviço
# (docker-compose.yml); esta imagem só tem o bot.
#
#   docker compose up -d --build
#
# Sem segredos aqui: o .env entra em tempo de execução (env_file no compose) e o
# .dockerignore impede que ele (ou credentials.json/token.json) entre no contexto.

# 3.11: todos os pinos do requirements.txt têm wheel (o numpy 1.24.4 não tem para 3.12).
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# tesseract: OCR de PDFs escaneados e imagens (read_pdf/read_image);
# git: ferramenta git do agente. bash já vem na imagem (o run_shell usa /bin/bash).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates git tesseract-ocr tesseract-ocr-por \
    && rm -rf /var/lib/apt/lists/*

# Usuário sem privilégios: o terminal do agente roda como ele, não como root.
RUN useradd --create-home --uid 1000 --shell /bin/bash jarvis

WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY jarvis/ ./jarvis/
COPY scripts/ ./scripts/

# data/ (histórico, métricas, índice do RAG, uploads, tokens) e a pasta de trabalho
# são volumes; o código em /app fica somente leitura para o usuário jarvis.
RUN mkdir -p /app/data /home/jarvis/workspace \
    && chown -R jarvis:jarvis /app/data /home/jarvis

USER jarvis

# Padrões do contêiner (o compose e o .env podem sobrescrever).
ENV JARVIS_DB_PATH=/app/data/jarvis.sqlite3 \
    JARVIS_WORKSPACE=/home/jarvis/workspace \
    JARVIS_ALLOWED_ROOTS=/home/jarvis/workspace \
    OLLAMA_URL=http://ollama:11434

VOLUME ["/app/data"]

# Long polling: o bot só faz conexões de saída, nenhuma porta é exposta.
CMD ["python", "-m", "jarvis"]
