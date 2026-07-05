# Deployment

## Local

```bash
make install
make ingest-demo
make evaluate
make run-api
make run-app
```

## Docker Compose

```bash
cp .env.example .env
docker compose up --build
```

The API runs on `http://localhost:8000` and Streamlit runs on `http://localhost:8501`.

Real SEC ingestion requires `SEC_EDGAR_USER_AGENT` with a valid name and email. Groq, HuggingFace, and LoRA are optional providers; the extractive provider is always available and labels itself as fallback mode.
