.PHONY: llm ui all

all: llm ui

llm:
	@echo "Starting LLM services..."
	flask --app services.llm.app run --debug --host 0.0.0.0 --port 12001

ui:
	@echo "Starting UI..."
	flask --app services.ui run --debug --host 0.0.0.0 --port 12002

retrieve:
	@echo "Starting RAG: embedding document"
	python -m services.rag.app

chat:
	@echo "CLI Chatting..."
	python -m services.llm.app