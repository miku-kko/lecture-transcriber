.PHONY: setup dev run install clean

install:
	pip3 install -r requirements.txt

setup: install
	@echo "Copying .env.example to .env (if not exists)..."
	@cp -n .env.example .env 2>/dev/null || true
	@echo ""
	@echo "Setup complete! Edit .env with your API keys:"
	@echo "  - DEEPGRAM_API_KEY (required)"
	@echo "  - GEMINI_API_KEY (optional)"
	@echo ""
	@echo "Also make sure Ollama is running with Bielik model:"
	@echo "  brew install ollama"
	@echo "  ollama pull SpeakLeash/bielik-7b-instruct-v0.1-gguf"
	@echo "  ollama serve"
	@echo ""
	@echo "And install PortAudio for audio capture:"
	@echo "  brew install portaudio"

dev:
	python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

run:
	python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
