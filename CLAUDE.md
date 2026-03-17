# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time Polish lecture transcription app with AI-powered text correction. Captures microphone audio, streams it to Deepgram for STT, corrects transcripts via Ollama (Bielik) or Gemini, and saves results as Obsidian-compatible Markdown notes. Polish-language UI and prompts throughout.

## Commands

```bash
make setup        # Install deps + create .env from example
make dev          # Dev server with hot reload (localhost:8000)
make run          # Production server (localhost:8000)
make clean        # Remove __pycache__

# Tests
python3 -m pytest                    # Run all tests
python3 -m pytest tests/test_models.py -k test_name  # Single test
```

**Prerequisites:** PortAudio (`brew install portaudio`), Ollama with Bielik model (`ollama pull SpeakLeash/bielik-7b-instruct-v0.1-gguf && ollama serve`).

## Architecture

### Data Pipeline

```
Mic (sounddevice) -> AudioCapture (gain/AGC) -> DeepgramSTTClient (WebSocket, Polish nova-3)
    -> TranscriptAssembler (interim/final) -> CorrectionManager (batches ~200 words)
    -> OllamaCorrector / GeminiCorrector -> WebSocket broadcast to frontend
```

On session stop: segments saved to Obsidian vault as Markdown + indexed in ChromaDB for RAG search.

### Backend (`backend/`)

- **`main.py`** — FastAPI app, WebSocket endpoint (`/ws`), REST endpoints, `SessionOrchestrator` that wires all services together. All services are module-level singletons.
- **`audio/capture.py`** — Mic capture with software gain and AGC. Runs callback on audio thread, uses `sounddevice`. Auto-detects built-in MacBook mic. Sets macOS system input volume via osascript.
- **`stt/deepgram_client.py`** — Raw WebSocket connection to Deepgram (not SDK). Runs send/recv in background threads with asyncio queue bridge. Groups words by speaker (diarization). Auto-reconnects up to 3 times.
- **`stt/transcript_assembler.py`** — Merges interim/final segments, maintains ordered segment list.
- **`correction/correction_manager.py`** — Batches segments until ~200 words or 15s timeout, then sends to AI with history of previously corrected blocks as context (up to 10 blocks). Tries Ollama first, falls back to Gemini.
- **`correction/ollama_corrector.py`** — Ollama `/api/chat` with Bielik model, JSON format, Polish system prompt.
- **`correction/gemini_corrector.py`** — Google GenAI client with rate limiting (12 RPM). Uses `run_in_executor` for sync API.
- **`correction/base.py`** — `BaseCorrector` ABC: `correct()` and `is_available()`.
- **`storage/session_manager.py`** — Holds current `TranscriptSession`, manages mode (automatic/interactive).
- **`storage/vault_manager.py`** — Saves sessions to `obsidian-vault/Lectures/YYYY/MM-Miesiąc/` as Markdown. Path traversal protection on reads.
- **`storage/markdown_writer.py`** — Generates Obsidian Markdown with YAML frontmatter, speaker labels, correction summaries.
- **`rag/indexer.py`** — Chunks text by sentences (~500 chars) and indexes in ChromaDB.
- **`rag/retriever.py`** — Cosine similarity search over ChromaDB with multilingual embeddings.
- **`rag/embeddings.py`** — Uses `paraphrase-multilingual-MiniLM-L12-v2` via sentence-transformers.
- **`models/`** — Dataclasses: `TranscriptSegment`, `TranscriptSession`, `Word`, `CorrectionResult`, `CorrectionItem`, `WSMessage`, `WSMessageType`.

### Frontend (`frontend/`)

Vanilla HTML/CSS/JS, no build step. Served as static files at `/static/`, app at `/app`.

- **`index.html`** — 3-panel layout: left (RAG search + lecture list), center (live transcript), right (corrections + keyterms + speaker map).
- **`js/app.js`** — WebSocket client, DOM rendering, state management. All communication via typed JSON messages over WS.
- **`css/main.css`** — Dark theme UI.

### WebSocket Protocol

All client-server communication uses JSON messages with `{type, payload, timestamp}`. Types prefixed `cmd_` are client->server commands, others are server->client events. Batch correction IDs are pipe-separated (`id1|id2|id3`).

### Storage

- **`obsidian-vault/`** — Output directory. Lectures stored in `Lectures/YYYY/MM-Miesiąc/*.md` with Polish month names.
- **`obsidian-vault/.rag-index/`** — ChromaDB persistent storage.

## Key Design Decisions

- Audio capture runs on a separate thread (sounddevice callback); audio bytes are pushed directly to Deepgram's send queue without going through asyncio.
- Deepgram connection uses `websockets.sync.client` in threads (not async) for reliability and timing control.
- RAG components are lazy-initialized to avoid slow startup from loading sentence-transformers.
- Correction batching with history context: each batch gets the last 10 corrected blocks as context for terminology/style consistency.
- Ollama availability is cached for 60s to avoid per-batch HTTP overhead.
