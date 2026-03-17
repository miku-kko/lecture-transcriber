from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.audio.capture import AudioCapture, AudioConfig
from backend.config import Settings
from backend.correction.correction_manager import CorrectionManager
from backend.correction.gemini_corrector import GeminiCorrector
from backend.correction.ollama_corrector import OllamaCorrector
from backend.models.messages import WSMessage, WSMessageType
from backend.models.transcript import SpeakerRole
from backend.rag.indexer import LectureIndexer
from backend.rag.retriever import LectureRetriever
from backend.storage.session_manager import SessionManager
from backend.storage.vault_manager import VaultManager
from backend.stt.deepgram_client import DeepgramSTTClient
from backend.stt.transcript_assembler import TranscriptAssembler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

settings = Settings()

# --- Service singletons ---
audio_capture = AudioCapture(AudioConfig())
deepgram_client = DeepgramSTTClient(settings)
transcript_assembler = TranscriptAssembler()
ollama_corrector = OllamaCorrector(
    base_url=settings.ollama_base_url,
    model=settings.ollama_model,
)
gemini_corrector: Optional[GeminiCorrector] = (
    GeminiCorrector(api_key=settings.gemini_api_key)
    if settings.gemini_api_key
    else None
)
correction_manager = CorrectionManager(
    ollama=ollama_corrector,
    gemini=gemini_corrector,
)
session_manager = SessionManager()
vault_manager = VaultManager(settings.vault_path)

# RAG is lazy-initialized to avoid slow startup when not needed
_indexer: Optional[LectureIndexer] = None
_retriever: Optional[LectureRetriever] = None


def get_indexer() -> LectureIndexer:
    global _indexer
    if _indexer is None:
        _indexer = LectureIndexer(settings.chroma_persist_dir)
    return _indexer


def get_retriever() -> LectureRetriever:
    global _retriever
    if _retriever is None:
        _retriever = LectureRetriever(settings.chroma_persist_dir)
    return _retriever


# --- WebSocket Connection Manager ---

class ConnectionManager:
    def __init__(self):
        self._connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.append(websocket)
        logger.info(f"Client connected. Total: {len(self._connections)}")

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self._connections:
            self._connections.remove(websocket)
        logger.info(f"Client disconnected. Total: {len(self._connections)}")

    async def broadcast(self, message: WSMessage) -> None:
        data = message.to_json()
        disconnected = []
        for ws in self._connections:
            try:
                await ws.send_text(data)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self.disconnect(ws)


ws_manager = ConnectionManager()


# --- Session Orchestrator ---

class SessionOrchestrator:
    def __init__(self):
        self._stt_to_ui_task: Optional[asyncio.Task] = None
        self._correction_to_ui_task: Optional[asyncio.Task] = None

    async def start_session(self, title: str = "", metadata: dict | None = None):
        if not settings.deepgram_api_key:
            raise ValueError("Deepgram API key is not configured. Set DEEPGRAM_API_KEY in .env")
        if session_manager.is_active:
            await self.stop_session()

        session = session_manager.create_session(title, metadata)
        transcript_assembler.reset()

        await deepgram_client.connect()

        # Wire audio directly from mic thread to Deepgram (bypasses asyncio loop)
        audio_capture.set_audio_callback(deepgram_client.send_audio_sync)

        # Wire audio level reporting to broadcast to UI
        self._loop = asyncio.get_running_loop()
        audio_capture.set_level_callback(self._on_audio_level)

        await audio_capture.start()
        await correction_manager.start()

        self._stt_to_ui_task = asyncio.create_task(self._stt_to_ui_loop())
        self._correction_to_ui_task = asyncio.create_task(self._correction_to_ui_loop())

        await ws_manager.broadcast(
            WSMessage(
                type=WSMessageType.SESSION_STARTED,
                payload={"session_id": session.session_id, "title": session.title},
            )
        )
        logger.info(f"Session started: {session.title}")

    def _on_audio_level(self, rms: float, peak: float) -> None:
        """Called from audio thread — schedule broadcast on the event loop."""
        try:
            self._loop.call_soon_threadsafe(
                asyncio.ensure_future,
                ws_manager.broadcast(
                    WSMessage(
                        type=WSMessageType.AUDIO_LEVEL,
                        payload={
                            "rms": round(rms, 4),
                            "peak": round(peak, 4),
                            "gain": round(audio_capture.gain, 1),
                            "agc_gain": round(audio_capture.effective_gain / max(audio_capture.gain, 0.1), 1),
                            "effective_gain": round(audio_capture.effective_gain, 1),
                        },
                    )
                ),
            )
        except Exception:
            pass  # event loop may be closed during shutdown

    async def stop_session(self):
        await audio_capture.stop()

        for task in [
            self._stt_to_ui_task,
            self._correction_to_ui_task,
        ]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        await deepgram_client.disconnect()
        await correction_manager.stop()

        session = session_manager.end_session()
        if session and session.segments:
            filepath = vault_manager.save_session(session)

            # Index for RAG
            full_text = " ".join(s.text for s in session.segments)
            try:
                get_indexer().index_lecture(
                    lecture_id=session.session_id,
                    text=full_text,
                    metadata={
                        "title": session.title,
                        "date": time.strftime("%Y-%m-%d"),
                    },
                )
            except Exception as e:
                logger.error(f"Failed to index lecture: {e}")

            await ws_manager.broadcast(
                WSMessage(
                    type=WSMessageType.SESSION_STOPPED,
                    payload={
                        "session_id": session.session_id,
                        "file_path": str(filepath),
                    },
                )
            )
            logger.info(f"Session saved: {filepath}")
        else:
            await ws_manager.broadcast(
                WSMessage(
                    type=WSMessageType.SESSION_STOPPED,
                    payload={"session_id": "", "file_path": ""},
                )
            )

    async def _stt_to_ui_loop(self):
        while True:
            try:
                segment = await asyncio.wait_for(
                    deepgram_client.get_segment(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            result = transcript_assembler.process_segment(segment)

            if result["action"] == "update_interim":
                await ws_manager.broadcast(
                    WSMessage(
                        type=WSMessageType.TRANSCRIPT_INTERIM,
                        payload={
                            "segment": segment.to_dict(),
                            "index": result["index"],
                        },
                    )
                )
            elif result["action"] == "finalize":
                if session_manager.current:
                    session_manager.current.segments.append(segment)

                await ws_manager.broadcast(
                    WSMessage(
                        type=WSMessageType.TRANSCRIPT_FINAL,
                        payload={
                            "segment": segment.to_dict(),
                            "index": result["index"],
                        },
                    )
                )

                context = transcript_assembler.get_text_for_correction()
                await correction_manager.submit(segment, context)

    async def _correction_to_ui_loop(self):
        while True:
            try:
                result = await asyncio.wait_for(
                    correction_manager.get_result(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            if session_manager.mode == "automatic":
                self._apply_correction(result)

            await ws_manager.broadcast(
                WSMessage(
                    type=WSMessageType.CORRECTION_RESULT,
                    payload={
                        **result.to_dict(),
                        "auto_applied": session_manager.mode == "automatic",
                    },
                )
            )

    def _apply_correction(self, result):
        if not session_manager.current:
            return
        # Batch corrections have pipe-separated IDs like "id1|id2|id3"
        target_ids = set(result.segment_id.split("|"))
        for seg in session_manager.current.segments:
            if seg.segment_id in target_ids:
                seg.correction = result
                seg.is_corrected = True

    async def accept_correction(self, segment_id: str):
        if not session_manager.current:
            return
        for seg in session_manager.current.segments:
            if seg.segment_id == segment_id and seg.correction:
                seg.is_corrected = True
                break

    async def reject_correction(self, segment_id: str):
        if not session_manager.current:
            return
        for seg in session_manager.current.segments:
            if seg.segment_id == segment_id:
                seg.correction = None
                seg.is_corrected = False
                break

    async def edit_segment(self, segment_id: str, new_text: str):
        if not session_manager.current:
            return
        for seg in session_manager.current.segments:
            if seg.segment_id == segment_id:
                seg.text = new_text
                seg.correction = None
                seg.is_corrected = False
                break

    def set_mode(self, mode: str):
        session_manager.mode = mode

    def set_speaker_role(self, speaker_id: int, role: str):
        session_manager.set_speaker_role(speaker_id, role)
        deepgram_client.set_speaker_map(
            session_manager.current.speaker_map if session_manager.current else {}
        )

    def set_title(self, title: str):
        session_manager.set_title(title)

    def set_keyterms(self, keyterms: list[str]):
        deepgram_client.set_keyterms(keyterms)

    def set_gain(self, gain: float, agc: bool | None = None):
        audio_capture.gain = gain
        if agc is not None:
            audio_capture.agc_enabled = agc

    async def rag_query(self, query: str) -> list[dict]:
        try:
            return await get_retriever().search(query)
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return []

    async def list_lectures(self) -> list[dict]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, vault_manager.list_lectures)


orchestrator = SessionOrchestrator()


# --- FastAPI App ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    vault_manager.initialize_vault()
    logger.info(f"Server starting on {settings.host}:{settings.port}")
    yield
    if session_manager.is_active:
        await orchestrator.stop_session()
    await ollama_corrector.close()
    logger.info("Server shutting down")


app = FastAPI(title="Lecture Transcriber", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
frontend_path = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


@app.get("/")
async def root():
    return {"status": "ok", "message": "Lecture Transcriber API"}


@app.get("/api/devices")
async def list_devices():
    return {"devices": AudioCapture.list_devices()}


@app.get("/api/status")
async def get_status():
    ollama_ok = await ollama_corrector.is_available()
    return {
        "recording": session_manager.is_active,
        "mode": session_manager.mode,
        "session_id": (
            session_manager.current.session_id if session_manager.current else None
        ),
        "session_title": (
            session_manager.current.title if session_manager.current else None
        ),
        "segment_count": (
            len(session_manager.current.segments) if session_manager.current else 0
        ),
        "ollama_available": ollama_ok,
        "gemini_available": gemini_corrector is not None,
        "deepgram_configured": bool(settings.deepgram_api_key),
        "gain": audio_capture.gain,
        "agc_enabled": audio_capture.agc_enabled,
        "effective_gain": audio_capture.effective_gain,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = WSMessage.from_json(raw)
                await _handle_ws_message(websocket, msg)
            except Exception as e:
                logger.error(f"WS message error: {e}")
                await websocket.send_text(
                    WSMessage(
                        type=WSMessageType.ERROR,
                        payload={"error": str(e)},
                    ).to_json()
                )
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


async def _handle_ws_message(websocket: WebSocket, msg: WSMessage):
    handlers = {
        WSMessageType.CMD_START_RECORDING: lambda p: orchestrator.start_session(
            p.get("title", ""), p.get("metadata", {})
        ),
        WSMessageType.CMD_STOP_RECORDING: lambda p: orchestrator.stop_session(),
        WSMessageType.CMD_SET_MODE: lambda p: _sync_wrap(
            orchestrator.set_mode, p["mode"]
        ),
        WSMessageType.CMD_ACCEPT_CORRECTION: lambda p: orchestrator.accept_correction(
            p["segment_id"]
        ),
        WSMessageType.CMD_REJECT_CORRECTION: lambda p: orchestrator.reject_correction(
            p["segment_id"]
        ),
        WSMessageType.CMD_EDIT_SEGMENT: lambda p: orchestrator.edit_segment(
            p["segment_id"], p["text"]
        ),
        WSMessageType.CMD_RAG_QUERY: lambda p: _handle_rag(websocket, p["query"]),
        WSMessageType.CMD_LIST_LECTURES: lambda p: _handle_list(websocket),
        WSMessageType.CMD_SET_SPEAKER_ROLE: lambda p: _sync_wrap(
            orchestrator.set_speaker_role, p["speaker_id"], p["role"]
        ),
        WSMessageType.CMD_SET_TITLE: lambda p: _sync_wrap(
            orchestrator.set_title, p["title"]
        ),
        WSMessageType.CMD_SET_KEYTERMS: lambda p: _sync_wrap(
            orchestrator.set_keyterms, p.get("keyterms", [])
        ),
        WSMessageType.CMD_SET_GAIN: lambda p: _sync_wrap(
            orchestrator.set_gain, p.get("gain", 3.0), p.get("agc")
        ),
    }

    handler = handlers.get(msg.type)
    if handler:
        result = handler(msg.payload)
        if asyncio.iscoroutine(result):
            await result
    else:
        logger.warning(f"Unknown WS command: {msg.type}")


def _sync_wrap(fn, *args):
    fn(*args)


async def _handle_rag(websocket: WebSocket, query: str):
    results = await orchestrator.rag_query(query)
    await websocket.send_text(
        WSMessage(
            type=WSMessageType.RAG_RESULT,
            payload={"query": query, "results": results},
        ).to_json()
    )


async def _handle_list(websocket: WebSocket):
    lectures = await orchestrator.list_lectures()
    await websocket.send_text(
        WSMessage(
            type=WSMessageType.LECTURE_LIST,
            payload={"lectures": lectures},
        ).to_json()
    )


# Serve index.html for the root page
from fastapi.responses import FileResponse, PlainTextResponse


@app.get("/app")
async def serve_app():
    return FileResponse(str(frontend_path / "index.html"))


@app.get("/api/lecture")
async def get_lecture(path: str):
    try:
        content = vault_manager.read_lecture(path)
        return PlainTextResponse(content)
    except FileNotFoundError:
        return PlainTextResponse("Plik nie znaleziony", status_code=404)
    except Exception as e:
        return PlainTextResponse(str(e), status_code=500)
