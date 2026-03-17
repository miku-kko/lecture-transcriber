import logging
import re

import chromadb

from backend.rag.embeddings import get_embedding_function

logger = logging.getLogger(__name__)


class LectureIndexer:
    def __init__(self, persist_dir: str):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name="lectures",
            embedding_function=get_embedding_function(),
            metadata={"hnsw:space": "cosine"},
        )

    def index_lecture(
        self,
        lecture_id: str,
        text: str,
        metadata: dict,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> int:
        chunks = self._split_text(text, chunk_size, chunk_overlap)
        if not chunks:
            return 0

        ids = [f"{lecture_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {**metadata, "chunk_index": i, "lecture_id": lecture_id}
            for i in range(len(chunks))
        ]

        self._collection.add(documents=chunks, ids=ids, metadatas=metadatas)
        logger.info(f"Indexed {len(chunks)} chunks for lecture {lecture_id}")
        return len(chunks)

    def delete_lecture(self, lecture_id: str) -> None:
        results = self._collection.get(where={"lecture_id": lecture_id})
        if results["ids"]:
            self._collection.delete(ids=results["ids"])

    def _split_text(
        self, text: str, chunk_size: int, overlap: int
    ) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Character-based overlap: take last N characters for context continuity
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                # Avoid cutting mid-word: find first space in overlap region
                space_idx = overlap_text.find(" ")
                if space_idx > 0:
                    overlap_text = overlap_text[space_idx + 1:]
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk += " " + sentence

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    @property
    def count(self) -> int:
        return self._collection.count()
