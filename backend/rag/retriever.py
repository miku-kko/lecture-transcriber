import asyncio
import logging
from typing import Optional

import chromadb

from backend.rag.embeddings import get_embedding_function

logger = logging.getLogger(__name__)


class LectureRetriever:
    def __init__(self, persist_dir: str):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name="lectures",
            embedding_function=get_embedding_function(),
            metadata={"hnsw:space": "cosine"},
        )

    async def search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> list[dict]:
        where = filter_metadata if filter_metadata else None
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self._collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            ),
        )

        output = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                output.append(
                    {
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                    }
                )
        return output

    async def search_with_context(self, query: str, n_results: int = 3) -> str:
        results = await self.search(query, n_results)
        if not results:
            return ""

        context_parts = []
        for r in results:
            lecture_id = r["metadata"].get("lecture_id", "unknown")
            context_parts.append(f"[Z wykladu {lecture_id}]: {r['text']}")
        return "\n\n".join(context_parts)
