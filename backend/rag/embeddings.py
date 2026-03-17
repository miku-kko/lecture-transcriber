from chromadb.utils import embedding_functions


def get_embedding_function():
    """
    Returns the multilingual embedding function for ChromaDB.
    paraphrase-multilingual-MiniLM-L12-v2 supports 50+ languages including Polish.
    """
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
