"""RAG pipeline: chunking, embedding, retrieval, and generation."""

import re

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# Shared embedding model (loaded once, runs locally — no API key needed)
_EMBEDDINGS = None


def _get_embeddings():
    """Lazy-load the HuggingFace embeddings model."""
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _EMBEDDINGS


def load_and_chunk_pdf(file_path: str) -> list[Document]:
    """Load a PDF and split it into chunks.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of Document objects (chunks).
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    return chunks


def build_vector_store(documents: list[Document]) -> FAISS:
    """Build a FAISS vector store from a list of Document chunks.

    Args:
        documents: List of Document objects to index.

    Returns:
        A FAISS vector store instance.
    """
    embeddings = _get_embeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store


def retrieve_context(vector_store: FAISS, query: str, k: int = 4) -> list[Document]:
    """Retrieve the top-k most relevant document chunks for a query.

    Args:
        vector_store: The FAISS vector store to search.
        query: User query string.
        k: Number of results to return.

    Returns:
        List of relevant Document chunks.
    """
    results = vector_store.similarity_search(query, k=k)
    return results


def generate_answer(llm, query: str, context_docs: list[Document]) -> str:
    """Generate an answer using the LLM with retrieved context.

    Args:
        llm: A LangChain chat model instance.
        query: The user's question.
        context_docs: List of retrieved Document chunks.

    Returns:
        The generated answer string.
    """
    # Build numbered context blocks
    context_blocks = []
    for i, doc in enumerate(context_docs, 1):
        page = doc.metadata.get("page", "unknown")
        context_blocks.append(f"[Context {i}] (Page {page}):\n{doc.page_content}")

    context_str = "\n\n".join(context_blocks)

    prompt = f"""You are a helpful document Q&A assistant. Answer the user's question based ONLY on the provided context. If the context does not contain enough information to answer, say so clearly.

After your answer, cite which context block(s) you used (e.g., [Context 1], [Context 3]).

--- CONTEXT ---
{context_str}

--- QUESTION ---
{query}

--- ANSWER ---"""

    response = llm.invoke(prompt)

    # Handle both string and AIMessage responses
    if hasattr(response, "content"):
        answer = response.content
    else:
        answer = str(response)

    # Strip <think>...</think> reasoning tags from models that include them
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

    return answer
