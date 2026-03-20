"""In-memory FAISS vector store management, keyed by session ID."""

from typing import Any

_STORES: dict[str, Any] = {}


def save_store(session_id: str, store) -> None:
    """Save a FAISS vector store under the given session ID."""
    _STORES[session_id] = store


def get_store(session_id: str):
    """Retrieve the FAISS vector store for a session, or None if not found."""
    return _STORES.get(session_id)


def delete_store(session_id: str) -> bool:
    """Delete the vector store for a session. Returns True if it existed."""
    if session_id in _STORES:
        del _STORES[session_id]
        return True
    return False
