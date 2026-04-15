"""
Backend-friendly RAG service layer.

Provides a stable structured response schema that can be consumed by
FastAPI, Streamlit, CLI, bots, or any future channel.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List, Optional

from chain import RAGChain


@dataclass
class Citation:
    chunk_id: Optional[str]
    score: float
    scheme_id: Optional[str]
    scheme_name: Optional[str]
    scheme_url: Optional[str]
    location_type: Optional[str]
    location_name: Optional[str]
    category_id: Optional[int]
    category_name: Optional[str]
    chunk_type: Optional[str]
    chunk_index: Optional[int]
    snippet: str


@dataclass
class SchemeLink:
    scheme_name: str
    scheme_url: str


@dataclass
class QueryError:
    code: str
    message: str


@dataclass
class QueryResponse:
    success: bool
    query: str
    answer: str
    citations: List[Citation] = field(default_factory=list)
    cited_schemes: List[SchemeLink] = field(default_factory=list)
    profile: Dict[str, Any] = field(default_factory=dict)
    debug: Dict[str, Any] = field(default_factory=dict)
    error: Optional[QueryError] = None
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if self.error is None:
            payload.pop("error", None)
        return payload


class RAGService:
    """Session-aware wrapper around `RAGChain` with stable JSON schema."""

    def __init__(self):
        self._default_chain = RAGChain()
        self._chains_by_session: Dict[str, RAGChain] = {}
        self._lock = Lock()

    def _get_chain(self, session_id: Optional[str]) -> RAGChain:
        if not session_id:
            return self._default_chain

        with self._lock:
            if session_id not in self._chains_by_session:
                self._chains_by_session[session_id] = RAGChain()
            return self._chains_by_session[session_id]

    def reset_session(self, session_id: Optional[str] = None) -> None:
        chain = self._get_chain(session_id)
        chain.reset()

    def answer(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        include_debug: bool = True,
    ) -> Dict[str, Any]:
        """Single backend function for answering queries with structured JSON."""
        chain = self._get_chain(session_id)

        if not query or not query.strip():
            return QueryResponse(
                success=False,
                query=query,
                answer="",
                error=QueryError(code="INVALID_QUERY", message="Query must not be empty."),
            ).to_dict()

        try:
            structured = chain.query_structured(user_query=query, k=top_k)
            citations = [Citation(**item) for item in structured.get("citations", [])]

            cited_schemes: List[SchemeLink] = []
            seen = set()
            for item in structured.get("scheme_links", []):
                name = (item.get("scheme_name") or "").strip()
                url = (item.get("scheme_url") or "").strip()
                if not url:
                    continue
                key = (name, url)
                if key in seen:
                    continue
                seen.add(key)
                cited_schemes.append(SchemeLink(scheme_name=name or "Unknown Scheme", scheme_url=url))

            response = QueryResponse(
                success=True,
                query=query,
                answer=structured.get("answer", ""),
                citations=citations,
                cited_schemes=cited_schemes,
                profile=structured.get("profile", {}),
                debug=structured.get("debug", {}) if include_debug else {},
            )
            return response.to_dict()
        except Exception as exc:
            return QueryResponse(
                success=False,
                query=query,
                answer="",
                error=QueryError(code="RAG_PIPELINE_ERROR", message=str(exc)),
                debug={"session_id": session_id} if include_debug else {},
            ).to_dict()


_service = RAGService()


def get_service() -> RAGService:
    """Get singleton RAG service instance."""
    return _service


def answer_query(
    query: str,
    session_id: Optional[str] = None,
    top_k: Optional[int] = None,
    include_debug: bool = True,
) -> Dict[str, Any]:
    """Convenience wrapper for API handlers and quick scripts."""
    return _service.answer(
        query=query,
        session_id=session_id,
        top_k=top_k,
        include_debug=include_debug,
    )


if __name__ == "__main__":
    demo_query = "I am a 19 year old student in Gujarat. Suggest schemes."
    result = answer_query(demo_query, session_id="demo")
    print(result)
