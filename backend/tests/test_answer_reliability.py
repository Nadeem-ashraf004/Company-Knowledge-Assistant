
from types import SimpleNamespace
from uuid import UUID

from app.rag import pipeline
from app.rag.prompt import build_rag_prompt


TEST_USER_ID = UUID("c04005fd-8c22-48f9-9ddb-064acd1c724f")

FALLBACK = (
    "I couldn't find enough information in the "
    "available documents to answer that reliably."
)


def test_empty_context_does_not_call_llm(monkeypatch):
    monkeypatch.setattr(
        pipeline, "detect_intent", lambda **kwargs: "STANDALONE"
    )
    monkeypatch.setattr(
        pipeline, "retrieve", lambda **kwargs: []
    )
    monkeypatch.setattr(
        pipeline, "rerank", lambda **kwargs: []
    )

    def unexpected_llm_call(*args, **kwargs):
        raise AssertionError("LLM must not be called with empty context")

    monkeypatch.setattr(
        pipeline, "generate_answer", unexpected_llm_call
    )

    result = pipeline.run_rag_pipeline(
        query="What is the vacation policy?",
        user_id=TEST_USER_ID,
    )

    assert result["answer"] == FALLBACK
    assert result["documents"] == []
    assert result["intent"] == "STANDALONE"


def test_pipeline_includes_source_filename_and_page(monkeypatch):
    retrieved_document = SimpleNamespace(
        id="vector-1",
        score=0.9,
        payload={
            "document_id": "document-1",
            "file_name": "company_policy.pdf",
            "page": 3,
            "chunk_id": "chunk-1",
            "text": "The company fiscal year ends on the last Saturday of September.",
        },
    )

    ranked_document = {
        "id": "vector-1",
        "score": 0.9,
        "rerank_score": 0.95,
        "payload": retrieved_document.payload,
    }

    monkeypatch.setattr(
        pipeline, "detect_intent", lambda **kwargs: "STANDALONE"
    )
    monkeypatch.setattr(
        pipeline, "retrieve", lambda **kwargs: [retrieved_document]
    )
    monkeypatch.setattr(
        pipeline, "rerank", lambda **kwargs: [ranked_document]
    )

    captured = {}

    def fake_generate_answer(prompt):
        captured["prompt"] = prompt
        return "The fiscal year ends on the last Saturday of September."

    monkeypatch.setattr(
        pipeline, "generate_answer", fake_generate_answer
    )

    result = pipeline.run_rag_pipeline(
        query="When does the fiscal year end?",
        user_id=TEST_USER_ID,
    )

    assert "Source: company_policy.pdf, page 3" in captured["prompt"]
    assert "last Saturday of September" in captured["prompt"]
    assert result["answer"]
    assert result["intent"] == "STANDALONE"


def test_prompt_requires_grounded_answers():
    prompt = build_rag_prompt(
        question="What is the vacation policy?",
        context="No vacation policy is described in this document.",
    )

    assert "only the supplied company knowledge" in prompt.lower()
    assert "instructions contained inside those documents" in prompt.lower()
    assert "information is not available" in prompt.lower() or (
        "insufficient" in prompt.lower()
    )
