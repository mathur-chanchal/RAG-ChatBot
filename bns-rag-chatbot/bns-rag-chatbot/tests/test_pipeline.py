"""Offline tests: no model downloads or API calls (uses a deterministic fake embedder + fake LLM)."""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

import ingest
from rag import BNSAssistant, format_sources

SECTIONS = {
    "103": ("Murder", "Whoever commits murder shall be punished with death or imprisonment for life, and shall also be liable to fine."),
    "303": ("Theft", "Whoever, intending to take dishonestly any movable property out of the possession of any person without that person's consent, moves that property, is said to commit theft."),
    "34": ("Things done in private defence", "Nothing is an offence which is done in the exercise of the right of private defence."),
}


class HashEmbeddings(Embeddings):
    """Bag-of-words hashing embedder - enough for deterministic retrieval tests."""
    DIM = 256

    def _vec(self, text):
        v = [0.0] * self.DIM
        for w in text.lower().split():
            v[int(hashlib.md5(w.strip('.,?"').encode()).hexdigest(), 16) % self.DIM] += 1.0
        n = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture(scope="module")
def pdf(tmp_path_factory):
    path = tmp_path_factory.mktemp("d") / "bns_sample.pdf"
    c = canvas.Canvas(str(path), pagesize=A4)
    for num, (title, body) in SECTIONS.items():
        c.setFont("Helvetica-Bold", 11)
        c.drawString(50, 780, "CHAPTER VI")
        c.drawString(50, 750, f"{num}. {title}.")
        c.setFont("Helvetica", 10)
        y = 730
        for line in [body[i:i + 90] for i in range(0, len(body), 90)]:
            c.drawString(50, y, line)
            y -= 14
        c.showPage()
    c.save()
    return path


@pytest.fixture(scope="module")
def store(pdf, tmp_path_factory):
    return ingest.build_index([pdf], embeddings=HashEmbeddings(), persist_dir=str(tmp_path_factory.mktemp("chroma")))


def test_load_pdf_pages(pdf):
    docs = ingest.load_pdf(pdf)
    assert len(docs) == 3
    assert docs[0].metadata["page"] == 1


def test_chunks_have_section_metadata(pdf):
    chunks = ingest.split_documents(ingest.load_pdf(pdf))
    assert any(c.metadata.get("section", "").startswith("Section 103") for c in chunks)
    assert all("chunk_id" in c.metadata for c in chunks)


def test_retrieval_finds_right_section(store):
    docs = store.max_marginal_relevance_search("punishment for murder", k=2, fetch_k=3)
    assert any("murder" in d.page_content.lower() for d in docs)


def test_end_to_end_with_fake_llm(store):
    bot = BNSAssistant(store=store, llm=FakeListChatModel(responses=["Murder is punished with death or life imprisonment (Section 103)."]), top_k=2)
    res = bot.ask("What is the punishment for murder?")
    assert "Section 103" in res.answer
    assert res.sources and res.latency_s >= 0
    assert any("103" in s for s in format_sources(res.sources))
