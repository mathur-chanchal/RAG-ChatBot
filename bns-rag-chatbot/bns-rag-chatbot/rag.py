"""Retrieval-Augmented Generation chain: Chroma retriever -> prompt -> Hugging Face LLM."""
import os
import time
from dataclasses import dataclass, field

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import config

SYSTEM_PROMPT = """You are an assistant that answers questions about the Bharatiya Nyaya Sanhita (BNS), \
India's criminal code, using ONLY the context excerpts provided.

Rules:
- Base every statement on the context. Do not use outside knowledge.
- If the context does not contain the answer, reply exactly: "I could not find this in the provided BNS documents."
- Cite the section numbers you rely on, e.g. "(Section 103)", when they appear in the context.
- Be concise and use plain language. Do not invent section numbers, punishments or case law.
- You provide general information, not legal advice."""

HUMAN_PROMPT = """Context:
{context}

Question: {question}

Answer:"""

NOT_FOUND = "I could not find this in the provided BNS documents."


@dataclass
class RAGResult:
    answer: str
    sources: list[Document] = field(default_factory=list)
    latency_s: float = 0.0


def format_context(docs: list[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, start=1):
        label = d.metadata.get("section") or f"page {d.metadata.get('page', '?')}"
        parts.append(f"[{i}] ({label}, p.{d.metadata.get('page', '?')})\n{d.page_content}")
    return "\n\n".join(parts)


def format_sources(docs: list[Document]) -> list[str]:
    seen, out = set(), []
    for d in docs:
        key = (d.metadata.get("source"), d.metadata.get("page"), d.metadata.get("section"))
        if key in seen:
            continue
        seen.add(key)
        label = d.metadata.get("section") or "Excerpt"
        out.append(f"{label} — {d.metadata.get('source', 'document')}, p.{d.metadata.get('page', '?')}")
    return out


def get_llm():
    """Chat model served through Hugging Face Inference (needs HUGGINGFACEHUB_API_TOKEN)."""
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

    if not (os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")):
        raise RuntimeError(
            "Set HUGGINGFACEHUB_API_TOKEN (create one at https://huggingface.co/settings/tokens)."
        )
    endpoint = HuggingFaceEndpoint(
        repo_id=config.HF_LLM_REPO,
        task="text-generation",
        max_new_tokens=config.LLM_MAX_NEW_TOKENS,
        temperature=config.LLM_TEMPERATURE,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN"),
    )
    return ChatHuggingFace(llm=endpoint)


class BNSAssistant:
    def __init__(self, store: Chroma | None = None, llm=None, top_k: int | None = None):
        if store is None:
            from ingest import get_embeddings

            store = Chroma(
                collection_name=config.COLLECTION_NAME,
                embedding_function=get_embeddings(),
                persist_directory=config.CHROMA_DIR,
            )
        self.store = store
        self.top_k = top_k or config.TOP_K
        self.llm = llm or get_llm()
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def retrieve(self, question: str) -> list[Document]:
        # MMR reduces near-duplicate chunks from overlapping windows.
        return self.store.max_marginal_relevance_search(
            question, k=self.top_k, fetch_k=self.top_k * 4
        )

    def ask(self, question: str) -> RAGResult:
        t0 = time.time()
        docs = self.retrieve(question)
        if not docs:
            return RAGResult(NOT_FOUND, [], time.time() - t0)
        answer = self.chain.invoke({"context": format_context(docs), "question": question}).strip()
        return RAGResult(answer, docs, time.time() - t0)
