"""Document ingestion pipeline: PDF parsing -> chunking -> embeddings -> Chroma index.

Usage:
    python ingest.py                      # ingest every PDF in ./data
    python ingest.py path/to/bns.pdf      # ingest specific file(s)
    python ingest.py --reset              # wipe the index first
"""
import argparse
import re
import shutil
import sys
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

import config

_SECTION_RE = re.compile(r"^\s*(\d{1,3})\.\s+([A-Z][^\n.]{3,120})\.", re.MULTILINE)
_CHAPTER_RE = re.compile(r"CHAPTER\s+([IVXLC]+)", re.IGNORECASE)


def clean_text(text: str) -> str:
    """Normalise whitespace and remove common PDF artefacts."""
    text = text.replace("\x00", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)        # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(path: Path) -> list[Document]:
    """Parse a PDF into one Document per page (page numbers are 1-based)."""
    reader = PdfReader(str(path))
    docs = []
    for i, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            docs.append(Document(page_content=text, metadata={"source": path.name, "page": i}))
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """Chunk pages, keeping section/chapter hints in the metadata for citations."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=config.SEPARATORS,
        is_separator_regex=True,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)

    current_chapter = None
    for n, chunk in enumerate(chunks):
        ch = _CHAPTER_RE.search(chunk.page_content)
        if ch:
            current_chapter = f"Chapter {ch.group(1).upper()}"
        sec = _SECTION_RE.search(chunk.page_content)
        chunk.metadata["chunk_id"] = n
        if sec:
            chunk.metadata["section"] = f"Section {sec.group(1)}: {sec.group(2).strip()}"
        if current_chapter:
            chunk.metadata["chapter"] = current_chapter
    return chunks


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def build_index(pdf_paths: list[Path], embeddings=None, persist_dir: str | None = None,
                batch_size: int = 128) -> Chroma:
    """Run the full pipeline and return the populated vector store."""
    embeddings = embeddings or get_embeddings()
    persist_dir = persist_dir or config.CHROMA_DIR

    t0 = time.time()
    docs: list[Document] = []
    for p in pdf_paths:
        pages = load_pdf(p)
        print(f"  parsed {p.name}: {len(pages)} pages")
        docs.extend(pages)
    if not docs:
        raise ValueError("No extractable text found. Scanned PDFs need OCR first.")

    chunks = split_documents(docs)
    print(f"  created {len(chunks)} chunks (size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")

    store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        store.add_documents(batch, ids=[f"{b.metadata['source']}-{b.metadata['chunk_id']}" for b in batch])
        print(f"  indexed {min(i + batch_size, len(chunks))}/{len(chunks)}")

    print(f"Done in {time.time() - t0:.1f}s -> {persist_dir}")
    return store


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdfs", nargs="*", help="PDF files (default: every PDF in ./data)")
    ap.add_argument("--reset", action="store_true", help="delete the existing index first")
    args = ap.parse_args()

    paths = [Path(p) for p in args.pdfs] or sorted(config.DATA_DIR.glob("*.pdf"))
    if not paths:
        sys.exit(f"No PDFs found. Put the BNS PDF in {config.DATA_DIR}/ or pass a path.")
    if args.reset and Path(config.CHROMA_DIR).exists():
        shutil.rmtree(config.CHROMA_DIR)
        print("Existing index removed.")

    print("Ingesting...")
    build_index(paths)


if __name__ == "__main__":
    main()
