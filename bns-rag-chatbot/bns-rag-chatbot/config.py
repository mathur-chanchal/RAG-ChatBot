"""Central configuration. Override any value with an environment variable."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"                       # put the BNS PDF(s) here
CHROMA_DIR = os.getenv("CHROMA_DIR", str(ROOT / "chroma_db"))
COLLECTION_NAME = "bns_documents"

# --- Chunking -------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))        # characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
# Legal text is organised by numbered sections ("103. Punishment for murder.")
# and chapters, so we try to split on those boundaries before falling back to
# paragraphs, lines, sentences and words.
SEPARATORS = [
    r"\n(?=\d{1,3}\.\s)",       # start of a numbered section
    r"\nCHAPTER\s+[IVXLC]+",    # chapter heading
    "\n\n",
    "\n",
    ". ",
    " ",
    "",
]

# --- Models ---------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_LLM_REPO = os.getenv("HF_LLM_REPO", "Qwen/Qwen2.5-7B-Instruct")
LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", 512))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.1))

# --- Retrieval ------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", 4))
