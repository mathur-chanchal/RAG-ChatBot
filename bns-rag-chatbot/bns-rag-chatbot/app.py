"""Gradio chat UI for the BNS AI Assistant (runs locally and on Hugging Face Spaces)."""
import gradio as gr

import config
from rag import BNSAssistant, format_sources

_assistant: BNSAssistant | None = None

DISCLAIMER = (
    "General information based on the Bharatiya Nyaya Sanhita text only — "
    "not legal advice. Consult a qualified lawyer for your situation."
)
EXAMPLES = [
    "What is the punishment for murder under the BNS?",
    "How does the BNS define theft?",
    "What is the difference between culpable homicide and murder?",
    "What does the BNS say about the right of private defence?",
]


def ensure_index():
    """On first run (e.g. a fresh Space) build the index from PDFs in ./data."""
    from pathlib import Path

    if Path(config.CHROMA_DIR).exists() and any(Path(config.CHROMA_DIR).iterdir()):
        return
    pdfs = sorted(config.DATA_DIR.glob("*.pdf"))
    if not pdfs:
        raise RuntimeError(f"No index and no PDFs found. Add the BNS PDF to {config.DATA_DIR}/ first.")
    from ingest import build_index

    print("No index found — building one from ./data ...")
    build_index(pdfs)


def get_assistant() -> BNSAssistant:
    global _assistant
    if _assistant is None:
        ensure_index()
        _assistant = BNSAssistant()
    return _assistant


def respond(message, history):
    try:
        result = get_assistant().ask(message)
    except Exception as e:  # surface config problems (missing token, no index) in the UI
        return f"⚠️ {e}"
    text = result.answer
    sources = format_sources(result.sources)
    if sources:
        text += "\n\n**Sources**\n" + "\n".join(f"- {s}" for s in sources)
    text += f"\n\n<sub>{result.latency_s:.1f}s</sub>"
    return text


demo = gr.ChatInterface(
    fn=respond,
    title="⚖️ BNS AI Assistant",
    description="Ask questions about the Bharatiya Nyaya Sanhita. Answers are grounded in the "
    f"indexed documents and cite their sources.\n\n_{DISCLAIMER}_",
    examples=EXAMPLES,
)

if __name__ == "__main__":
    demo.launch()
