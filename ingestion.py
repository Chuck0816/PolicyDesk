"""Load policy text files and the default HR guidance document."""

from pathlib import Path

from llama_index.core import Document, SimpleDirectoryReader

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_documents():
    documents = SimpleDirectoryReader(
        input_dir=str(DATA_DIR), required_exts=[".txt"], raise_on_error=True
    ).load_data()
    # A Document can also be created directly instead of coming from a Reader.
    documents.append(
        Document(
            text="Contact HR when the available company policies do not cover your situation.",
            metadata={"file_name": "manual_faq"},
        )
    )
    return documents
