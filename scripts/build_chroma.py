"""Build the Korean Basic Dictionary grammar vector index."""

import json
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


INPUT_FILE = Path("data/processed/grammar_clean.json")
CHROMA_DIR = Path("data/chroma")
COLLECTION_NAME = "korean_grammar"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


def make_document(entry: dict) -> Document:
    related = ", ".join(
        item["grammar"]
        for item in entry.get("related", [])
        if item.get("grammar")
    )
    retrieval_text = (
        "passage:\n"
        f"Korean grammar: {entry.get('grammar') or ''}\n"
        f"English meaning: {entry.get('definition_en') or ''}\n"
        f"Korean meaning: {entry.get('definition_ko') or ''}\n"
        f"Related grammar: {related}"
    )
    return Document(
        page_content=retrieval_text,
        metadata={
            "grammar": entry.get("grammar") or "",
            "entry_id": str(entry.get("id", "")),
            "sense_id": str(entry.get("sense_id", "")),
            "sense_number": int(entry.get("sense_number", 1)),
            "source": entry.get("source", "Korean Basic Dictionary"),
        },
    )


def make_id(entry: dict, index: int) -> str:
    return (
        f"{entry.get('id', index)}_"
        f"{entry.get('sense_id', '')}_"
        f"{entry.get('sense_number', 1)}"
    )


def main() -> None:
    with INPUT_FILE.open(encoding="utf-8") as file:
        entries = json.load(file)

    documents = [make_document(entry) for entry in entries]
    ids = [make_id(entry, index) for index, entry in enumerate(entries)]
    print(f"Prepared {len(documents)} documents.")

    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    vectorstore.add_documents(documents=documents, ids=ids)
    print(f"Stored {len(documents)} documents in {CHROMA_DIR}.")


if __name__ == "__main__":
    main()
