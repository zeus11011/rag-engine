"""Run: python test_rag.py — swaps in fake embeddings/LLM, no Ollama or keys needed."""
import tempfile

from langchain_chroma import Chroma
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel

from app import rag

fake_store = Chroma("test", embedding_function=DeterministicFakeEmbedding(size=64), persist_directory=tempfile.mkdtemp())
rag.store = lambda: fake_store
rag.llm = lambda: FakeListChatModel(responses=["fake answer"])

assert rag.ingest("", "empty.txt") == 0
n = rag.ingest("RAG engines retrieve context. " * 100, "notes.md")
assert n > 1, n  # ~3000 chars must split into several chunks

answer, docs = rag.ask("RAG engines retrieve context.", k=2)
assert answer == "fake answer"
assert len(docs) == 2 and docs[0].metadata["source"] == "notes.md"

docs, tokens = rag.ask_stream("RAG engines retrieve context.", k=2)
assert len(docs) == 2 and "".join(tokens) == "fake answer"

rag.ingest("Page one. " * 100, "a.pdf p.1")
rag.ingest("Page two. " * 100, "a.pdf p.2")
assert set(rag.documents()) == {"notes.md", "a.pdf"}  # PDF pages grouped under one file
assert rag.delete("a.pdf") > 0 and set(rag.documents()) == {"notes.md"}
assert rag.delete("missing.txt") == 0

rag.clear()
assert rag.ask("anything")[1] == []

# --- context enrichment: every chunk gets the LLM context prepended ---
rag.llm = lambda: FakeListChatModel(responses=["  From the setup chapter.  "])
n = rag.ingest("Install the engine first. " * 100, "doc.pdf p.1", doc="whole document")
stored = fake_store.get()
assert len(stored["documents"]) == n > 1
assert all(d.startswith("From the setup chapter.\n\nInstall") for d in stored["documents"]), stored["documents"][0]
assert all(m["context"] == "From the setup chapter." for m in stored["metadatas"])
rag.clear()

# --- PDF extraction ---
import io

from fastapi import HTTPException, UploadFile

from app.main import read_pdf


def make_pdf(pages: list[str]) -> bytes:
    """Smallest valid PDF with one line of text per page."""
    n = len(pages)
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", f"<< /Type /Pages /Kids [{kids}] /Count {n} >>"]
    font = 3 + 2 * n
    for i, text in enumerate(pages):
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 100] /Contents {4 + 2 * i} 0 R "
                    f"/Resources << /Font << /F1 {font} 0 R >> >> >>")
        stream = f"BT /F1 12 Tf 10 50 Td ({text}) Tj ET"
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    objs.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out, offsets = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    return out + f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()


pages = read_pdf(UploadFile(io.BytesIO(make_pdf(["Hello page one", "Second page"])), filename="doc.pdf"))
assert [s for _, s, _ in pages] == ["doc.pdf p.1", "doc.pdf p.2"], pages
assert "Hello page one" in pages[0][0] and "Second page" in pages[1][0], pages
assert all("Hello page one" in d and "Second page" in d for _, _, d in pages), pages  # full doc on every page

try:
    read_pdf(UploadFile(io.BytesIO(b"not a pdf"), filename="bad.pdf"))
    raise AssertionError("garbage PDF accepted")
except HTTPException as e:
    assert e.status_code == 400
print("ok")
