import json

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app import rag

app = FastAPI(title="RAG Engine")


class Query(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(4, ge=1, le=20)


@app.get("/health")
def health():
    return {"status": "ok"}


# Sync handlers: FastAPI runs them in a threadpool, so blocking embed/LLM calls don't stall the server.
@app.post("/documents")
def upload(files: list[UploadFile]):
    texts = []
    for f in files:  # validate everything before storing anything
        name = f.filename.lower()
        if name.endswith(".pdf"):
            texts += read_pdf(f)
        elif name.endswith((".txt", ".md")):
            try:
                texts.append((f.file.read().decode("utf-8"), f.filename, None))
            except UnicodeDecodeError:
                raise HTTPException(400, f"{f.filename}: not valid UTF-8")
        else:
            raise HTTPException(400, f"{f.filename}: only .txt, .md and .pdf are supported")
    return {"chunks": sum(rag.ingest(text, name, doc) for text, name, doc in texts)}


def read_pdf(f: UploadFile) -> list[tuple[str, str, str]]:
    """One (text, source, full_doc) per page: per-page sources cite page numbers, full_doc drives chunk enrichment."""
    try:
        reader = PdfReader(f.file)
        if reader.is_encrypted:
            raise HTTPException(400, f"{f.filename}: encrypted PDFs are not supported")
        # ponytail: no OCR, scanned PDFs yield no text; add pytesseract if you need them
        pages = [(page.extract_text() or "", f"{f.filename} p.{i}") for i, page in enumerate(reader.pages, 1)]
        doc = "\n\n".join(text for text, _ in pages)
        return [(text, source, doc) for text, source in pages]
    except PdfReadError:
        raise HTTPException(400, f"{f.filename}: not a valid PDF")


def sources(docs):
    return [{"source": d.metadata["source"], "text": d.page_content} for d in docs]


@app.post("/query")
def query(q: Query):
    answer, docs = rag.ask(q.question, q.k)
    return {"answer": answer, "sources": sources(docs)}


@app.post("/query/stream")
def query_stream(q: Query):
    """NDJSON: first line {"sources": [...]}, then {"token": "..."} lines, or {"error": "..."} if the model fails mid-stream."""
    docs, tokens = rag.ask_stream(q.question, q.k)

    def lines():
        yield json.dumps({"sources": sources(docs)}) + "\n"
        try:
            for t in tokens:
                if t:
                    yield json.dumps({"token": t}) + "\n"
        except Exception as e:  # headers are already sent, so report in-band
            yield json.dumps({"error": str(e)}) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson")


@app.get("/documents")
def list_documents():
    return [{"name": name, "chunks": n} for name, n in sorted(rag.documents().items())]


@app.delete("/documents/{name:path}")
def delete_document(name: str):
    if not rag.delete(name):
        raise HTTPException(404, f"{name}: no such document")
    return {"status": "deleted"}


@app.delete("/documents")
def clear():
    rag.clear()
    return {"status": "cleared"}
