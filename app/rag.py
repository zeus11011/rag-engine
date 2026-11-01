import os
import re
from collections import Counter
from functools import lru_cache

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

MODEL = os.getenv("MODEL", "ollama:llama3.2")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "ollama:nomic-embed-text")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_data")

PROMPT = (
    "Answer the question using only the context below. "
    "Cite sources by their [filename]. If the context doesn't contain the answer, say you don't know."
)

CONTEXT_PROMPT = """<document>
{doc}
</document>
Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>
Give a short succinct context to situate this chunk within the overall document for the purposes of \
improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

# ponytail: whole doc goes into every context call, truncated to fit; Ollama's default num_ctx is small,
# raise this (and num_ctx) for big-context models, or switch to a sliding window of nearby pages.
MAX_DOC_CHARS = 12_000

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)


@lru_cache
def store():
    return Chroma("docs", embedding_function=init_embeddings(EMBEDDING_MODEL), persist_directory=CHROMA_DIR)


@lru_cache
def llm():
    return init_chat_model(MODEL)


def ingest(text: str, source: str, doc: str | None = None) -> int:
    """If `doc` (the full document) is given, each chunk gets an LLM-written context prepended before embedding."""
    chunks = splitter.split_documents([Document(text, metadata={"source": source})])
    if chunks and doc:
        prompts = [CONTEXT_PROMPT.format(doc=doc[:MAX_DOC_CHARS], chunk=c.page_content) for c in chunks]
        for c, reply in zip(chunks, llm().batch(prompts)):  # batch = concurrent calls
            c.metadata["context"] = reply.content.strip()
            c.page_content = f"{c.metadata['context']}\n\n{c.page_content}"
    if chunks:
        store().add_documents(chunks)
    return len(chunks)


def retrieve(question: str, k: int = 4) -> tuple[list, list[Document]]:
    """Returns (chat messages, retrieved docs)."""
    docs = store().similarity_search(question, k=k)
    context = "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in docs)
    return [("system", PROMPT), ("human", f"Context:\n{context}\n\nQuestion: {question}")], docs


def ask(question: str, k: int = 4) -> tuple[str, list[Document]]:
    messages, docs = retrieve(question, k)
    return llm().invoke(messages).content, docs


def ask_stream(question: str, k: int = 4):
    """Returns (docs, generator of answer text pieces)."""
    messages, docs = retrieve(question, k)
    return docs, (chunk.content for chunk in llm().stream(messages))


def clear():
    store().reset_collection()


def file_of(source: str) -> str:
    """PDF chunks are cited per page ("a.pdf p.3"); strip the page to get the uploaded file name."""
    return re.sub(r"(?i)(\.pdf) p\.\d+$", r"\1", source)


# ponytail: both scan every chunk's metadata; store a "file" field and filter with `where` if the store gets big.
def documents() -> dict[str, int]:
    """Uploaded file name -> chunk count."""
    return dict(Counter(file_of(m["source"]) for m in store().get(include=["metadatas"])["metadatas"]))


def delete(file: str) -> int:
    got = store().get(include=["metadatas"])
    ids = [i for i, m in zip(got["ids"], got["metadatas"]) if file_of(m["source"]) == file]
    if ids:
        store().delete(ids)
    return len(ids)
