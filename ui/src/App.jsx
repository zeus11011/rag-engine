import { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";

async function call(url, opts) {
  const res = await fetch(url, opts);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof body.detail === "string" ? body.detail : `HTTP ${res.status}`);
  return body;
}

// Same rule as file_of() in app/rag.py: PDF sources are cited per page ("a.pdf p.3").
const fileOf = (source) => source.replace(/(\.pdf) p\.\d+$/i, "$1");

function App() {
  const [files, setFiles] = useState([]);
  const [over, setOver] = useState(false);
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState("");
  const [docs, setDocs] = useState([]);

  const refresh = async () => setDocs(await call("/documents"));
  useEffect(() => { refresh().catch((err) => setError(err.message)); }, []);

  const run = (label, fn) => async (e) => {
    e?.preventDefault();
    setBusy(label); setError(""); setStatus("");
    try { await fn(); } catch (err) { setError(err.message); } finally { setBusy(""); }
  };

  const upload = run("Uploading and indexing…", async () => {
    const form = new FormData();
    for (const f of files) form.append("files", f);
    const { chunks } = await call("/documents", { method: "POST", body: form });
    setStatus(`Indexed ${files.length} file(s) into ${chunks} chunks.`);
    setFiles([]);
    await refresh();
  });

  const ask = run("Thinking…", async () => {
    setResult(null);
    const res = await fetch("/query/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(typeof body.detail === "string" ? body.detail : `HTTP ${res.status}`);
    }
    // NDJSON: {"sources"} first, then {"token"} lines, maybe {"error"}.
    const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += value;
      const lines = buf.split("\n");
      buf = lines.pop();
      for (const line of lines.filter(Boolean)) {
        const msg = JSON.parse(line);
        if (msg.error) throw new Error(msg.error);
        if (msg.sources) setResult({ answer: "", sources: msg.sources });
        if (msg.token) setResult((r) => ({ ...r, answer: r.answer + msg.token }));
      }
    }
  });

  const clear = run("Clearing…", async () => {
    if (!confirm("Delete all stored documents?")) return;
    await call("/documents", { method: "DELETE" });
    setResult(null); setStatus("All documents cleared.");
    await refresh();
  });

  const remove = (name) => run(`Removing ${name}…`, async () => {
    if (!confirm(`Remove ${name}?`)) return;
    await call(`/documents/${encodeURIComponent(name)}`, { method: "DELETE" });
    setStatus(`Removed ${name}.`);
    await refresh();
  })();

  const onDrop = (e) => {
    e.preventDefault(); setOver(false);
    setFiles([...e.dataTransfer.files]);
  };

  return (
    <div className="wrap">
      <header>
        <h1>RAG Engine</h1>
        <p>Upload your documents, then ask questions about them.</p>
      </header>

      <section className="card">
        <h2>📄 Documents</h2>
        <label
          className={`drop ${over ? "over" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setOver(true); }}
          onDragLeave={() => setOver(false)}
          onDrop={onDrop}
        >
          <input type="file" multiple accept=".txt,.md,.pdf" onChange={(e) => setFiles([...e.target.files])} />
          <div><strong>Click to choose</strong> or drag files here</div>
          <div className="hint">.txt, .md, .pdf</div>
        </label>
        {files.length > 0 && (
          <div className="chips">{files.map((f) => <span className="chip" key={f.name}>{f.name}</span>)}</div>
        )}
        <div className="row">
          <button onClick={upload} disabled={!!busy || !files.length}>Upload</button>
          <button className="ghost" onClick={clear} disabled={!!busy || !docs.length}>Clear all</button>
        </div>
        {docs.length > 0 && (
          <ul className="docs">
            {docs.map((d) => (
              <li key={d.name}>
                <span className="name">{d.name}</span>
                <span className="hint">{d.chunks} chunks</span>
                <button className="icon" onClick={() => remove(d.name)} disabled={!!busy}
                  aria-label={`Remove ${d.name}`} title="Remove">✕</button>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="card">
        <h2>💬 Ask a question</h2>
        <form onSubmit={ask}>
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) ask(e); }}
            placeholder="What is in my notes?"
          />
          <div className="row" style={{ justifyContent: "space-between", alignItems: "center" }}>
            <span className="hint">Enter to send · Shift+Enter for new line</span>
            <button disabled={!!busy || !question.trim()}>Ask</button>
          </div>
        </form>
      </section>

      {busy && <div className="msg busy" role="status"><span className="spin" />{busy}</div>}
      {status && <div className="msg ok" role="status">{status}</div>}
      {error && <div className="msg err" role="alert">{error}</div>}

      {result && (
        <section className="card">
          <h2>✨ Answer</h2>
          <div className="answer" aria-live="polite">{result.answer}{busy && <span className="cursor">▍</span>}</div>
          {result.sources.length > 0 && (
            <div className="sources">
              <h3>Sources</h3>
              {result.sources.map((s, i) => {
                const file = fileOf(s.source);
                const gone = !docs.some((d) => d.name === file);
                return (
                  <details key={i} className={gone ? "gone" : ""}>
                    <summary>
                      <span className="name">{s.source}{gone && " (removed)"}</span>
                      {!gone && (
                        <button className="icon" disabled={!!busy} aria-label={`Remove ${file}`} title={`Remove ${file}`}
                          onClick={(e) => { e.preventDefault(); remove(file); }}>✕</button>
                      )}
                    </summary>
                    <pre>{s.text}</pre>
                  </details>
                );
              })}
            </div>
          )}
        </section>
      )}
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
