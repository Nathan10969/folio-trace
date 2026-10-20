# API and configuration

## Local entry point

Start `rlm_qa.http_app:app` with Uvicorn on port 8787. Its browser console and
upload/chat endpoints belong to the same application.

| Endpoint | Purpose |
| --- | --- |
| `GET /` | Local document console |
| `GET /healthz` | Process health |
| `POST /upload` | Multipart upload with a `file` field |
| `GET /documents/{source_id}` | Latest document processing status |
| `POST /chat` | Complete answer JSON with evidence and debug fields |

```bash
curl -F 'file=@examples/sample-paper.md' http://127.0.0.1:8787/upload
```

Use the returned `source_id` in your chat request:

```json
{
  "source_ids": ["SOURCE_ID_FROM_UPLOAD"],
  "session_id": "local-review",
  "question": "Summarize this document and compare retained adhesion."
}
```

`source_id` is also accepted for a single document. The local endpoint returns the
whole JSON answer; its `stream` request field does not enable SSE. Session IDs are
stored with run artifacts, not automatically reused as platform conversation memory.
PDF uploads queue background GLM OCR jobs; TXT/Markdown ingestion does not call OCR.
TXT/Markdown page numbers are virtual pages, not printed PDF page numbers.

## Platform document contract

The platform entry point reads already processed document directories. It does not
upload PDFs or run OCR workers. Set `RLM_PLATFORM_DATA_ROOT` to a local data directory:

```text
data/platform/
  paper-a/
    status.json
    metadata.json
    elements.jsonl
  paper-b/
    status.json
    metadata.json
    text_blocks.jsonl
```

Minimum `status.json`:

```json
{"status": "ok", "pages": 2}
```

Example `metadata.json`:

```json
{"fileId": 101, "fileName": "paper-a.pdf"}
```

Each `elements.jsonl` or fallback `text_blocks.jsonl` line is a page-indexed object:

```json
{"page": 1, "index": 0, "label": "text", "content": "A supporting passage."}
```

Page numbers are 1-based and bounded by `status.json.pages`. Elements are sorted by
page and index. Readable text, tables, and formulas enter the context; bare image
URLs do not establish textual evidence. Input directory names must be safe, direct
children of the configured root; path traversal and symlink escapes are rejected.
Content fingerprints invalidate normalized document caches when artifacts change.

## Platform endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /healthz` | Process health |
| `GET /readyz` | Configured providers, data root, and runtime readiness |
| `POST /chat` | Enhanced answer contract, or SSE when `stream=true` |
| `POST /v1/chat/completions` | OpenAI-style answer envelope, or SSE |

Readiness checks do not make a live model call. A healthy/ready response does not
verify document answers or provider compatibility.

```bash
curl -N http://127.0.0.1:8788/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"precise","messages":[{"role":"user","content":"Summarize the evidence."}],"metadata":{"chat_id":"review-01","dir_names":["paper-a"],"doc_ids":["101"]},"stream":true}'
```

The API takes the last nonempty user message as the current question. Use a stable
`metadata.chat_id` to continue a conversation; different IDs isolate stored history.
`doc_ids`, when supplied, must align with `dir_names`; they are business identifiers,
not filesystem paths. The client `model` field does not select an arbitrary provider
model. The server settings control provider/model routing.

SSE carries answer deltas and reference annotations. Preserve annotations to resolve
inline markers to files and pages. Completed exchanges enter conversation memory;
cancelled partial exchanges do not.

## Model roles

| Role | Configuration |
| --- | --- |
| Generation | `GLM_GENERATION_API_KEYS`, `GLM_GENERATION_MODEL` |
| Generation fallback | `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL` |
| Map/subagent | `DEEPSEEK_SUBAGENT_API_KEYS`, `DEEPSEEK_SUBAGENT_MODEL` |
| PDF OCR | `GLM_OCR_API_KEYS` or `GLM_OCR_API_KEY`, `GLM_OCR_MODEL` |

Key pools accept comma-separated keys. The source also accepts legacy finalizer
settings; current Map-Reduce finalization uses the generation route. See
`src/rlm_qa/llm.py` for the authoritative routing implementation.

Both environment variables and a working-directory `.env` are supported; nonempty
environment variables take precedence. Set models available to your account rather
than treating source defaults as a provider availability guarantee. Keys and runtime
directories must remain local. Live inference sends document content to providers.

## Execution and storage

| Setting | Default | Purpose |
| --- | --- | --- |
| `RLM_MAP_EXECUTION_MODE` | `staged` | `staged`, `pipelined`, or `one_pass` |
| `RLM_REDUCE_CONTEXT_MODE` | `expanded` | `expanded` or `compact` |
| `RLM_BATCH_MAX_WORKERS` | `4` | Map/subagent concurrency |
| `RLM_MAX_DEEP_PACKS` | `8` | Deep-pack selection limit |
| `RLMQA_RUNTIME_DIR` | `runtime/rlm_qa` | Uploads, documents, and run artifacts |
| `RLM_PLATFORM_CACHE_DIR` | `runtime/platform_docs` | Normalized OCR artifact cache |
| `RLM_CONVERSATION_MEMORY_DB` | unset | Override platform SQLite path |
| `RLM_CONVERSATION_MEMORY_ENABLED` | `true` | Enable platform chat history |

One-pass may need repair calls. `compact` reduces finalizer context, but latency and
evidence coverage depend on document selection, provider behavior, and configuration.
The historical performance note is not a general speed guarantee.

The local REPL is in-process rather than an isolated sandbox. The service has no
built-in HTTP authentication; bind to loopback for local use. The example systemd
unit is a deployment template, not a verified production environment.
