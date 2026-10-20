# Source provenance

FolioTrace is a source distribution of its document question-answering
application, bundled runtime, local console, tests, and deployment template.

## Source baseline

The application baseline was supplied as a source archive containing 97
nonempty files. Its SHA-256 digest is:

```text
06385fb5dae0c2e594bfd92bb2fc8739721cc57eca1a0aad8df36b795f451d8f
```

That archive contained no Git metadata. This release therefore starts a new,
inspectable source history rather than reconstructing an earlier repository.
The environment and performance records in [performance.md](performance.md)
describe their recorded experimental setup.

## Included source

- `src/rlm_qa/`: document ingestion, evidence handling, model routing, and APIs.
- `src/rlm_runtime/`: the runtime maintained and distributed with FolioTrace.
- `web/static/`: the local document upload and question-answering console.
- `tests/` and `examples/`: tests, synthetic documents, and a fixed offline demo.
- `deploy/`: a deployment template using an unprivileged account and a loopback listener.

The project and distribution name is FolioTrace. The Python import names
`rlm_qa` and `rlm_runtime` and the documented environment variables are retained
for compatibility.

## Design reference and license

Recursive Language Models is the upstream design reference for REPL-based
document analysis and recursive model calls. Its repository, attribution, and
MIT permission notice are documented in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). The upstream package is not
an installation dependency of FolioTrace.

## Data boundary

Examples and the console screenshot use synthetic material. Provider credentials,
research documents, runtime databases, caches, and generated run artifacts are
configured or created locally and excluded from the source distribution.
