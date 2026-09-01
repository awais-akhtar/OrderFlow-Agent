# Local Models and RAG

I make model execution explicit. The deterministic agent, evaluation runner, catalog, order history, and operator screens can run without a provider. Customer-facing assistant messages require a provider with a verified streaming contract;

## KernelLoom

`requirements.txt` installs `kernelloom[genai]==0.4.1` from PyPI. In Python mode, OrderFlow-Agent imports `KernelLoomModel` and `ModelConfig`, loads a configured model lazily, and consumes `KernelLoomModel.stream(...)` fragments. The adapter passes backend, device, CPU profile, reserved-core, and workspace data settings to the package.

```dotenv
GENAI_PROVIDER=kernelloom
KERNELLOOM_TRANSPORT=python
KERNELLOOM_CHAT_MODEL_PATH=C:\models\compatible-openvino-model
KERNELLOOM_BACKEND=openvino
KERNELLOOM_DEVICE=CPU
KERNELLOOM_DATA_DIR=data/kernelloom
```

this path locally with KernelLoom 0.4.1 and Qwen2.5 1.5B Instruct INT4 OpenVINO weights. The repository does not contain those weights, download a model, choose quantization, or claim every KernelLoom backend is available on every machine.

KernelLoom HTTP mode expects a separately running service and implements:

- `GET /health` and `GET /v1/models`;
- streaming and buffered `POST /v1/chat/completions`;
- `POST /v1/embeddings`.

Chat and embedding model identifiers are configured independently.

## OpenAI

The OpenAI customer path uses the Responses API with `store=False` and consumes `response.output_text.delta` events. The adapter also implements embeddings, completed-turn audio transcription, and speech synthesis. It requires an explicit `OPENAI_API_KEY` and response model id.

```dotenv
GENAI_PROVIDER=openai
OPENAI_API_KEY=your-key
GENAI_RESPONSE_MODEL=your-response-model-id
```

The staff Settings view keeps a typed key only in process memory. It is excluded from databases, traces, exports, and evaluation artifacts. Environment-based secret management is the appropriate boundary for shared deployment.

## Hugging Face

The hosted adapter requires an absolute endpoint base URL ending in `/v1`, a token, and a model id. It consumes server-sent chat-completion deltas from `/chat/completions`.

## OpenAgent

The OpenAgent adapter implements `/api/health`, `/api/chat`, and explicit LatticeRAG status, sync, and query routes. The service contract used here does not expose a verified token stream, so the adapter refuses customer-chat streaming instead of simulating it. OpenAgent remains useful as a linked local chat/RAG service when the caller uses its buffered boundary.

## Built-In Dual Retrieval

The staff Knowledge view extracts UTF-8 text, Markdown, CSV, JSON, and text-based PDF, then stores deterministic chunks and lineage in SQLite. One retrieval lane runs BM25. The second uses local latent semantic analysis over TF-IDF, a term-vector fallback for very small corpora, or embeddings from a compatible active provider.

Both lanes run concurrently. Reciprocal-rank fusion combines their ordered candidates. The trace records lane names, counts, selected chunk ids, and measured timing. The fusion value is a ranking score, not a calibrated probability.

When a generation provider is available, the answer prompt labels sources and permits only supplied evidence. Without one, the UI returns source-labelled extracts. In both cases, selected passages remain visible for inspection.
