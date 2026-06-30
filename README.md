# MLXKit

Swift helpers for building **local** LLM apps on Apple platforms using the `mlx-swift` ecosystem.

MLXKit now has two ways to use the same local MLX model runtime:

- **Embed `MLXKit`** in a Swift app and receive streamed tokens/tool calls directly.
- Run **`MLXKitServer`** as an executable for terminal chat or a small OpenAI-compatible local HTTP service.

It focuses on the annoying parts you hit immediately when building a real app:

- **Download** MLX Community models from Hugging Face
- **Store + list** downloaded models (in your app’s `Documents/models/…`)
- **Load** an `MLXLLM` model container
- **Chat + stream tokens** and optionally handle **tool calls**

This package is used by my example agent app **ComfyPilot** (browser-controlled agent + in-app model download + tool calling).

## Requirements

- Swift tools: `swift-tools-version: 6.2`
- Platforms (as currently set in `Package.swift`):
  - macOS 26+
  - iOS 26+

## Install (Swift Package Manager)

Add `MLXKit` as a dependency in your app’s `Package.swift`:

```swift
.package(url: "<your-repo-url-for-MLXKit>", branch: "main")
```

Then add the product to your target:

```swift
.product(name: "MLXKit", package: "MLXKit")
```

MLXKit itself depends on:

- `mlx-swift`
- `mlx-swift-lm`

## Quick Start

### Embedded in a Swift app

`MLXChatService` is `@MainActor`, which makes it natural to own from a SwiftUI
view model. It can load a local MLX model folder or download/cache a Hugging
Face MLX repository directly.

```swift
import MLXKit

@MainActor
func answer() async throws {
    let chat = MLXChatService()
    try await chat.loadModel(
        from: .hub(id: "mlx-community/Llama-3.2-3B-Instruct-4bit"),
        defaultPrompt: "You are a concise assistant."
    )

    let response = try await chat.respond(
        messages: [.init(role: .user, content: "Explain async/await in Swift.")],
        options: .init(maxTokens: 512, temperature: 0.4),
        onToken: { chunk in
            // Append `chunk` to your UI as it arrives.
            print(chunk, terminator: "")
        },
        onToolCall: { call in
            print("Requested tool:", call.functionName, call.arguments)
        }
    )
    print("\nFinal answer:", response)
}
```

Use a downloaded folder instead when the model is managed by your app:

```swift
try await chat.loadModel(from: .directory(localModelURL))
```

`getResponse(...)` remains available as the compatibility API used by existing
MLXKit apps. Its `tokens` and `temperature` settings now feed the actual MLX
generation parameters.

### Command-line executable

Build or run the executable with SwiftPM:

```bash
swift run MLXKitServer \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --prompt "Write a haiku about Metal."
```

Pass a filesystem path to `--model` to use a model that is already downloaded:

```bash
swift run MLXKitServer \
  --model ~/Models/Llama-3.2-3B-Instruct-4bit \
  --interactive
```

The executable streams terminal output by default. `--no-stream`,
`--temperature`, `--top-p`, `--top-k`, `--max-tokens`, and `--system` map to
the same generation settings as `MLXGenerationOptions`.

### C / C++ dynamic library

Build the C ABI product with:

```bash
swift build --product MLXKitC
```

This produces `libMLXKitC.dylib` in `.build/arm64-apple-macosx/debug/`. Its
header is [MLXKitC.h](Sources/MLXKitC/include/MLXKitC.h). The ABI is
asynchronous: calls return after being accepted, then a completion callback
reports success or failure. Generation sends UTF-8 token fragments through a
separate callback.

```c
#include "MLXKitC.h"

void finished(void *context, int32_t status, const char *error) {
    // status == 0 on success; copy `error` before returning if non-NULL.
}

void token(void *context, const char *fragment) {
    // Copy or consume this streamed UTF-8 fragment before returning.
}

void *runtime = mlxkit_runtime_create();
mlxkit_runtime_load_model(runtime, "/path/to/MLX-model", finished, NULL);
// After `finished` reports success:
mlxkit_runtime_chat(
    runtime,
    "[{\"role\":\"user\",\"content\":\"Hello\"}]",
    0.6f, 512, token, finished, NULL
);
// Destroy only after outstanding callbacks have completed.
mlxkit_runtime_destroy(runtime);
```

The dynamic library also requires the generated
`MLXKit_MLXKitMetalResources.bundle` alongside the host executable/app's
resources when you distribute it; that bundle contains MLX's Metal shaders.
The C ABI currently supports text chat, local model folders, Hugging Face
model IDs, and token streaming.

### Local HTTP service

For local integration testing, start a server with a model loaded once:

```bash
swift run MLXKitServer \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --serve --port 8080
```

It exposes `GET /v1/models` and an OpenAI-compatible
`POST /v1/chat/completions` endpoint. Set `stream` to `true` for Server-Sent
Events (SSE) `data:` chunks followed by `data: [DONE]`—the format expected by
clients such as MLXStudio:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "local-mlx",
    "messages": [{"role": "user", "content": "Hello from curl"}],
    "max_tokens": 128,
    "temperature": 0.4
  }'
```

For a streaming request:

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "local-mlx",
    "messages": [{"role": "user", "content": "Hello from curl"}],
    "stream": true
  }'
```

Function-tool schemas in the OpenAI `tools` field are passed to the model.
When it requests a tool, the stream emits `tool_calls` chunks and finishes with
`"finish_reason":"tool_calls"`; send the assistant tool-call message and the
corresponding `tool` result back in the next chat request.

### Metal runtime resource

`MLXKitServer` includes MLX's `default.metallib` shader library as a SwiftPM
resource. This is required for `swift run` to initialize Metal; without it,
MLX reports `Failed to load the default metallib` before a model is loaded.

If you update the pinned `mlx-swift` package, regenerate the committed resource
with Xcode's Metal Toolchain installed:

```bash
./Scripts/generate-mlx-metallib.sh
```

### 1) Download + Select A Model (UI-Friendly)

Use `ModelLoaderService` to:

- list downloaded models
- download a new model by name (from `mlx-community`)
- prompt the user before downloading the full file set

```swift
import MLXKit

@MainActor
let loader = ModelLoaderService(selectFirst: true)

// Refresh local models from Documents/models
loader.sync()

// Download a model from mlx-community/<name>
Task {
  await loader.download(named: "Llama-3.2-3B-Instruct")
}
```

On macOS you can also open the models folder:

```swift
#if os(macOS)
loader.openModelFolder()
#endif
```

Where models are stored:

`Documents/models/<model-name>/...`

In Your App Container, For Example:
`~/Library/Containers/com.whatever.app/Data/Documents/models`

### 2) Load The Model

```swift
import MLXKit

@MainActor
let chat = MLXChatService()

if let model = loader.selected {
  await chat.loadModel(
    at: model.url,
    defaultPrompt: "You are a helpful assistant."
  )
}
```

### 3) Stream A Response (+ Tool Calls)

MLXKit streams assistant tokens via `completion` and surfaces tool calls via `toolcallCompletionHandler`.

```swift
import MLXKit
import MLXLMCommon

let tools: [[String: any Sendable]] = [
  [
    "type": "function",
    "function": [
      "name": "search",
      "description": "Search the web for information",
      "parameters": [
        "type": "object",
        "properties": [
          "query": ["type": "string"]
        ]
      ]
    ]
  ]
]

let messages: [ModelMessage] = [
  .init(role: .user, content: "Search for the weather in Chicago.")
]

let _ = try await chat.getResponse(
  messages: messages,
  tools: tools,
  completion: { tokenChunk in
    // Append streamed tokens to your UI
    print(tokenChunk, terminator: "")
  },
  toolcallCompletionHandler: { toolCall in
    // Route tool calls to your app (browser, network, filesystem, etc.)
    print("Tool:", toolCall.functionName, "args:", toolCall.arguments)
  }
)
```

## What’s Included

- `MLXChatService`
  - loads an MLX LLM container
  - streams responses
  - emits tool calls
- `ModelLoaderService`
  - lists installed models
  - downloads models from `mlx-community` on Hugging Face
  - supports a “confirm before download” UI flow
- `MLXChatModel`
  - lightweight model folder reference (`Documents/models/<name>`)

## Notes / Caveats

- Model storage is currently based on `URL.documentsDirectory` (iOS/macOS sandbox documents).
- The `Package.swift` platform minimums are currently set to v26. If you need older OS support, you’ll want to lower those and verify `Observation/@Observable` usage. I do it cuz I have had no need to support older Versions.
- Please Open a issue if you need an older OS Version.
