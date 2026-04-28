# MLXKit

Swift helpers for building **local** LLM apps on Apple platforms using the `mlx-swift` ecosystem.

MLXKit focuses on the annoying parts you hit immediately when building a real app:

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

