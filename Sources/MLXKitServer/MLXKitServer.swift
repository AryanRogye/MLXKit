import Foundation
import MLXKit
import MLXKitMetalResources
@preconcurrency import NIOCore
@preconcurrency import NIOHTTP1
@preconcurrency import NIOPosix

@main
struct MLXKitServer {
    static func main() async {
        do {
            MLXKitMetalResources.registerBundle()

            let configuration = try CommandLineConfiguration(arguments: Array(CommandLine.arguments.dropFirst()))
            if configuration.showHelp {
                print(CommandLineConfiguration.usage)
                return
            }

            guard let model = configuration.model else {
                throw CommandLineError.missingModel
            }

            let service = await MainActor.run { MLXChatService() }
            let source = modelSource(for: model)
            print("Loading \(model)…")
            try await service.loadModel(from: source, defaultPrompt: configuration.systemPrompt ?? "hello")

            if configuration.serve {
                try await HTTPServer(
                    service: service,
                    modelName: model,
                    defaultOptions: configuration.options
                ).run(host: configuration.host, port: configuration.port)
            } else if configuration.interactive {
                try await runInteractiveChat(service: service, configuration: configuration)
            } else if let prompt = configuration.prompt {
                try await runOneShot(prompt: prompt, service: service, configuration: configuration)
            } else {
                throw CommandLineError.missingPrompt
            }
        } catch {
            fputs("MLXKitServer: \(error.localizedDescription)\n", stderr)
            fputs("\(CommandLineConfiguration.usage)\n", stderr)
            return
        }
    }

    private static func modelSource(for value: String) -> MLXModelSource {
        let url = URL(fileURLWithPath: value)
        if FileManager.default.fileExists(atPath: url.path) {
            return .directory(url)
        }
        return .hub(id: value)
    }

    @MainActor
    private static func runOneShot(
        prompt: String,
        service: MLXChatService,
        configuration: CommandLineConfiguration
    ) async throws {
        var messages = [ModelMessage]()
        if let systemPrompt = configuration.systemPrompt {
            messages.append(.init(role: .system, content: systemPrompt))
        }
        messages.append(.init(role: .user, content: prompt))

        let writeChunk: @Sendable (String) -> Void = { chunk in
            FileHandle.standardOutput.write(Data(chunk.utf8))
        }
        let onToken: @Sendable (String) -> Void
        if configuration.stream {
            onToken = writeChunk
        } else {
            onToken = { _ in }
        }
        let response: String = try await service.respond(
            messages: messages,
            options: configuration.options,
            onToken: onToken
        )
        if !configuration.stream {
            print(response)
        } else {
            print()
        }
    }

    @MainActor
    private static func runInteractiveChat(
        service: MLXChatService,
        configuration: CommandLineConfiguration
    ) async throws {
        var messages = [ModelMessage]()
        if let systemPrompt = configuration.systemPrompt {
            messages.append(.init(role: .system, content: systemPrompt))
        }

        print("MLXKit interactive chat — type /exit to quit.")
        while true {
            print("you> ", terminator: "")
            guard let prompt = readLine(), !prompt.isEmpty else { continue }
            if prompt == "/exit" || prompt == "/quit" { return }

            messages.append(.init(role: .user, content: prompt))
            print("assistant> ", terminator: "")
            let writeChunk: @Sendable (String) -> Void = { chunk in
                FileHandle.standardOutput.write(Data(chunk.utf8))
            }
            let response: String = try await service.respond(
                messages: messages,
                options: configuration.options,
                onToken: writeChunk
            )
            print()
            messages.append(.init(role: .assistant, content: response))
        }
    }
}

private enum CommandLineError: LocalizedError {
    case missingModel
    case missingPrompt
    case invalidOption(String)
    case missingValue(String)

    var errorDescription: String? {
        switch self {
        case .missingModel: "--model is required."
        case .missingPrompt: "Pass --prompt or use --interactive / --serve."
        case .invalidOption(let option): "Unknown or invalid option: \(option)"
        case .missingValue(let option): "Missing value for \(option)."
        }
    }
}

private struct CommandLineConfiguration {
    var model: String?
    var prompt: String?
    var systemPrompt: String?
    var interactive = false
    var serve = false
    var stream = true
    var host = "127.0.0.1"
    var port = 8080
    var options = MLXGenerationOptions()
    var showHelp = false

    init(arguments: [String]) throws {
        var index = 0
        while index < arguments.count {
            let argument = arguments[index]
            switch argument {
            case "-h", "--help":
                showHelp = true
            case "-m", "--model":
                model = try value(after: argument, arguments: arguments, index: &index)
            case "-p", "--prompt":
                prompt = try value(after: argument, arguments: arguments, index: &index)
            case "--system":
                systemPrompt = try value(after: argument, arguments: arguments, index: &index)
            case "-i", "--interactive":
                interactive = true
            case "--serve":
                serve = true
            case "--no-stream":
                stream = false
            case "--host":
                host = try value(after: argument, arguments: arguments, index: &index)
            case "--port":
                let raw = try value(after: argument, arguments: arguments, index: &index)
                guard let parsed = Int(raw), (1...65_535).contains(parsed) else {
                    throw CommandLineError.invalidOption("--port \(raw)")
                }
                port = parsed
            case "--max-tokens":
                let raw = try value(after: argument, arguments: arguments, index: &index)
                guard let parsed = Int(raw), parsed > 0 else {
                    throw CommandLineError.invalidOption("--max-tokens \(raw)")
                }
                options.maxTokens = parsed
            case "--temperature":
                let raw = try value(after: argument, arguments: arguments, index: &index)
                guard let parsed = Float(raw), parsed >= 0 else {
                    throw CommandLineError.invalidOption("--temperature \(raw)")
                }
                options.temperature = parsed
            case "--top-p":
                let raw = try value(after: argument, arguments: arguments, index: &index)
                guard let parsed = Float(raw), (0...1).contains(parsed) else {
                    throw CommandLineError.invalidOption("--top-p \(raw)")
                }
                options.topP = parsed
            case "--top-k":
                let raw = try value(after: argument, arguments: arguments, index: &index)
                guard let parsed = Int(raw), parsed >= 0 else {
                    throw CommandLineError.invalidOption("--top-k \(raw)")
                }
                options.topK = parsed
            default:
                throw CommandLineError.invalidOption(argument)
            }
            index += 1
        }
    }

    private func value(after option: String, arguments: [String], index: inout Int) throws -> String {
        index += 1
        guard index < arguments.count else { throw CommandLineError.missingValue(option) }
        return arguments[index]
    }

    static let usage = """
    Usage:
      MLXKitServer --model <local-path|huggingface-id> --prompt <text> [options]
      MLXKitServer --model <local-path|huggingface-id> --interactive [options]
      MLXKitServer --model <local-path|huggingface-id> --serve [options]

    Options:
      -m, --model <value>       Local MLX model folder or Hugging Face ID (required)
      -p, --prompt <text>       Generate one response
      -i, --interactive          Start a multi-turn terminal chat
          --serve                Start an OpenAI-compatible HTTP endpoint on /v1/chat/completions
          --system <text>        System prompt
          --max-tokens <n>       Maximum generated tokens (default: 1024)
          --temperature <n>      Sampling temperature (default: 0.6)
          --top-p <n>            Nucleus sampling threshold (default: 1.0)
          --top-k <n>            Top-k sampling; 0 disables it (default: 0)
          --no-stream            Print one-shot output after generation completes
          --host <host>          HTTP bind host (default: 127.0.0.1)
          --port <port>          HTTP bind port (default: 8080)
      -h, --help                 Show this help
    """
}
