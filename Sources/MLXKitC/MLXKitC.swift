import Foundation
import MLXKit
import MLXKitMetalResources

/// Completion callback used by model-loading and generation calls.
///
/// `errorMessage` is valid only for the duration of the callback. Copy it if
/// it must outlive the call. A `status` of zero indicates success.
public typealias MLXKitCompletionCallback = @convention(c) (
    UnsafeMutableRawPointer?, Int32, UnsafePointer<CChar>?
) -> Void

/// Token callback used by `mlxkit_runtime_chat`.
///
/// `token` is a UTF-8 fragment that is valid only for the duration of the
/// callback. Callbacks may arrive from a Swift concurrency executor, so the
/// C caller must make its own state thread-safe.
public typealias MLXKitTokenCallback = @convention(c) (
    UnsafeMutableRawPointer?, UnsafePointer<CChar>?
) -> Void

private let invalidArgument: Int32 = -1

/// Opaque state owned by the C runtime handle.
private actor MLXKitCRuntime {
    private var service: MLXChatService?

    func loadModel(directory: String) async throws {
        let service = await chatService()
        try await service.loadModel(from: .directory(URL(fileURLWithPath: directory)))
    }

    func loadModel(id: String, revision: String) async throws {
        let service = await chatService()
        try await service.loadModel(from: .hub(id: id, revision: revision))
    }

    func chat(
        messagesJSON: String,
        temperature: Float,
        maxTokens: Int32,
        tokenTarget: CTokenTarget
    ) async throws {
        let data = Data(messagesJSON.utf8)
        let messages = try JSONDecoder().decode([CMessage].self, from: data).map { message in
            guard let role = Role(rawValue: message.role) else {
                throw CAPIError.invalidRole(message.role)
            }
            return ModelMessage(role: role, content: message.content ?? "")
        }

        let service = await chatService()
        let options = MLXGenerationOptions(
            maxTokens: maxTokens > 0 ? Int(maxTokens) : nil,
            temperature: temperature >= 0 ? temperature : 0.6
        )
        _ = try await service.respond(messages: messages, options: options) { token in
            tokenTarget.call(token)
        }
    }

    private func chatService() async -> MLXChatService {
        if let service {
            return service
        }
        let created = await MainActor.run { MLXChatService() }
        service = created
        return created
    }
}

private struct CMessage: Decodable {
    let role: String
    let content: String?
}

private enum CAPIError: LocalizedError {
    case invalidRole(String)

    var errorDescription: String? {
        switch self {
        case .invalidRole(let role): "Unsupported message role: \(role)"
        }
    }
}

/// C pointers and function pointers are valid to pass across this bridge, but
/// Swift cannot prove that the foreign caller has made their context safe.
private struct CTokenTarget: @unchecked Sendable {
    let callback: MLXKitTokenCallback?
    let context: UnsafeMutableRawPointer?

    func call(_ token: String) {
        token.withCString { pointer in
            callback?(context, pointer)
        }
    }
}

private struct CCompletionTarget: @unchecked Sendable {
    let callback: MLXKitCompletionCallback?
    let context: UnsafeMutableRawPointer?

    func succeed() {
        callback?(context, 0, nil)
    }

    func fail(_ error: Error) {
        error.localizedDescription.withCString { pointer in
            callback?(context, 1, pointer)
        }
    }
}

/// Creates a runtime handle. Pass the result to the other `mlxkit_runtime_*`
/// functions and eventually release it with `mlxkit_runtime_destroy`.
@_cdecl("mlxkit_runtime_create")
public func mlxkit_runtime_create() -> UnsafeMutableRawPointer {
    MLXKitMetalResources.registerBundle()
    return Unmanaged.passRetained(MLXKitCRuntime()).toOpaque()
}

/// Releases a runtime handle created by `mlxkit_runtime_create`.
@_cdecl("mlxkit_runtime_destroy")
public func mlxkit_runtime_destroy(_ handle: UnsafeMutableRawPointer?) {
    guard let handle else { return }
    Unmanaged<MLXKitCRuntime>.fromOpaque(handle).release()
}

/// Loads a local MLX model directory asynchronously.
///
/// Returns zero after accepting the work. Wait for `completion` before
/// calling `mlxkit_runtime_chat` with this handle.
@_cdecl("mlxkit_runtime_load_model")
public func mlxkit_runtime_load_model(
    _ handle: UnsafeMutableRawPointer?,
    _ directory: UnsafePointer<CChar>?,
    _ completion: MLXKitCompletionCallback?,
    _ context: UnsafeMutableRawPointer?
) -> Int32 {
    guard let handle, let directory else { return invalidArgument }
    let runtime = Unmanaged<MLXKitCRuntime>.fromOpaque(handle).takeUnretainedValue()
    let path = String(cString: directory)
    let completionTarget = CCompletionTarget(callback: completion, context: context)
    Task { [runtime, completionTarget] in
        do {
            try await runtime.loadModel(directory: path)
            completionTarget.succeed()
        } catch {
            completionTarget.fail(error)
        }
    }
    return 0
}

/// Loads a Hugging Face MLX repository asynchronously.
@_cdecl("mlxkit_runtime_load_model_id")
public func mlxkit_runtime_load_model_id(
    _ handle: UnsafeMutableRawPointer?,
    _ modelID: UnsafePointer<CChar>?,
    _ revision: UnsafePointer<CChar>?,
    _ completion: MLXKitCompletionCallback?,
    _ context: UnsafeMutableRawPointer?
) -> Int32 {
    guard let handle, let modelID else { return invalidArgument }
    let runtime = Unmanaged<MLXKitCRuntime>.fromOpaque(handle).takeUnretainedValue()
    let id = String(cString: modelID)
    let revision = revision.map(String.init(cString:)) ?? "main"
    let completionTarget = CCompletionTarget(callback: completion, context: context)
    Task { [runtime, completionTarget] in
        do {
            try await runtime.loadModel(id: id, revision: revision)
            completionTarget.succeed()
        } catch {
            completionTarget.fail(error)
        }
    }
    return 0
}

/// Generates from a JSON array of chat messages asynchronously.
///
/// Example `messagesJSON`: `[ {"role":"user","content":"Hello"} ]`.
/// Token fragments are delivered to `onToken`; `completion` marks the end of
/// the request or reports its error.
@_cdecl("mlxkit_runtime_chat")
public func mlxkit_runtime_chat(
    _ handle: UnsafeMutableRawPointer?,
    _ messagesJSON: UnsafePointer<CChar>?,
    _ temperature: Float,
    _ maxTokens: Int32,
    _ onToken: MLXKitTokenCallback?,
    _ completion: MLXKitCompletionCallback?,
    _ context: UnsafeMutableRawPointer?
) -> Int32 {
    guard let handle, let messagesJSON else { return invalidArgument }
    let runtime = Unmanaged<MLXKitCRuntime>.fromOpaque(handle).takeUnretainedValue()
    let json = String(cString: messagesJSON)
    let tokenTarget = CTokenTarget(callback: onToken, context: context)
    let completionTarget = CCompletionTarget(callback: completion, context: context)
    Task { [runtime, tokenTarget, completionTarget] in
        do {
            try await runtime.chat(
                messagesJSON: json,
                temperature: temperature,
                maxTokens: maxTokens,
                tokenTarget: tokenTarget
            )
            completionTarget.succeed()
        } catch {
            completionTarget.fail(error)
        }
    }
    return 0
}
