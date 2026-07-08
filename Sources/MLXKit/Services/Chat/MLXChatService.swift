//
//  MLXChatService.swift
//  MLXKit
//
//  Created by Aryan Rogye on 4/5/26.
//

import Foundation
import SwiftUI
import MLX
import MLXLLM
import MLXLMCommon
import HuggingFace

public enum MLXModelChatVideoModelError: Error {
    case modelDoesntExist
    case errorWhileLoadingContainer(String)
    case containerNotConfigured
    case cantGenerateResponseNotLoaded
    case cantReload(String)
}

/// Where an MLX language model should be loaded from.
///
/// Use `.directory` for a model already present on disk, or `.hub` to let
/// `mlx-swift-lm` download and cache a Hugging Face repository.
public enum MLXModelSource: Sendable {
    case directory(URL)
    case hub(id: String, revision: String = "main")
}

/// Controls text sampling for a generation request.
///
/// These map directly to `MLXLMCommon.GenerateParameters`, so the same
/// settings work in a SwiftUI app and in `MLXKitServer`.
public struct MLXGenerationOptions: Sendable {
    public var maxTokens: Int?
    public var temperature: Float
    public var topP: Float
    public var topK: Int
    public var minP: Float
    public var repetitionPenalty: Float?
    public var repetitionContextSize: Int

    public init(
        maxTokens: Int? = 1_024,
        temperature: Float = 0.6,
        topP: Float = 1.0,
        topK: Int = 0,
        minP: Float = 0.0,
        repetitionPenalty: Float? = nil,
        repetitionContextSize: Int = 20
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
    }

    var generateParameters: GenerateParameters {
        GenerateParameters(
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            topK: topK,
            minP: minP,
            repetitionPenalty: repetitionPenalty,
            repetitionContextSize: repetitionContextSize
        )
    }
}

/**
 * Represents a message formatted for a language model.
 */
public struct ModelMessage {
    public var role: Role
    public var content: String
    public var toolCalls: [[String: any Sendable]]?
    public var toolCallID: String?
    
    public var representation: [String: any Sendable] {
        var dict: [String: any Sendable] = [
            "role": role.rawValue,
            "content": content
        ]
        
        if let toolCalls {
            dict["tool_calls"] = toolCalls
        }
        if let toolCallID {
            dict["tool_call_id"] = toolCallID
        }
        
        return dict
    }
    
    public init(
        role: Role,
        content: String,
        toolCalls: [[String: any Sendable]]? = nil,
        toolCallID: String? = nil
    ) {
        self.role = role
        self.content = content
        self.toolCalls = toolCalls
        self.toolCallID = toolCallID
    }
}

/**
 * The role of a message in a language model conversation.
 */
public enum Role: String, Equatable, Sendable {
    case user
    case assistant
    case system
    case tool
}

/**
 * Container for a tool call response.
 *
 * Encapsulates the function name and its arguments.
 */
public struct ToolCallResponse: Sendable {
    public let functionName: String
    public let arguments: [String: JSONValue]
    public let rawToolCall: [String: any Sendable]?
    
    public init(
        _ functionName: String,
        _ arguments: [String : JSONValue],
        _ rawToolCall: [String: any Sendable]?
    ) {
        self.functionName = functionName
        self.arguments = arguments
        self.rawToolCall = rawToolCall
    }
}

@Observable
@MainActor
public final class MLXChatService {
    
    public var modelPath: URL?
    public private(set) var modelSource: MLXModelSource?
    public var defaultPrompt: String?
    
    public var modelConfig: ModelConfiguration?
    public var container: ModelContainer?
    
    public var tokens: Int = 1024
    public var temperature: Float = 0.5
    
    public var isLoaded: Bool {
        container != nil && modelConfig != nil
    }
    
    public init() {
    }
    
    public func setMLXMemory(limitInMB: Int) {
        let bytes = limitInMB * 1024 * 1024
        MLX.Memory.memoryLimit = bytes
        MLX.Memory.cacheLimit = bytes

        // Pro Tip: Clear the current cache so the new limit
        // is enforced against a fresh slate.
        MLX.Memory.clearCache()
    }
}

// MARK: - Load Model
extension MLXChatService {
    /**
     * Load Model
     * using default for defaultPrompt as hello is the same thing what mlx does
     */
    public func loadModel(
        at url: URL,
        defaultPrompt: String = "hello"
    ) async throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw MLXModelChatVideoModelError.modelDoesntExist
        }
        try await loadModel(from: .directory(url), defaultPrompt: defaultPrompt)
    }

    /// Loads either a local model folder or a Hugging Face MLX repository.
    ///
    /// The Hugging Face form is useful for command-line tools because model
    /// files are fetched into the standard Hub cache automatically.
    public func loadModel(
        from source: MLXModelSource,
        defaultPrompt: String = "hello"
    ) async throws {
        let modelConfig: ModelConfiguration
        switch source {
        case .directory(let url):
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw MLXModelChatVideoModelError.modelDoesntExist
            }
            modelConfig = ModelConfiguration(directory: url, defaultPrompt: defaultPrompt)
        case .hub(let id, let revision):
            modelConfig = ModelConfiguration(id: id, revision: revision, defaultPrompt: defaultPrompt)
        }
        
        do {
            let container = try await LLMModelFactory.shared.loadContainer(configuration: modelConfig)
            self.defaultPrompt = defaultPrompt
            self.modelPath = {
                if case .directory(let url) = source { return url }
                return nil
            }()
            self.modelSource = source
            self.modelConfig = modelConfig
            self.container = container
        } catch {
            throw MLXModelChatVideoModelError.errorWhileLoadingContainer(error.localizedDescription)
        }
    }

    /// Convenience overload for a model from Hugging Face, for example
    /// `mlx-community/Llama-3.2-3B-Instruct-4bit`.
    public func loadModel(
        modelID: String,
        revision: String = "main",
        defaultPrompt: String = "hello"
    ) async throws {
        try await loadModel(
            from: .hub(id: modelID, revision: revision),
            defaultPrompt: defaultPrompt
        )
    }
}

// MARK: - Loading/Unloading
extension MLXChatService {
    public func unload() {
        modelConfig = nil
        container = nil
        modelPath = nil
        modelSource = nil
        MLX.Memory.clearCache()
    }
    public func reload() async throws {
        guard let modelSource else {
            throw MLXModelChatVideoModelError.cantReload("Model source is nil")
        }
        guard let defaultPrompt else {
            throw MLXModelChatVideoModelError.cantReload("Default Prompt is nil")
        }
        try await loadModel(from: modelSource, defaultPrompt: defaultPrompt)
    }
}

// MARK: - Get Response
extension MLXChatService {
    /// Generates a response and delivers text, tool calls, and final metrics
    /// as they become available. This is the shared API used by the CLI and
    /// by apps embedding MLXKit.
    @discardableResult
    public func respond(
        messages: [ModelMessage],
        tools: [[String: any Sendable]] = [],
        options: MLXGenerationOptions = .init(),
        onToken: @Sendable @escaping (String) -> Void = { _ in },
        onToolCall: @Sendable @escaping (ToolCallResponse) -> Void = { _ in },
        onCompletion: @Sendable @escaping (GenerateCompletionInfo) -> Void = { _ in }
    ) async throws -> String {
        guard isLoaded else {
            throw MLXModelChatVideoModelError.cantGenerateResponseNotLoaded
        }
        guard let container else {
            throw MLXModelChatVideoModelError.containerNotConfigured
        }

        let safeMessages = messages.map(\.representation)
        let input = try await container.prepare(
            input: .init(messages: safeMessages, tools: tools)
        )
        let stream = try await container.generate(
            input: input,
            parameters: options.generateParameters
        )

        var output = ""
        for await generation in stream {
            if let info = generation.info {
                onCompletion(info)
            }
            if let chunk = generation.chunk {
                output += chunk
                onToken(chunk)
            }
            if let tool = generation.toolCall {
                let functionName = tool.function.name
                let arguments = tool.function.arguments
                let rawToolCall: [String: any Sendable] = [
                    "type": "function",
                    "function": [
                        "name": functionName,
                        "arguments": arguments
                    ] as [String: any Sendable]
                ]
                onToolCall(ToolCallResponse(functionName, arguments, rawToolCall))
            }
        }
        return output
    }

    public func getResponse(
        messages: [ModelMessage],
        tools: [[String: any Sendable]],
        completion: @Sendable @escaping (String) -> Void,
        toolcallCompletionHandler: @Sendable @escaping (ToolCallResponse) -> Void,
        infoCompletionHandler: @Sendable @escaping (GenerateCompletionInfo) -> Void
    ) async throws -> String {
        try await respond(
            messages: messages,
            tools: tools,
            options: .init(maxTokens: tokens, temperature: temperature),
            onToken: completion,
            onToolCall: toolcallCompletionHandler,
            onCompletion: infoCompletionHandler
        )
    }
}
