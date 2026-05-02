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

/**
 * Represents a message formatted for a language model.
 */
public struct ModelMessage {
    public var role: Role
    public var content: String
    public var toolCalls: [[String: any Sendable]]?
    
    public var representation: [String: any Sendable] {
        var dict: [String: any Sendable] = [
            "role": role.rawValue,
            "content": content
        ]
        
        if let toolCalls {
            dict["tool_calls"] = toolCalls
        }
        
        return dict
    }
    
    public init(
        role: Role,
        content: String,
        toolCalls: [[String: any Sendable]]? = nil
    ) {
        self.role = role
        self.content = content
        self.toolCalls = toolCalls
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
    public var defaultPrompt: String?
    
    public var modelConfig: ModelConfiguration?
    public var container: ModelContainer?
    
    public var tokens: Int = 1024
    public var temperature: Float = 0.5
    
    public var isLoaded: Bool {
        container != nil && modelConfig != nil && modelPath != nil
    }
    
    public init() {
    }
    
    public func setMLXMemory(limitInMB: Int) {
        let bytes = limitInMB * 1024 * 1024
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
        let modelURL = url
        let modelConfig = ModelConfiguration(
            directory: url,
            defaultPrompt: defaultPrompt
        )
        
        do {
            let container = try await LLMModelFactory.shared.loadContainer(configuration: modelConfig)
            self.defaultPrompt = defaultPrompt
            self.modelPath = modelURL
            self.modelConfig = modelConfig
            self.container = container
        } catch {
            throw MLXModelChatVideoModelError.errorWhileLoadingContainer(error.localizedDescription)
        }
    }
}

// MARK: - Loading/Unloading
extension MLXChatService {
    public func unload() {
        modelConfig = nil
        container = nil
        MLX.Memory.clearCache()
    }
    public func reload() async throws {
        guard let modelPath else {
            throw MLXModelChatVideoModelError.cantReload("Model Path is nil")
        }
        guard let defaultPrompt else {
            throw MLXModelChatVideoModelError.cantReload("Default Prompt is nil")
        }
        try await loadModel(
            at: modelPath,
            defaultPrompt: defaultPrompt
        )
    }
}

// MARK: - Get Response
extension MLXChatService {
    public func getResponse(
        messages: [ModelMessage],
        tools: [[String: any Sendable]],
        completion: @Sendable @escaping (String) -> Void,
        toolcallCompletionHandler: @Sendable @escaping (ToolCallResponse) -> Void,
        infoCompletionHandler: @Sendable @escaping (GenerateCompletionInfo) -> Void
    ) async throws -> String {
        guard isLoaded else {
            throw MLXModelChatVideoModelError.cantGenerateResponseNotLoaded
        }
        guard let container else {
            throw MLXModelChatVideoModelError.containerNotConfigured
        }
        let safeMessages = messages.map { msg in
            msg.representation
        }
        return try await container.perform { context in
            let input = try await context
                .processor
                .prepare(
                    input: .init(
                        messages: safeMessages,
                        tools: tools
                    )
                )
            
            let stream = try await generate(
                input: input,
                parameters: GenerateParameters(
                    temperature: temperature
                ),
                context: context
            )
            var output = ""
            for await generation in stream {
                if let info = generation.info {
                    infoCompletionHandler(info)
                }
                if let chunk = generation.chunk {
                    
                    output += chunk
                    await MainActor.run {
                        completion(chunk)
                    }
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
                    
                    toolcallCompletionHandler(
                        ToolCallResponse(
                            functionName,
                            arguments,
                            rawToolCall
                        )
                    )
                }
            }
            return output
        }
    }
}
