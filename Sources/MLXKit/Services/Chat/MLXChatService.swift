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
}

/**
 * Represents a message formatted for a language model.
 */
public struct ModelMessage {
    public var role: Role
    public var content: String
    
    public var representation: [String: any Sendable] {
        return [
            "role": role.rawValue,
            "content": content
        ]
    }
    
    public init(role: Role, content: String) {
        self.role = role
        self.content = content
    }
}

/**
 * The role of a message in a language model conversation.
 */
public enum Role: String, Equatable, Sendable {
    case user
    case assistant
    case system
}

/**
 * Container for a tool call response.
 *
 * Encapsulates the function name and its arguments.
 */
public struct ToolCallResponse: Sendable {
    public let functionName: String
    public let arguments: [String: JSONValue]
    
    public init(_ functionName: String, _ arguments: [String : JSONValue]) {
        self.functionName = functionName
        self.arguments = arguments
    }
}

@Observable
@MainActor
public final class MLXChatService {
    
    public var modelPath: URL?
    public var modelConfig: ModelConfiguration?
    public var container: ModelContainer?
    
    public var tokens: Int = 1024
    public var temperature: Float = 0.5
    
    public var isLoaded: Bool {
        container != nil && modelConfig != nil && modelPath != nil
    }
    
    public init() {
    }
}

// MARK: - Load Model
extension MLXChatService {
    public func loadModel(
        at url: URL
    ) async throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw MLXModelChatVideoModelError.modelDoesntExist
        }
        let modelURL = url
        let modelConfig = ModelConfiguration(directory: url)
        
        do {
            let container = try await LLMModelFactory.shared.loadContainer(configuration: modelConfig)
            self.modelPath = modelURL
            self.modelConfig = modelConfig
            self.container = container
        } catch {
            throw MLXModelChatVideoModelError.errorWhileLoadingContainer(error.localizedDescription)
        }
    }
}

// MARK: - Get Response
extension MLXChatService {
    public func getResponse(
        messages: [ModelMessage],
        tools: [[String: any Sendable]],
        completion: @Sendable @escaping (String) -> Void,
        toolcallCompletionHandler: @Sendable @escaping (ToolCallResponse) -> Void
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
                if let chunk = generation.chunk {
                    
                    output += chunk
                    await MainActor.run {
                        completion(chunk)
                    }
                }
                if let tool = generation.toolCall {
                    let functionName = tool.function.name
                    toolcallCompletionHandler(
                        ToolCallResponse(functionName, tool.function.arguments)
                    )
                }
            }
            return output
        }
    }
}
