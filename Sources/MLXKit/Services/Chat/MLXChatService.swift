//
//  MLXChatService.swift
//  ModelChat
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
 * This is the message getting sent to every LLM
 */
public struct ModelMessage {
    public var role: String
    public var content: String
    public var toolCalls: [ToolCall]?
    
    public var isToolCall: Bool {
        toolCalls != nil
    }
    
    public var message: [String: any Sendable] {
        return [
            "role": role,
            "content": content
        ]
    }
    
    /**
     Prompt:
     There was currently a new video by SideQuest Drew about exploring epsteins new island, can u look up the free version of the video?
     */
    static let searchTool: [String: any Sendable] = [
        "type": "function",
        "function": [
            "name": "search",
            "description": "Search the web for information",
            "parameters": [
                "type": "object",
                "properties": [
                    "query": [
                        "type": "string",
                        "description": "The search query"
                    ] as [String: any Sendable]
                ] as [String: any Sendable],
                "required": ["query"]
            ] as [String: any Sendable]
        ] as [String: any Sendable]
    ]
    
    static let clickLinkTool: [String: any Sendable] = [
        "type": "function",
        "function": [
            "name": "clickLink",
            "description": "Open one of the numbered links from the current browser page.",
            "parameters": [
                "type": "object",
                "properties": [
                    "index": [
                        "type": "integer",
                        "description": "The 1-based number of the link to open from the current page's Links list."
                    ] as [String: any Sendable]
                ] as [String: any Sendable],
                "required": ["index"]
            ] as [String: any Sendable]
        ] as [String: any Sendable]
    ]
    
    public init(role: String, content: String) {
        self.role = role
        self.content = content
    }
}

public struct ToolCall {
    public var id: String
    public var name: String
    public var arguments: String
}

public struct SearchToolArguments: Codable {
    public let query: String
    
    public init(query: String) {
        self.query = query
    }
}

public struct ClickTookArguments: Codable {
    public let index: Int
    
    public init(index: Int) {
        self.index = index
    }
}

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
    
    
    public func getResponse(
        messages: [ModelMessage],
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
            msg.message
        }
        return try await container.perform { context in
            let input = try await context
                .processor
                .prepare(
                    input: .init(
                        messages: safeMessages,
                        tools: [
                            ModelMessage.searchTool,
                            ModelMessage.clickLinkTool
                        ]
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
