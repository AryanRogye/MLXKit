import Foundation
import MLXKit
@preconcurrency import NIOCore
@preconcurrency import NIOHTTP1
@preconcurrency import NIOPosix

/// A deliberately small OpenAI-compatible surface for local development.
/// It supports `GET /v1/models` and streaming/non-streaming chat completions.
final class HTTPServer: @unchecked Sendable {
    private let service: MLXChatService
    private let modelName: String
    private let defaultOptions: MLXGenerationOptions

    init(service: MLXChatService, modelName: String, defaultOptions: MLXGenerationOptions) {
        self.service = service
        self.modelName = modelName
        self.defaultOptions = defaultOptions
    }

    func run(host: String, port: Int) async throws {
        let group = MultiThreadedEventLoopGroup(numberOfThreads: 1)

        let server = self
        let channel = try await ServerBootstrap(group: group)
            .serverChannelOption(ChannelOptions.backlog, value: 256)
            .serverChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
            .childChannelInitializer { channel in
                channel.pipeline.configureHTTPServerPipeline().flatMap {
                    channel.pipeline.addHandler(HTTPHandler(server: server))
                }
            }
            .bind(host: host, port: port)
            .get()

        print("MLXKitServer listening at http://\(host):\(port)/v1")
        try await channel.closeFuture.get()
        try await group.shutdownGracefully()
    }

    @MainActor
    fileprivate func complete(_ request: ChatCompletionRequest) async throws -> ChatCompletionResponse {
        let (messages, options) = try generationInputs(for: request)

        let content = try await service.respond(messages: messages, options: options)
        return ChatCompletionResponse(model: request.model ?? modelName, content: content)
    }

    @MainActor
    fileprivate func stream(
        _ request: ChatCompletionRequest,
        emit: @Sendable @escaping (ChatCompletionStreamChunk) -> Void
    ) async throws {
        let (messages, options) = try generationInputs(for: request)
        let id = "chatcmpl-\(UUID().uuidString.lowercased())"
        let model = request.model ?? modelName

        emit(.role(id: id, model: model))
        _ = try await service.respond(
            messages: messages,
            options: options,
            onToken: { text in
                emit(.content(id: id, model: model, text: text))
            },
            onToolCall: { call in
                let arguments = (try? JSONEncoder().encode(call.arguments))
                    .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
                emit(.toolCall(
                    id: id,
                    model: model,
                    toolCall: .init(
                        index: 0,
                        id: UUID().uuidString.lowercased(),
                        type: "function",
                        function: .init(name: call.functionName, arguments: arguments)
                    )
                ))
            }
        )
        emit(.finished(id: id, model: model))
    }

    @MainActor
    private func generationInputs(
        for request: ChatCompletionRequest
    ) throws -> ([ModelMessage], MLXGenerationOptions) {
        let messages = try request.messages.map { message in
            guard let role = Role(rawValue: message.role) else {
                throw HTTPError.badRequest("Unsupported message role: \(message.role)")
            }
            return ModelMessage(role: role, content: message.content)
        }

        var options = defaultOptions
        if let temperature = request.temperature { options.temperature = temperature }
        if let topP = request.topP { options.topP = topP }
        if let maxTokens = request.maxTokens { options.maxTokens = maxTokens }
        return (messages, options)
    }

    fileprivate func modelsResponse() -> ModelsResponse {
        ModelsResponse(model: modelName)
    }
}

private final class HTTPHandler: ChannelInboundHandler, @unchecked Sendable {
    typealias InboundIn = HTTPServerRequestPart
    typealias OutboundOut = HTTPServerResponsePart

    private let server: HTTPServer
    private var head: HTTPRequestHead?
    private var body = ByteBuffer()

    init(server: HTTPServer) {
        self.server = server
    }

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        switch unwrapInboundIn(data) {
        case .head(let head):
            self.head = head
            body.clear()
        case .body(var buffer):
            body.writeBuffer(&buffer)
        case .end:
            guard let head else { return }
            handle(head: head, body: body, context: context)
            self.head = nil
            body.clear()
        }
    }

    private func handle(head: HTTPRequestHead, body: ByteBuffer, context: ChannelHandlerContext) {
        let path = head.uri.split(separator: "?", maxSplits: 1).first.map(String.init) ?? head.uri
        if head.method == .GET && path == "/v1/models" {
            sendJSON(server.modelsResponse(), status: .ok, context: context)
            return
        }
        guard head.method == .POST, path == "/v1/chat/completions" else {
            sendError(status: .notFound, message: "Use GET /v1/models or POST /v1/chat/completions.", context: context)
            return
        }

        do {
            let data = Data(body.getBytes(at: body.readerIndex, length: body.readableBytes) ?? [])
            let request = try JSONDecoder().decode(ChatCompletionRequest.self, from: data)
            let loopBoundContext = NIOLoopBound(context, eventLoop: context.eventLoop)
            if request.stream {
                startStream(request: request, context: loopBoundContext)
                return
            }
            Task { [weak self] in
                guard let self else { return }
                do {
                    let response = try await self.server.complete(request)
                    loopBoundContext.eventLoop.execute {
                        self.sendJSON(response, status: .ok, context: loopBoundContext.value)
                    }
                } catch let error as HTTPError {
                    loopBoundContext.eventLoop.execute {
                        self.sendError(status: error.status, message: error.message, context: loopBoundContext.value)
                    }
                } catch {
                    let message = error.localizedDescription
                    loopBoundContext.eventLoop.execute {
                        self.sendError(status: .internalServerError, message: message, context: loopBoundContext.value)
                    }
                }
            }
        } catch {
            sendError(status: .badRequest, message: "Invalid JSON: \(error.localizedDescription)", context: context)
        }
    }

    private func startStream(
        request: ChatCompletionRequest,
        context: NIOLoopBound<ChannelHandlerContext>
    ) {
        let writer = SSEWriter(context: context)
        writer.start()
        Task { [weak self] in
            guard let self else { return }
            do {
                try await self.server.stream(request) { chunk in
                    writer.send(chunk)
                }
                writer.finish()
            } catch {
                writer.send(APIErrorResponse(message: error.localizedDescription))
                writer.finish()
            }
        }
    }

    private func sendJSON<T: Encodable>(_ value: T, status: HTTPResponseStatus, context: ChannelHandlerContext) {
        do {
            let data = try JSONEncoder().encode(value)
            var buffer = context.channel.allocator.buffer(capacity: data.count)
            buffer.writeBytes(data)
            let headers = HTTPHeaders([
                ("content-type", "application/json; charset=utf-8"),
                ("content-length", "\(data.count)")
            ])
            context.write(wrapOutboundOut(.head(.init(version: .http1_1, status: status, headers: headers))), promise: nil)
            context.writeAndFlush(wrapOutboundOut(.body(.byteBuffer(buffer))), promise: nil)
            context.writeAndFlush(wrapOutboundOut(.end(nil)), promise: nil)
        } catch {
            sendError(status: .internalServerError, message: "Could not encode response.", context: context)
        }
    }

    private func sendError(status: HTTPResponseStatus, message: String, context: ChannelHandlerContext) {
        sendJSON(APIErrorResponse(message: message), status: status, context: context)
    }
}

/// Safely marshals SSE writes back onto NIO's event loop while MLX generates
/// from its actor-isolated runtime.
private final class SSEWriter: @unchecked Sendable {
    private let context: NIOLoopBound<ChannelHandlerContext>

    init(context: NIOLoopBound<ChannelHandlerContext>) {
        self.context = context
    }

    func start() {
        let headers = HTTPHeaders([
            ("content-type", "text/event-stream; charset=utf-8"),
            ("cache-control", "no-cache"),
            ("connection", "keep-alive")
        ])
        context.value.writeAndFlush(
            NIOAny(HTTPServerResponsePart.head(.init(
                version: .http1_1,
                status: .ok,
                headers: headers
            ))),
            promise: nil
        )
    }

    func send<T: Encodable>(_ value: T) {
        guard let data = try? JSONEncoder().encode(value),
              let payload = String(data: data, encoding: .utf8)
        else { return }
        sendPayload(payload)
    }

    func finish() {
        sendPayload("[DONE]")
        context.eventLoop.execute {
            self.context.value.writeAndFlush(NIOAny(HTTPServerResponsePart.end(nil)), promise: nil)
        }
    }

    private func sendPayload(_ payload: String) {
        let event = "data: \(payload)\n\n"
        context.eventLoop.execute {
            var buffer = self.context.value.channel.allocator.buffer(capacity: event.utf8.count)
            buffer.writeString(event)
            self.context.value.writeAndFlush(
                NIOAny(HTTPServerResponsePart.body(.byteBuffer(buffer))),
                promise: nil
            )
        }
    }
}

private enum HTTPError: Error {
    case badRequest(String)

    var status: HTTPResponseStatus { .badRequest }
    var message: String {
        switch self { case .badRequest(let message): message }
    }
}

private struct ChatCompletionRequest: Decodable, Sendable {
    let model: String?
    let messages: [ChatMessage]
    let temperature: Float?
    let topP: Float?
    let maxTokens: Int?
    let stream: Bool

    enum CodingKeys: String, CodingKey {
        case model, messages, temperature, stream
        case topP = "top_p"
        case maxTokens = "max_tokens"
    }

    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        model = try values.decodeIfPresent(String.self, forKey: .model)
        messages = try values.decode([ChatMessage].self, forKey: .messages)
        temperature = try values.decodeIfPresent(Float.self, forKey: .temperature)
        topP = try values.decodeIfPresent(Float.self, forKey: .topP)
        maxTokens = try values.decodeIfPresent(Int.self, forKey: .maxTokens)
        stream = try values.decodeIfPresent(Bool.self, forKey: .stream) ?? false
    }
}

private struct ChatMessage: Decodable, Sendable {
    let role: String
    let content: String
}

private struct ChatCompletionResponse: Encodable {
    let id = "chatcmpl-\(UUID().uuidString.lowercased())"
    let object = "chat.completion"
    let created = Int(Date().timeIntervalSince1970)
    let model: String
    let choices: [Choice]

    init(model: String, content: String) {
        self.model = model
        choices = [.init(content: content)]
    }

    struct Choice: Encodable {
        let index = 0
        let message: AssistantMessage
        let finishReason = "stop"

        init(content: String) { message = .init(content: content) }

        enum CodingKeys: String, CodingKey {
            case index, message
            case finishReason = "finish_reason"
        }
    }

    struct AssistantMessage: Encodable {
        let role = "assistant"
        let content: String
    }
}

private struct ChatCompletionStreamChunk: Encodable {
    let id: String
    let object = "chat.completion.chunk"
    let created = Int(Date().timeIntervalSince1970)
    let model: String
    let choices: [Choice]

    static func role(id: String, model: String) -> Self {
        .init(id: id, model: model, choices: [.init(delta: .init(role: "assistant"))])
    }

    static func content(id: String, model: String, text: String) -> Self {
        .init(id: id, model: model, choices: [.init(delta: .init(content: text))])
    }

    static func toolCall(id: String, model: String, toolCall: ToolCallDelta) -> Self {
        .init(id: id, model: model, choices: [.init(delta: .init(toolCalls: [toolCall]))])
    }

    static func finished(id: String, model: String) -> Self {
        .init(id: id, model: model, choices: [.init(delta: .init(), finishReason: "stop")])
    }

    struct Choice: Encodable {
        let index = 0
        let delta: Delta
        let finishReason: String?

        init(delta: Delta, finishReason: String? = nil) {
            self.delta = delta
            self.finishReason = finishReason
        }

        enum CodingKeys: String, CodingKey {
            case index, delta
            case finishReason = "finish_reason"
        }
    }

    struct Delta: Encodable {
        let role: String?
        let content: String?
        let toolCalls: [ToolCallDelta]?

        init(role: String? = nil, content: String? = nil, toolCalls: [ToolCallDelta]? = nil) {
            self.role = role
            self.content = content
            self.toolCalls = toolCalls
        }

        enum CodingKeys: String, CodingKey {
            case role, content
            case toolCalls = "tool_calls"
        }
    }

    struct ToolCallDelta: Encodable {
        let index: Int
        let id: String
        let type: String
        let function: Function

        struct Function: Encodable {
            let name: String
            let arguments: String
        }
    }
}

private struct ModelsResponse: Encodable {
    let object = "list"
    let data: [Model]

    init(model: String) { data = [.init(id: model)] }

    struct Model: Encodable {
        let id: String
        let object = "model"
        let ownedBy = "mlxkit"

        enum CodingKeys: String, CodingKey {
            case id, object
            case ownedBy = "owned_by"
        }
    }
}

private struct APIErrorResponse: Encodable {
    let error: Details

    init(message: String) { error = .init(message: message) }

    struct Details: Encodable {
        let message: String
        let type = "invalid_request_error"
    }
}
