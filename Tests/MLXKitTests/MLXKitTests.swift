import XCTest
@testable import MLXKit

final class MLXKitTests: XCTestCase {
    func testModelMessageBuildsChatTemplateRepresentation() {
        let message = ModelMessage(role: .user, content: "Hello")

        XCTAssertEqual(message.representation["role"] as? String, "user")
        XCTAssertEqual(message.representation["content"] as? String, "Hello")
    }

    func testGenerationOptionsPreserveSamplingValues() {
        let options = MLXGenerationOptions(
            maxTokens: 64,
            temperature: 0.2,
            topP: 0.9,
            topK: 40,
            minP: 0.1,
            repetitionPenalty: 1.1,
            repetitionContextSize: 32
        )

        XCTAssertEqual(options.maxTokens, 64)
        XCTAssertEqual(options.temperature, 0.2)
        XCTAssertEqual(options.topP, 0.9)
        XCTAssertEqual(options.topK, 40)
        XCTAssertEqual(options.minP, 0.1)
        XCTAssertEqual(options.repetitionPenalty, 1.1)
        XCTAssertEqual(options.repetitionContextSize, 32)
    }
}
