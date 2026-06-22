import XCTest
@testable import MLXKitC

final class MLXKitCTests: XCTestCase {
    func testRuntimeHandleCanBeCreatedAndReleased() {
        let handle = mlxkit_runtime_create()
        mlxkit_runtime_destroy(handle)
    }

    func testInvalidArgumentsAreRejectedSynchronously() {
        XCTAssertEqual(mlxkit_runtime_load_model(nil, nil, nil, nil), -1)
        XCTAssertEqual(mlxkit_runtime_chat(nil, nil, 0.6, 32, nil, nil, nil), -1)
    }
}
