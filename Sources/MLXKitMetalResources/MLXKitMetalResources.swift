import Foundation

/// Registers the resource bundle containing MLX's precompiled Metal shaders.
///
/// Cmlx discovers the nested `mlx-swift_Cmlx.bundle` via `Bundle.allBundles`.
/// Accessing `Bundle.module` makes this SwiftPM resource bundle visible before
/// MLX initializes its Metal device.
public enum MLXKitMetalResources {
    public static func registerBundle() {
        _ = Bundle.module
    }
}
