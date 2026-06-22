#!/usr/bin/env bash
set -euo pipefail

# Rebuild the MLX Metal library when the pinned mlx-swift version changes.
# Requires Xcode's Metal Toolchain component.

root="$(cd "$(dirname "$0")/.." && pwd)"
source_root="$root/.build/checkouts/mlx-swift/Source/Cmlx/mlx"
kernel_root="$source_root/mlx/backend/metal/kernels"
destination="$root/Sources/MLXKitServer/Resources/mlx-swift_Cmlx.bundle/default.metallib"

if [[ ! -d "$kernel_root" ]]; then
  echo "mlx-swift is not resolved. Run 'swift package resolve' first." >&2
  exit 1
fi

metal="$(xcrun --find metal)"
metallib="$(xcrun --find metallib)"
sdk="$(xcrun --sdk macosx --show-sdk-path)"
work_directory="$(mktemp -d)"
trap 'rm -rf "$work_directory"' EXIT

mkdir -p "$(dirname "$destination")" "$work_directory/cache"

while IFS= read -r source; do
  relative_path="${source#"$kernel_root"/}"
  output_name="${relative_path//\//_}"
  output_name="${output_name%.metal}.air"

  "$metal" \
    -x metal \
    -Wall -Wextra -fno-fast-math \
    -Wno-c++17-extensions -Wno-c++20-extensions \
    -fmodules-cache-path="$work_directory/cache" \
    -isysroot "$sdk" \
    -I"$source_root" \
    -c "$source" \
    -o "$work_directory/$output_name"
done < <(find "$kernel_root" -name '*.metal' -type f | sort)

"$metallib" "$work_directory"/*.air -o "$destination"
echo "Wrote $destination"
