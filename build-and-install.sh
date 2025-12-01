#!/bin/bash
set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANDROID_NDK="${ANDROID_NDK_HOME:-$HOME/android/ndk/27.2.12479018}"
INSTALL_PREFIX="${HANDSFREEAI_DEV_HOME:-$HOME/audio/hfai_dev}/whisper.cpp"

echo "=== Whisper.cpp Build and Install Script ==="
echo "Source directory: $SCRIPT_DIR"
echo "Android NDK: $ANDROID_NDK"
echo "Install prefix: $INSTALL_PREFIX"
echo ""

# Create install directory structure
echo "Creating install directory structure..."
mkdir -p "$INSTALL_PREFIX/include"
mkdir -p "$INSTALL_PREFIX/linux-x86_64"
mkdir -p "$INSTALL_PREFIX/android/arm64-v8a"
mkdir -p "$INSTALL_PREFIX/android/armeabi-v7a"
mkdir -p "$INSTALL_PREFIX/android/x86_64"
mkdir -p "$INSTALL_PREFIX/android/x86"

# Function to build for a specific platform
build_platform() {
    local platform=$1
    local build_dir=$2
    shift 2
    local cmake_args=("$@")

    echo ""
    echo "=== Building for $platform ==="
    mkdir -p "$build_dir"
    cd "$build_dir"

    cmake "${cmake_args[@]}" \
        -DBUILD_SHARED_LIBS=ON \
        -DWHISPER_BUILD_EXAMPLES=OFF \
        -DWHISPER_BUILD_TESTS=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        "$SCRIPT_DIR"

    cmake --build . --config Release -j$(nproc)

    echo "✓ Build complete for $platform"
    cd "$SCRIPT_DIR"
}

# Build Linux x86_64 (native)
build_platform "Linux x86_64" \
    "$SCRIPT_DIR/build-linux-x86_64"

# Build Android arm64-v8a
build_platform "Android arm64-v8a" \
    "$SCRIPT_DIR/build-android-arm64-v8a" \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-21

# Build Android armeabi-v7a
build_platform "Android armeabi-v7a" \
    "$SCRIPT_DIR/build-android-armeabi-v7a" \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=armeabi-v7a \
    -DANDROID_PLATFORM=android-21

# Build Android x86_64
build_platform "Android x86_64" \
    "$SCRIPT_DIR/build-android-x86_64" \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=x86_64 \
    -DANDROID_PLATFORM=android-21

# Build Android x86
build_platform "Android x86" \
    "$SCRIPT_DIR/build-android-x86" \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=x86 \
    -DANDROID_PLATFORM=android-21

echo ""
echo "=== Installing libraries and headers ==="

# Copy headers
echo "Copying headers..."
cp "$SCRIPT_DIR/include/whisper.h" "$INSTALL_PREFIX/include/"
cp "$SCRIPT_DIR/ggml/include/ggml.h" "$INSTALL_PREFIX/include/"
cp "$SCRIPT_DIR/ggml/include/ggml-cpu.h" "$INSTALL_PREFIX/include/"

# Copy Linux library
echo "Copying Linux x86_64 library..."
cp "$SCRIPT_DIR/build-linux-x86_64/src/libwhisper.so" "$INSTALL_PREFIX/linux-x86_64/"
cp "$SCRIPT_DIR/build-linux-x86_64/ggml/src/libggml.so" "$INSTALL_PREFIX/linux-x86_64/"
cp "$SCRIPT_DIR/build-linux-x86_64/ggml/src/libggml-base.so" "$INSTALL_PREFIX/linux-x86_64/"
cp "$SCRIPT_DIR/build-linux-x86_64/ggml/src/libggml-cpu.so" "$INSTALL_PREFIX/linux-x86_64/"

# Copy Android libraries
echo "Copying Android arm64-v8a libraries..."
cp "$SCRIPT_DIR/build-android-arm64-v8a/src/libwhisper.so" "$INSTALL_PREFIX/android/arm64-v8a/"
cp "$SCRIPT_DIR/build-android-arm64-v8a/ggml/src/libggml.so" "$INSTALL_PREFIX/android/arm64-v8a/"
cp "$SCRIPT_DIR/build-android-arm64-v8a/ggml/src/libggml-base.so" "$INSTALL_PREFIX/android/arm64-v8a/"
cp "$SCRIPT_DIR/build-android-arm64-v8a/ggml/src/libggml-cpu.so" "$INSTALL_PREFIX/android/arm64-v8a/"

echo "Copying Android armeabi-v7a libraries..."
cp "$SCRIPT_DIR/build-android-armeabi-v7a/src/libwhisper.so" "$INSTALL_PREFIX/android/armeabi-v7a/"
cp "$SCRIPT_DIR/build-android-armeabi-v7a/ggml/src/libggml.so" "$INSTALL_PREFIX/android/armeabi-v7a/"
cp "$SCRIPT_DIR/build-android-armeabi-v7a/ggml/src/libggml-base.so" "$INSTALL_PREFIX/android/armeabi-v7a/"
cp "$SCRIPT_DIR/build-android-armeabi-v7a/ggml/src/libggml-cpu.so" "$INSTALL_PREFIX/android/armeabi-v7a/"

echo "Copying Android x86_64 libraries..."
cp "$SCRIPT_DIR/build-android-x86_64/src/libwhisper.so" "$INSTALL_PREFIX/android/x86_64/"
cp "$SCRIPT_DIR/build-android-x86_64/ggml/src/libggml.so" "$INSTALL_PREFIX/android/x86_64/"
cp "$SCRIPT_DIR/build-android-x86_64/ggml/src/libggml-base.so" "$INSTALL_PREFIX/android/x86_64/"
cp "$SCRIPT_DIR/build-android-x86_64/ggml/src/libggml-cpu.so" "$INSTALL_PREFIX/android/x86_64/"

echo "Copying Android x86 libraries..."
cp "$SCRIPT_DIR/build-android-x86/src/libwhisper.so" "$INSTALL_PREFIX/android/x86/"
cp "$SCRIPT_DIR/build-android-x86/ggml/src/libggml.so" "$INSTALL_PREFIX/android/x86/"
cp "$SCRIPT_DIR/build-android-x86/ggml/src/libggml-base.so" "$INSTALL_PREFIX/android/x86/"
cp "$SCRIPT_DIR/build-android-x86/ggml/src/libggml-cpu.so" "$INSTALL_PREFIX/android/x86/"

echo ""
echo "=== Installation Summary ==="
echo "Libraries and headers installed to: $INSTALL_PREFIX"
echo ""
echo "Directory structure:"
tree -L 2 "$INSTALL_PREFIX" 2>/dev/null || find "$INSTALL_PREFIX" -type f
echo ""
echo "✓ Build and installation complete!"
