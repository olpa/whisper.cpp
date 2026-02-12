#!/bin/bash
set -eu

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANDROID_NDK="${ANDROID_NDK_HOME:-$HOME/android/ndk/27.2.12479018}"
echo Install prefix: $HANDSFREEVC_DEV_HOME
INSTALL_PREFIX="${HANDSFREEVC_DEV_HOME:-$HOME/p/handsfree_vc/hfvc_dev}/whisper.cpp"

# Parse platform arguments (default: all platforms)
BUILD_LINUX=0
BUILD_ANDROID_ARM64=0
BUILD_ANDROID_ARM7=0
BUILD_ANDROID_X64=0
BUILD_ANDROID_X86=0

if [ $# -eq 0 ]; then
    # No arguments, build all platforms
    BUILD_LINUX=1
    BUILD_ANDROID_ARM64=1
    BUILD_ANDROID_ARM7=1
    BUILD_ANDROID_X64=1
    BUILD_ANDROID_X86=1
else
    # Parse platform arguments
    for arg in "$@"; do
        case "$arg" in
            linux)
                BUILD_LINUX=1
                ;;
            android|android-arm64)
                BUILD_ANDROID_ARM64=1
                ;;
            android-arm7)
                BUILD_ANDROID_ARM7=1
                ;;
            android-x64)
                BUILD_ANDROID_X64=1
                ;;
            android-x86)
                BUILD_ANDROID_X86=1
                ;;
            android-all)
                BUILD_ANDROID_ARM64=1
                BUILD_ANDROID_ARM7=1
                BUILD_ANDROID_X64=1
                BUILD_ANDROID_X86=1
                ;;
            all)
                BUILD_LINUX=1
                BUILD_ANDROID_ARM64=1
                BUILD_ANDROID_ARM7=1
                BUILD_ANDROID_X64=1
                BUILD_ANDROID_X86=1
                ;;
            *)
                echo "Unknown platform: $arg"
                echo "Usage: $0 [linux] [android|android-arm64] [android-arm7] [android-x64] [android-x86] [android-all] [all]"
                exit 1
                ;;
        esac
    done
fi

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
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        "$SCRIPT_DIR"

    cmake --build . --config RelWithDebInfo -j$(nproc)

    echo "✓ Build complete for $platform"
    cd "$SCRIPT_DIR"
}

# Build Linux x86_64 (native)
if [ $BUILD_LINUX -eq 1 ]; then
    build_platform "Linux x86_64" \
        "$SCRIPT_DIR/build-linux-x86_64"
fi

# Build Android arm64-v8a
if [ $BUILD_ANDROID_ARM64 -eq 1 ]; then
    build_platform "Android arm64-v8a" \
        "$SCRIPT_DIR/build-android-arm64-v8a" \
        -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
        -DANDROID_ABI=arm64-v8a \
        -DANDROID_PLATFORM=android-21 \
        -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-z,max-page-size=16384"
fi

# Build Android armeabi-v7a
if [ $BUILD_ANDROID_ARM7 -eq 1 ]; then
    build_platform "Android armeabi-v7a" \
        "$SCRIPT_DIR/build-android-armeabi-v7a" \
        -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
        -DANDROID_ABI=armeabi-v7a \
        -DANDROID_PLATFORM=android-21
fi

# Build Android x86_64
if [ $BUILD_ANDROID_X64 -eq 1 ]; then
    build_platform "Android x86_64" \
        "$SCRIPT_DIR/build-android-x86_64" \
        -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
        -DANDROID_ABI=x86_64 \
        -DANDROID_PLATFORM=android-21
fi

# Build Android x86
if [ $BUILD_ANDROID_X86 -eq 1 ]; then
    build_platform "Android x86" \
        "$SCRIPT_DIR/build-android-x86" \
        -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
        -DANDROID_ABI=x86 \
        -DANDROID_PLATFORM=android-21
fi

echo ""
echo "=== Installing libraries and headers ==="

# Create VERSION file
echo "Creating VERSION file..."
echo "1.8.2" > "$INSTALL_PREFIX/VERSION"

# Copy headers
echo "Copying headers..."
cp "$SCRIPT_DIR/include/whisper.h" "$INSTALL_PREFIX/include/"
cp "$SCRIPT_DIR/ggml/include/ggml.h" "$INSTALL_PREFIX/include/"
cp "$SCRIPT_DIR/ggml/include/ggml-cpu.h" "$INSTALL_PREFIX/include/"
cp "$SCRIPT_DIR/ggml/include/ggml-backend.h" "$INSTALL_PREFIX/include/"
cp "$SCRIPT_DIR/ggml/include/ggml-alloc.h" "$INSTALL_PREFIX/include/"

# Copy Linux library
if [ $BUILD_LINUX -eq 1 ]; then
    echo "Copying Linux x86_64 library..."
    # Copy whisper with version symlinks
    cp -P "$SCRIPT_DIR/build-linux-x86_64/src/libwhisper.so"* "$INSTALL_PREFIX/linux-x86_64/"
    # Copy ggml libraries (these don't have versioning yet, but copy as-is)
    cp "$SCRIPT_DIR/build-linux-x86_64/ggml/src/libggml.so" "$INSTALL_PREFIX/linux-x86_64/"
    cp "$SCRIPT_DIR/build-linux-x86_64/ggml/src/libggml-base.so" "$INSTALL_PREFIX/linux-x86_64/"
    cp "$SCRIPT_DIR/build-linux-x86_64/ggml/src/libggml-cpu.so" "$INSTALL_PREFIX/linux-x86_64/"
fi

# Copy Android libraries
if [ $BUILD_ANDROID_ARM64 -eq 1 ]; then
    echo "Copying Android arm64-v8a libraries..."
    cp "$SCRIPT_DIR/build-android-arm64-v8a/src/libwhisper.so" "$INSTALL_PREFIX/android/arm64-v8a/"
    cp "$SCRIPT_DIR/build-android-arm64-v8a/ggml/src/libggml.so" "$INSTALL_PREFIX/android/arm64-v8a/"
    cp "$SCRIPT_DIR/build-android-arm64-v8a/ggml/src/libggml-base.so" "$INSTALL_PREFIX/android/arm64-v8a/"
    cp "$SCRIPT_DIR/build-android-arm64-v8a/ggml/src/libggml-cpu.so" "$INSTALL_PREFIX/android/arm64-v8a/"
fi

if [ $BUILD_ANDROID_ARM7 -eq 1 ]; then
    echo "Copying Android armeabi-v7a libraries..."
    cp "$SCRIPT_DIR/build-android-armeabi-v7a/src/libwhisper.so" "$INSTALL_PREFIX/android/armeabi-v7a/"
    cp "$SCRIPT_DIR/build-android-armeabi-v7a/ggml/src/libggml.so" "$INSTALL_PREFIX/android/armeabi-v7a/"
    cp "$SCRIPT_DIR/build-android-armeabi-v7a/ggml/src/libggml-base.so" "$INSTALL_PREFIX/android/armeabi-v7a/"
    cp "$SCRIPT_DIR/build-android-armeabi-v7a/ggml/src/libggml-cpu.so" "$INSTALL_PREFIX/android/armeabi-v7a/"
fi

if [ $BUILD_ANDROID_X64 -eq 1 ]; then
    echo "Copying Android x86_64 libraries..."
    cp "$SCRIPT_DIR/build-android-x86_64/src/libwhisper.so" "$INSTALL_PREFIX/android/x86_64/"
    cp "$SCRIPT_DIR/build-android-x86_64/ggml/src/libggml.so" "$INSTALL_PREFIX/android/x86_64/"
    cp "$SCRIPT_DIR/build-android-x86_64/ggml/src/libggml-base.so" "$INSTALL_PREFIX/android/x86_64/"
    cp "$SCRIPT_DIR/build-android-x86_64/ggml/src/libggml-cpu.so" "$INSTALL_PREFIX/android/x86_64/"
fi

if [ $BUILD_ANDROID_X86 -eq 1 ]; then
    echo "Copying Android x86 libraries..."
    cp "$SCRIPT_DIR/build-android-x86/src/libwhisper.so" "$INSTALL_PREFIX/android/x86/"
    cp "$SCRIPT_DIR/build-android-x86/ggml/src/libggml.so" "$INSTALL_PREFIX/android/x86/"
    cp "$SCRIPT_DIR/build-android-x86/ggml/src/libggml-base.so" "$INSTALL_PREFIX/android/x86/"
    cp "$SCRIPT_DIR/build-android-x86/ggml/src/libggml-cpu.so" "$INSTALL_PREFIX/android/x86/"
fi

echo ""
echo "=== Installation Summary ==="
echo "Libraries and headers installed to: $INSTALL_PREFIX"
echo ""
echo "Directory structure:"
tree -L 2 "$INSTALL_PREFIX" 2>/dev/null || find "$INSTALL_PREFIX" -type f
echo ""
echo "✓ Build and installation complete!"
