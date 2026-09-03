#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 Arm Limited
# SPDX-License-Identifier: MIT

# ----------------------------------------------------------------------------
# Configuration

# Exit immediately if any component command errors
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAYER_DIR="${SCRIPT_DIR}/.."

BUILD_DIR="${LAYER_DIR}/build_android"
BUILD_DIR_PACK="${LAYER_DIR}/build_package/arm64/android/lib"

# ----------------------------------------------------------------------------
# Process command line options
if [ "$#" -lt 1 ]; then
    BUILD_TYPE=Release
else
    BUILD_TYPE=$1
fi

if [ "$#" -lt 2 ]; then
    PACKAGE=0
else
    PACKAGE=$2
fi

if [ "${PACKAGE}" -gt "0" ]; then
    echo "Building a ${BUILD_TYPE} build with packaging"
else
    echo "Building a ${BUILD_TYPE} build without packaging"
fi

# ----------------------------------------------------------------------------
# Build the 64-bit layer
mkdir -p ${BUILD_DIR}
pushd ${BUILD_DIR}

# Tests are host executables validated by native builds; Android packaging only needs the layer library.
cmake \
    -S "${LAYER_DIR}" \
    -B "${BUILD_DIR}" \
    -DCMAKE_SYSTEM_NAME=Android \
    -DANDROID_PLATFORM=29 \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_TOOLCHAIN=clang \
    -DANDROID_STL=c++_static \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DBUILD_TESTING=OFF \
    -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK_HOME}/build/cmake/android.toolchain.cmake" \
    -DCMAKE_WARN_DEPRECATED=OFF

cmake --build ${BUILD_DIR} -j$(nproc)

popd

# ----------------------------------------------------------------------------
# Build the release package
if [ "${PACKAGE}" -gt "0" ]; then
    # Setup the package directories
    mkdir -p ${BUILD_DIR_PACK}

    # Install the 64-bit layer
    cp ${BUILD_DIR}/source/libVkLayerNeuralStatistics.so ${BUILD_DIR_PACK}
fi
