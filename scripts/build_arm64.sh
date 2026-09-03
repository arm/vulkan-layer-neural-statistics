#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 Arm Limited
# SPDX-License-Identifier: MIT

# ----------------------------------------------------------------------------
# Configuration

# Exit immediately if any component command errors
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAYER_DIR="${SCRIPT_DIR}/.."

BUILD_DIR="${LAYER_DIR}/build_arm64"
BUILD_DIR_PACK="${LAYER_DIR}/build_package/arm64/linux/lib"

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

cmake \
    -S ${LAYER_DIR} \
    -B ${BUILD_DIR} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_TOOLCHAIN_FILE="${LAYER_DIR}/cmake/toolchain/linux-aarch64-gcc.cmake" \
    -DCMAKE_WARN_DEPRECATED=OFF

cmake --build ${BUILD_DIR} -j$(nproc)

popd

# ----------------------------------------------------------------------------
# Build the release package
if [ "${PACKAGE}" -gt "0" ]; then
    # Setup the package directories
    mkdir -p ${BUILD_DIR_PACK}/lib/linux/arm64

    # Install the 64-bit layer
    cp ${BUILD_DIR}/source/libVkLayerNeuralStatistics.so ${BUILD_DIR_PACK}
    cp ${LAYER_DIR}/vk_layer_settings.txt ${BUILD_DIR_PACK}
    cp ${LAYER_DIR}/manifest.json ${BUILD_DIR_PACK}/layer_neural_statistics.json
fi
