#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
# SPDX-License-Identifier: MIT

# ----------------------------------------------------------------------------
# Configuration

# Exit immediately if any component command errors
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAYER_DIR="${SCRIPT_DIR}/.."

BUILD_DIR="${LAYER_DIR}/build_native"

HOST_ARCH_RAW="$(uname -m)"
case "${HOST_ARCH_RAW}" in
    x86_64|amd64)
        PACKAGE_ARCH="x86_64"
        ;;
    aarch64|arm64)
        PACKAGE_ARCH="arm64"
        ;;
    *)
        PACKAGE_ARCH="${HOST_ARCH_RAW}"
        echo "Warning: packaging using unrecognized host architecture '${PACKAGE_ARCH}'" >&2
        ;;
esac

BUILD_DIR_PACK="${LAYER_DIR}/build_package/${PACKAGE_ARCH}/linux/lib"

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
    echo "Building a native ${BUILD_TYPE} build with packaging for ${PACKAGE_ARCH}"
else
    echo "Building a native ${BUILD_TYPE} build without packaging for ${PACKAGE_ARCH}"
fi

# ----------------------------------------------------------------------------
# Build the native layer
mkdir -p "${BUILD_DIR}"
pushd "${BUILD_DIR}" >/dev/null

cmake \
    -S "${LAYER_DIR}" \
    -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_WARN_DEPRECATED=OFF

cmake --build "${BUILD_DIR}" -j"$(nproc)"

popd >/dev/null

# ----------------------------------------------------------------------------
# Build the release package
if [ "${PACKAGE}" -gt "0" ]; then
    # Setup the package directories
    mkdir -p "${BUILD_DIR_PACK}"

    # Install the native layer
    cp "${BUILD_DIR}/source/libVkLayerNeuralStatistics.so" "${BUILD_DIR_PACK}"
    cp "${LAYER_DIR}/vk_layer_settings.txt" "${BUILD_DIR_PACK}"
    cp "${LAYER_DIR}/manifest.json" "${BUILD_DIR_PACK}/layer_neural_statistics.json"
fi
