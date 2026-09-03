#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Arm Limited
# SPDX-License-Identifier: MIT

"""Generate a curated CycloneDX SBOM for release package artifacts.

The distributed package is a small native binary archive, so scanner-only SBOMs
do not see the source repositories and generated inputs that went into it. This
script records the archive, files inside the archive, and the known build inputs
from the assembled libGPULayers workspace.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from cyclonedx.model import (
    ExternalReference,
    ExternalReferenceType,
    HashAlgorithm,
    HashType,
    Property,
    XsUri,
)
from cyclonedx.model.bom import Bom, BomMetaData, Tool
from cyclonedx.model.bom_ref import BomRef
from cyclonedx.model.component import Component, ComponentScope, ComponentType
from cyclonedx.model.dependency import Dependency
from cyclonedx.model.license import DisjunctiveLicense, LicenseExpression
from cyclonedx.output import OutputFormat, SchemaVersion, make_outputter


SCRIPT_VERSION = "1"
SBOM_DISCLAIMER = (
    "THIS SOFTWARE BILL OF MATERIALS (SBOM) IS PROVIDED BY ARM LIMITED "
    "\"AS IS\" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT "
    "LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A "
    "PARTICULAR PURPOSE, AND NONINFRINGEMENT ARE DISCLAIMED. IN NO EVENT "
    "SHALL ARM LIMITED BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, "
    "SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED "
    "TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR "
    "PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF "
    "LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING "
    "NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SBOM, "
    "EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
)


def run_command(args: list[str], cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            args,
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def git_value(repo: Path, *args: str) -> str | None:
    if not (repo / ".git").exists():
        return None
    return run_command(["git", *args], cwd=repo)


def git_commit(repo: Path) -> str | None:
    return git_value(repo, "rev-parse", "HEAD")


def git_remote(repo: Path) -> str | None:
    return git_value(repo, "config", "--get", "remote.origin.url")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_project_version(source_root: Path) -> str | None:
    cmake = source_root / "CMakeLists.txt"
    if not cmake.exists():
        return None
    match = re.search(
        r"project\s*\(\s*VkLayerNeuralStatistics\s+VERSION\s+([^\s\)]+)",
        cmake.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    return match.group(1) if match else None


def normalize_url(url: str | None) -> str | None:
    if not url:
        return None
    if url.startswith("git@github.com:"):
        return "https://github.com/" + url.removeprefix("git@github.com:")
    return url.removesuffix(".git") if url.startswith("https://") else url


def safe_ref(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "-", text).strip("-")


def bom_ref(value: str) -> BomRef:
    return BomRef(value)


def license_id(identifier: str) -> LicenseExpression:
    return LicenseExpression(identifier)


def license_name(name: str) -> DisjunctiveLicense:
    return DisjunctiveLicense(name=name)


def property_item(name: str, value: str | None) -> Property | None:
    if value is None or value == "":
        return None
    return Property(name=name, value=value)


def properties(items: list[tuple[str, str | None]]) -> list[Property]:
    return [
        item
        for item in (property_item(name, value) for name, value in items)
        if item is not None
    ]


COMPONENT_TYPES = {
    "application": ComponentType.APPLICATION,
    "file": ComponentType.FILE,
    "library": ComponentType.LIBRARY,
}


def component_scope(value: str | None) -> ComponentScope | None:
    if value is None:
        return None
    return {
        "excluded": ComponentScope.EXCLUDED,
        "required": ComponentScope.REQUIRED,
    }[value]


def hash_items(items: list[dict[str, str]] | None) -> list[HashType] | None:
    if items is None:
        return None
    return [
        HashType(alg=HashAlgorithm.SHA_256, content=item["content"])
        for item in items
        if item["alg"] == "SHA-256"
    ]


def component(
    *,
    name: str,
    bom_ref: str,
    component_type: str,
    version: str | None = None,
    scope: str | None = None,
    description: str | None = None,
    licenses: list[LicenseExpression | DisjunctiveLicense] | None = None,
    hashes: list[dict[str, str]] | None = None,
    external_refs: list[ExternalReference] | None = None,
    props: list[Property] | None = None,
) -> Component:
    return Component(
        name=name,
        bom_ref=BomRef(bom_ref),
        type=COMPONENT_TYPES[component_type],
        version=version,
        scope=component_scope(scope),
        description=description,
        licenses=licenses,
        hashes=hash_items(hashes),
        external_references=external_refs,
        properties=props,
    )


def vcs_ref(url: str | None) -> list[ExternalReference]:
    normalized = normalize_url(url)
    if not normalized:
        return []
    return [ExternalReference(type=ExternalReferenceType.SCM, url=XsUri(normalized))]


def get_archive_files(package: Path) -> list[Component]:
    refs: list[Component] = []

    if tarfile.is_tarfile(package):
        with tarfile.open(package, "r:*") as archive:
            for member in sorted(archive.getmembers(), key=lambda item: item.name):
                if not member.isfile():
                    continue
                file_obj = archive.extractfile(member)
                if file_obj is None:
                    continue
                data = file_obj.read()
                refs.append(
                    component(
                        name=member.name,
                        bom_ref=f"file:{safe_ref(member.name)}",
                        component_type="file",
                        scope="required",
                        hashes=[{"alg": "SHA-256", "content": sha256_bytes(data)}],
                        props=properties(
                            [
                                ("arm:sbom:archiveMember", "true"),
                                ("arm:sbom:size", str(member.size)),
                            ]
                        ),
                    )
                )
        return refs

    if zipfile.is_zipfile(package):
        with zipfile.ZipFile(package) as archive:
            for info in sorted(archive.infolist(), key=lambda item: item.filename):
                if info.is_dir():
                    continue
                data = archive.read(info)
                refs.append(
                    component(
                        name=info.filename,
                        bom_ref=f"file:{safe_ref(info.filename)}",
                        component_type="file",
                        scope="required",
                        hashes=[{"alg": "SHA-256", "content": sha256_bytes(data)}],
                        props=properties(
                            [
                                ("arm:sbom:archiveMember", "true"),
                                ("arm:sbom:size", str(info.file_size)),
                            ]
                        ),
                    )
                )
        return refs

    raise ValueError(f"Unsupported package archive: {package}")


def tool_version(command: str) -> str | None:
    if shutil.which(command) is None:
        return None
    output = run_command([command, "--version"])
    return output.splitlines()[0] if output else None


def ndk_version() -> tuple[str | None, str | None]:
    ndk_root = (
        os.environ.get("ANDROID_NDK_HOME")
        or os.environ.get("ANDROID_NDK_ROOT")
        or os.environ.get("ANDROID_NDK_LATEST_HOME")
    )
    if not ndk_root:
        return None, None

    source_properties = Path(ndk_root) / "source.properties"
    if not source_properties.exists():
        return ndk_root, None

    for line in source_properties.read_text(encoding="utf-8").splitlines():
        if line.startswith("Pkg.Revision"):
            return ndk_root, line.split("=", 1)[1].strip()
    return ndk_root, None


def repo_component(
    *,
    name: str,
    bom_ref: str,
    repo: Path,
    license_expr: str,
    description: str,
) -> Component:
    commit = git_commit(repo)
    remote = git_remote(repo)
    return component(
        name=name,
        bom_ref=bom_ref,
        component_type="library",
        version=commit,
        scope="required",
        description=description,
        licenses=[license_id(license_expr)],
        external_refs=vcs_ref(remote),
        props=properties(
            [
                ("arm:sbom:sourcePath", str(repo)),
                ("arm:sbom:gitCommit", commit),
                ("arm:sbom:gitRemote", normalize_url(remote)),
            ]
        ),
    )


def build_tool_components(target: str) -> list[Component]:
    components: list[Component] = []

    cmake_version = tool_version("cmake")
    if cmake_version:
        components.append(
            component(
                name="CMake",
                bom_ref="tool:cmake",
                component_type="application",
                version=cmake_version,
                scope="excluded",
                description="Build system used to configure the native layer.",
                licenses=[license_id("BSD-3-Clause")],
            )
        )

    if target.startswith("linux"):
        gcc_version = tool_version("aarch64-linux-gnu-g++")
        if gcc_version:
            components.append(
                component(
                    name="GNU aarch64 cross compiler",
                    bom_ref="tool:aarch64-linux-gnu-g++",
                    component_type="application",
                    version=gcc_version,
                    scope="excluded",
                    description=(
                        "Linux aarch64 cross compiler used to build the "
                        "distributed shared object."
                    ),
                )
            )

    if target.startswith("android"):
        ndk_root, version = ndk_version()
        components.append(
            component(
                name="Android NDK",
                bom_ref="tool:android-ndk",
                component_type="application",
                version=version,
                scope="excluded",
                description=(
                    "Android native toolchain used to build the distributed "
                    "shared object."
                ),
                licenses=[license_name("Android Software Development Kit License")],
                props=properties([("arm:sbom:sourcePath", ndk_root)]),
            )
        )

    return components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a curated CycloneDX SBOM for a release artifact"
    )
    parser.add_argument(
        "--package",
        required=True,
        type=Path,
        help="Distributed package archive, e.g. dist/VkLayer...tar.gz",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Package target, e.g. linux-arm64 or android-arm64-v8a",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output SBOM path. Defaults to PACKAGE with .cdx.json suffix.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path.cwd(),
        help="Layer source root. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--libgpu-layers-root",
        type=Path,
        default=Path.cwd().parent,
        help="Assembled libGPULayers checkout root. Defaults to parent dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package = args.package.resolve()
    source_root = args.source_root.resolve()
    libgpu_root = args.libgpu_layers_root.resolve()
    output = args.output or package.with_suffix("").with_suffix(".cdx.json")

    if not package.exists():
        raise FileNotFoundError(package)
    if not source_root.exists():
        raise FileNotFoundError(source_root)
    if not libgpu_root.exists():
        raise FileNotFoundError(libgpu_root)

    project_version = read_project_version(source_root) or os.environ.get(
        "GITHUB_REF_NAME"
    )
    source_commit = git_commit(source_root) or os.environ.get("GITHUB_SHA")
    source_remote = git_remote(source_root)
    if source_remote is None and os.environ.get("GITHUB_REPOSITORY"):
        source_remote = (
            os.environ.get("GITHUB_SERVER_URL", "https://github.com")
            + "/"
            + os.environ["GITHUB_REPOSITORY"]
        )

    package_hash = sha256_file(package)
    package_ref = f"artifact:{safe_ref(package.name)}"
    root_ref = f"pkg:generic/VkLayerNeuralStatistics@{project_version or 'unknown'}"
    source_ref = "source:vulkan-layer-neural-statistics"
    generated_ref = "source:generated-vulkan-utility-sources"

    archive_files = get_archive_files(package)
    base_components = [
        component(
            name=package.name,
            bom_ref=package_ref,
            component_type="file",
            scope="required",
            version=f"sha256:{package_hash}",
            description="Distributed release archive.",
            hashes=[{"alg": "SHA-256", "content": package_hash}],
            props=properties(
                [
                    ("arm:sbom:target", args.target),
                    ("arm:sbom:size", str(package.stat().st_size)),
                ]
            ),
        ),
        component(
            name="vulkan_layer_neural_statistics",
            bom_ref=source_ref,
            component_type="application",
            version=source_commit or project_version,
            scope="required",
            description="Source repository for the Vulkan neural statistics layer.",
            licenses=[license_id("MIT")],
            external_refs=vcs_ref(source_remote),
            props=properties(
                [
                    ("arm:sbom:sourcePath", str(source_root)),
                    ("arm:sbom:gitCommit", source_commit),
                ]
            ),
        ),
        repo_component(
            name="libGPULayers",
            bom_ref="source:libGPULayers",
            repo=libgpu_root,
            license_expr="MIT",
            description="Layer framework and build integration used by the package.",
        ),
        repo_component(
            name="Vulkan-Headers",
            bom_ref="source:vulkan-headers",
            repo=libgpu_root / "source_third_party/khronos/vulkan",
            license_expr="Apache-2.0",
            description="Khronos Vulkan headers and registry used by the build.",
        ),
        repo_component(
            name="Vulkan-Utility-Libraries",
            bom_ref="source:vulkan-utility-libraries",
            repo=libgpu_root / "source_third_party/khronos/vulkan-utilities",
            license_expr="Apache-2.0",
            description="Khronos Vulkan utility library sources used by the build.",
        ),
        component(
            name="Generated Vulkan utility sources",
            bom_ref=generated_ref,
            component_type="library",
            scope="required",
            description=(
                "Generated Vulkan utility sources produced from the Khronos "
                "Vulkan registry before compiling the layer."
            ),
            licenses=[license_id("Apache-2.0")],
        ),
    ]

    tool_components = build_tool_components(args.target)
    all_components = base_components + archive_files + tool_components

    tool_refs = [str(item.bom_ref) for item in tool_components]
    archive_file_refs = [str(item.bom_ref) for item in archive_files]

    root_component = component(
        name=f"VkLayerNeuralStatistics-{args.target}",
        bom_ref=root_ref,
        component_type="application",
        version=project_version,
        scope="required",
        description=(
            "Release package for the VkLayerNeuralStatistics Vulkan "
            f"layer targeting {args.target}."
        ),
        licenses=[license_id("MIT")],
        hashes=[{"alg": "SHA-256", "content": package_hash}],
        props=properties(
            [
                ("arm:sbom:target", args.target),
                ("arm:sbom:artifact", package.name),
                ("arm:sbom:creator", "Arm Limited"),
                ("arm:sbom:disclaimer", SBOM_DISCLAIMER),
            ]
        ),
    )

    bom = Bom(
        metadata=BomMetaData(
            timestamp=datetime.now(timezone.utc),
            tools=[
                Tool(
                    vendor="Arm Limited",
                    name="generate-release-sbom.py",
                    version=SCRIPT_VERSION,
                )
            ],
            component=root_component,
        ),
        components=all_components,
        dependencies=[
            Dependency(
                ref=bom_ref(root_ref),
                dependencies=[
                    Dependency(ref=bom_ref(dep_ref))
                    for dep_ref in [
                        package_ref,
                        source_ref,
                        "source:libGPULayers",
                        "source:vulkan-headers",
                        "source:vulkan-utility-libraries",
                        generated_ref,
                        *tool_refs,
                    ]
                ],
            ),
            Dependency(
                ref=bom_ref(package_ref),
                dependencies=[
                    Dependency(ref=bom_ref(dep_ref)) for dep_ref in archive_file_refs
                ],
            ),
            Dependency(
                ref=bom_ref(source_ref),
                dependencies=[
                    Dependency(ref=bom_ref(dep_ref))
                    for dep_ref in [
                        "source:libGPULayers",
                        "source:vulkan-headers",
                        "source:vulkan-utility-libraries",
                        generated_ref,
                    ]
                ],
            ),
            Dependency(
                ref=bom_ref(generated_ref),
                dependencies=[
                    Dependency(ref=bom_ref("source:vulkan-headers")),
                    Dependency(ref=bom_ref("source:vulkan-utility-libraries")),
                ],
            ),
        ],
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    outputter = make_outputter(bom, OutputFormat.JSON, SchemaVersion.V1_6)
    output.write_text(outputter.output_as_string() + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
