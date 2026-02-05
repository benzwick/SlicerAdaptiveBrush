#!/usr/bin/env python3
"""Package SlicerAdaptiveBrush extension for distribution.

Creates platform-specific packages (.tar.gz for Linux/macOS, .zip for Windows)
for specified Slicer revisions.
"""

import argparse
import re
import shutil
import tarfile
import zipfile
from pathlib import Path

# Extension metadata
EXTENSION_NAME = "SlicerAdaptiveBrush"
EXTENSION_CATEGORY = "Segmentation"
EXTENSION_DESCRIPTION = (
    "Adaptive brush segment editor effect that segments based on image intensity similarity"
)
EXTENSION_HOMEPAGE = "https://github.com/benzwick/SlicerAdaptiveBrush"
EXTENSION_CONTRIBUTORS = "Ben Zwick"
EXTENSION_DEPENDS = "NA"

# Slicer revision to version mapping
SLICER_VERSIONS = {
    "33241": "5.8.1",
    "34045": "5.10.0",
}


def get_slicer_version(revision: str) -> str:
    """Get Slicer version string for a revision number."""
    return SLICER_VERSIONS.get(revision, f"r{revision}")


def parse_cmake_module_files(cmake_path: Path) -> tuple[list[str], list[str]]:
    """Parse CMakeLists.txt to extract module Python scripts and resources."""
    content = cmake_path.read_text()

    # Extract MODULE_PYTHON_SCRIPTS
    scripts_match = re.search(r"set\(MODULE_PYTHON_SCRIPTS\s+(.*?)\s*\)", content, re.DOTALL)
    scripts = []
    if scripts_match:
        scripts_text = scripts_match.group(1)
        # Replace ${MODULE_NAME} with actual name
        scripts_text = scripts_text.replace("${MODULE_NAME}", "SegmentEditorAdaptiveBrush")
        scripts = [s.strip() for s in scripts_text.split() if s.strip()]

    # Extract MODULE_PYTHON_RESOURCES
    resources_match = re.search(r"set\(MODULE_PYTHON_RESOURCES\s+(.*?)\s*\)", content, re.DOTALL)
    resources = []
    if resources_match:
        resources_text = resources_match.group(1)
        resources_text = resources_text.replace("${MODULE_NAME}", "SegmentEditorAdaptiveBrush")
        resources = [r.strip() for r in resources_text.split() if r.strip()]

    return scripts, resources


def create_s4ext_content(commit_hash: str) -> str:
    """Generate .s4ext extension description file content."""
    icon_url = (
        f"https://raw.githubusercontent.com/benzwick/{EXTENSION_NAME}/{commit_hash}/"
        f"SegmentEditorAdaptiveBrush/Resources/Icons/SegmentEditorAdaptiveBrush.png"
    )

    return f"""scm git
scmurl https://github.com/benzwick/{EXTENSION_NAME}.git
scmrevision {commit_hash}
depends {EXTENSION_DEPENDS}
build_subdirectory .
homepage {EXTENSION_HOMEPAGE}
contributors {EXTENSION_CONTRIBUTORS}
category {EXTENSION_CATEGORY}
iconurl {icon_url}
status Development
description {EXTENSION_DESCRIPTION}
"""


def get_slicer_major_minor(revision: str) -> str:
    """Get Slicer major.minor version for a revision number (e.g., '5.10')."""
    version = get_slicer_version(revision)
    parts = version.split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return version


def get_platform_paths(platform: str, revision: str, base_dir: Path) -> tuple[Path, Path]:
    """Get platform-specific paths for modules and share directories.

    Returns:
        Tuple of (modules_dir, share_dir) where:
        - modules_dir: Where qt-scripted-modules go
        - share_dir: Where .s4ext file goes
    """
    slicer_version = get_slicer_major_minor(revision)

    if platform == "macos":
        # macOS structure: Slicer.app/Contents/Extensions-{rev}/{ExtName}/lib/...
        ext_root = base_dir / "Slicer.app" / "Contents" / f"Extensions-{revision}" / EXTENSION_NAME
        modules_dir = ext_root / "lib" / f"Slicer-{slicer_version}" / "qt-scripted-modules"
        share_dir = ext_root / "share" / f"Slicer-{slicer_version}"
    else:
        # Linux/Windows structure: lib/Slicer-X.Y/qt-scripted-modules/
        modules_dir = base_dir / "lib" / f"Slicer-{slicer_version}" / "qt-scripted-modules"
        share_dir = base_dir / "share" / f"Slicer-{slicer_version}"

    return modules_dir, share_dir


def copy_module_files(src_dir: Path, modules_dir: Path) -> None:
    """Copy module files according to CMakeLists.txt configuration.

    Slicer expects scripted modules directly in qt-scripted-modules/:
    - qt-scripted-modules/ModuleName.py
    - qt-scripted-modules/ModuleNameLib/...
    - qt-scripted-modules/Resources/...

    Args:
        src_dir: Source directory containing SegmentEditorAdaptiveBrush module
        modules_dir: Destination qt-scripted-modules directory
    """
    module_dir = src_dir / "SegmentEditorAdaptiveBrush"
    cmake_path = module_dir / "CMakeLists.txt"

    if not cmake_path.exists():
        raise FileNotFoundError(f"CMakeLists.txt not found at {cmake_path}")

    scripts, resources = parse_cmake_module_files(cmake_path)

    # Copy Python scripts directly to modules_dir (not a subdirectory)
    for script in scripts:
        src_file = module_dir / script
        dest_file = modules_dir / script
        if src_file.exists():
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dest_file)
        else:
            print(f"Warning: Script not found: {src_file}")

    # Copy resources directly to modules_dir
    for resource in resources:
        src_file = module_dir / resource
        dest_file = modules_dir / resource
        if src_file.exists():
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dest_file)
        else:
            print(f"Warning: Resource not found: {src_file}")


def create_package(
    platform: str,
    revision: str,
    version: str,
    commit_hash: str,
    src_dir: Path,
    output_dir: Path,
) -> Path:
    """Create a package for the specified platform and Slicer revision.

    Creates platform-specific directory structures:
    - Linux/Windows: lib/Slicer-X.Y/qt-scripted-modules/
    - macOS: Slicer.app/Contents/Extensions-{rev}/{ExtName}/lib/Slicer-X.Y/qt-scripted-modules/
    """
    slicer_version = get_slicer_version(revision)
    package_name = f"{EXTENSION_NAME}-{slicer_version}-{revision}-{platform}-{version}"

    # Create temporary staging directory
    staging_dir = output_dir / f"staging-{platform}-{revision}"
    package_content_dir = staging_dir / EXTENSION_NAME

    # Clean up any existing staging directory
    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    package_content_dir.mkdir(parents=True)

    # Get platform-specific paths
    modules_dir, share_dir = get_platform_paths(platform, revision, package_content_dir)

    # Create directories
    modules_dir.mkdir(parents=True, exist_ok=True)
    share_dir.mkdir(parents=True, exist_ok=True)

    # Copy module files to platform-specific location
    copy_module_files(src_dir, modules_dir)

    # Create .s4ext file in share directory
    s4ext_content = create_s4ext_content(commit_hash)
    s4ext_path = share_dir / f"{EXTENSION_NAME}.s4ext"
    s4ext_path.write_text(s4ext_content)

    # Create archive
    output_dir.mkdir(parents=True, exist_ok=True)

    if platform == "windows":
        archive_path = output_dir / f"{package_name}.zip"
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in package_content_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(staging_dir)
                    zf.write(file_path, arcname)
    else:
        archive_path = output_dir / f"{package_name}.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tf:
            tf.add(package_content_dir, arcname=EXTENSION_NAME)

    # Clean up staging directory
    shutil.rmtree(staging_dir)

    print(f"Created: {archive_path}")
    return archive_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Package SlicerAdaptiveBrush extension")
    parser.add_argument(
        "--platform",
        required=True,
        choices=["linux", "macos", "windows"],
        help="Target platform",
    )
    parser.add_argument(
        "--slicer-revisions",
        required=True,
        help="Comma-separated list of Slicer revisions",
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Extension version (e.g., 1.0.0)",
    )
    parser.add_argument(
        "--commit-hash",
        required=True,
        help="Git commit hash",
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Build date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for packages",
    )

    args = parser.parse_args()

    # Get source directory (repository root)
    script_dir = Path(__file__).parent.resolve()
    src_dir = script_dir.parent.parent  # .github/scripts -> repo root

    output_dir = Path(args.output_dir).resolve()
    revisions = [r.strip() for r in args.slicer_revisions.split(",")]

    print(f"Packaging {EXTENSION_NAME} v{args.version}")
    print(f"Platform: {args.platform}")
    print(f"Slicer revisions: {revisions}")
    print(f"Commit: {args.commit_hash[:8]}")
    print(f"Date: {args.date}")
    print()

    for revision in revisions:
        create_package(
            platform=args.platform,
            revision=revision,
            version=args.version,
            commit_hash=args.commit_hash,
            src_dir=src_dir,
            output_dir=output_dir,
        )

    print()
    print("Packaging complete!")


if __name__ == "__main__":
    main()
