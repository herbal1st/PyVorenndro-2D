"""
Bundling utility for aggregating logic and specific data files for LLM analysis.
"""

import os
from typing import List, Set, Optional

# --- Configuration ---

# Set to a filename to bundle only that specific file.
TARGET_SPECIFIC_FILE: Optional[str] = ""  # Name

# Set to a folder name to bundle only files within that directory.
TARGET_SPECIFIC_FOLDER: Optional[str] = ""  # Name

# The name of the resulting text bundle.
OUTPUT_BUNDLE_NAME: str = "codebase.txt"  # Name

# Extension Whitelist: Only logic files are processed by default.
EXTENSION_WHITELIST: Set[str] = {".py", ".md", ".txt", ".yaml"}  # Filter

# File Whitelist: Explicitly permit these files regardless of extension.
FILE_EXCEPTIONS_WHITELIST: Set[str] = {
    "voxels.yaml",
    "brain_library.yaml",
    "README.txt",
    "MANUAL.txt",
    "DEVELOPER.md"
}  # Filter

# Filename Blacklist: Globally ignored files.
FILENAME_BLACKLIST: Set[str] = {
    "_bundle_full.py",
    "_bundle_core.py",
    "_clean_up.py",
    "_get_folder_tree.py",
    "_get_import_tree.py",
    "_line_counter.py",
    "_txt_to_pdf.py"
}  # Filter

# Directory Blacklist: Folders to skip entirely.
DIRECTORY_BLACKLIST: Set[str] = {
    "__pycache__",
}  # Filter


def bundle_codebase() -> None:
    """
    Orchestrates the discovery and serialization of files into a text bundle.
    """
    script_directory: str = os.path.dirname(os.path.abspath(__file__))
    output_path: str = os.path.join(script_directory, OUTPUT_BUNDLE_NAME)

    print(f"Bundling codebase logic into: {OUTPUT_BUNDLE_NAME}")

    resolved_files: List[str] = _gather_file_paths()

    if not resolved_files:
        print("No files matched the current filtering criteria.")
        return

    _write_bundle_to_disk(output_path, resolved_files)
    print(f"Aggregation complete. {len(resolved_files)} files processed.")


def _gather_file_paths() -> List[str]:
    """
    Walks the project structure to find files passing all filters.
    """
    valid_paths: List[str] = []

    for current_root, directories, filenames in os.walk("."):
        # Prune blacklisted directories to prevent recursive entry
        directories[:] = [
            directory_name
            for directory_name in directories
            if directory_name not in DIRECTORY_BLACKLIST
        ]

        # Skip logic if we are not within the target folder scope
        if not _is_within_target_scope(current_root):
            continue

        for filename in filenames:
            file_path: str = os.path.join(current_root, filename)
            if _should_include_file(filename):
                valid_paths.append(file_path)

    return valid_paths


def _is_within_target_scope(current_root: str) -> bool:
    """
    Checks if the current path resides within the configured target folder.
    """
    if not TARGET_SPECIFIC_FOLDER:
        return True

    normalized_path: str = os.path.normpath(current_root)
    path_segments: List[str] = normalized_path.split(os.sep)

    return TARGET_SPECIFIC_FOLDER in path_segments


def _should_include_file(filename: str) -> bool:
    """
    Determines if a file is allowed based on inclusion and exclusion rules.
    """
    # Specific file override takes precedence
    if TARGET_SPECIFIC_FILE:
        return filename == TARGET_SPECIFIC_FILE

    # Priority inclusion for specific data files
    if filename in FILE_EXCEPTIONS_WHITELIST:
        return True

    # Validate extension
    _, file_extension = os.path.splitext(filename)
    if file_extension not in EXTENSION_WHITELIST:
        return False

    # Filter out blacklisted names
    if filename in FILENAME_BLACKLIST:
        return False

    # Ignore hidden system files
    if filename.startswith("."):
        return False

    return True


def _write_bundle_to_disk(
    destination_path: str, source_file_list: List[str]
) -> None:
    """
    Iterates through resolved paths and writes content to the output stream.
    """
    try:
        with open(destination_path, "w", encoding="utf-8") as bundle_stream:
            for source_path in source_file_list:
                _process_and_append_file(bundle_stream, source_path)
    except Exception as write_error:
        print(f"Critical failure during bundle write: {write_error}")


def _process_and_append_file(output_stream, source_path: str) -> None:
    """
    Reads a file's content and appends it to the bundle with a header.
    """
    try:
        with open(source_path, "r", encoding="utf-8", errors="ignore") as f:
            file_content: str = f.read()
            output_stream.write(f"\n--- FILE: {source_path} ---\n")
            output_stream.write(file_content)
            output_stream.write("\n")
    except Exception as read_error:
        print(f"Warning: Skipping {source_path} due to error: {read_error}")


if __name__ == "__main__":
    bundle_codebase()
    input("\nProcess finished! Press Enter to close...")