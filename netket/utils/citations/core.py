# Copyright 2025 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Core citation system implementation for NetKet.

This module provides the internal implementation for citation tracking and bibliography generation.
"""

import re
from pathlib import Path

from netket._version import __version__

from rich.console import Console
from rich.text import Text

# Global citation registry
_citation_registry: dict[str, dict[str, str]] = {}


def _parse_bibtex_entry(bibtex_string: str) -> dict[str, str]:
    """
    Parse a simple BibTeX entry to extract key information.

    Args:
        bibtex_string: BibTeX entry as string

    Returns:
        Dictionary with parsed BibTeX information

    Raises:
        ValueError: If BibTeX entry is malformed
    """
    bibtex_string = bibtex_string.strip()

    if not bibtex_string.startswith("@"):
        raise ValueError("BibTeX entry must start with '@'")

    # Extract entry type and key
    first_line_match = re.match(r"@(\w+)\s*\{\s*([^,\s]+)", bibtex_string)
    if not first_line_match:
        raise ValueError("Malformed BibTeX entry: cannot parse entry type and key")

    entry_type = first_line_match.group(1).lower()
    key = first_line_match.group(2)

    # Extract fields (simple parsing)
    fields = {}
    field_pattern = r"(\w+)\s*=\s*\{([^}]*)\}"
    for match in re.finditer(field_pattern, bibtex_string):
        field_name = match.group(1).lower()
        field_value = match.group(2)
        fields[field_name] = field_value

    return {"type": entry_type, "key": key, "bibtex": bibtex_string, **fields}


def _load_centralized_bib() -> dict[str, dict[str, str]]:
    """
    Load the internal citations bibliography file.

    Returns:
        Dictionary mapping citation keys to their BibTeX information
    """
    # Find the internal_citations.bib file in the same directory
    current_file = Path(__file__)
    citations_file = current_file.parent / "internal_citations.bib"

    if not citations_file.exists():
        return {}

    citations = {}
    try:
        with open(citations_file, encoding="utf-8") as f:
            content = f.read()

        # Split into individual entries
        entries = re.split(r"\n(?=@)", content)
        for entry in entries:
            entry = entry.strip()
            if entry and entry.startswith("@"):
                try:
                    parsed = _parse_bibtex_entry(entry)
                    citations[parsed["key"]] = parsed
                except ValueError:
                    # Skip malformed entries
                    continue
    except (OSError, UnicodeDecodeError):
        # If we can't read the file, return empty dict
        pass

    return citations


def _validate_citation_key(citation_key: str) -> bool:
    """
    Validate that a citation key exists in the internal bibliography.

    Args:
        citation_key: The citation key to validate

    Returns:
        True if key exists, False otherwise
    """
    centralized_bib = _load_centralized_bib()
    return citation_key in centralized_bib


def register_citation(
    citation: str | list[str], condition: str | None = None, message: str | None = None
) -> None:
    """
    Register citation(s) in the global registry.

    Args:
        citation: Either a citation key, full BibTeX entry, or list of them
        condition: Optional condition for when to cite this reference
        message: Optional custom citation message (defaults to "This work used the methods described in Ref.")
    """
    citations_list = citation if isinstance(citation, list) else [citation]

    # Create a unique group key for this registration
    import uuid

    group_id = str(uuid.uuid4())

    citation_keys = []

    for single_citation in citations_list:
        if single_citation.strip().startswith("@"):
            # Full BibTeX entry
            try:
                parsed = _parse_bibtex_entry(single_citation)
                key = parsed["key"]
                citation_keys.append(key)
                _citation_registry[key] = {
                    "bibtex": single_citation,
                    "condition": condition or "",
                    "message": message
                    or "This work used the methods described in Ref.~",
                    "type": "bibtex",
                    "group_id": group_id,
                }
            except ValueError as e:
                raise ValueError(f"Invalid BibTeX entry: {e}")
        else:
            # Citation key - store without validation
            citation_keys.append(single_citation)
            _citation_registry[single_citation] = {
                "key": single_citation,
                "condition": condition or "",
                "message": message or "This work used the methods described in Ref.~",
                "type": "key",
                "group_id": group_id,
            }

    # Store the group information
    if len(citation_keys) > 1:
        for key in citation_keys:
            _citation_registry[key]["group_keys"] = citation_keys


def reference(
    citation: str | list[str], condition: str | None = None, message: str | None = None
):
    """
    Decorator to add citation tracking to functions and classes.

    This decorator registers the citation(s) and returns the original function/class unchanged.

    Args:
        citation: Either a citation key, full BibTeX entry, or list of them
        condition: Optional condition description for when to cite
        message: Optional custom citation message

    Example:
        @reference("netket3:2021")
        def some_function():
            pass

        @reference(["key1", "key2"], condition="when using multiple methods")
        def another_function():
            pass

        @reference("@article{key, ...}", condition="when using momentum",
                   message="This work implemented the algorithm from")
        def third_function():
            pass
    """

    def decorator(obj):
        # Register the citation(s) - this is all we need to do
        register_citation(citation, condition, message)

        # Return the original function/class unchanged
        return obj

    return decorator


def get_all_citations() -> dict[str, dict[str, str]]:
    """
    Get all registered citations.

    Returns:
        Dictionary mapping citation keys to their information
    """
    all_citations = {}
    centralized_bib = _load_centralized_bib()

    for key, citation_info in _citation_registry.items():
        citation_info = citation_info.copy()

        # If it's a key reference, get the BibTeX from centralized file
        if citation_info.get("type") == "key" and key in centralized_bib:
            citation_info["bibtex"] = centralized_bib[key]["bibtex"]
            citation_info.update(centralized_bib[key])
        elif citation_info.get("type") == "bibtex":
            # For BibTeX entries, parse and add the fields
            try:
                parsed = _parse_bibtex_entry(citation_info["bibtex"])
                citation_info.update(parsed)
            except ValueError:
                # If parsing fails, just keep the original info
                pass

        all_citations[key] = citation_info

    return all_citations


def cite(
    output_file: str | None = None, bib: bool = False, from_cli: bool = False
) -> None:
    """
    Display or export all known citations from decorated functions and classes.

    Args:
        output_file: If provided, write BibTeX to this file. Otherwise print to console.
        bib: If True, generate a 'references.bib' file in the current directory.
        from_cli: If True, show CLI-appropriate instructions instead of Python API instructions.
    """

    netket_version_parts = __version__.split(".")[:2]  # Get major.minor
    netket_version = ".".join(netket_version_parts)

    # Register NetKet's own references as a group if not already present
    netket_papers = ["netket3:2021", "netket2:2019"]
    if not any(paper_key in _citation_registry for paper_key in netket_papers):
        register_citation(
            netket_papers, message=f"This work used NetKet {netket_version}"
        )

    all_citations = get_all_citations()
    if not all_citations:
        console = Console()
        console.print("[yellow]No citations registered.[/yellow]")
        return

    # Handle bib file generation
    bib_filename = output_file or ("references.bib" if bib else None)
    if bib_filename:
        # Write BibTeX file
        with open(bib_filename, "w", encoding="utf-8") as f:
            # Write all citations
            for key, info in all_citations.items():
                if "bibtex" in info:
                    f.write(info["bibtex"] + "\n\n")

        console = Console()
        console.print(
            f"[green]✓[/green] BibTeX file saved to [bold]{bib_filename}[/bold]"
        )

    # Display formatted citation instructions
    console = Console()

    # Group citations by their group_id to handle multiple refs in single registration
    citation_groups = {}

    for key, info in all_citations.items():
        group_id = info.get("group_id", key)  # Use key as fallback group_id
        if group_id not in citation_groups:
            citation_groups[group_id] = {
                "keys": [],
                "condition": info.get("condition", "").strip(),
                "message": info.get(
                    "message", "This work used the methods described in Ref.~"
                ),
            }
        citation_groups[group_id]["keys"].append(key)

    # Separate groups into those with and without conditions
    unconditional_citations = []
    conditional_citations = []

    for group_id, group_info in citation_groups.items():
        keys = group_info["keys"]
        condition = group_info["condition"]
        message = group_info["message"]

        # Create citation with multiple keys if needed
        citation_keys = ",".join(keys)
        citation_text = f"{message}~\\cite{{{citation_keys}}}."

        if not condition:
            # No condition - add to unconditional list
            unconditional_citations.append(citation_text)
        else:
            # Has condition - add to conditional list
            condition_comment = f"%% * {condition}"
            conditional_citations.append(f"{condition_comment}\n% {citation_text}")

    # Build LaTeX content starting with unconditional citations
    latex_content_parts = []

    if unconditional_citations:
        latex_content_parts.append("\n\n".join(unconditional_citations))

    if conditional_citations:
        latex_content_parts.append("\n".join(conditional_citations))

    latex_content = "\n\n".join(latex_content_parts)

    # Create Rich formatted output with improved colors
    title = Text("Code and Algorithmic Acknowledgements", style="bold bright_magenta")

    # Only show instruction if we're displaying citations (not writing to file)
    instruction = Text()
    if from_cli:
        instruction.append("💡 Call with ", style="bright_green")
        instruction.append("netket-cite --bib", style="bold bright_yellow")
        instruction.append(
            " to generate the accompanying bib file in the current directory",
            style="bright_green",
        )
    else:
        instruction.append("💡 Call with ", style="bright_green")
        instruction.append(".cite(bib=True)", style="bold bright_yellow")
        instruction.append(
            " to generate the accompanying bib file in the current directory",
            style="bright_green",
        )

    # Display everything
    console.print()
    console.print(title, justify="center")
    console.print()
    console.print(
        "If you use NetKet for your research, we ask that you acknowledge all relevant publications in your publication.",
        style="bright_cyan",
    )
    console.print(
        "This includes both the NetKet library itself, as well as any specific algorithm and implementation you use.",
        style="bright_cyan",
    )
    console.print(
        "See more information at https://www.netket.org/cite/ ",
        style="bright_cyan",
    )
    console.print()
    console.print(instruction)
    console.print()

    # Print the LaTeX content with indentation (no box for easy copy-paste)
    for line in latex_content.split("\n"):
        if line.strip():  # Only indent non-empty lines
            console.print(f"    {line}")
        else:
            console.print()  # Empty line

    console.print()


def validate_references() -> bool:
    """
    Validate all registered references against the internal bibliography.

    Returns:
        True if all references are valid, False otherwise
    """
    centralized_bib = _load_centralized_bib()
    all_valid = True

    for key, info in _citation_registry.items():
        if info.get("type") == "key":
            if key not in centralized_bib:
                print(f"ERROR: Citation key '{key}' not found in internal bibliography")
                all_valid = False
        elif info.get("type") == "bibtex":
            try:
                _parse_bibtex_entry(info["bibtex"])
            except ValueError as e:
                print(f"ERROR: Invalid BibTeX for key '{key}': {e}")
                all_valid = False

    return all_valid
