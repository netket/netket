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
Documentation-specific citation utilities for NetKet.

This module provides functions for generating formatted citations for documentation.
"""

import re
from netket._version import __version__
from .core import get_all_citations, register_citation, _citation_registry


def _extract_doi_from_bibtex(bibtex_string: str) -> str | None:
    """
    Extract DOI from a BibTeX entry.

    Args:
        bibtex_string: BibTeX entry as string

    Returns:
        DOI string if found, None otherwise
    """
    # Look for doi field in various formats
    doi_patterns = [
        r"doi\s*=\s*\{([^}]+)\}",
        r'doi\s*=\s*"([^"]+)"',
        r"doi\s*=\s*([^,\s}]+)",
    ]

    for pattern in doi_patterns:
        match = re.search(pattern, bibtex_string, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None


def _extract_url_from_bibtex(bibtex_string: str) -> str | None:
    """
    Extract URL from a BibTeX entry.

    Args:
        bibtex_string: BibTeX entry as string

    Returns:
        URL string if found, None otherwise
    """
    # Look for url field in various formats
    url_patterns = [
        r"url\s*=\s*\{([^}]+)\}",
        r'url\s*=\s*"([^"]+)"',
        r"url\s*=\s*([^,\s}]+)",
    ]

    for pattern in url_patterns:
        match = re.search(pattern, bibtex_string, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None


def _extract_title_from_bibtex(bibtex_string: str) -> str | None:
    """
    Extract title from a BibTeX entry.

    Args:
        bibtex_string: BibTeX entry as string

    Returns:
        Title string if found, None otherwise
    """
    # Look for title field in various formats
    title_patterns = [
        r"title\s*=\s*\{([^}]+)\}",
        r'title\s*=\s*"([^"]+)"',
    ]

    for pattern in title_patterns:
        match = re.search(pattern, bibtex_string, re.IGNORECASE)
        if match:
            # Remove LaTeX formatting
            title = match.group(1).strip()
            title = re.sub(r"\{([^}]*)\}", r"\1", title)  # Remove braces
            return title

    return None


def get_citations_for_docs(format_type: str = "detailed") -> str:
    """
    Generate HTML formatted citations for documentation.

    Args:
        format_type: Either 'detailed' or 'simple'

    Returns:
        HTML string with formatted citations
    """
    # NetKet is already imported at this point since this function is called from within NetKet

    netket_version_parts = __version__.split(".")[:2]
    netket_version = ".".join(netket_version_parts)

    # Register NetKet's own references if not already present
    netket_papers = ["netket3:2021", "netket2:2019"]
    if not any(paper_key in _citation_registry for paper_key in netket_papers):
        register_citation(
            netket_papers, message=f"This work used NetKet {netket_version}"
        )

    all_citations = get_all_citations()
    if not all_citations:
        return '<div class="admonition note"><p class="admonition-title">Note</p><p>No citations registered.</p></div>'

    # Group citations by their group_id
    citation_groups = {}
    for key, info in all_citations.items():
        group_id = info.get("group_id", key)
        if group_id not in citation_groups:
            citation_groups[group_id] = {
                "keys": [],
                "condition": info.get("condition", "").strip(),
                "message": info.get("message", "").strip(),
                "bibtex_entries": [],
                "titles_and_links": [],
            }
        citation_groups[group_id]["keys"].append(key)
        if "bibtex" in info:
            bibtex = info["bibtex"]
            citation_groups[group_id]["bibtex_entries"].append(bibtex)

            # Extract title, DOI, and URL for linking
            title = _extract_title_from_bibtex(bibtex)
            doi = _extract_doi_from_bibtex(bibtex)
            url = _extract_url_from_bibtex(bibtex)

            # Determine link URL (prioritize DOI over URL)
            link_url = None
            if doi:
                link_url = f"https://doi.org/{doi}"
            elif url:
                link_url = url

            citation_groups[group_id]["titles_and_links"].append(
                {"key": key, "title": title, "link_url": link_url}
            )

    html_parts = []
    html_parts.append('<div class="netket-citations">')
    html_parts.append("<h3>Registered Citations</h3>")

    if format_type == "detailed":
        html_parts.append(
            "<p>The following citations have been registered based on your NetKet usage:</p>"
        )

        for group_id, group_info in citation_groups.items():
            html_parts.append('<div class="citation-group">')

            # Show condition if present
            if group_info["condition"]:
                html_parts.append(
                    f'<p><strong>When:</strong> {group_info["condition"]}</p>'
                )

            # Show message if present
            if group_info["message"]:
                html_parts.append(
                    f'<p><strong>Citation message:</strong> {group_info["message"]}</p>'
                )

            # Show citation keys with linked titles if available
            if group_info["titles_and_links"]:
                refs_parts = []
                for entry in group_info["titles_and_links"]:
                    if entry["title"] and entry["link_url"]:
                        refs_parts.append(
                            f'<code>{entry["key"]}</code>: <a href="{entry["link_url"]}" target="_blank">{entry["title"]}</a>'
                        )
                    elif entry["title"]:
                        refs_parts.append(
                            f'<code>{entry["key"]}</code>: {entry["title"]}'
                        )
                    else:
                        refs_parts.append(f'<code>{entry["key"]}</code>')
                html_parts.append(
                    f'<p><strong>References:</strong> {"; ".join(refs_parts)}</p>'
                )
            else:
                keys_str = ", ".join(
                    f"<code>{key}</code>" for key in group_info["keys"]
                )
                html_parts.append(f"<p><strong>References:</strong> {keys_str}</p>")

            # Show BibTeX entries if available
            if group_info["bibtex_entries"]:
                html_parts.append("<details><summary>BibTeX entries</summary>")
                for bibtex in group_info["bibtex_entries"]:
                    html_parts.append(
                        f'<pre><code class="language-bibtex">{bibtex}</code></pre>'
                    )
                html_parts.append("</details>")

            html_parts.append("</div><hr/>")

    else:  # simple format
        html_parts.append("<ul>")
        for group_id, group_info in citation_groups.items():
            keys_str = ", ".join(group_info["keys"])
            condition_str = (
                f" ({group_info['condition']})" if group_info["condition"] else ""
            )
            html_parts.append(f"<li><code>{keys_str}</code>{condition_str}</li>")
        html_parts.append("</ul>")

    html_parts.append("</div>")

    return "\n".join(html_parts)
